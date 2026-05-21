"""
scripts/train_lora.py

Qwen3-1.7B LoRA 微调脚本（面向 A100 服务器，单卡或少量卡）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
依赖安装（在服务器上执行一次）：

    pip install transformers>=4.45.0 peft>=0.12.0 trl>=0.11.0 \
                accelerate>=0.34.0 datasets>=2.20.0

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
用法（在项目根目录执行）：

  # 单卡，使用第 0 张 GPU
  CUDA_VISIBLE_DEVICES=0 python scripts/train_lora.py

  # 指定任意一张空闲 GPU（查看空闲情况：nvidia-smi）
  CUDA_VISIBLE_DEVICES=3 python scripts/train_lora.py

  # 双卡（抢到两张空卡时可用，速度约 2×）
  CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes=2 scripts/train_lora.py

  # 常用参数示例
  CUDA_VISIBLE_DEVICES=0 python scripts/train_lora.py \\
      --base-model Qwen/Qwen3-1.7B \\  # 也可以是本地路径
      --output output/lora_v1 \\
      --epochs 5 \\
      --lr 1e-4

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
资源估算（单张 A100-40GB）：

  - 显存占用：~8-10 GB（1.7B bf16 模型 + LoRA 参数 + 激活值）
  - 训练时长：~30-60 分钟（1054 条样本，3 个 epoch）
  - 输出目录：output/lora/final/（adapter 权重 + tokenizer）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输出文件说明：

  output/lora/
  ├── checkpoint-<step>/     每个 epoch 结束时保存的检查点（最多保留 2 个）
  ├── final/
  │   ├── adapter_config.json        LoRA 配置
  │   ├── adapter_model.safetensors  LoRA 权重（~50MB）
  │   └── tokenizer*                 tokenizer 文件（拷贝自基础模型）
  └── trainer_state.json             训练历史（loss 曲线等）
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# 脚本位于 scripts/ 下，需要把项目根目录加入路径才能 import 本项目模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# 默认超参数
# ---------------------------------------------------------------------------

# LoRA 架构参数
# r=16 对 1.7B 模型 + 小数据集是经典选择；若过拟合可减小到 8，欠拟合可增到 32
LORA_R            = 16
LORA_ALPHA        = 32        # 通常设为 2*r
LORA_DROPOUT      = 0.05
# Qwen3 的注意力 + FFN 全部加 LoRA，覆盖所有可训练线性层
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",   # 注意力层
    "gate_proj", "up_proj", "down_proj",       # FFN 层
]

# 训练超参数
DEFAULT_EPOCHS      = 3
DEFAULT_LR          = 2e-4
# global batch 的设计目标是 16，通过 PER_DEVICE_BATCH × GRAD_ACCUM × GPU数 维持
# 推荐 2 张 GPU：PER_DEVICE=8, ACCUM=1, global=16
# 单卡备用：PER_DEVICE=8, ACCUM=2, global=16（同效果，慢一倍）
# 如果 OOM：PER_DEVICE 减半，ACCUM 翻倍（保持 global=16）
PER_DEVICE_BATCH    = 8
GRAD_ACCUM          = 1   # 单卡时改为 2；2 卡时保持 1
MAX_SEQ_LENGTH      = 2048    # system+user+assistant 总长度上限

# 路径
DEFAULT_BASE_MODEL  = "/mnt/data/model/Qwen3-1.7B"
TRAIN_PATH          = Path("data/finetune/train.jsonl")
VAL_PATH            = Path("data/finetune/val.jsonl")
DEFAULT_OUTPUT_DIR  = Path("output/lora")


# ---------------------------------------------------------------------------
# 数据处理
# ---------------------------------------------------------------------------

def build_dataset(tokenizer: AutoTokenizer) -> dict:
    """
    加载 JSONL 数据集并应用 Qwen3 chat template，返回包含 'text' 字段的 DatasetDict。

    training data 的每条记录格式：
        {"messages": [
            {"role": "system",    "content": "..."},
            {"role": "user",      "content": "...\n输出：/no_think"},
            {"role": "assistant", "content": "{...json...}"}
        ]}

    apply_chat_template 会把 messages 转换为 Qwen3 的 ChatML 格式字符串，
    enable_thinking=False 告知 tokenizer 不插入 <think> 标记，
    与训练数据中的 /no_think 指令保持一致。
    """
    dataset = load_dataset(
        "json",
        data_files={"train": str(TRAIN_PATH), "validation": str(VAL_PATH)},
    )

    def apply_template(examples: dict) -> dict:
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,  # 训练时包含完整对话（含 assistant 回复）
                enable_thinking=False,
            )
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(
        apply_template,
        batched=True,
        remove_columns=["messages"],
        desc="应用 chat template",
    )
    return dataset


# ---------------------------------------------------------------------------
# 主训练函数
# ---------------------------------------------------------------------------

def train(
    base_model: str,
    output_dir: Path,
    epochs: int,
    lr: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 检测是否处于 accelerate launch 的多进程环境
    is_main_process = int(os.environ.get("LOCAL_RANK", 0)) == 0
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if is_main_process:
        print(f"\n{'='*55}")
        print(f"  基础模型   : {base_model}")
        print(f"  训练集     : {TRAIN_PATH}")
        print(f"  验证集     : {VAL_PATH}")
        print(f"  输出目录   : {output_dir}")
        print(f"  GPU 数量   : {world_size}")
        print(f"  Epoch 数   : {epochs}")
        print(f"  学习率     : {lr}")
        print(f"  等效 batch : {PER_DEVICE_BATCH * GRAD_ACCUM * world_size}")
        print(f"{'='*55}\n")

    # ── tokenizer ─────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    # Qwen3 的 pad_token 默认是 eos_token，右填充与 Causal LM 训练兼容
    tokenizer.padding_side = "right"

    # ── 基础模型 ──────────────────────────────────────────────────────────
    # 注意：training 阶段不传 device_map，由 SFTTrainer + accelerate 负责设备分配。
    # device_map="auto" 会做 tensor parallel（切割模型层），与 DDP 不兼容。
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,   # A100 原生支持 bf16，精度与 fp32 接近
        trust_remote_code=True,
        # 不设 device_map，由 Trainer 统一管理
    )
    # 启用梯度检查点（以少量重计算换显存，A100 上几乎无速度损失）
    model.enable_input_require_grads()

    # ── LoRA 配置 ─────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # ── 数据集 ────────────────────────────────────────────────────────────
    dataset = build_dataset(tokenizer)
    if is_main_process:
        print(f"训练集：{len(dataset['train'])} 条  验证集：{len(dataset['validation'])} 条\n")

    # ── SFTTrainer ────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=lora_config,
        args=SFTConfig(
            # 基础设置
            output_dir=str(output_dir),
            num_train_epochs=epochs,

            # Batch size：单卡 A100-40GB 跑 1.7B 可用 8，OOM 时改为 4 并把 grad_accum 翻倍
            per_device_train_batch_size=PER_DEVICE_BATCH,
            per_device_eval_batch_size=PER_DEVICE_BATCH,
            gradient_accumulation_steps=GRAD_ACCUM,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},

            # 优化器
            learning_rate=lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            weight_decay=0.01,
            optim="adamw_torch",

            # 精度
            bf16=True,
            fp16=False,

            # 评估与保存
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,            # 只保留最近 2 个 checkpoint，节省磁盘
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,

            # 日志
            logging_steps=10,
            logging_dir=str(output_dir / "logs"),
            report_to="none",              # 不上传 wandb / tensorboard（可改为 "tensorboard"）

            # 数据
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LENGTH,
            dataloader_num_workers=4,      # Linux 服务器可用多进程加速数据加载

            # 其他
            seed=42,
            data_seed=42,
        ),
    )

    # ── 开始训练 ──────────────────────────────────────────────────────────
    if is_main_process:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        print(f"可训练参数：{trainable:,}（占比 {trainable/total*100:.2f}%）")
        print(f"总参数量  ：{total:,}\n")

    trainer.train()

    # ── 保存最终结果 ──────────────────────────────────────────────────────
    # 只保存 LoRA adapter，体积约 50MB（远小于完整模型权重 ~3.4GB）
    final_dir = output_dir / "final"
    trainer.save_model(final_dir)
    

    if is_main_process:
        tokenizer.save_pretrained(str(final_dir))
        print(f"\n{'='*55}")
        print(f"  训练完成！")
        print(f"  Adapter 已保存到：{final_dir}")
        print(f"  下一步：更新 model/generator.py 加载本地 adapter")
        print(f"{'='*55}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qwen3-1.7B LoRA 微调（单卡或多卡）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--base-model", type=str, default=DEFAULT_BASE_MODEL,
        help=f"基础模型名称或本地路径（默认：{DEFAULT_BASE_MODEL}）",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"输出目录（默认：{DEFAULT_OUTPUT_DIR}）",
    )
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help=f"训练轮数（默认：{DEFAULT_EPOCHS}）",
    )
    parser.add_argument(
        "--lr", type=float, default=DEFAULT_LR,
        help=f"初始学习率（默认：{DEFAULT_LR}）",
    )
    args = parser.parse_args()

    train(
        base_model=args.base_model,
        output_dir=args.output,
        epochs=args.epochs,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
