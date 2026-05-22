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
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

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
DEFAULT_EPOCHS      = 5
DEFAULT_LR          = 1e-4
# global batch 的设计目标是 16，通过 PER_DEVICE_BATCH × GRAD_ACCUM × GPU数 维持
# 注意：system prompt 约 2400 token，全序列平均 2800 token，MAX_SEQ_LENGTH 必须 >= 3500
# 单卡 A100-40GB 参考配置：PER_DEVICE=4, ACCUM=4, global=16
# 如果 OOM：PER_DEVICE 减半，ACCUM 翻倍（保持 global=16）
PER_DEVICE_BATCH    = 4
GRAD_ACCUM          = 4   # 单卡有效 batch = 4×4 = 16；2 卡时 ACCUM 改为 2
MAX_SEQ_LENGTH      = 4096    # 必须大于最长序列（实测最大约 3067 token）

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
    加载 JSONL 数据集并预计算 input_ids 和 labels。

    labels 中 prompt 部分（system + user + assistant 起始标记）全部设为 -100，
    只在 assistant 的 JSON 回复 token 上计算 loss。
    这比 DataCollatorForCompletionOnlyLM 的模板匹配更可靠，避免特殊 token 编码
    不一致导致全部 label 被 mask、loss 归零的问题。
    """
    dataset = load_dataset(
        "json",
        data_files={"train": str(TRAIN_PATH), "validation": str(VAL_PATH)},
    )

    def apply_template(examples: dict) -> dict:
        input_ids_list: list[list[int]] = []
        labels_list: list[list[int]] = []

        for messages in examples["messages"]:
            # 完整序列（含 assistant 回复）
            full_text: str = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            # Prompt 部分（system + user），末尾自动附加 <|im_start|>assistant\n
            # 这正好是 assistant 回复开始之前的全部内容
            prompt_text: str = tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            # 两段文本均不额外添加 BOS/EOS，因为 apply_chat_template 已包含所有特殊 token
            full_ids: list[int] = tokenizer(
                full_text, add_special_tokens=False
            )["input_ids"]
            prompt_len: int = len(
                tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            )

            # prompt 部分 mask 为 -100，只在 assistant 回复 token 上产生梯度
            labels: list[int] = [-100] * prompt_len + full_ids[prompt_len:]

            input_ids_list.append(full_ids[:MAX_SEQ_LENGTH])
            labels_list.append(labels[:MAX_SEQ_LENGTH])

        return {"input_ids": input_ids_list, "labels": labels_list}

    dataset = dataset.map(
        apply_template,
        batched=True,
        remove_columns=["messages"],
        desc="应用 chat template 并计算 labels",
    )

    # ── 诊断：检查 label 是否全部被 mask（如出现则说明 MAX_SEQ_LENGTH 不够）──
    for split in dataset:
        n_all_masked = sum(
            1 for s in dataset[split] if all(l == -100 for l in s["labels"])
        )
        avg_valid = sum(
            sum(1 for l in s["labels"] if l != -100) for s in dataset[split]
        ) / len(dataset[split])
        print(
            f"[{split}] 全 mask 样本: {n_all_masked}/{len(dataset[split])}，"
            f"平均有效 label token 数: {avg_valid:.1f}"
        )
        if n_all_masked > 0:
            raise RuntimeError(
                f"{n_all_masked} 条样本的 labels 全部为 -100，"
                f"请增大 MAX_SEQ_LENGTH（当前 {MAX_SEQ_LENGTH}）"
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
    # 注意：training 阶段不传 device_map，由 Trainer + accelerate 负责设备分配。
    # device_map="auto" 会做 tensor parallel（切割模型层），与 DDP 不兼容。
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,   # A100 原生支持 bf16，精度与 fp32 接近
        trust_remote_code=True,
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
    model = get_peft_model(model, lora_config)

    # ── 数据集 ────────────────────────────────────────────────────────────
    dataset = build_dataset(tokenizer)
    if is_main_process:
        print(f"训练集：{len(dataset['train'])} 条  验证集：{len(dataset['validation'])} 条\n")

    # ── DataCollator ──────────────────────────────────────────────────────
    # labels 已在 build_dataset 中预计算（prompt 部分为 -100）。
    # 使用自定义 collator 而非 DataCollatorForSeq2Seq，避免其 seq2seq 专属行为
    # （如 decoder_input_ids 创建、label/input 长度不一致处理）干扰 causal LM 训练。
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def collate_fn(features: list[dict]) -> dict[str, torch.Tensor]:
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        labels    = [torch.tensor(f["labels"],    dtype=torch.long) for f in features]
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
        labels_padded    = pad_sequence(labels,    batch_first=True, padding_value=-100)
        attention_mask   = (input_ids_padded != pad_id).long()
        return {
            "input_ids":      input_ids_padded,
            "attention_mask": attention_mask,
            "labels":         labels_padded,
        }

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collate_fn,
        args=TrainingArguments(
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
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,

            # 日志
            logging_steps=10,
            logging_dir=str(output_dir / "logs"),
            report_to="none",

            # 数据
            dataloader_num_workers=4,
            remove_unused_columns=False,   # 保留 input_ids / labels 列

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
    if is_main_process:
        model.save_pretrained(str(final_dir))   # 只写 adapter 权重
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
