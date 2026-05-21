"""
A线接口：generate_spec()
Plan A 实现：Qwen3-1.7B + LoRA adapter 本地推理（在服务器上运行）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from model.prompts import (
    SYSTEM_DELTA_FINETUNE,
    SYSTEM_FIRST_FINETUNE,
    format_user_message,
    format_user_message_delta,
)

if TYPE_CHECKING:
    from peft import PeftModel
    from transformers import AutoTokenizer

# 模型路径（服务器上的绝对路径 + 项目内相对路径）
_BASE_MODEL = "/mnt/data/model/Qwen3-1.7B"
_LORA_CKPT = str(Path(__file__).parent.parent / "output" / "lora" / "checkpoint-198")

DEBUG = True  # True 时将原始模型响应打印到终端

# 延迟加载状态（第一次调用 generate_spec 时初始化）
_tokenizer: AutoTokenizer | None = None
_model: PeftModel | None = None

# 从 data_context 字符串中提取 cache://xxxxxxxx 格式的缓存键
_CACHE_KEY_RE = re.compile(r"cache://[a-f0-9]+")


def _load_model() -> tuple[Any, Any]:
    """延迟加载 tokenizer 和 LoRA 模型（进程内只执行一次）。"""
    global _tokenizer, _model
    if _model is not None:
        return _tokenizer, _model

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[A线] 加载基础模型：{_BASE_MODEL}")
    _tokenizer = AutoTokenizer.from_pretrained(_BASE_MODEL, trust_remote_code=True)

    base = AutoModelForCausalLM.from_pretrained(
        _BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",      # 推理阶段使用 auto，单卡自动映射到 cuda:0
        trust_remote_code=True,
    )

    print(f"[A线] 加载 LoRA adapter：{_LORA_CKPT}")
    _model = PeftModel.from_pretrained(base, _LORA_CKPT).eval()
    print("[A线] 模型加载完成")
    return _tokenizer, _model


def _strip_markdown(text: str) -> str:
    """去除模型可能输出的 markdown 代码块标记。"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # 去掉 ```json 或 ``` 开头行
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _extract_cache_key(data_context: str) -> str | None:
    """从 data_context 摘要字符串中提取缓存键（cache://xxxxxxxx）。"""
    m = _CACHE_KEY_RE.search(data_context)
    return m.group(0) if m else None


def generate_spec(
    user_input: str,
    data_context: str,
    current_spec: dict | None = None,
) -> dict:
    """
    根据用户自然语言输入和数据摘要生成 PlotSpec 或 delta。

    Args:
        user_input:    用户的自然语言输入字符串。
        data_context:  DataLoader 生成的数据摘要字符串，注入 prompt 供模型参考。
        current_spec:  当前 PlotSpec dict；首轮为 None，修改轮传入当前值。

    Returns:
        首轮：包含所有 REQUIRED_FIELDS 的完整 PlotSpec dict。
        修改轮：仅包含变更字段的 delta dict。
        返回值已经过 JSON 解析，不是字符串。
    """
    # Mock实现，A线替换
    import torch  # 延迟导入：仅在安装了 torch 的服务器环境中运行

    tokenizer, model = _load_model()

    if current_spec is None:
        system_msg = SYSTEM_FIRST_FINETUNE
        user_msg = format_user_message(user_input, data_context)
    else:
        system_msg = SYSTEM_DELTA_FINETUNE
        user_msg = format_user_message_delta(user_input, data_context, current_spec)

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # 与训练数据的 /no_think 指令保持一致
    )

    device = next(model.parameters()).device
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,                    # greedy decoding，JSON 输出更稳定
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    if DEBUG:
        round_label = "首轮" if current_spec is None else "修改轮"
        print(f"\n{'='*50}")
        print(f"[Qwen3 LoRA {round_label}] 用户输入: {user_input}")
        print(f"[Qwen3 LoRA 原始响应]:\n{raw_text}")
        print(f"{'='*50}\n")

    parsed: dict = json.loads(_strip_markdown(raw_text))

    # 从工具调用包装结构中提取 arguments
    if "tool" in parsed and "arguments" in parsed:
        tool_name = parsed["tool"]
        result = parsed["arguments"]
        if DEBUG:
            print(f"[A线] tool={tool_name}")
    else:
        # 兼容旧格式（裸 PlotSpec），避免格式迁移期间推理完全失败
        result = parsed

    # 首轮推理时注入 data_source（Plan A 模型不输出该字段，由此处自动注入）
    if current_spec is None and "data_source" not in result:
        cache_key = _extract_cache_key(data_context)
        if cache_key:
            result["data_source"] = cache_key

    return result
