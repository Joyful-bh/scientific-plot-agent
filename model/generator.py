"""
A线接口：generate_spec()
Plan A 实现：Qwen3-1.7B + LoRA adapter 本地推理（在服务器上运行）。
Plan B 实现：DeepSeek API（OpenAI 兼容接口），无需本地 GPU，用于调试和对比。
通过 set_backend() 在运行时切换，默认使用 Plan A。
"""

from __future__ import annotations

import json
import os
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

# ---------------------------------------------------------------------------
# 后端选择
# ---------------------------------------------------------------------------

_backend: str = "plan_b"  # "plan_a" | "plan_b"


def set_backend(backend: str) -> None:
    """切换推理后端（"plan_a" 或 "plan_b"），进程内全局生效。"""
    global _backend
    if backend not in ("plan_a", "plan_b"):
        raise ValueError(f"未知后端：{backend}，可选：plan_a / plan_b")
    _backend = backend


def get_backend() -> str:
    return _backend


# ---------------------------------------------------------------------------
# 公共工具函数
# ---------------------------------------------------------------------------

DEBUG = True  # True 时将原始模型响应打印到终端

_CACHE_KEY_RE = re.compile(r"cache://[a-f0-9]+")


def _strip_markdown(text: str) -> str:
    """去除模型可能输出的 markdown 代码块标记。"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _extract_json_object(text: str) -> str:
    """
    从文本中提取第一个完整的 JSON 对象（{...}），忽略闭合大括号后的多余内容。
    模型有时在 JSON 后追加自然语言说明，导致 json.loads 失败；此函数通过括号计数
    精确定位闭合位置，不依赖正则贪婪匹配。
    """
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    in_str = False
    escape = False
    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def _extract_cache_key(data_context: str) -> str | None:
    """从 data_context 摘要字符串中提取缓存键（cache://xxxxxxxx）。"""
    m = _CACHE_KEY_RE.search(data_context)
    return m.group(0) if m else None


def _parse_tool_response(
    raw_text: str,
    data_context: str,
    current_spec: dict | None,
    backend_label: str,
) -> dict:
    """
    共享解析逻辑：从原始模型输出中提取 PlotSpec 或 delta dict。
    处理工具调用格式 {"tool":..., "arguments":...} 和裸 JSON 兼容格式。
    """
    parsed: dict = json.loads(_extract_json_object(_strip_markdown(raw_text)))

    if "tool" in parsed and "arguments" in parsed:
        tool_name = parsed["tool"]
        if DEBUG:
            print(f"[{backend_label}] tool={tool_name}")
        if tool_name == "ask_user":
            return {
                "__ask_user__": True,
                "question": parsed["arguments"].get("question", "请补充更多信息"),
            }
        result = parsed["arguments"]
    else:
        # 兼容旧格式（裸 PlotSpec），避免格式迁移期间推理完全失败
        result = parsed

    # 首轮时注入 data_source（Plan A/B 均不输出该字段，由此处自动注入）
    if current_spec is None and "data_source" not in result:
        cache_key = _extract_cache_key(data_context)
        if cache_key:
            result["data_source"] = cache_key

    return result


# ---------------------------------------------------------------------------
# Plan A：Qwen3-1.7B + LoRA 本地推理
# ---------------------------------------------------------------------------

_BASE_MODEL = "/mnt/data/model/Qwen3-1.7B"
_LORA_CKPT = str(Path(__file__).parent.parent / "output" / "lora" / "checkpoint-210")

_tokenizer: AutoTokenizer | None = None
_model: PeftModel | None = None


def _load_model() -> tuple[Any, Any]:
    """延迟加载 tokenizer 和 LoRA 模型（进程内只执行一次）。"""
    global _tokenizer, _model
    if _model is not None:
        return _tokenizer, _model

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[Plan A] 加载基础模型：{_BASE_MODEL}")
    _tokenizer = AutoTokenizer.from_pretrained(_BASE_MODEL, trust_remote_code=True)

    base = AutoModelForCausalLM.from_pretrained(
        _BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"[Plan A] 加载 LoRA adapter：{_LORA_CKPT}")
    _model = PeftModel.from_pretrained(base, _LORA_CKPT).eval()
    print("[Plan A] 模型加载完成")
    return _tokenizer, _model


def _generate_plan_a(
    user_input: str,
    data_context: str,
    current_spec: dict | None,
) -> dict:
    """Plan A：本地 Qwen3-1.7B + LoRA 推理。"""
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
            do_sample=False,  # greedy decoding，JSON 输出更稳定
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    if DEBUG:
        round_label = "首轮" if current_spec is None else "修改轮"
        print(f"\n{'='*50}")
        print(f"[Plan A {round_label}] 用户输入: {user_input}")
        print(f"[Plan A 原始响应]:\n{raw_text}")
        print(f"{'='*50}\n")

    return _parse_tool_response(raw_text, data_context, current_spec, "Plan A")


# ---------------------------------------------------------------------------
# Plan B：DeepSeek API（OpenAI 兼容接口）
# ---------------------------------------------------------------------------

_PLAN_B_BASE_URL = "http://192.168.1.3:8000/v1"
_PLAN_B_MODEL = "qwen3-plot-lora"


def _generate_plan_b(
    user_input: str,
    data_context: str,
    current_spec: dict | None,
) -> dict:
    """Plan B：DeepSeek API，无需本地 GPU，用于调试和对比。"""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Plan B 需要安装 openai 库：pip install openai") from exc

    # 本地或私有 API 服务器通常不校验 Key；DEEPSEEK_API_KEY 未设置时用占位值
    api_key = os.environ.get("DEEPSEEK_API_KEY") or "local"
    
    client = OpenAI(api_key=api_key, base_url=_PLAN_B_BASE_URL)

    if current_spec is None:
        system_msg = SYSTEM_FIRST_FINETUNE
        user_msg = format_user_message(user_input, data_context)
    else:
        system_msg = SYSTEM_DELTA_FINETUNE
        user_msg = format_user_message_delta(user_input, data_context, current_spec)

    response = client.chat.completions.create(
        model=_PLAN_B_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=512,
    )

    raw_text = response.choices[0].message.content or ""

    if DEBUG:
        round_label = "首轮" if current_spec is None else "修改轮"
        print(f"\n{'='*50}")
        print(f"[Plan B {round_label}] 用户输入: {user_input}")
        print(f"[Plan B 原始响应]:\n{raw_text}")
        print(f"{'='*50}\n")

    return _parse_tool_response(raw_text, data_context, current_spec, "Plan B")


# ---------------------------------------------------------------------------
# 公共接口（签名不得修改）
# ---------------------------------------------------------------------------

def generate_spec(
    user_input: str,
    data_context: str,
    current_spec: dict | None = None,
) -> dict:
    """
    根据用户自然语言输入和数据摘要生成 PlotSpec 或 delta。
    当前后端通过 set_backend() 切换，默认为 plan_a。

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
    if _backend == "plan_b":
        return _generate_plan_b(user_input, data_context, current_spec)
    return _generate_plan_a(user_input, data_context, current_spec)
