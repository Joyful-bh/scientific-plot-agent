"""
scripts/pack_finetune.py

读取首轮配对和修改轮配对，打包为 Qwen3 微调格式的 JSONL 文件。

输入：
    data/pairs/valid_pairs.jsonl   首轮配对（record_type="first"）
    data/pairs/delta_pairs.jsonl   修改轮配对（record_type="delta"，若存在则合并）

输出：
    data/finetune/train.jsonl   训练集（90%）
    data/finetune/val.jsonl     验证集（10%）

每条记录格式（Qwen3 ChatML / messages 格式）：
    {
      "messages": [
        {"role": "system",    "content": "<根据 record_type 选择的系统提示词>"},
        {"role": "user",      "content": "<DataContext>\\n\\n用户需求：...\\n输出：/no_think"},
        {"role": "assistant", "content": "{\"tool\":\"create_plot\",\"arguments\":{...plotspec...}}"}
      ]
    }

assistant 内容为工具调用格式：
    首轮（record_type="first"）：{"tool":"create_plot","arguments":{完整 PlotSpec（不含 data_source）}}
    修改轮（record_type="delta"）：{"tool":"update_plot","arguments":{仅含变更字段的 delta dict}}

系统提示词按记录类型（record_type）动态选择：
    "first" → SYSTEM_FIRST_FINETUNE（要求输出 create_plot 工具调用）
    "delta" → SYSTEM_DELTA_FINETUNE（要求输出 update_plot 工具调用）

valid_pairs.jsonl 字段（首轮）：record_type, user_input, data_context, plotspec, csv_path
delta_pairs.jsonl 字段（修改轮）：record_type, user_input, data_context, current_spec, plotspec(delta), csv_path

用法：
    python scripts/pack_finetune.py
    python scripts/pack_finetune.py --val-ratio 0.15   # 调整验证集比例
    python scripts/pack_finetune.py --seed 123          # 调整随机分割种子
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.prompts import (
    SYSTEM_DELTA_FINETUNE,
    SYSTEM_FIRST_FINETUNE,
    format_user_message,
    format_user_message_delta,
)

_SYSTEM_MAP: dict[str, str] = {
    "first": SYSTEM_FIRST_FINETUNE,
    "delta": SYSTEM_DELTA_FINETUNE,
}

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

PAIRS_DIR    = Path("data/pairs")
FINETUNE_DIR = Path("data/finetune")
FINETUNE_DIR.mkdir(parents=True, exist_ok=True)

VALID_PAIRS_PATH = PAIRS_DIR / "valid_pairs.jsonl"
DELTA_PAIRS_PATH = PAIRS_DIR / "delta_pairs.jsonl"
TRAIN_PATH = FINETUNE_DIR / "train.jsonl"
VAL_PATH   = FINETUNE_DIR / "val.jsonl"

DEFAULT_VAL_RATIO = 0.10
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# 构建单条训练样本
# ---------------------------------------------------------------------------

def build_messages(
    user_input: str,
    data_context: str,
    plotspec: dict,
    record_type: str = "first",
    current_spec: dict | None = None,
) -> dict:
    """
    构建一条微调样本的 messages 结构（含 system message）。

    系统提示词根据 record_type 从 _SYSTEM_MAP 中选取，内嵌到每条记录，
    使不同类型的记录（首轮/修改轮）各自携带正确的提示词，无需训练时统一注入。

    assistant message 使用工具调用格式：
        首轮：{"tool":"create_plot","arguments":{...完整 PlotSpec...}}
        修改轮：{"tool":"update_plot","arguments":{...delta 字段...}}

    Args:
        user_input:   用户请求（首轮或修改请求）。
        data_context: 数据摘要字符串（由 tools.loader._build_context 生成）。
        plotspec:     首轮为完整 PlotSpec dict，修改轮为 delta dict（均不含 data_source）。
        record_type:  "first"=首轮（默认），"delta"=修改轮。
        current_spec: 修改轮专用，当前生效的完整 PlotSpec dict（不含 data_source）。
                      record_type="first" 时忽略此参数。

    Returns:
        {"messages": [...]} 格式的 dict，直接可序列化为 JSONL。
    """
    system_prompt = _SYSTEM_MAP.get(record_type, SYSTEM_FIRST_FINETUNE)

    if record_type == "delta" and current_spec is not None:
        user_content = format_user_message_delta(user_input, data_context, current_spec)
    else:
        user_content = format_user_message(user_input, data_context)

    tool_name = "create_plot" if record_type == "first" else "update_plot"
    assistant_content = json.dumps(
        {"tool": tool_name, "arguments": plotspec},
        ensure_ascii=False, separators=(",", ":"),
    )

    return {
        "messages": [
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------

def pack(val_ratio: float, seed: int) -> None:
    if not VALID_PAIRS_PATH.exists():
        print(f"✗ 找不到 {VALID_PAIRS_PATH}，请先运行 validate_pairs.py")
        return

    records: list[dict] = []
    with VALID_PAIRS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print("✗ valid_pairs.jsonl 为空，没有可打包的数据")
        return

    n_first = len(records)

    # 合并修改轮数据（若存在）
    if DELTA_PAIRS_PATH.exists():
        with DELTA_PAIRS_PATH.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        n_delta = len(records) - n_first
        print(f"读取 {n_first} 条首轮配对 + {n_delta} 条修改轮配对，共 {len(records)} 条")
    else:
        print(f"读取 {n_first} 条有效配对（未找到 delta_pairs.jsonl，仅首轮数据）")

    # 随机分割
    rng = random.Random(seed)
    rng.shuffle(records)
    n_val = max(1, round(len(records) * val_ratio))
    val_records   = records[:n_val]
    train_records = records[n_val:]

    # 构建并写出
    def _write(path: Path, recs: list[dict], split_name: str) -> None:
        with path.open("w", encoding="utf-8") as f:
            for rec in recs:
                record_type = rec.get("record_type", "first")
                sample = build_messages(
                    user_input=rec["user_input"],
                    data_context=rec["data_context"],
                    plotspec=rec["plotspec"],
                    record_type=record_type,
                    current_spec=rec.get("current_spec"),
                )
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"  {split_name}: {len(recs)} 条  →  {path}")

    _write(TRAIN_PATH, train_records, "train")
    _write(VAL_PATH,   val_records,   "val  ")

    print(f"\n完成！训练集 {len(train_records)} 条，验证集 {len(val_records)} 条。")
    print(f"验证集比例：{len(val_records)/len(records)*100:.1f}%（目标 {val_ratio*100:.0f}%）")

    # 打印一条样本供确认
    print("\n── 样本预览（第 1 条 train）──")
    sample_rec = train_records[0]
    sample = build_messages(
        user_input=sample_rec["user_input"],
        data_context=sample_rec["data_context"],
        plotspec=sample_rec["plotspec"],
        record_type=sample_rec.get("record_type", "first"),
        current_spec=sample_rec.get("current_spec"),
    )
    for msg in sample["messages"]:
        role = msg["role"]
        content = msg["content"]
        preview = content[:120].replace("\n", "↵") + ("..." if len(content) > 120 else "")
        print(f"  [{role:9s}] {preview}")


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="打包有效配对为 Qwen3 微调 JSONL")
    parser.add_argument(
        "--val-ratio", type=float, default=DEFAULT_VAL_RATIO,
        help=f"验证集比例（默认 {DEFAULT_VAL_RATIO}）",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"随机分割种子（默认 {DEFAULT_SEED}）",
    )
    args = parser.parse_args()
    pack(val_ratio=args.val_ratio, seed=args.seed)


if __name__ == "__main__":
    main()
