"""
scripts/pack_finetune.py

将 data/pairs/valid_pairs.jsonl 打包为 Qwen3 微调格式的 JSONL 文件。

输出：
    data/finetune/train.jsonl   训练集（90%）
    data/finetune/val.jsonl     验证集（10%）

每条记录格式（Qwen3 ChatML / messages 格式）：
    {
      "messages": [
        {"role": "system",    "content": "<SYSTEM_FIRST_FINETUNE>"},
        {"role": "user",      "content": "数据摘要：\\n...\\n\\n用户需求：...\\n输出：/no_think"},
        {"role": "assistant", "content": "{plotspec json}"}
      ]
    }

用法：
    python scripts/pack_finetune.py
    python scripts/pack_finetune.py --val-ratio 0.15   # 调整验证集比例
    python scripts/pack_finetune.py --seed 123          # 调整随机分割种子

扩展说明（多轮数据）：
    多轮微调数据（修改轮）格式：在 messages 中追加多个 user/assistant 对。
    当前版本只生成首轮数据（messages 长度固定为 3）。
    如需添加多轮，在 build_messages() 中扩展 turn_list 即可，接口已预留。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.prompts import SYSTEM_FIRST_FINETUNE, format_user_message

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

PAIRS_DIR   = Path("data/pairs")
FINETUNE_DIR = Path("data/finetune")
FINETUNE_DIR.mkdir(parents=True, exist_ok=True)

VALID_PAIRS_PATH = PAIRS_DIR / "valid_pairs.jsonl"
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
    turn_list: list[dict] | None = None,
) -> dict:
    """
    构建一条微调样本的 messages 结构。

    Args:
        user_input:   首轮用户请求。
        data_context: 数据摘要字符串（由 tools.loader._build_context 生成）。
        plotspec:     目标 PlotSpec dict（不含 data_source）。
        turn_list:    后续修改轮列表，每个元素为
                      {"user_input": str, "delta": dict}。
                      当前版本传 None，预留多轮扩展接口。

    Returns:
        {"messages": [...]} 格式的 dict，直接可序列化为 JSONL。
    """
    messages = [
        {"role": "system", "content": SYSTEM_FIRST_FINETUNE},
        {
            "role": "user",
            "content": format_user_message(user_input, data_context),
        },
        {
            "role": "assistant",
            "content": json.dumps(plotspec, ensure_ascii=False, separators=(",", ":")),
        },
    ]

    # 多轮扩展预留：追加后续 user/assistant 对
    if turn_list:
        for turn in turn_list:
            messages.append({
                "role": "user",
                "content": turn["user_input"] + " /no_think",
            })
            messages.append({
                "role": "assistant",
                "content": json.dumps(turn["delta"], ensure_ascii=False, separators=(",", ":")),
            })

    return {"messages": messages}


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

    print(f"读取 {len(records)} 条有效配对")

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
                sample = build_messages(
                    user_input=rec["user_input"],
                    data_context=rec["data_context"],
                    plotspec=rec["plotspec"],
                    turn_list=None,   # 首轮只有，多轮扩展时传入
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
