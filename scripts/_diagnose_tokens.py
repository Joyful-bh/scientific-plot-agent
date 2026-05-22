"""
scripts/_diagnose_tokens.py

在服务器上运行，实际测量 token 数量分布，回答两个关键问题：
1. 系统提示词实际占用多少 token？
2. MAX_SEQ_LENGTH=2048 是否真的截掉了 assistant 回复？

用法（在服务器项目根目录）：
    python scripts/_diagnose_tokens.py --base-model /mnt/data/model/Qwen3-1.7B
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="/mnt/data/model/Qwen3-1.7B")
    parser.add_argument("--train-jsonl", default="data/finetune/train.jsonl")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    print(f"加载 tokenizer：{args.base_model}")
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    from model.prompts import SYSTEM_FIRST_FINETUNE, SYSTEM_DELTA_FINETUNE

    # ── 1. 系统提示词实际 token 数 ─────────────────────────────────────────
    sys1_ids = tok(SYSTEM_FIRST_FINETUNE, add_special_tokens=False)["input_ids"]
    sys2_ids = tok(SYSTEM_DELTA_FINETUNE, add_special_tokens=False)["input_ids"]
    print(f"\n系统提示词（首轮）：{len(SYSTEM_FIRST_FINETUNE)} 字符 → {len(sys1_ids)} token")
    print(f"系统提示词（修改轮）：{len(SYSTEM_DELTA_FINETUNE)} 字符 → {len(sys2_ids)} token")

    # ── 2. 对训练集抽样测量全序列 token 分布 ──────────────────────────────
    with open(args.train_jsonl, encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]

    sys_lens, user_lens, asst_lens, total_lens, prompt_lens = [], [], [], [], []

    for rec in records:
        msgs = rec["messages"]
        sys_msg  = msgs[0]["content"]
        user_msg = msgs[1]["content"]
        asst_msg = msgs[-1]["content"]

        # 完整序列 token 数
        full_text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        full_ids = tok(full_text, add_special_tokens=False)["input_ids"]

        # prompt 部分 token 数（system + user + <|im_start|>assistant\n）
        prompt_text = tok.apply_chat_template(
            msgs[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]

        sys_len  = len(tok(sys_msg,  add_special_tokens=False)["input_ids"])
        user_len = len(tok(user_msg, add_special_tokens=False)["input_ids"])
        asst_len = len(tok(asst_msg, add_special_tokens=False)["input_ids"])

        sys_lens.append(sys_len)
        user_lens.append(user_len)
        asst_lens.append(asst_len)
        total_lens.append(len(full_ids))
        prompt_lens.append(len(prompt_ids))

    def stats(name: str, arr: list[int]) -> None:
        arr_s = sorted(arr)
        n = len(arr_s)
        print(
            f"  {name}: min={arr_s[0]}  avg={sum(arr)//n}"
            f"  p50={arr_s[n//2]}  p95={arr_s[int(n*0.95)]}  max={arr_s[-1]}"
        )

    print(f"\n── token 分布（共 {len(records)} 条）──")
    stats("system  ", sys_lens)
    stats("user    ", user_lens)
    stats("asst    ", asst_lens)
    stats("prompt  ", prompt_lens)
    stats("total   ", total_lens)

    # ── 3. 关键诊断：MAX_SEQ_LENGTH=2048 下截断了多少 assistant token ──────
    print(f"\n── MAX_SEQ_LENGTH={args.max_seq_length} 截断分析 ──")
    n_asst_gone    = sum(1 for p, t in zip(prompt_lens, total_lens) if p >= args.max_seq_length)
    n_asst_partial = sum(1 for p, t in zip(prompt_lens, total_lens)
                         if p < args.max_seq_length < t)
    n_asst_full    = sum(1 for t in total_lens if t <= args.max_seq_length)

    print(f"  assistant 回复完全丢失（prompt_len >= {args.max_seq_length}）: "
          f"{n_asst_gone}/{len(records)} ({n_asst_gone/len(records)*100:.1f}%)")
    print(f"  assistant 回复部分截断（total > {args.max_seq_length} 但 prompt 未超）: "
          f"{n_asst_partial}/{len(records)} ({n_asst_partial/len(records)*100:.1f}%)")
    print(f"  assistant 回复完整保留（total <= {args.max_seq_length}）: "
          f"{n_asst_full}/{len(records)} ({n_asst_full/len(records)*100:.1f}%)")

    # ── 4. 用 MAX_SEQ_LENGTH=4096 重做一遍 ─────────────────────────────────
    max2 = 4096
    n2_gone = sum(1 for p in prompt_lens if p >= max2)
    n2_full = sum(1 for t in total_lens if t <= max2)
    print(f"\n── MAX_SEQ_LENGTH={max2} 截断分析 ──")
    print(f"  assistant 完全丢失: {n2_gone}/{len(records)}")
    print(f"  assistant 完整保留: {n2_full}/{len(records)} ({n2_full/len(records)*100:.1f}%)")


if __name__ == "__main__":
    main()
