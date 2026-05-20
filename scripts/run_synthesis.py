"""
scripts/run_synthesis.py

训练数据合成流水线的一键运行脚本。按顺序执行以下四步：

  步骤 1  gen_csv.py       —— 生成 51 个场景 CSV（含列名随机化）
  步骤 2  gen_pairs.py     —— 调用 DeepSeek API 合成 (user_input, plotspec) 配对
  步骤 3  validate_pairs.py —— 三级校验（字段 + 列名 + 渲染）
  步骤 4  pack_finetune.py  —— 打包为 Qwen3 ChatML 微调 JSONL

前置条件：
  - 在 .env 或环境变量中设置 DEEPSEEK_API_KEY
  - pip install -r requirements.txt

用法示例：

  # 快速测试（只合成 2 个 CSV 的配对，跳过渲染校验）
  python scripts/run_synthesis.py --limit 2 --no-render

  # 完整运行（指定模型）
  python scripts/run_synthesis.py --model deepseek-chat

  # 跳过 CSV 生成（data/train/ 已存在时）
  python scripts/run_synthesis.py --skip-csv --model deepseek-chat

  # 只重新打包（已有 valid_pairs.jsonl）
  python scripts/run_synthesis.py --only-pack
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# 项目根目录（scripts/ 的上一级）
ROOT = Path(__file__).parent.parent
SCRIPTS = Path(__file__).parent

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"{'='*62}")


def _run(label: str, script: str, extra_args: list[str]) -> bool:
    """
    用当前 Python 解释器运行 scripts/<script>，传入额外参数。
    返回 True 表示成功（returncode == 0）。
    """
    _banner(f"步骤：{label}")
    cmd = [sys.executable, str(SCRIPTS / script)] + extra_args
    print(f"命令：{' '.join(cmd)}\n")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - t0
    ok = result.returncode == 0
    status = "✓ 完成" if ok else "✗ 失败"
    print(f"\n{status}  用时 {elapsed:.1f}s")
    return ok


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="训练数据合成一键脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="gen_pairs 只处理前 N 个 CSV（快速测试用）",
    )
    parser.add_argument(
        "--model", type=str, default="deepseek-chat",
        help="DeepSeek 模型名（默认：deepseek-chat）",
    )
    parser.add_argument(
        "--no-render", action="store_true",
        help="validate_pairs 跳过渲染校验（更快，但不保证 PlotSpec 可真实出图）",
    )
    parser.add_argument(
        "--skip-csv", action="store_true",
        help="跳过步骤1（data/train/ 已有 CSV 时使用）",
    )
    parser.add_argument(
        "--skip-pairs", action="store_true",
        help="跳过步骤2（raw_pairs.jsonl 已存在时）",
    )
    parser.add_argument(
        "--only-pack", action="store_true",
        help="只执行步骤4（valid_pairs.jsonl 已存在时）",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.10,
        help="验证集比例（默认 0.10）",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="gen_pairs 追加写入（而非覆盖）raw_pairs.jsonl",
    )
    args = parser.parse_args()

    # 打印摘要
    print("\n训练数据合成流水线")
    print(f"  模型      : {args.model}")
    print(f"  CSV 限制  : {'全量' if args.limit is None else f'前 {args.limit} 个'}")
    print(f"  渲染校验  : {'关闭' if args.no_render else '开启'}")
    print(f"  验证集比例: {args.val_ratio:.0%}")

    t_start = time.time()
    success_steps: list[str] = []
    failed_step: str | None = None

    # ── 步骤 1：生成 CSV ──────────────────────────────────────────────────
    if args.only_pack or args.skip_csv or args.skip_pairs:
        print("\n[步骤1] 跳过 CSV 生成")
    else:
        ok = _run("生成训练 CSV（gen_csv.py）", "gen_csv.py", [])
        if ok:
            success_steps.append("步骤1 生成CSV")
        else:
            failed_step = "步骤1 gen_csv"
            _print_summary(success_steps, failed_step, t_start)
            sys.exit(1)

    # ── 步骤 2：合成配对 ──────────────────────────────────────────────────
    if args.only_pack or args.skip_pairs:
        print("\n[步骤2] 跳过配对合成")
    else:
        extra: list[str] = ["--model", args.model]
        if args.limit:
            extra += ["--limit", str(args.limit)]
        if args.append:
            extra.append("--append")
        ok = _run("合成配对（gen_pairs.py）", "gen_pairs.py", extra)
        if ok:
            success_steps.append("步骤2 合成配对")
        else:
            failed_step = "步骤2 gen_pairs"
            _print_summary(success_steps, failed_step, t_start)
            sys.exit(1)

    # ── 步骤 3：校验过滤 ──────────────────────────────────────────────────
    if args.only_pack:
        print("\n[步骤3] 跳过校验")
    else:
        extra = ["--no-render"] if args.no_render else []
        ok = _run("校验过滤（validate_pairs.py）", "validate_pairs.py", extra)
        if ok:
            success_steps.append("步骤3 校验过滤")
        else:
            failed_step = "步骤3 validate_pairs"
            _print_summary(success_steps, failed_step, t_start)
            sys.exit(1)

    # ── 步骤 4：打包 JSONL ────────────────────────────────────────────────
    extra = ["--val-ratio", str(args.val_ratio)]
    ok = _run("打包微调数据（pack_finetune.py）", "pack_finetune.py", extra)
    if ok:
        success_steps.append("步骤4 打包JSONL")
    else:
        failed_step = "步骤4 pack_finetune"

    _print_summary(success_steps, failed_step, t_start)
    if failed_step:
        sys.exit(1)


def _print_summary(
    success_steps: list[str],
    failed_step: str | None,
    t_start: float,
) -> None:
    total = time.time() - t_start
    _banner("流水线结束")
    for s in success_steps:
        print(f"  ✓  {s}")
    if failed_step:
        print(f"  ✗  {failed_step}  ← 在此停止")
    print(f"\n总用时：{total:.1f}s")
    if not failed_step:
        print("\n输出文件：")
        for p in [
            Path("data/finetune/train.jsonl"),
            Path("data/finetune/val.jsonl"),
        ]:
            full = ROOT / p
            if full.exists():
                size_kb = full.stat().st_size / 1024
                n_lines = sum(1 for _ in full.open(encoding="utf-8"))
                print(f"  {p}  ({n_lines} 条, {size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
