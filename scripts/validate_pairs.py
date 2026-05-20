"""
scripts/validate_pairs.py

读取 data/pairs/raw_pairs.jsonl，对每条记录执行三级校验：
  1. 字段完整性 + 枚举值（fill_defaults + validator）
  2. 列名存在性（data_x/data_y/data_group_by/data_error 必须在 CSV 中存在）
  3. 渲染校验（实际调用 render_plot，验证能出图，出图后删除 PNG）

通过的写入 data/pairs/valid_pairs.jsonl，
失败的写入 data/pairs/reject_log.jsonl（含拒绝原因）。

用法：
    python scripts/validate_pairs.py
    python scripts/validate_pairs.py --no-render   # 跳过渲染校验（快速模式）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from system.merger import fill_defaults
from system.validator import validate
from tools.loader import DataLoadError, _CACHE, load_data
from tools.renderer import RenderError, render_plot

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

PAIRS_DIR = Path("data/pairs")
RAW_PAIRS_PATH  = PAIRS_DIR / "raw_pairs.jsonl"
VALID_PAIRS_PATH = PAIRS_DIR / "valid_pairs.jsonl"
REJECT_LOG_PATH  = PAIRS_DIR / "reject_log.jsonl"

# data_x / data_y 等可能引用数据列的字段
_COL_REF_FIELDS = ("data_x", "data_group_by", "data_error", "params_heatmap_value")


# ---------------------------------------------------------------------------
# 校验逻辑
# ---------------------------------------------------------------------------

def _check_col_exists(spec: dict, df_cols: set[str]) -> str | None:
    """
    检查 spec 中引用列名的字段是否都在 DataFrame 中存在。
    data_y 允许是字符串或列表；heatmap 宽表时 data_x 可以是概念名（不在列中）。
    返回 None 表示通过，返回错误描述字符串表示失败。
    """
    # data_y：支持字符串或列表
    data_y = spec.get("data_y")
    if data_y is not None:
        y_cols = [data_y] if isinstance(data_y, str) else list(data_y)
        for col in y_cols:
            if col not in df_cols:
                return f"data_y 列 '{col}' 不存在于数据中"

    # data_x：heatmap 宽表时可能是概念名，不作强制检查
    data_x = spec.get("data_x")
    if data_x and spec.get("chart_type") != "heatmap":
        if data_x not in df_cols:
            return f"data_x 列 '{data_x}' 不存在于数据中"

    # 其他引用列的字段
    for field in _COL_REF_FIELDS:
        val = spec.get(field)
        if val and val not in df_cols:
            return f"字段 '{field}' 引用列 '{val}' 不存在于数据中"

    return None


def validate_one(
    record: dict,
    do_render: bool,
) -> tuple[bool, str, dict | None]:
    """
    对单条记录执行完整校验。

    Args:
        record:    raw_pairs.jsonl 中的一条记录。
        do_render: 是否执行渲染校验。

    Returns:
        (passed, reason, enriched_record)
        passed=True 时 enriched_record 含 data_context 字段，供 pack 使用。
        passed=False 时 enriched_record 为 None。
    """
    csv_path = record.get("csv_path", "")
    plotspec = record.get("plotspec", {})

    # ── 加载 CSV，获取 data_context 和 cache_key ──────────────────────────
    try:
        data_context, cache_key = load_data(csv_path)
    except DataLoadError as e:
        return False, f"load_data 失败：{e}", None

    df = _CACHE[cache_key]
    df_cols = set(df.columns)

    # ── 1. data_source 不得存在（strip 后继续）───────────────────────────
    clean_spec = {k: v for k, v in plotspec.items() if k != "data_source"}

    # ── 2. fill_defaults + validator ──────────────────────────────────────
    # 注入 data_source 仅用于 fill_defaults/validate/render，不写入 valid_pairs
    spec_with_source = fill_defaults({**clean_spec, "data_source": cache_key})
    result = validate(spec_with_source)
    if not result.ok:
        reason_parts = []
        if result.missing_required:
            reason_parts.append(f"缺少必填字段：{result.missing_required}")
        if result.type_errors:
            reason_parts.append(f"类型错误：{result.type_errors}")
        return False, "validator 失败：" + "；".join(reason_parts), None

    # ── 3. 列名存在性 ────────────────────────────────────────────────────
    col_err = _check_col_exists(spec_with_source, df_cols)
    if col_err:
        return False, col_err, None

    # ── 4. 渲染校验 ──────────────────────────────────────────────────────
    if do_render:
        try:
            img_path = render_plot(spec_with_source, cache_key)
        except RenderError as e:
            return False, f"渲染失败：{e}", None
        except Exception as e:
            return False, f"渲染异常：{e}", None
        finally:
            pass
        # 删除生成的测试图片
        try:
            Path(img_path).unlink(missing_ok=True)
        except Exception:
            pass

    enriched = {
        **record,
        "plotspec": clean_spec,      # 不含 data_source 的干净版本
        "data_context": data_context,
    }
    return True, "ok", enriched


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------

def run_validation(do_render: bool) -> None:
    if not RAW_PAIRS_PATH.exists():
        print(f"✗ 找不到 {RAW_PAIRS_PATH}，请先运行 gen_pairs.py")
        return

    records = []
    with RAW_PAIRS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"读取 {len(records)} 条原始记录")
    print(f"渲染校验：{'开启' if do_render else '关闭（快速模式）'}\n")

    passed_list: list[dict] = []
    rejected_list: list[dict] = []

    for i, record in enumerate(records, 1):
        rid = record.get("id", f"record_{i}")
        ok, reason, enriched = validate_one(record, do_render=do_render)
        status = "✓" if ok else "✗"
        print(f"  [{i:3d}/{len(records)}] {status} {rid}  —  {reason}")
        if ok:
            passed_list.append(enriched)
        else:
            rejected_list.append({**record, "reject_reason": reason})

    # 写出结果
    with VALID_PAIRS_PATH.open("w", encoding="utf-8") as f:
        for rec in passed_list:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with REJECT_LOG_PATH.open("w", encoding="utf-8") as f:
        for rec in rejected_list:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total = len(records)
    n_ok = len(passed_list)
    n_fail = len(rejected_list)
    pass_rate = n_ok / total * 100 if total else 0

    print(f"\n校验完成：{n_ok}/{total} 通过（{pass_rate:.1f}%），{n_fail} 条被拒绝")
    print(f"有效配对：{VALID_PAIRS_PATH}")
    print(f"拒绝日志：{REJECT_LOG_PATH}")


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="校验 raw_pairs.jsonl 中的 PlotSpec 配对")
    parser.add_argument(
        "--no-render", action="store_true",
        help="跳过渲染校验（快速模式，只做字段和列名校验）",
    )
    args = parser.parse_args()
    run_validation(do_render=not args.no_render)


if __name__ == "__main__":
    main()
