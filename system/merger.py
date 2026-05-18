"""
C线模块：DeltaMerger
将多轮对话中的 delta 合并到当前 PlotSpec，并填充可选字段默认值。
"""

from __future__ import annotations

from schema import OPTIONAL_DEFAULTS


def merge_delta(current_spec: dict, delta: dict) -> dict:
    """
    将 delta 中的字段覆盖到 current_spec。

    Args:
        current_spec: 当前完整 PlotSpec dict（不被修改）。
        delta:        仅含变更字段的 dict。

    Returns:
        合并后的新 PlotSpec dict。
    """
    merged = dict(current_spec)
    merged.update(delta)
    return merged


def fill_defaults(spec: dict) -> dict:
    """
    将 OPTIONAL_DEFAULTS 中存在但 spec 中缺失的字段填入默认值。

    Args:
        spec: 可能缺少 optional 字段的 PlotSpec dict（不被修改）。

    Returns:
        填充了所有 optional 默认值的完整 PlotSpec dict。
    """
    result = dict(spec)
    for key, default_val in OPTIONAL_DEFAULTS.items():
        if key not in result:
            result[key] = default_val
    return result
