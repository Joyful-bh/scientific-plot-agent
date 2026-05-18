"""
B线工具：PlotRenderer
接收 PlotSpec 和数据源，输出 PNG 图表文件。
当前为 Mock 实现，B线完成后替换各子渲染器函数体，render_plot 接口签名不变。
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tools.loader import get_dataframe, load_data
from tools.themes import ThemeConfig, apply_theme

OUTPUT_DIR = Path("output")


class RenderError(Exception):
    """图表渲染失败时抛出。"""


def render_plot(spec: dict, data_source: str) -> str:
    """
    主渲染入口。

    Args:
        spec:        已通过校验且 optional 字段已填充默认值的 PlotSpec dict。
        data_source: CSV 文件路径（首次加载）或 cache_key（已缓存）。

    Returns:
        生成图表的 PNG 文件路径字符串。

    Raises:
        RenderError: 渲染过程中出现任何错误时。
    """
    # Mock实现，B线替换
    placeholder = Path("data") / "placeholder.png"
    return str(placeholder)


# ---------------------------------------------------------------------------
# 子渲染器骨架 —— B线填入真实实现
# ---------------------------------------------------------------------------

def _render_bar(df: pd.DataFrame, spec: dict, theme: ThemeConfig) -> plt.Figure:
    """渲染柱状图。Mock实现，B线替换"""
    raise NotImplementedError


def _render_line(df: pd.DataFrame, spec: dict, theme: ThemeConfig) -> plt.Figure:
    """渲染折线图。Mock实现，B线替换"""
    raise NotImplementedError


def _render_scatter(df: pd.DataFrame, spec: dict, theme: ThemeConfig) -> plt.Figure:
    """渲染散点图。Mock实现，B线替换"""
    raise NotImplementedError


def _render_box(df: pd.DataFrame, spec: dict, theme: ThemeConfig) -> plt.Figure:
    """渲染箱线图。Mock实现，B线替换"""
    raise NotImplementedError


def _render_heatmap(df: pd.DataFrame, spec: dict, theme: ThemeConfig) -> plt.Figure:
    """渲染热力图。Mock实现，B线替换"""
    raise NotImplementedError


RENDERERS: dict[str, object] = {
    "bar":     _render_bar,
    "line":    _render_line,
    "scatter": _render_scatter,
    "box":     _render_box,
    "heatmap": _render_heatmap,
}


def _ensure_output_dir() -> None:
    """首次调用时自动创建 output/ 目录。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _output_path(chart_type: str) -> Path:
    """生成带时间戳的输出文件路径。"""
    _ensure_output_dir()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return OUTPUT_DIR / f"plot_{timestamp}_{chart_type}.png"


def _resolve_dataframe(data_source: str, spec: dict) -> pd.DataFrame:
    """根据 data_source 取回 DataFrame（缓存键或 CSV 路径均支持）。"""
    if data_source.startswith("cache://"):
        return get_dataframe(data_source)
    _, cache_key = load_data(data_source)
    return get_dataframe(cache_key)
