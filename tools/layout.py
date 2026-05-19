"""
B线工具：LayoutEngine
根据数据规模和结构动态计算布局参数，与 ThemeConfig 的静态风格属性互补。

分工：
  ThemeConfig  → 风格（配色 / 字体族 / 线宽基准 / 网格 / 轴脊 / 背景色 / DPI / 基准图幅 / 宽高比）
  LayoutParams → 布局（最终图幅尺寸 / 字体绝对大小 / 柱宽 / 标签旋转 / 图例位置 / Y轴起点）
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tools.themes import ThemeConfig


@dataclass
class LayoutParams:
    """compute_layout() 的输出——每次渲染时动态计算的布局参数。"""
    figure_width: float       # 最终图幅宽度（英寸）
    figure_height: float      # 最终图幅高度（英寸）
    font_size: float          # 最终字体大小（磅），夹在 [6, 14]
    legend_fontsize: float    # = font_size × 0.85
    tick_rotation: int        # X 轴刻度旋转角度；0 = 渲染层 post-draw 自动检测，非零 = 用户明确指定
    tick_ha: str              # 刻度标签水平对齐："right"（旋转时）或 "center"
    legend_loc: str           # "inside_best" | "outside_right" | "none"
    y_min: float | None       # Y 轴下限（已合并用户指定值与自动计算值）；None = matplotlib 自动
    y_max: float | None       # Y 轴上限（来自 spec.axes_y_max）；None = matplotlib 自动
    bar_width: float          # 单根柱子/箱体宽度
    marker_size: float        # line: markersize 直径(pts)；scatter: s 面积(pts²)，各自由对应 layout 函数计算
    annot_fontsize: float     # 热力图单元格注释字体大小
    capsize: float            # 误差棒帽宽（磅）
    x_max_ticks: int | None   # 折线图 X 轴最多刻度数；None = 不限制
    rotate_labels: bool       # True=间距不足时旋转标签(30°–45°)；False=缩小字号（来自 axes_x_rotate_labels）


def compute_layout(
    df: pd.DataFrame,
    spec: dict,
    theme: ThemeConfig,
) -> LayoutParams:
    """
    根据数据结构和 PlotSpec 动态计算布局参数。
    全程用 try/except 包裹；任何计算失败时 fallback 到 theme 基准值。
    """
    chart_type = spec.get("chart_type", "bar")
    try:
        return _DISPATCHERS[chart_type](df, spec, theme)
    except Exception:
        return _fallback(theme)


# ---------------------------------------------------------------------------
# 内部工具函数
# ---------------------------------------------------------------------------

def _fallback(theme: ThemeConfig) -> LayoutParams:
    base_font = float(theme.font_size)
    return LayoutParams(
        figure_width=theme.figure_width,
        figure_height=theme.figure_width * theme.aspect_ratio,
        font_size=base_font,
        legend_fontsize=round(base_font * 0.85, 1),
        tick_rotation=0,
        tick_ha="center",
        legend_loc="inside_best",
        y_min=None,
        y_max=None,
        bar_width=0.6,
        marker_size=5.0,
        annot_fontsize=base_font,
        capsize=float(theme.line_width) * 2,
        x_max_ticks=None,
        rotate_labels=False,
    )


def _scale_font(base: float, fig_w: float, base_w: float) -> float:
    """图幅扩大时适度增大字号，结果夹在 [6, 14] 磅。"""
    ratio = (fig_w / max(base_w, 0.1)) ** 0.4
    return round(max(6.0, min(14.0, base * ratio)), 1)



def _auto_y_bottom(df: pd.DataFrame, y_col: str | list) -> float | None:
    """
    若数据远离 0（最小值 > 最大值的 50%），自动收紧 Y 轴下限。
    返回收紧后的下限，或 None（交给 matplotlib 自动决定）。
    """
    cols = [y_col] if isinstance(y_col, str) else list(y_col)
    valid = [c for c in cols if c in df.columns]
    if not valid:
        return None
    vals = pd.concat([df[c] for c in valid], ignore_index=True).dropna()
    if vals.empty:
        return None
    vmin, vmax = float(vals.min()), float(vals.max())
    if vmin <= 0 or vmax <= 0 or vmin / vmax <= 0.5:
        return None
    cushion = (vmax - vmin) * 0.15
    raw = vmin - cushion
    if raw <= 0:
        return 0.0
    magnitude = 10 ** int(np.log10(max(abs(raw), 1)))
    return float(np.floor(raw / magnitude) * magnitude)


def _compute_y_range(
    df: pd.DataFrame, spec: dict, y_col: str | list
) -> tuple[float | None, float | None]:
    """
    返回 (y_min, y_max)，合并用户指定值与自动计算值。
    用户指定值优先；y_min 未指定时尝试自动收紧。
    """
    y_min = spec.get("axes_y_min")
    if y_min is None:
        y_min = _auto_y_bottom(df, y_col)
    y_max = spec.get("axes_y_max")
    return y_min, y_max


def _legend_params(n_series: int) -> str:
    """
    根据图例条目数返回图例位置字符串。
    "none" → 无图例；"inside_best" → 图内最优；"outside_right" → 图外右侧。
    """
    if n_series <= 0:
        return "none"
    if n_series <= 5:
        return "inside_best"
    return "outside_right"


def _resolve_legend_loc(default_loc: str, spec: dict) -> str:
    """用 spec.legend_loc 覆写 LayoutEngine 的默认图例位置。
    用户值 "inside" 映射到内部值 "inside_best"；None/"auto" 保留默认。
    """
    user_val = spec.get("legend_loc")
    if not user_val or user_val == "auto":
        return default_loc
    return "inside_best" if user_val == "inside" else user_val


# ---------------------------------------------------------------------------
# 各图表类型布局计算
# ---------------------------------------------------------------------------

def _layout_bar(df: pd.DataFrame, spec: dict, theme: ThemeConfig) -> LayoutParams:
    x_col = spec.get("data_x", "")
    y_col = spec.get("data_y", "")
    group_col = spec.get("data_group_by")
    horizontal = spec.get("params_orientation", "vertical") == "horizontal"

    n_cat = int(df[x_col].nunique()) if x_col in df.columns else 6
    if isinstance(y_col, list):
        n_groups = len(y_col)
    elif group_col and group_col in df.columns:
        n_groups = int(df[group_col].nunique())
    else:
        n_groups = 1

    # 图幅：每个"槽"约 0.20 英寸 + 类别间距 + 边距
    min_width = n_cat * n_groups * 0.20 + n_cat * 0.12 + 1.5
    fig_w = max(theme.figure_width, min_width)

    n_legend = n_groups if n_groups > 1 else 0
    legend_loc = _resolve_legend_loc(_legend_params(n_legend), spec)
    if legend_loc == "outside_right":
        fig_w += 1.2  # 为图外图例留空间

    fig_h = fig_w * theme.aspect_ratio
    bar_width = max(0.08, min(0.4, 0.75 / max(n_groups, 1)))
    font_size = _scale_font(theme.font_size, fig_w, theme.figure_width)
    legend_fontsize = round(font_size * 0.85, 1)

    rotate_labels = bool(spec.get("axes_x_rotate_labels", False))

    if horizontal:
        # 水平柱状图：类别在 Y 轴，X 轴是数值，不旋转 X 刻度
        tick_rotation, tick_ha = 0, "center"
    else:
        user_rot = spec.get("axes_x_tick_rotation")
        if user_rot:  # 用户明确指定了非零旋转，直接使用；否则 post-draw 自动检测
            tick_rotation = int(user_rot)
            tick_ha = "right" if tick_rotation > 0 else "center"
        else:
            tick_rotation, tick_ha = 0, "center"

    y_min, y_max = _compute_y_range(df, spec, y_col)

    return LayoutParams(
        figure_width=fig_w, figure_height=fig_h,
        font_size=font_size, legend_fontsize=legend_fontsize,
        tick_rotation=tick_rotation, tick_ha=tick_ha,
        legend_loc=legend_loc,
        y_min=y_min, y_max=y_max,
        bar_width=bar_width,
        marker_size=5.0,
        annot_fontsize=font_size,
        capsize=float(theme.line_width) * 2,
        x_max_ticks=None,
        rotate_labels=rotate_labels,
    )


def _layout_line(df: pd.DataFrame, spec: dict, theme: ThemeConfig) -> LayoutParams:
    x_col = spec.get("data_x", "")
    y_col = spec.get("data_y", "")
    group_col = spec.get("data_group_by")

    n_points = len(df)
    if isinstance(y_col, list):
        n_series = len(y_col)
    elif group_col and group_col in df.columns:
        n_series = int(df[group_col].nunique())
    else:
        n_series = 1

    fig_w = theme.figure_width
    # 折线图数据覆盖全图区域，3 条及以上系列改为图外放置以减少遮挡
    if n_series <= 1:
        _default_legend = "none"
    elif n_series <= 2:
        _default_legend = "inside_best"
    else:
        _default_legend = "outside_right"
    legend_loc = _resolve_legend_loc(_default_legend, spec)
    if legend_loc == "outside_right":
        fig_w += 1.2
    fig_h = fig_w * theme.aspect_ratio

    font_size = _scale_font(theme.font_size, fig_w, theme.figure_width)
    legend_fontsize = round(font_size * 0.85, 1)

    # marker 大小（直径，pts）随数据密度调整；用户指定时尊重用户
    user_msize = spec.get("params_marker_size")
    if user_msize is not None:
        marker_size = float(user_msize)
    elif n_points > 50:
        marker_size = 3.0
    elif n_points > 20:
        marker_size = 4.0
    else:
        marker_size = 5.0

    # 用唯一 X 值数量（而非总行数）判断是否需要限制刻度数量
    n_x = int(df[x_col].nunique()) if x_col in df.columns else n_points
    x_max_ticks = 10 if n_x > 15 else None

    user_rot = spec.get("axes_x_tick_rotation")
    if user_rot:  # 用户明确指定了非零旋转，直接使用；否则 post-draw 自动检测
        tick_rotation = int(user_rot)
        tick_ha = "right" if tick_rotation > 0 else "center"
    else:
        tick_rotation, tick_ha = 0, "center"

    rotate_labels = bool(spec.get("axes_x_rotate_labels", False))
    y_min, y_max = _compute_y_range(df, spec, y_col)

    return LayoutParams(
        figure_width=fig_w, figure_height=fig_h,
        font_size=font_size, legend_fontsize=legend_fontsize,
        tick_rotation=tick_rotation, tick_ha=tick_ha,
        legend_loc=legend_loc,
        y_min=y_min, y_max=y_max,
        bar_width=0.6,
        marker_size=marker_size,
        annot_fontsize=font_size,
        capsize=float(theme.line_width) * 2,
        x_max_ticks=x_max_ticks,
        rotate_labels=rotate_labels,
    )


def _layout_scatter(df: pd.DataFrame, spec: dict, theme: ThemeConfig) -> LayoutParams:
    n_points = len(df)
    group_col = spec.get("data_group_by")
    n_series = int(df[group_col].nunique()) if group_col and group_col in df.columns else 1

    fig_w = theme.figure_width
    legend_loc = _resolve_legend_loc(_legend_params(n_series if n_series > 1 else 0), spec)
    if legend_loc == "outside_right":
        fig_w += 1.2
    fig_h = fig_w * theme.aspect_ratio

    font_size = _scale_font(theme.font_size, fig_w, theme.figure_width)
    legend_fontsize = round(font_size * 0.85, 1)

    # scatter 的 s 参数是面积（pts²），公式随点密度收缩，夹在 [10, 80]
    user_msize = spec.get("params_marker_size")
    if user_msize is not None:
        marker_size = float(user_msize) ** 2  # 用户指定直径，转为面积
    else:
        marker_size = max(10.0, min(80.0, 60.0 / max(n_points ** 0.5, 1)))

    user_rot = spec.get("axes_x_tick_rotation")
    if user_rot:  # 用户明确指定了非零旋转，直接使用；否则 post-draw 自动检测
        tick_rotation = int(user_rot)
        tick_ha = "right" if tick_rotation > 0 else "center"
    else:
        tick_rotation, tick_ha = 0, "center"

    rotate_labels = bool(spec.get("axes_x_rotate_labels", False))
    y_col = spec.get("data_y", "")
    y_min, y_max = _compute_y_range(df, spec, y_col)

    return LayoutParams(
        figure_width=fig_w, figure_height=fig_h,
        font_size=font_size, legend_fontsize=legend_fontsize,
        tick_rotation=tick_rotation, tick_ha=tick_ha,
        legend_loc=legend_loc,
        y_min=y_min, y_max=y_max,
        bar_width=0.6,
        marker_size=marker_size,
        annot_fontsize=font_size,
        capsize=float(theme.line_width) * 2,
        x_max_ticks=None,
        rotate_labels=rotate_labels,
    )


def _layout_box(df: pd.DataFrame, spec: dict, theme: ThemeConfig) -> LayoutParams:
    x_col = spec.get("data_x", "")
    y_col = spec.get("data_y", "")
    n_cat = int(df[x_col].nunique()) if x_col in df.columns else 4

    min_width = n_cat * 0.5 + 1.2
    fig_w = max(theme.figure_width, min_width)
    legend_loc = _resolve_legend_loc("none", spec)
    if legend_loc == "outside_right":
        fig_w += 1.2
    fig_h = fig_w * theme.aspect_ratio
    bar_width = max(0.1, min(0.5, 0.6 / max(n_cat, 1)))

    font_size = _scale_font(theme.font_size, fig_w, theme.figure_width)
    legend_fontsize = round(font_size * 0.85, 1)

    user_rot = spec.get("axes_x_tick_rotation")
    if user_rot:  # 用户明确指定了非零旋转，直接使用；否则 post-draw 自动检测
        tick_rotation = int(user_rot)
        tick_ha = "right" if tick_rotation > 0 else "center"
    else:
        tick_rotation, tick_ha = 0, "center"

    rotate_labels = bool(spec.get("axes_x_rotate_labels", False))
    y_min, y_max = _compute_y_range(df, spec, y_col)

    return LayoutParams(
        figure_width=fig_w, figure_height=fig_h,
        font_size=font_size, legend_fontsize=legend_fontsize,
        tick_rotation=tick_rotation, tick_ha=tick_ha,
        legend_loc=legend_loc,
        y_min=y_min, y_max=y_max,
        bar_width=bar_width,
        marker_size=float(theme.font_size) * 0.4,
        annot_fontsize=font_size,
        capsize=float(theme.line_width) * 2,
        x_max_ticks=None,
        rotate_labels=rotate_labels,
    )


def _layout_heatmap(df: pd.DataFrame, spec: dict, theme: ThemeConfig) -> LayoutParams:
    x_col = spec.get("data_x", "")
    y_col = spec.get("data_y", "")
    filter_expr = spec.get("data_filter")
    dff = df.query(filter_expr) if filter_expr else df

    n_cols = int(dff[x_col].nunique()) if x_col in dff.columns else 5
    n_rows = int(dff[y_col].nunique()) if y_col in dff.columns else 5

    cell_size = 0.5  # 每格约 0.5 英寸
    fig_w = max(theme.figure_width, n_cols * cell_size + 1.5)
    legend_loc = _resolve_legend_loc("none", spec)
    if legend_loc == "outside_right":
        fig_w += 1.2
    fig_h = max(theme.figure_width * theme.aspect_ratio, n_rows * cell_size + 1.0)

    font_size = _scale_font(theme.font_size, fig_w, theme.figure_width)
    legend_fontsize = round(font_size * 0.85, 1)

    # 格子多时缩小注释字体；格子数 > 20 时缩到 6.0
    n_cells = n_cols * n_rows
    annot_fontsize = 6.0 if n_cells > 20 else round(font_size * 0.85, 1)

    user_rot = spec.get("axes_x_tick_rotation")
    if user_rot:  # 用户明确指定了非零旋转，直接使用；否则 post-draw 自动检测
        tick_rotation = int(user_rot)
        tick_ha = "right" if tick_rotation > 0 else "center"
    else:
        tick_rotation, tick_ha = 0, "center"

    rotate_labels = bool(spec.get("axes_x_rotate_labels", False))

    return LayoutParams(
        figure_width=fig_w, figure_height=fig_h,
        font_size=font_size, legend_fontsize=legend_fontsize,
        tick_rotation=tick_rotation, tick_ha=tick_ha,
        legend_loc=legend_loc,
        y_min=None, y_max=None,
        bar_width=0.6,
        marker_size=5.0,
        annot_fontsize=annot_fontsize,
        capsize=float(theme.line_width) * 2,
        x_max_ticks=None,
        rotate_labels=rotate_labels,
    )


_DISPATCHERS = {
    "bar":     _layout_bar,
    "line":    _layout_line,
    "scatter": _layout_scatter,
    "box":     _layout_box,
    "heatmap": _layout_heatmap,
}
