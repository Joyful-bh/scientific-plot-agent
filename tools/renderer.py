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
    # ============================================================
    # B线实现指引：删除下方两行 Mock，换成下面的真实实现框架
    # ============================================================
    #
    #   try:
    #       # 1. 取回 DataFrame（支持 CSV 路径和 cache_key 两种来源）
    #       df = _resolve_dataframe(data_source, spec)
    #
    #       # 2. 取主题配置（apply_theme 已实现，直接调用）
    #       theme = apply_theme(
    #           spec["style_theme"],
    #           spec.get("style_palette_override"),   # None 表示不覆盖配色
    #       )
    #
    #       # 3. 分发到对应子渲染器
    #       chart_type = spec["chart_type"]
    #       if chart_type not in RENDERERS:
    #           raise RenderError(f"不支持的图表类型：{chart_type}")
    #       fig = RENDERERS[chart_type](df, spec, theme)
    #
    #       # 4. 保存文件并返回路径
    #       out_path = _output_path(chart_type)
    #       fig.savefig(out_path, dpi=theme.dpi, bbox_inches="tight")
    #       plt.close(fig)
    #       return str(out_path)
    #
    #   except RenderError:
    #       raise
    #   except Exception as exc:
    #       raise RenderError(f"渲染失败：{exc}") from exc
    #
    # ============================================================
    placeholder = Path("data") / "placeholder.png"
    return str(placeholder)


# ---------------------------------------------------------------------------
# 公共辅助：将 ThemeConfig 应用到已有的 Figure/Axes —— B线实现
# ---------------------------------------------------------------------------

def _apply_theme_to_fig(fig: plt.Figure, ax: plt.Axes, theme: ThemeConfig) -> None:
    """
    将 ThemeConfig 的排版参数应用到 Figure/Axes。
    所有子渲染器在 return fig 前必须调用此函数，避免重复代码。
    """
    # ============================================================
    # B线实现指引
    # ============================================================
    #
    # 1. 字体
    #    import matplotlib as mpl
    #    mpl.rcParams["font.family"] = theme.font_family
    #    mpl.rcParams["font.size"]   = theme.font_size
    #
    # 2. 轴脊：只保留 theme.spines 里的，其余全部隐藏
    #    for side in ["left", "right", "top", "bottom"]:
    #        ax.spines[side].set_visible(side in theme.spines)
    #        if side in theme.spines:
    #            ax.spines[side].set_linewidth(theme.line_width)
    #
    # 3. 网格
    #    if theme.grid:
    #        ax.grid(True, linestyle=theme.grid_style, linewidth=0.5, alpha=0.7)
    #    else:
    #        ax.grid(False)
    #
    # 4. 图幅尺寸（英寸）
    #    fig.set_size_inches(theme.figure_width, theme.figure_height)
    #
    # 5. 刻度线方向（建议内向，符合 Nature/IEEE 风格）
    #    ax.tick_params(direction="in", width=theme.line_width,
    #                   labelsize=theme.font_size)
    #
    # ============================================================
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 子渲染器骨架 —— B线填入真实实现
# ---------------------------------------------------------------------------

def _render_bar(df: pd.DataFrame, spec: dict, theme: ThemeConfig) -> plt.Figure:
    """渲染柱状图。Mock实现，B线替换"""
    # ============================================================
    # B线实现指引
    # ============================================================
    #
    # 需要读取的 spec 字段：
    #   spec["data_x"]                   X轴列名，如 "method"
    #   spec["data_y"]                   Y轴列名，如 "accuracy"
    #   spec.get("data_group_by")        分组列名，如 "dataset"；None=不分组
    #   spec.get("data_error")           误差棒列名，如 "std"；None=不画误差棒
    #   spec.get("label_title")          图标题
    #   spec.get("label_x")              X轴标签
    #   spec.get("label_y")              Y轴标签
    #   spec.get("axes_y_min/max")       Y轴范围，None=自动
    #   spec.get("axes_x_tick_rotation") X轴刻度旋转角度，默认 0
    #   spec.get("params_orientation")   "vertical"（默认）或 "horizontal"
    #   spec.get("params_stacked")       True=堆叠，默认 False
    #   spec.get("params_sort")          "asc"/"desc"/None
    #   spec.get("params_show_values")   True=柱顶标数值，默认 False
    #
    # ── 无分组最简实现 ──────────────────────────────────────────
    #
    #   fig, ax = plt.subplots()
    #   x_col, y_col = spec["data_x"], spec["data_y"]
    #   err_col = spec.get("data_error")
    #
    #   if spec.get("params_sort") == "asc":
    #       df = df.sort_values(y_col)
    #   elif spec.get("params_sort") == "desc":
    #       df = df.sort_values(y_col, ascending=False)
    #
    #   bars = ax.bar(
    #       df[x_col], df[y_col],
    #       yerr=df[err_col] if err_col else None,
    #       color=theme.palette[0],
    #       linewidth=theme.line_width,
    #       capsize=3,
    #       error_kw={"linewidth": theme.line_width},
    #   )
    #
    #   if spec.get("params_show_values"):
    #       for bar in bars:
    #           ax.text(
    #               bar.get_x() + bar.get_width() / 2,
    #               bar.get_height(),
    #               f"{bar.get_height():.1f}",
    #               ha="center", va="bottom",
    #               fontsize=theme.font_size,
    #           )
    #
    # ── 有分组时推荐 seaborn ─────────────────────────────────────
    #
    #   group_col = spec.get("data_group_by")
    #   if group_col:
    #       import seaborn as sns
    #       palette = {g: c for g, c in zip(df[group_col].unique(), theme.palette)}
    #       sns.barplot(
    #           data=df, x=x_col, y=y_col, hue=group_col,
    #           palette=palette,
    #           linewidth=theme.line_width,
    #           ax=ax,
    #       )
    #       ax.legend(frameon=theme.legend_frameon, fontsize=theme.legend_fontsize)
    #
    # ── 通用收尾（所有分支都要执行）────────────────────────────
    #
    #   ax.set_title(spec.get("label_title", ""), fontsize=theme.font_size + 1)
    #   ax.set_xlabel(spec.get("label_x", ""),    fontsize=theme.font_size)
    #   ax.set_ylabel(spec.get("label_y", ""),    fontsize=theme.font_size)
    #   if spec.get("axes_y_min") is not None: ax.set_ylim(bottom=spec["axes_y_min"])
    #   if spec.get("axes_y_max") is not None: ax.set_ylim(top=spec["axes_y_max"])
    #   plt.xticks(rotation=spec.get("axes_x_tick_rotation", 0))
    #   _apply_theme_to_fig(fig, ax, theme)   # ← 必须调用
    #   return fig
    #
    # ============================================================
    raise NotImplementedError


def _render_line(df: pd.DataFrame, spec: dict, theme: ThemeConfig) -> plt.Figure:
    """渲染折线图。Mock实现，B线替换"""
    # ============================================================
    # B线实现指引
    # ============================================================
    #
    # 需要读取的 spec 字段：
    #   spec["data_x"]                 X轴列名，如 "step"
    #   spec["data_y"]                 Y轴列名，如 "val_loss"
    #   spec.get("data_group_by")      分组列名；None=画单条线
    #   spec.get("data_error")         误差带列名；None=不画置信区间
    #   spec.get("label_title/x/y")   标签
    #   spec.get("axes_y_min/max")     轴范围
    #   spec.get("params_markers")     True=画数据点标记，默认 True
    #   spec.get("params_smooth")      True=做平滑处理
    #
    # ── 无分组最简实现 ──────────────────────────────────────────
    #
    #   fig, ax = plt.subplots()
    #   x_col, y_col = spec["data_x"], spec["data_y"]
    #   marker = "o" if spec.get("params_markers", True) else None
    #
    #   ax.plot(
    #       df[x_col], df[y_col],
    #       marker=marker, markersize=3,
    #       linewidth=theme.line_width,
    #       color=theme.palette[0],
    #   )
    #
    #   # 误差带（data_error 列作为上下浮动宽度）
    #   err_col = spec.get("data_error")
    #   if err_col:
    #       ax.fill_between(
    #           df[x_col],
    #           df[y_col] - df[err_col],
    #           df[y_col] + df[err_col],
    #           alpha=0.2, color=theme.palette[0],
    #       )
    #
    # ── 有分组时按组循环 ─────────────────────────────────────────
    #
    #   group_col = spec.get("data_group_by")
    #   if group_col:
    #       for i, (gval, gdf) in enumerate(df.groupby(group_col)):
    #           color = theme.palette[i % len(theme.palette)]
    #           ax.plot(gdf[x_col], gdf[y_col],
    #                   label=str(gval), color=color,
    #                   linewidth=theme.line_width,
    #                   marker=marker, markersize=3)
    #       ax.legend(frameon=theme.legend_frameon, fontsize=theme.legend_fontsize)
    #
    # ── 通用收尾 ─────────────────────────────────────────────────
    #
    #   ax.set_title(spec.get("label_title", ""), fontsize=theme.font_size + 1)
    #   ax.set_xlabel(spec.get("label_x", ""),    fontsize=theme.font_size)
    #   ax.set_ylabel(spec.get("label_y", ""),    fontsize=theme.font_size)
    #   if spec.get("axes_y_min") is not None: ax.set_ylim(bottom=spec["axes_y_min"])
    #   if spec.get("axes_y_max") is not None: ax.set_ylim(top=spec["axes_y_max"])
    #   _apply_theme_to_fig(fig, ax, theme)   # ← 必须调用
    #   return fig
    #
    # ============================================================
    raise NotImplementedError


def _render_scatter(df: pd.DataFrame, spec: dict, theme: ThemeConfig) -> plt.Figure:
    """渲染散点图。Mock实现，B线替换"""
    # ============================================================
    # B线实现指引
    # ============================================================
    #
    # 需要读取的 spec 字段：
    #   spec["data_x"]                      X轴列名
    #   spec["data_y"]                      Y轴列名
    #   spec.get("data_group_by")           分组列名；None=单色
    #   spec.get("label_title/x/y")         标签
    #   spec.get("params_alpha", 0.8)       点透明度
    #   spec.get("params_show_regression")  True=叠加线性回归线
    #
    # ── 无分组最简实现 ──────────────────────────────────────────
    #
    #   fig, ax = plt.subplots()
    #   x_col, y_col = spec["data_x"], spec["data_y"]
    #
    #   ax.scatter(
    #       df[x_col], df[y_col],
    #       alpha=spec.get("params_alpha", 0.8),
    #       color=theme.palette[0], s=20,
    #       linewidths=theme.line_width,
    #   )
    #
    #   if spec.get("params_show_regression"):
    #       import numpy as np
    #       x_vals, y_vals = df[x_col].values, df[y_col].values
    #       coef = np.polyfit(x_vals, y_vals, 1)
    #       x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
    #       ax.plot(x_line, np.poly1d(coef)(x_line),
    #               "--", color=theme.palette[1], linewidth=theme.line_width)
    #
    #   ax.set_title(spec.get("label_title", ""), fontsize=theme.font_size + 1)
    #   ax.set_xlabel(spec.get("label_x", ""),    fontsize=theme.font_size)
    #   ax.set_ylabel(spec.get("label_y", ""),    fontsize=theme.font_size)
    #   _apply_theme_to_fig(fig, ax, theme)   # ← 必须调用
    #   return fig
    #
    # ============================================================
    raise NotImplementedError


def _render_box(df: pd.DataFrame, spec: dict, theme: ThemeConfig) -> plt.Figure:
    """渲染箱线图。Mock实现，B线替换"""
    # ============================================================
    # B线实现指引
    # ============================================================
    #
    # 需要读取的 spec 字段：
    #   spec["data_x"]                      分组列名（X轴类别）
    #   spec["data_y"]                      数值列名
    #   spec.get("label_title/x/y")         标签
    #   spec.get("params_show_points")      "all"/"outliers"（默认）/"none"
    #   spec.get("params_notch")            True=缺口箱线图，默认 False
    #
    # ── 推荐用 seaborn ──────────────────────────────────────────
    #
    #   import seaborn as sns
    #   fig, ax = plt.subplots()
    #
    #   show_pts = spec.get("params_show_points", "outliers")
    #   # seaborn boxplot 默认显示离群点；"none" 时通过 flierprops 隐藏
    #   flierprops = {"marker": ""} if show_pts == "none" else {}
    #
    #   sns.boxplot(
    #       data=df, x=spec["data_x"], y=spec["data_y"],
    #       notch=spec.get("params_notch", False),
    #       palette=theme.palette,
    #       linewidth=theme.line_width,
    #       flierprops=flierprops,
    #       ax=ax,
    #   )
    #
    #   # "all" 时额外叠加所有数据点（stripplot）
    #   if show_pts == "all":
    #       sns.stripplot(
    #           data=df, x=spec["data_x"], y=spec["data_y"],
    #           color="black", alpha=0.4, size=2, ax=ax,
    #       )
    #
    #   ax.set_title(spec.get("label_title", ""), fontsize=theme.font_size + 1)
    #   ax.set_xlabel(spec.get("label_x", ""),    fontsize=theme.font_size)
    #   ax.set_ylabel(spec.get("label_y", ""),    fontsize=theme.font_size)
    #   _apply_theme_to_fig(fig, ax, theme)   # ← 必须调用
    #   return fig
    #
    # ============================================================
    raise NotImplementedError


def _render_heatmap(df: pd.DataFrame, spec: dict, theme: ThemeConfig) -> plt.Figure:
    """渲染热力图。Mock实现，B线替换"""
    # ============================================================
    # B线实现指引
    # ============================================================
    #
    # 需要读取的 spec 字段：
    #   spec["data_x"]                      透视后列方向的列名（columns）
    #   spec["data_y"]                      透视后行方向的列名（index）
    #   spec.get("data_filter")             pandas query 字符串；None=不过滤
    #   spec.get("style_palette_override")  若为 "coolwarm"，cmap="coolwarm"
    #   spec.get("params_annot", True)      True=单元格标数值
    #   spec.get("params_fmt", ".2f")       数值格式字符串
    #
    # ── 长表转宽表后用 seaborn heatmap ──────────────────────────
    #
    #   import seaborn as sns
    #   from tools.themes import PALETTES
    #
    #   # 1. 行过滤
    #   filter_expr = spec.get("data_filter")
    #   if filter_expr:
    #       df = df.query(filter_expr)
    #
    #   # 2. 确定数值列（取 df 中第一个数值型列作为热力值）
    #   x_col, y_col = spec["data_x"], spec["data_y"]
    #   num_cols = [c for c in df.columns if c not in (x_col, y_col)
    #               and pd.api.types.is_numeric_dtype(df[c])]
    #   if not num_cols:
    #       raise RenderError("heatmap 需要至少一个数值列作为热力值")
    #   val_col = num_cols[0]
    #
    #   # 3. 长表 → 宽表
    #   pivot = df.pivot(index=y_col, columns=x_col, values=val_col)
    #
    #   # 4. 确定 cmap
    #   override = spec.get("style_palette_override")
    #   cmap = "coolwarm" if override == "coolwarm" else "Blues"
    #
    #   # 5. 绘图
    #   fig, ax = plt.subplots()
    #   sns.heatmap(
    #       pivot,
    #       annot=spec.get("params_annot", True),
    #       fmt=spec.get("params_fmt", ".2f"),
    #       cmap=cmap,
    #       linewidths=0.3,
    #       ax=ax,
    #   )
    #
    #   ax.set_title(spec.get("label_title", ""), fontsize=theme.font_size + 1)
    #   _apply_theme_to_fig(fig, ax, theme)   # ← 必须调用
    #   return fig
    #
    # ============================================================
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
    """生成带时间戳的输出文件路径。格式：output/plot_20240101_120000_bar.png"""
    _ensure_output_dir()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return OUTPUT_DIR / f"plot_{timestamp}_{chart_type}.png"


def _resolve_dataframe(data_source: str, spec: dict) -> pd.DataFrame:
    """根据 data_source 取回 DataFrame（缓存键或 CSV 路径均支持）。"""
    if data_source.startswith("cache://"):
        return get_dataframe(data_source)
    _, cache_key = load_data(data_source)
    return get_dataframe(cache_key)
