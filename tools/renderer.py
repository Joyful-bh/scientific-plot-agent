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
import numpy as np

from tools.loader import get_dataframe, load_data
from tools.themes import ThemeConfig, apply_theme

OUTPUT_DIR = Path("output")


class RenderError(Exception):
    """图表渲染失败时抛出。"""
#new_in function
def _prepare_axes(
    spec: dict,
    theme: ThemeConfig,
) -> tuple[plt.Figure, plt.Axes]:
    """
    创建 Figure/Axes 并统一处理：
    - figsize
    - dpi
    - title
    - xlabel/ylabel
    - 坐标轴范围
    """

    fig, ax = plt.subplots(
        figsize=(theme.figure_width, theme.figure_height),
        dpi=theme.dpi,
    )

    ax.set_title(
        spec.get("label_title", ""),
        fontsize=theme.font_size + 1,
    )

    ax.set_xlabel(
        spec.get("label_x", ""),
        fontsize=theme.font_size,
    )

    ax.set_ylabel(
        spec.get("label_y", ""),
        fontsize=theme.font_size,
    )

    return fig, ax

def _apply_axis_limits(
    ax: plt.Axes,
    spec: dict,
    horizontal: bool,
) -> None:
    """
    应用坐标轴范围限制。
    """

    y_min = spec.get("axes_y_min")
    y_max = spec.get("axes_y_max")

    if horizontal:
        ax.set_xlim(
            left=y_min,
            right=y_max,
        )
    else:
        ax.set_ylim(
            bottom=y_min,
            top=y_max,
        )

def _add_bar_values(
    ax: plt.Axes,
    bars,
    horizontal: bool,
    theme: ThemeConfig,
) -> None:
    """
    为柱状图添加数值标签。
    """

    for bar in bars:

        if horizontal:

            value = bar.get_width()

            ax.text(
                value,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}",
                va="center",
                ha="left",
                fontsize=theme.font_size,
            )

        else:

            value = bar.get_height()

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=theme.font_size,
            )


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
    try:
    #       # 1. 取回 DataFrame（支持 CSV 路径和 cache_key 两种来源）
        df = _resolve_dataframe(data_source, spec)
    #
    #       # 2. 取主题配置（apply_theme 已实现，直接调用）
        theme = apply_theme(
            spec["style_theme"],
            spec.get("style_palette_override"),   # None 表示不覆盖配色
       )
    #
    #       # 3. 分发到对应子渲染器
        chart_type = spec["chart_type"]
       
        if chart_type not in RENDERERS:
           raise RenderError(
                f"不支持的图表类型：{chart_type}"
            )

        renderer = RENDERERS[chart_type]
        fig = renderer(
            df=df,
            spec=spec,
            theme=theme,
        )
    #       # 4. 保存文件并返回路径
        out_path = _output_path(chart_type)
        fig.savefig(out_path, dpi=theme.dpi, bbox_inches="tight")
        plt.close(fig)
        return str(out_path)
    #
    except RenderError:
           raise
    except Exception as exc:
           raise RenderError(f"渲染失败：{exc}") from exc
    #
    # ============================================================


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
    import matplotlib as mpl
    mpl.rcParams["font.family"] = theme.font_family
    mpl.rcParams["font.size"]   = theme.font_size
    #
    # 2. 轴脊：只保留 theme.spines 里的，其余全部隐藏
    for side in ["left", "right", "top", "bottom"]:
        
        ax.spines[side].set_visible(side in theme.spines)
        
        if side in theme.spines:
            ax.spines[side].set_linewidth(theme.line_width)
    # 3. 网格
    if theme.grid:
        ax.grid(True, linestyle=theme.grid_style, linewidth=0.5, alpha=0.7)
    else:
        ax.grid(False)
    #
    # 4. 图幅尺寸（英寸）
    fig.set_size_inches(theme.figure_width, theme.figure_height)
    #
    # 5. 刻度线方向（建议内向，符合 Nature/IEEE 风格）
    ax.tick_params(direction="in", width=theme.line_width,
                   labelsize=theme.font_size)
    #
    # 6. 背景色与文字色（dark 主题时非白底，需整体设置）
    fig.patch.set_facecolor(theme.bg_color)
    ax.set_facecolor(theme.bg_color)
    if theme.bg_color != "white":
        for item in [ax.title, ax.xaxis.label, ax.yaxis.label]:
            item.set_color(theme.text_color)
        ax.tick_params(colors=theme.text_color)
        for spine in ax.spines.values():
            spine.set_edgecolor(theme.text_color)
    #
    # ============================================================


# ---------------------------------------------------------------------------
# 子渲染器骨架 —— B线填入真实实现
# ---------------------------------------------------------------------------

def _render_bar(df: pd.DataFrame, spec: dict, theme: ThemeConfig) -> plt.Figure:
    """渲染柱状图。Mock实现，B线替换"""

    
    x_col = spec["data_x"]
    y_col = spec["data_y"]

    group_col = spec.get("data_group_by")
    err_col = spec.get("data_error")

    horizontal = (
        spec.get("params_orientation", "vertical")
        == "horizontal"
    )

    stacked = spec.get("params_stacked", False)
    sort_mode = spec.get("params_sort")
    show_values = spec.get("params_show_values", False)

    # ============================================================
    # 排序
    # ============================================================

    if sort_mode == "asc":
        df = df.sort_values(y_col)

    elif sort_mode == "desc":
        df = df.sort_values(y_col, ascending=False)

    # ============================================================
    # Figure / Axes
    # ============================================================

    fig, ax = _prepare_axes(spec, theme)

    # ============================================================
    # 无分组
    # ============================================================

    if group_col is None:

        if horizontal:

            bars = ax.barh(
                df[x_col],
                df[y_col],
                xerr=df[err_col] if err_col else None,
                color=theme.palette[0],
                linewidth=theme.line_width,
                capsize=3,
            )

        else:

            bars = ax.bar(
                df[x_col],
                df[y_col],
                yerr=df[err_col] if err_col else None,
                color=theme.palette[0],
                linewidth=theme.line_width,
                capsize=3,
            )

    # ============================================================
    # 分组
    # ============================================================

    else:

        pivot = df.pivot(
            index=x_col,
            columns=group_col,
            values=y_col,
        )

        x = np.arange(len(pivot.index))
        groups = list(pivot.columns)

        # --------------------------------------------------------
        # 堆叠柱状图
        # --------------------------------------------------------

        if stacked:

            acc = np.zeros(len(pivot.index))

            for i, g in enumerate(groups):

                values = pivot[g].values

                if horizontal:

                    bars = ax.barh(
                        pivot.index,
                        values,
                        left=acc,
                        color=theme.palette[i % len(theme.palette)],
                        linewidth=theme.line_width,
                        label=str(g),
                    )

                else:

                    bars = ax.bar(
                        x,
                        values,
                        bottom=acc,
                        color=theme.palette[i % len(theme.palette)],
                        linewidth=theme.line_width,
                        label=str(g),
                    )

                acc += values

        # --------------------------------------------------------
        # 并列柱状图
        # --------------------------------------------------------

        else:

            width = 0.8 / len(groups)

            for i, g in enumerate(groups):

                values = pivot[g].values

                offset = (
                    (i - len(groups) / 2) * width
                    + width / 2
                )

                if horizontal:

                    bars = ax.barh(
                        x + offset,
                        values,
                        height=width,
                        color=theme.palette[i % len(theme.palette)],
                        linewidth=theme.line_width,
                        label=str(g),
                    )

                    ax.set_yticks(x)
                    ax.set_yticklabels(pivot.index)

                else:

                    bars = ax.bar(
                        x + offset,
                        values,
                        width=width,
                        color=theme.palette[i % len(theme.palette)],
                        linewidth=theme.line_width,
                        label=str(g),
                    )

                    ax.set_xticks(x)
                    ax.set_xticklabels(pivot.index)

        ax.legend(
            frameon=theme.legend_frameon,
            fontsize=theme.legend_fontsize,
        )

    # ============================================================
    # 数值标注
    # ============================================================

    if show_values:
        _add_bar_values(
            ax=ax,
            bars=bars,
            horizontal=horizontal,
            theme=theme,
        )

    # ============================================================
    # 坐标轴范围
    # ============================================================

    _apply_axis_limits(
        ax=ax,
        spec=spec,
        horizontal=horizontal,
    )

    # ============================================================
    # 刻度旋转
    # ============================================================

    plt.xticks(
        rotation=spec.get(
            "axes_x_tick_rotation",
            0,
        )
    )

    # ============================================================
    # 应用主题
    # ============================================================

    _apply_theme_to_fig(fig, ax, theme)

    return fig
    


def _render_line(df: pd.DataFrame, spec: dict, theme: ThemeConfig) -> plt.Figure:
    """渲染折线图。Mock实现，B线替换"""
    # ============================================================
    # B线实现指引
    # ============================================================
    
    # 需要读取的 spec 字段：
    #   spec["data_x"]                      X轴列名（字符串），如 "step"
    #   spec["data_y"]                      Y轴列名，可以是：
    #                                         - 字符串：单条线，如 "val_loss"
    #                                         - 字符串列表：多条线，如 ["train_loss", "val_loss"]
    #   spec.get("data_group_by")           分组列名；data_y 是列表时忽略此字段
    #   spec.get("data_error")              误差带列名；None=不画置信区间
    #   spec.get("label_title/x/y")         标签
    #   spec.get("axes_y_min/max")          轴范围
    #   spec.get("params_markers")          True=画数据点标记，默认 True；False=纯折线
    #   spec.get("params_smooth")           True=做平滑处理（可用 scipy.interpolate）
    #   spec.get("params_linestyle")        线型："solid"/"dashed"/"dotted"/"dashdot"
    #                                       所有线统一，默认 "solid"
    #   spec.get("params_line_colors")      颜色列表，如 ["#E64B35","#4DBBD5"]
    #                                       按线顺序覆盖 theme.palette；None=使用主题配色
    #   spec.get("params_marker_style")     标记样式，如 "o" "s" "^" "D" "v" "P" "*"
    #                                       None 时默认 "o"；params_markers=False 时忽略
    #   spec.get("params_marker_size", 4)   标记大小（点径，非面积）
    x_col = spec["data_x"]
    y_spec = spec["data_y"]

    group_col = spec.get("data_group_by")
    err_col = spec.get("data_error")
    # ── 线型与颜色的使用方式 ─────────────────────────────────────
    #
    #linestyle   = spec.get("params_linestyle", "solid")
    #line_colors = spec.get("params_line_colors")       # None 或 list[str]
    #palette     = line_colors if line_colors else theme.palette
    linestyle = spec.get(
        "params_linestyle",
        "solid",
    )

    line_colors = spec.get(
        "params_line_colors"
    )

    palette = (
        line_colors
        if line_colors
        else theme.palette
    )

    use_markers = spec.get(
        "params_markers",
        True,
    )
    #   # matplotlib linestyle 值与 ax.plot 的 linestyle 参数直接对应：
    #   # "solid" → "-"，"dashed" → "--"，"dotted" → ":"，"dashdot" → "-."
    #   # 也可以直接传 "solid"/"dashed" 等字符串，matplotlib 同样接受
    #
    #   ax.plot(..., linestyle=linestyle, color=palette[i % len(palette)])
    #
    # ── 标记样式的使用方式 ───────────────────────────────────────
    #
    #use_markers = spec.get("params_markers", True)
    #marker      = (spec.get("params_marker_style") or "o") if use_markers else None
    msize       = spec.get("params_marker_size", 4)   # ax.plot 的 markersize 是点径
    marker = (
        spec.get("params_marker_style") or "o"
    ) if use_markers else None

    msize = spec.get(
        "params_marker_size",
        4,
    )

    smooth = spec.get(
        "params_smooth",
        False,
    )
    # ============================================================
    # Figure / Axes
    # ============================================================

    fig, ax = _prepare_axes(spec, theme)

    # ============================================================
    # 平滑函数
    # ============================================================

    def _smooth_xy(x, y):

        if not smooth:
            return x, y

        try:

            import numpy as np
            from scipy.interpolate import make_interp_spline

            x = np.asarray(x)
            y = np.asarray(y)

            x_new = np.linspace(
                x.min(),
                x.max(),
                300,
            )

            spline = make_interp_spline(x, y)

            y_new = spline(x_new)

            return x_new, y_new

        except Exception:

            return x, y
    # ── 分支一：data_y 是字符串（单条线）───────────────────────
    if isinstance(y_spec, list):

        y_cols = y_spec

        for i, y_col in enumerate(y_cols):

            color = palette[i % len(palette)]

            x_data, y_data = _smooth_xy(
                df[x_col],
                df[y_col],
            )

            ax.plot(
                x_data,
                y_data,
                label=y_col,
                color=color,
                linestyle=linestyle,
                linewidth=theme.line_width,
                marker=marker,
                markersize=msize,
            )

        ax.legend(
            frameon=theme.legend_frameon,
            fontsize=theme.legend_fontsize,
        )

    # ── 分支二：data_y 是字符串列表（多条线）────────────────────
    else:

        y_col = y_spec

        # --------------------------------------------------------
        # 分组折线
        # --------------------------------------------------------

        if group_col:

            for i, (gval, gdf) in enumerate(
                df.groupby(group_col)
            ):

                color = palette[i % len(palette)]

                x_data, y_data = _smooth_xy(
                    gdf[x_col],
                    gdf[y_col],
                )

                ax.plot(
                    x_data,
                    y_data,
                    label=str(gval),
                    color=color,
                    linestyle=linestyle,
                    linewidth=theme.line_width,
                    marker=marker,
                    markersize=msize,
                )

            ax.legend(
                frameon=theme.legend_frameon,
                fontsize=theme.legend_fontsize,
            )

        else:

            x_data, y_data = _smooth_xy(
                df[x_col],
                df[y_col],
            )

            ax.plot(
                x_data,
                y_data,
                color=palette[0],
                linestyle=linestyle,
                linewidth=theme.line_width,
                marker=marker,
                markersize=msize,
            )

            # ----------------------------------------------------
            # 误差带
            # ----------------------------------------------------

            if err_col:

                ax.fill_between(
                    df[x_col],
                    df[y_col] - df[err_col],
                    df[y_col] + df[err_col],
                    alpha=0.2,
                    color=palette[0],
                )

    # ============================================================
    # 坐标轴范围
    # ============================================================

    y_min = spec.get("axes_y_min")
    y_max = spec.get("axes_y_max")

    ax.set_ylim(
        bottom=y_min,
        top=y_max,
    )

    # ============================================================
    # 应用主题
    # ============================================================

    _apply_theme_to_fig(
        fig,
        ax,
        theme,
    )

    return fig
    
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



def _render_scatter(df: pd.DataFrame, spec: dict, theme: ThemeConfig) -> plt.Figure:
    """渲染散点图。Mock实现，B线替换"""
    # ============================================================
    # B线实现指引
    # ============================================================
    x_col = spec["data_x"]
    y_col = spec["data_y"]
    # 需要读取的 spec 字段：
    #   spec["data_x"]                      X轴列名
    #   spec["data_y"]                      Y轴列名
    #   spec.get("data_group_by")           分组列名；None=单色
    #   spec.get("label_title/x/y")         标签
    #   spec.get("params_alpha", 0.8)       点透明度
    #   spec.get("params_show_regression")  True=叠加线性回归线
    #   spec.get("params_marker_style")     标记样式，如 "o" "s" "^" "D" "v" "P" "*"
    #                                       None 时使用 matplotlib 默认（"o"）
    #   spec.get("params_marker_size", 4)   标记大小（点径）
    group_col = spec.get("data_group_by")

    alpha = spec.get(
        "params_alpha",
        0.8,
    )

    show_regression = spec.get(
        "params_show_regression",
        False,
    )

    marker = (
        spec.get("params_marker_style")
        or "o"
    )

    msize = spec.get(
        "params_marker_size",
        4,
    )
    fig, ax = _prepare_axes(spec, theme)
    if group_col is None:

        ax.scatter(
            df[x_col],
            df[y_col],
            marker=marker,
            s=msize ** 2,
            alpha=alpha,
            color=theme.palette[0],
            linewidths=theme.line_width,
        )
    if show_regression:

            x_vals = df[x_col].values
            y_vals = df[y_col].values

            coef = np.polyfit(
                x_vals,
                y_vals,
                1,
            )

            x_line = np.linspace(
                x_vals.min(),
                x_vals.max(),
                100,
            )

            y_line = np.poly1d(coef)(x_line)

            ax.plot(
                x_line,
                y_line,
                "--",
                color=theme.palette[1],
                linewidth=theme.line_width,
            )
    else:

        for i, (gval, gdf) in enumerate(
            df.groupby(group_col)
        ):

            color = theme.palette[
                i % len(theme.palette)
            ]

            ax.scatter(
                gdf[x_col],
                gdf[y_col],
                marker=marker,
                s=msize ** 2,
                alpha=alpha,
                color=color,
                linewidths=theme.line_width,
                label=str(gval),
            )

            # ----------------------------------------------------
            # 分组回归线
            # ----------------------------------------------------

            if show_regression:

                x_vals = gdf[x_col].values
                y_vals = gdf[y_col].values

                if len(x_vals) >= 2:

                    coef = np.polyfit(
                        x_vals,
                        y_vals,
                        1,
                    )

                    x_line = np.linspace(
                        x_vals.min(),
                        x_vals.max(),
                        100,
                    )

                    y_line = np.poly1d(coef)(
                        x_line
                    )

                    ax.plot(
                        x_line,
                        y_line,
                        "--",
                        color=color,
                        linewidth=theme.line_width,
                    )

        ax.legend(
            frameon=theme.legend_frameon,
            fontsize=theme.legend_fontsize,
        )
    # ============================================================
    # 坐标轴范围
    # ============================================================

    y_min = spec.get("axes_y_min")
    y_max = spec.get("axes_y_max")

    ax.set_ylim(
        bottom=y_min,
        top=y_max,
    )

    # ============================================================
    # 应用主题
    # ============================================================

    _apply_theme_to_fig(
        fig,
        ax,
        theme,
    )

    return fig
    

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
    import seaborn as sns

    x_col = spec["data_x"]
    y_col = spec["data_y"]
    show_points = spec.get(
        "params_show_points",
        "outliers",
    )

    notch = spec.get(
        "params_notch",
        False,
    )

    # ============================================================
    # Figure / Axes
    # ============================================================

    fig, ax = _prepare_axes(spec, theme)

    flierprops = (
        {"marker": ""}
        if show_points == "none"
        else {}
    )
    # ============================================================
    # 箱线图
    # ============================================================

    sns.boxplot(
        data=df,
        x=x_col,
        y=y_col,
        notch=notch,
        palette=theme.palette,
        linewidth=theme.line_width,
        flierprops=flierprops,
        ax=ax,
    )

    # ============================================================
    # 显示全部数据点
    # ============================================================

    if show_points == "all":

        sns.stripplot(
            data=df,
            x=x_col,
            y=y_col,
            color="black",
            alpha=0.4,
            size=2,
            ax=ax,
        )

    # ============================================================
    # Y轴范围
    # ============================================================

    y_min = spec.get("axes_y_min")
    y_max = spec.get("axes_y_max")

    ax.set_ylim(
        bottom=y_min,
        top=y_max,
    )

    # ============================================================
    # 应用主题
    # ============================================================

    _apply_theme_to_fig(
        fig,
        ax,
        theme,
    )

    return fig


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
    import seaborn as sns
    from tools.themes import PALETTES
    #
    #   # 1. 行过滤
    filter_expr = spec.get("data_filter")
    if filter_expr:
        df = df.query(filter_expr)
    #
    #   # 2. 确定数值列（取 df 中第一个数值型列作为热力值）
    x_col = spec["data_x"]
    y_col = spec["data_y"]
    
    num_cols = [c for c in df.columns if c not in (x_col, y_col)
                and pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        raise RenderError("heatmap 需要至少一个数值列作为热力值")
    val_col = num_cols[0]
    #
    #   # 3. 长表 → 宽表
    pivot = df.pivot(index=y_col, columns=x_col, values=val_col)
    #
    #   # 4. 确定 cmap
    override = spec.get("style_palette_override")
    cmap = "coolwarm" if override == "coolwarm" else "Blues"
    #
    #   # 5. 绘图
    fig, ax = plt.subplots()
    sns.heatmap(
        pivot,
        annot=spec.get("params_annot", True),
        fmt=spec.get("params_fmt", ".2f"),
        cmap=cmap,
        linewidths=0.3,
        ax=ax,
    )
    #
    ax.set_title(spec.get("label_title", ""), fontsize=theme.font_size + 1)
    _apply_theme_to_fig(fig, ax, theme)   # ← 必须调用
    return fig
    #
    # ============================================================


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
