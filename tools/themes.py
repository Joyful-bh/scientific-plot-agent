"""
B线工具：主题与调色板注册表。
提供 ThemeConfig dataclass、THEMES 字典、PALETTES 字典及 apply_theme() 函数。
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import matplotlib.pyplot as plt

from schema import STYLE_THEMES, PALETTE_OVERRIDES


@dataclass
class ThemeConfig:
    """单个主题的完整排版与配色配置（静态，数据无关）。"""

    font_family: str
    font_size: int          # 标准图幅下的基准字号（磅），LayoutEngine 按图幅缩放后得实际字号
    line_width: float
    figure_width: float     # 最小基准图幅宽度（英寸），LayoutEngine 只扩大不缩小
    aspect_ratio: float     # 宽高比：figure_height = figure_width × aspect_ratio
    dpi: int
    spines: list[str]
    grid: bool
    grid_style: str
    legend_frameon: bool
    palette: list[str]
    bg_color: str = "white"         # 背景色；dark 主题使用深色
    text_color: str = "black"       # 文字/刻度/轴脊颜色；dark 主题使用浅色
    hatch: str | None = None        # 柱子纹理（None=不使用）；仅 bar 图生效
    edgecolor: str | None = None    # 柱子/纹理边框颜色（None=matplotlib默认）；仅 bar 图生效
    hatch_linewidth: float = 0.5    # 纹理线宽；仅 hatch 不为 None 时生效


def _tab10_colors() -> list[str]:
    """从 matplotlib tab10 colormap 提取十六进制颜色列表。"""
    cmap = plt.cm.tab10
    return [
        "#{:02x}{:02x}{:02x}".format(
            int(cmap(i)[0] * 255),
            int(cmap(i)[1] * 255),
            int(cmap(i)[2] * 255),
        )
        for i in range(10)
    ]


THEMES: dict[str, ThemeConfig] = {
    "nature": ThemeConfig(
        font_family="Arial",
        font_size=7,
        line_width=0.75,
        figure_width=3.5,
        aspect_ratio=0.75,      # 原 3.5 × 2.625
        dpi=300,
        spines=["left", "bottom"],
        grid=False,
        grid_style="--",
        legend_frameon=False,
        palette=["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F", "#8491B4", "#91D1C2"],
    ),
    "ieee": ThemeConfig(
        font_family="Times New Roman",
        font_size=8,
        line_width=0.5,
        figure_width=3.5,
        aspect_ratio=0.75,      # 原 3.5 × 2.625
        dpi=300,
        spines=["left", "bottom"],
        grid=True,
        grid_style="--",
        legend_frameon=False,
        palette=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"],
    ),
    "vivid": ThemeConfig(
        font_family="DejaVu Sans",
        font_size=10,
        line_width=1.5,
        figure_width=6.0,
        aspect_ratio=0.667,     # 原 6.0 × 4.0
        dpi=150,
        spines=["left", "bottom"],
        grid=True,
        grid_style="--",
        legend_frameon=True,
        palette=["#E63946", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4", "#FF5722"],
    ),
    "morandi": ThemeConfig(
        font_family="Arial",
        font_size=9,
        line_width=1.0,
        figure_width=5.0,
        aspect_ratio=0.75,      # 原 5.0 × 3.75
        dpi=150,
        spines=["left", "bottom"],
        grid=False,
        grid_style="--",
        legend_frameon=False,
        palette=["#8B9BAB", "#C4A882", "#9CAF88", "#B89BAD", "#A89888", "#C4B8A8", "#88A0A8"],
    ),
    "clean": ThemeConfig(
        font_family="DejaVu Sans",
        font_size=10,
        line_width=1.0,
        figure_width=6.0,
        aspect_ratio=0.667,     # 原 6.0 × 4.0
        dpi=150,
        spines=["left", "bottom"],
        grid=False,
        grid_style="--",
        legend_frameon=True,
        palette=["#5C85A4", "#A4785C", "#7AA45C", "#A45C7A", "#7C5CA4", "#5CA48C", "#A4A45C"],
    ),
    "dark": ThemeConfig(
        font_family="DejaVu Sans",
        font_size=11,
        line_width=1.5,
        figure_width=7.0,
        aspect_ratio=0.667,     # 原 7.0 × 4.5，按设计规范取 0.667
        dpi=150,
        spines=["left", "bottom"],
        grid=True,
        grid_style="--",
        legend_frameon=True,
        palette=["#61DAFB", "#F7B731", "#A3E635", "#FB7185", "#C084FC", "#34D399", "#F97316"],
        bg_color="#1e1e2e",
        text_color="#cdd6f4",
    ),
}

# 配色覆盖注册表；"coolwarm" 是字符串，供 heatmap cmap 参数使用
PALETTES: dict[str, list[str] | str] = {
    "morandi":  ["#8B9BAB", "#C4A882", "#9CAF88", "#B89BAD", "#A89888"],
    "nature_d": ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F"],
    "tab10":    _tab10_colors(),
    "coolwarm": "coolwarm",
}


def apply_theme(
    theme_name: str,
    palette_override: str | None = None,
) -> ThemeConfig:
    """
    返回指定主题的 ThemeConfig；若有 palette_override 则替换调色板字段。

    Args:
        theme_name:       STYLE_THEMES 中的主题名称。
        palette_override: PALETTE_OVERRIDES 中的覆盖名称，或 None。

    Returns:
        ThemeConfig 实例（新对象，不修改注册表中的原始配置）。

    Raises:
        ValueError: 主题名或覆盖名不在合法枚举内时。
    """
    if theme_name not in STYLE_THEMES:
        raise ValueError(f"未知主题：{theme_name}，合法值：{STYLE_THEMES}")

    base = THEMES[theme_name]

    if palette_override is None:
        return dataclasses.replace(base)

    if palette_override not in PALETTE_OVERRIDES:
        raise ValueError(f"未知配色覆盖：{palette_override}，合法值：{PALETTE_OVERRIDES}")

    override_palette = PALETTES[palette_override]
    # coolwarm 是字符串（heatmap 专用），不能直接赋给 palette list
    new_palette = override_palette if isinstance(override_palette, list) else base.palette

    return dataclasses.replace(base, palette=new_palette)


# spec 字段名 → ThemeConfig 字段名的映射（style_* 覆写通道）
_STYLE_OVERRIDE_MAP: dict[str, str] = {
    "style_grid":           "grid",
    "style_line_width":     "line_width",
    "style_font_size":      "font_size",
    "style_hatch":          "hatch",
    "style_edgecolor":      "edgecolor",
    "style_hatch_linewidth": "hatch_linewidth",
    "style_dpi":            "dpi",
    "style_legend_frameon": "legend_frameon",
    "style_bg_color":       "bg_color",
    "style_text_color":     "text_color",
    "style_aspect_ratio":   "aspect_ratio",
    "style_figure_width":   "figure_width",
    "style_font_family":    "font_family",
    "style_spines":         "spines",
}


def apply_style_overrides(theme: ThemeConfig, spec: dict) -> ThemeConfig:
    """
    将 spec 中的 style_* 覆写字段应用到 ThemeConfig，返回新实例。
    值为 None 的字段跳过（保留主题默认）。
    在 apply_theme() 之后调用。
    """
    overrides = {
        theme_key: spec[spec_key]
        for spec_key, theme_key in _STYLE_OVERRIDE_MAP.items()
        if spec.get(spec_key) is not None
    }
    return dataclasses.replace(theme, **overrides) if overrides else theme
