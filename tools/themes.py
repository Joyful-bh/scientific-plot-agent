"""
B线工具：主题与调色板注册表。
提供 ThemeConfig dataclass、THEMES 字典、PALETTES 字典及 apply_theme() 函数。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt

from schema import STYLE_THEMES, PALETTE_OVERRIDES


@dataclass
class ThemeConfig:
    """单个主题的完整排版与配色配置。"""

    font_family: str
    font_size: int
    line_width: float
    figure_width: float
    figure_height: float
    dpi: int
    spines: list[str]
    grid: bool
    grid_style: str
    legend_frameon: bool
    legend_fontsize: int
    palette: list[str]
    bg_color: str = "white"    # 背景色；dark 主题使用深色
    text_color: str = "black"  # 文字/刻度/轴脊颜色；dark 主题使用浅色


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
        figure_height=2.625,
        dpi=300,
        spines=["left", "bottom"],
        grid=False,
        grid_style="--",
        legend_frameon=False,
        legend_fontsize=6,
        palette=["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F", "#8491B4", "#91D1C2"],
    ),
    "ieee": ThemeConfig(
        font_family="Times New Roman",
        font_size=8,
        line_width=0.5,
        figure_width=3.5,
        figure_height=2.625,
        dpi=300,
        spines=["left", "bottom"],
        grid=True,
        grid_style="--",
        legend_frameon=False,
        legend_fontsize=7,
        palette=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"],
    ),
    "vivid": ThemeConfig(
        font_family="DejaVu Sans",
        font_size=10,
        line_width=1.5,
        figure_width=6.0,
        figure_height=4.0,
        dpi=150,
        spines=["left", "bottom"],
        grid=True,
        grid_style="--",
        legend_frameon=True,
        legend_fontsize=9,
        palette=["#E63946", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4", "#FF5722"],
    ),
    "morandi": ThemeConfig(
        font_family="Arial",
        font_size=9,
        line_width=1.0,
        figure_width=5.0,
        figure_height=3.75,
        dpi=150,
        spines=["left", "bottom"],
        grid=False,
        grid_style="--",
        legend_frameon=False,
        legend_fontsize=8,
        palette=["#8B9BAB", "#C4A882", "#9CAF88", "#B89BAD", "#A89888", "#C4B8A8", "#88A0A8"],
    ),
    "clean": ThemeConfig(
        font_family="DejaVu Sans",
        font_size=10,
        line_width=1.0,
        figure_width=6.0,
        figure_height=4.0,
        dpi=150,
        spines=["left", "bottom"],
        grid=False,
        grid_style="--",
        legend_frameon=True,
        legend_fontsize=9,
        palette=["#5C85A4", "#A4785C", "#7AA45C", "#A45C7A", "#7C5CA4", "#5CA48C", "#A4A45C"],
    ),
    "dark": ThemeConfig(
        font_family="DejaVu Sans",
        font_size=11,
        line_width=1.5,
        figure_width=7.0,
        figure_height=4.5,
        dpi=150,
        spines=["left", "bottom"],
        grid=True,
        grid_style="--",
        legend_frameon=True,
        legend_fontsize=10,
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
        return ThemeConfig(**base.__dict__)

    if palette_override not in PALETTE_OVERRIDES:
        raise ValueError(f"未知配色覆盖：{palette_override}，合法值：{PALETTE_OVERRIDES}")

    override_palette = PALETTES[palette_override]
    # coolwarm 是字符串（heatmap 专用），不能直接赋给 palette list
    new_palette = override_palette if isinstance(override_palette, list) else base.palette

    return ThemeConfig(
        font_family=base.font_family,
        font_size=base.font_size,
        line_width=base.line_width,
        figure_width=base.figure_width,
        figure_height=base.figure_height,
        dpi=base.dpi,
        spines=base.spines,
        grid=base.grid,
        grid_style=base.grid_style,
        legend_frameon=base.legend_frameon,
        legend_fontsize=base.legend_fontsize,
        palette=new_palette,
        bg_color=base.bg_color,
        text_color=base.text_color,
    )
