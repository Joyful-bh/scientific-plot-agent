"""
全局 Schema 定义 —— 项目唯一共享契约。
所有模块从此处 import 常量，禁止在其他文件硬编码枚举值。
"""

# 图表类型
# 新增图表类型时：
#   1. 在此列表末尾追加新类型名（字符串）
#   2. 在下方 OPTIONAL_DEFAULTS 添加该类型专属的 params_xxx 字段（若有）
#   3. 在下方 CHART_PARAMS 添加该类型使用的参数字段列表
#   4. 在 tools/renderer.py 实现 _render_xxx() 并注册到 RENDERERS 字典
#   validator.py 自动感知此列表，无需修改
CHART_TYPES: list[str] = ["bar", "line", "scatter", "box", "heatmap"]

# 风格主题（排版 + 配色 + 图幅的完整配置）
# 新增主题时：
#   1. 在此列表末尾追加新主题名（字符串）
#   2. 在 tools/themes.py 的 THEMES 字典里添加对应的 ThemeConfig 实例
#   validator.py 自动感知此列表，无需修改
STYLE_THEMES: list[str] = ["clean", "vivid", "nature", "ieee", "morandi", "dark"]

# 配色覆盖（仅用户明确要求时使用，覆盖 theme 默认配色）
# 新增配色时：
#   1. 在此列表末尾追加新名称
#   2. 在 tools/themes.py 的 PALETTES 字典里添加对应的颜色列表（或 cmap 字符串）
PALETTE_OVERRIDES: list[str] = ["morandi", "nature_d", "tab10", "coolwarm"]

# Required 字段：缺失时系统必须回问用户，不能继续渲染
REQUIRED_FIELDS: list[str] = [
    "chart_type",
    "data_source",
    "data_x",
    "data_y",
    "style_theme",
]

# Optional 字段及其默认值
OPTIONAL_DEFAULTS: dict = {
    "data_group_by": None,
    "data_error": None,
    "data_filter": None,
    "label_title": "",
    "label_x": "",
    "label_y": "",
    "axes_x_scale": "linear",
    "axes_y_scale": "linear",
    "axes_y_min": None,
    "axes_y_max": None,
    "axes_x_tick_rotation": 0,
    "style_palette_override": None,
    # bar 专属
    "params_orientation": "vertical",
    "params_stacked": False,
    "params_sort": None,
    "params_show_values": False,
    "params_hatch": None,           # 柱子纹理，如 "/" "\\" "|" "-" "+" "x" "o" "." "*"；None=不使用
    "params_edgecolor": None,       # 柱子/纹理边框颜色，如 "white"/"black"；None=matplotlib默认
    "params_hatch_linewidth": 0.5,  # 纹理线宽；仅 params_hatch 不为 None 时生效
    # line 专属
    "params_markers": True,
    "params_smooth": False,
    "params_linestyle": "solid",    # 线型："solid"/"dashed"/"dotted"/"dashdot"；所有线统一
    "params_line_colors": None,     # 按线顺序的颜色列表，如 ["#E64B35","#4DBBD5"]；None=使用主题配色
    "params_marker_style": None,    # 标记样式，如 "o" "s" "^" "D" "v" "P" "*"；None=使用"o"
    "params_marker_size": 4,        # 标记大小（点径）
    # scatter 专属
    "params_alpha": 0.8,
    "params_show_regression": False,
    # scatter/line 共享上方 params_marker_style / params_marker_size
    # box 专属
    "params_show_points": "outliers",
    "params_notch": False,
    # heatmap 专属
    "params_annot": True,
    "params_fmt": ".2f",
}

# 每种图表类型对应的有效 params 字段
CHART_PARAMS: dict[str, list[str]] = {
    "bar":     [
        "params_orientation", "params_stacked", "params_sort", "params_show_values",
        "params_hatch", "params_edgecolor", "params_hatch_linewidth",
    ],
    "line":    ["params_markers", "params_smooth", "params_linestyle", "params_line_colors",
                "params_marker_style", "params_marker_size"],
    "scatter": ["params_alpha", "params_show_regression", "params_marker_style", "params_marker_size"],
    "box":     ["params_show_points", "params_notch"],
    "heatmap": ["params_annot", "params_fmt"],
}
