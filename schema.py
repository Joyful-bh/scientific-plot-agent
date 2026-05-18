"""
全局 Schema 定义 —— 项目唯一共享契约。
所有模块从此处 import 常量，禁止在其他文件硬编码枚举值。
"""

# 图表类型
CHART_TYPES: list[str] = ["bar", "line", "scatter", "box", "heatmap"]

# 风格主题（排版 + 配色 + 图幅的完整配置）
STYLE_THEMES: list[str] = ["nature", "ieee", "neurips", "clean", "morandi"]

# 配色覆盖（仅用户明确要求时使用，覆盖 theme 默认配色）
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
    "params_orientation": "vertical",
    "params_stacked": False,
    "params_sort": None,
    "params_show_values": False,
    "params_markers": True,
    "params_smooth": False,
    "params_alpha": 0.8,
    "params_show_regression": False,
    "params_show_points": "outliers",
    "params_notch": False,
    "params_annot": True,
    "params_fmt": ".2f",
}

# 每种图表类型对应的有效 params 字段
CHART_PARAMS: dict[str, list[str]] = {
    "bar":     ["params_orientation", "params_stacked", "params_sort", "params_show_values"],
    "line":    ["params_markers", "params_smooth"],
    "scatter": ["params_alpha", "params_show_regression"],
    "box":     ["params_show_points", "params_notch"],
    "heatmap": ["params_annot", "params_fmt"],
}
