"""
A线专用提示词。供 fine-tuned 小模型训练数据打包和 A线推理使用。

与 generator.py 中 _SYSTEM_FIRST（Plan B 路径）的核心区别：
- Plan B：data_source 是必填字段，由 DeepSeek 从 data_context 的"缓存key"读取后输出
- Plan A：data_source 由系统自动注入，模型不应输出该字段

⚠️ 格式警告（来自 tools/loader.py _build_context 的注释）：
A线推理时传给模型的 data_context 格式必须与 _build_context() 输出完全一致。
如果 _build_context() 的输出格式有变动，必须同步更新 SYSTEM_FIRST_FINETUNE 中的格式说明。
"""

from __future__ import annotations

import json

from schema import CHART_TYPES, OPTIONAL_DEFAULTS, PALETTE_OVERRIDES, REQUIRED_FIELDS, STYLE_THEMES

_REQUIRED_NO_SOURCE: frozenset[str] = frozenset(f for f in REQUIRED_FIELDS if f != "data_source")

_CT = "/".join(CHART_TYPES)           # "bar/line/scatter/box/heatmap"
_ST = "/".join(STYLE_THEMES)          # "normal/morandi/macaron/bright/rococo/earth/science/nature"
_PO = "/".join(PALETTE_OVERRIDES)     # "morandi/nature_d/tab10/coolwarm"
_ST_DESC = (
    "normal=简洁常规柔和  morandi=低饱和莫兰迪  macaron=马卡龙轻快色调  bright=高对比鲜艳  "
    "rococo=洛可可低对比淡色  earth=深沉大地色  science=Science期刊冷色  nature=Nature期刊暖色"
)

# ── 字段说明（首轮与修改轮共享，确保两套提示词对字段的描述始终一致）──────────────
_FIELD_SPEC: str = f"""\
【必填字段】
- chart_type: 图表类型，从 [{_CT}] 中选
- data_x: X轴列名（字符串，从数据摘要的列名中选）
- data_y: Y轴列名字符串，或列名字符串列表（如需同时展示多列：["acc","f1"]）
- style_theme: 视觉风格，从 [{_ST}] 中选
  · {_ST_DESC}

【数据相关可选字段】
- data_group_by: 分组列名（按类别绘制分组/堆叠图时使用）
- data_error: 误差棒列名（数值型列，如std/sem）
- data_filter: pandas query字符串，只保留满足条件的数据行
  · 列名直接写，字符串值需加引号：'method == "BERT"'
  · 多条件：'accuracy > 0.8 and dataset != "CoLA"'
  · 数值比较：'epoch >= 10'；null=不过滤（使用全部数据）

【标签可选字段】
- label_title: 图表标题
- label_x: X轴标签
- label_y: Y轴标签

【坐标轴可选字段】
- axes_y_min / axes_y_max: Y轴数值范围（如 axes_y_min=80）
- axes_x_tick_rotation: X轴刻度旋转角度（如 45，默认0）
- axes_x_rotate_labels: 标签拥挤时的处理方式，true=旋转标签（30°–45°），false=缩小字号（默认false）
- axes_y_scale: Y轴缩放，"linear"（默认）或 "log"
- legend_loc: 图例位置，"inside"=图内最优位置，"outside_right"=图外右侧，"none"=不显示，null=自动

【配色可选字段】（两个字段用途完全不同，不能混用）
- style_palette_override: 切换预设配色方案，值只能是以下字符串之一：
    {_PO}
  ⚠️ 不能填颜色列表，不能填十六进制颜色
- params_line_colors: 【仅line图】自定义每条线的颜色，值为十六进制颜色字符串列表
    示例：["#E64B35","#4DBBD5","#00A087","#3C5488"]
  ⚠️ 这个字段只用于 line 图；bar/scatter/box 图改颜色只能用 style_palette_override

【主题覆写字段】（在所选主题基础上修改单项视觉属性，null=保留主题默认）
- style_grid: true/false，是否显示网格
- style_line_width: 数值，线宽（磅）
- style_font_size: 整数，基准字号（磅）
- style_hatch: 柱子纹理，⚠️必须是字符串或字符串列表，不能是true/false（仅bar图）
    单个字符串：所有分组用同一纹理，合法值："/" "\\" "|" "-" "+" "x" "o" "." "*"
    字符串列表：各分组轮换使用不同纹理，如 ["/", "\\\\", "|"]（黑白打印区分分组）
- style_edgecolor: 柱子/纹理边框色，如"white"/"black"，null=默认（仅bar图）
- style_hatch_linewidth: 纹理线宽数值，null=0.5（仅style_hatch不为null时生效）
- style_dpi: 整数，输出分辨率
- style_legend_frameon: true/false，图例是否有边框
- style_bg_color: 背景色字符串，如"#f5f5f5"（浅灰）或"#ffffff"（白色）
- style_text_color: 文字/刻度颜色字符串
- style_aspect_ratio: 宽高比数值，如0.75（高=宽×该值）
- style_figure_width: 最小图幅宽度（英寸），如5.0；LayoutEngine仍可按数据扩大
- style_font_family: 字体族，如"Arial"/"Times New Roman"/"DejaVu Sans"
- style_spines: 保留的轴脊列表，如["left","bottom"]或["left","bottom","top","right"]

【图表专属参数】
bar图:
  params_orientation("vertical"默认/"horizontal") · params_stacked(true/false)
  params_sort("asc"/"desc"，按Y值大小排序柱子) · params_show_values(true/false，柱顶显示数值)

line图:
  params_show_markers(true/false，是否显示数据点标记，默认true)
  params_marker_style(标记形状，"o"/"s"/"^"/"D"/"v"/"P"/"*"，null=默认"o")
  params_marker_size(标记大小数值，如4；null=按数据密度自动调整)
  params_smooth(true/false) · params_linestyle("solid"/"dashed"/"dotted"/"dashdot")
  params_line_colors(自定义颜色列表，见配色字段说明)

scatter图: params_alpha(0~1) · params_show_regression(true/false)
  params_marker_style(标记形状) · params_marker_size(标记大小，null=按数据密度自动调整)

box图:
  params_show_points 取值是字符串，不是布尔值：
    "all"=显示所有数据点 / "outliers"=仅显示离群点（默认） / "none"=不显示
  params_notch(true/false，缺口箱线图)

heatmap:
  params_annot(true/false，是否在格子里显示数值，默认true)
  params_annot_fmt(数值格式字符串，如".2f"保留两位小数，默认".2f")
  params_heatmap_value(热力值列名字符串；null=自动取第一个非轴数值列)
  ⚠️ 宽表矩阵格式（列名本身是分类值，如 model|SST-2|MR|CoLA）：
     data_y 填行标签列名（如"model"），data_x 填列轴的概念名（任意字符串，如"dataset"）
     系统自动识别宽表，params_heatmap_value 留 null
"""


# ── 首轮系统提示词 ─────────────────────────────────────────────────────────────
SYSTEM_FIRST_FINETUNE: str = f"""\
你是一个科研绘图助手。根据用户需求和数据摘要，输出一个 PlotSpec JSON。
⚠️ 只输出用户明确要求或对出图有实质影响的字段，不需要的字段一律省略，系统会自动补全默认值。
⚠️ 不要在输出中包含 data_source 字段，系统会自动注入。

{_FIELD_SPEC}

【输出规则】
1. 必填字段（chart_type / data_x / data_y / style_theme）必须输出
2. 可选字段只在用户明确要求或对出图有实质影响时才输出，其余省略
3. data_x、data_y 填数据摘要中出现的列名；heatmap 宽表时 data_x 例外，填列轴概念名即可
4. style_palette_override 只能填预设名称字符串，绝对不能填颜色列表；line图自定义颜色用 params_line_colors
5. params_show_markers 是布尔(true/false)，params_marker_style 是形状字符串，二者不能互换
6. params_show_points 是字符串("all"/"outliers"/"none")，不是布尔值
7. 输出格式必须是以下工具调用结构，不要任何解释，不要 markdown 代码块：
   {"tool":"create_plot","arguments":{...字段...}}"""


# ── 修改轮系统提示词 ───────────────────────────────────────────────────────────
SYSTEM_DELTA_FINETUNE: str = f"""\
你是一个科研绘图助手。根据用户的修改请求和当前 PlotSpec，输出一个 delta JSON，只包含需要修改的字段。
⚠️ 不要输出未变更的字段；不要输出 data_source 字段。

{_FIELD_SPEC}

【delta 输出规则】
1. 只输出需要变更的字段，未变更的字段一律不输出
2. 若需将某字段恢复为默认值（如删除标题、关闭误差棒、取消过滤），将该字段输出为 null
3. data_x、data_y、data_group_by 等列名字段填列名字符串，不填具体数值
4. style_palette_override 只能填预设名称字符串，不能填颜色列表
5. params_show_markers 是布尔(true/false)，params_marker_style 是形状字符串，二者不能互换
6. params_show_points 是字符串("all"/"outliers"/"none")，不是布尔值
7. 输出格式必须是以下工具调用结构，不要任何解释，不要 markdown 代码块：
   {"tool":"update_plot","arguments":{...变更字段...}}"""


def format_user_message(user_input: str, data_context: str) -> str:
    """
    构造首轮传给 fine-tuned 模型的 user message。
    训练数据打包（pack_finetune.py）和 A线推理（generator.py 真实实现）
    必须使用同一个函数，确保格式完全一致。

    Args:
        user_input:   用户的自然语言绘图请求。
        data_context: tools.loader.load_data() 返回的数据摘要字符串。

    Returns:
        格式化后的 user message 字符串（含 /no_think 后缀）。
    """
    return (
        f"{data_context}\n\n"
        f"用户需求：{user_input}\n"
        "输出：/no_think"
    )


def format_user_message_delta(
    user_input: str,
    data_context: str,
    current_spec: dict,
) -> str:
    """
    构造修改轮传给 fine-tuned 模型的 user message。
    当前 PlotSpec 以 JSON 形式注入 user message，模型据此输出 delta。
    训练数据打包和 A线推理必须使用同一个函数。

    Args:
        user_input:   用户的自然语言修改请求。
        data_context: tools.loader.load_data() 返回的数据摘要字符串。
        current_spec: 当前生效的 PlotSpec dict（不含 data_source）。

    Returns:
        格式化后的 user message 字符串（含 /no_think 后缀）。
    """
    # 只序列化必填字段 + 实际非默认的可选字段，剔除冗余默认值以节省 token
    compact = {
        k: v for k, v in current_spec.items()
        if k in _REQUIRED_NO_SOURCE
        or (k in OPTIONAL_DEFAULTS and v != OPTIONAL_DEFAULTS[k])
    }
    spec_json = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    return (
        f"{data_context}\n\n"
        f"当前PlotSpec：{spec_json}\n\n"
        f"用户需求：{user_input}\n"
        "输出：/no_think"
    )
