"""
A线接口：generate_spec()
当前为 Plan B 实现（DeepSeek API），A线完成后替换函数体，接口签名不变。
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent.parent / ".env")


_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
_DEEPSEEK_MODEL = "deepseek-chat"
DEBUG = True  # True 时将原始 API 响应打印到终端，A线替换时可设为 False

_CLIENT: OpenAI | None = None

_FEW_SHOT = """\
【示例1 首轮 · 柱状图分组+误差棒】
数据摘要：缓存key：cache://a1b2c3d4，列：method（类别型）、dataset（类别型）、accuracy（数值型）、std（数值型）
用户需求：画柱状图对比各模型在不同数据集上的准确率，nature风格，按数据集分组，加误差棒
输出：{"chart_type":"bar","data_source":"cache://a1b2c3d4","data_x":"method","data_y":"accuracy","style_theme":"nature","data_group_by":"dataset","data_error":"std","label_title":"模型准确率对比","label_y":"Accuracy (%)"}

【示例2 首轮 · 折线图多列Y】
数据摘要：缓存key：cache://b2c3d4e5，列：epoch（数值型）、train_loss（数值型）、val_loss（数值型）
用户需求：画训练曲线，同时展示train_loss和val_loss，clean风格，平滑一下
输出：{"chart_type":"line","data_source":"cache://b2c3d4e5","data_x":"epoch","data_y":["train_loss","val_loss"],"style_theme":"clean","params_smooth":true,"label_title":"训练曲线","label_x":"Epoch","label_y":"Loss"}

【示例3 首轮 · 散点图+回归线】
数据摘要：缓存key：cache://c3d4e5f6，列：param_size（数值型）、accuracy（数值型）、method（类别型）
用户需求：scatter图看模型参数量和准确率的关系，加回归线，vivid风格
输出：{"chart_type":"scatter","data_source":"cache://c3d4e5f6","data_x":"param_size","data_y":"accuracy","style_theme":"vivid","data_group_by":"method","params_show_regression":true,"label_x":"参数量（M）","label_y":"准确率"}

【示例4 首轮 · 折线图+自定义颜色】
数据摘要：缓存key：cache://d4e5f6a7，列：method（类别型，唯一值6个）、dataset（类别型，唯一值4个）、accuracy（数值型）
用户需求：折线图，X轴是model，按dataset分组画多条线，颜色依次用红#E64B35、蓝#4DBBD5、绿#00A087、深蓝#3C5488，clean风格
输出：{"chart_type":"line","data_source":"cache://d4e5f6a7","data_x":"method","data_y":"accuracy","data_group_by":"dataset","style_theme":"clean","params_line_colors":["#E64B35","#4DBBD5","#00A087","#3C5488"],"label_x":"Model","label_y":"Accuracy (%)"}

【示例5 修改轮 · 换风格】
修改需求：换成ieee风格
输出：{"style_theme":"ieee"}

【示例6 修改轮 · 调整坐标轴】
修改需求：Y轴从80开始，最高到100，X轴标签旋转45度
输出：{"axes_y_min":80,"axes_y_max":100,"axes_x_tick_rotation":45}

【示例7 修改轮 · 改图表属性】
修改需求：改成横向柱状图，按数值从大到小排序
输出：{"params_orientation":"horizontal","params_sort":"desc"}"""

_SYSTEM_FIRST = """\
你是一个科研绘图助手。根据用户需求和数据摘要，输出一个 PlotSpec JSON。

【必填字段】
- chart_type: 图表类型，从 [bar, line, scatter, box, heatmap] 中选
- data_source: 数据摘要末尾"缓存key"的值（格式：cache://xxxxxxxx）
- data_x: X轴列名（字符串，从数据摘要的列名中选）
- data_y: Y轴列名字符串，或列名字符串列表（如需同时展示多列：["acc","f1"]）
- style_theme: 视觉风格，从 [clean, vivid, nature, ieee, morandi, dark] 中选
  · clean=简洁灰调  vivid=高饱和鲜艳  nature=Nature期刊风  ieee=IEEE论文风  morandi=莫兰迪低饱和  dark=深色背景

【数据相关可选字段】
- data_group_by: 分组列名（按类别绘制分组/堆叠图时使用）
- data_error: 误差棒列名（数值型列，如std/sem）
- data_filter: pandas query字符串，过滤数据行（如 "accuracy > 0.8"）

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
    "morandi"（莫兰迪低饱和）/ "nature_d"（Nature标志色）/ "tab10"（10色鲜艳）/ "coolwarm"（仅heatmap）
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
    字符串列表：各分组轮换使用不同纹理，如 ["/", "\\", "|"]（黑白打印区分分组）
- style_edgecolor: 柱子/纹理边框色，如"white"/"black"，null=默认（仅bar图）
- style_hatch_linewidth: 纹理线宽数值，null=0.5（仅style_hatch不为null时生效）
- style_dpi: 整数，输出分辨率
- style_legend_frameon: true/false，图例是否有边框
- style_bg_color: 背景色字符串，如"#1e1e2e"
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
  params_marker_size(标记大小数值，如4；null=按数据密度自动调整) ← 注意：与 params_show_markers 是完全不同的字段
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

【输出规则】
1. data_x、data_y 通常填数据摘要中出现的列名；heatmap 宽表时 data_x 例外，填列轴概念名即可
2. style_palette_override 只能填预设名称字符串，绝对不能填颜色列表；line图自定义颜色用 params_line_colors
3. params_show_markers 是布尔(true/false)，params_marker_style 是形状字符串，二者不能互换
4. params_show_points 是字符串("all"/"outliers"/"none")，不是布尔值
5. 只输出 JSON，不要任何解释，不要 markdown 代码块"""

_SYSTEM_DELTA = """\
你是一个科研绘图助手。根据修改需求，只返回需要变更的字段，格式为 JSON。

【可修改的字段范围】
数据: data_x · data_y · data_group_by · data_error · data_filter
标签: label_title · label_x · label_y
坐标轴: axes_y_min · axes_y_max · axes_x_tick_rotation · axes_x_rotate_labels(bool，标签拥挤时true=旋转/false=缩小字号) · axes_y_scale("linear"/"log") · legend_loc("inside"/"outside_right"/"none")
风格: style_theme[clean/vivid/nature/ieee/morandi/dark] · style_palette_override[morandi/nature_d/tab10/coolwarm，只能填这四个字符串之一]
主题覆写: style_hatch(⚠️必须是字符串或列表非布尔，仅bar；单串="/",多分组轮换=["/","\\","|"]) · style_edgecolor(仅bar) · style_hatch_linewidth · style_grid(bool) · style_line_width · style_font_size · style_dpi · style_legend_frameon(bool) · style_bg_color · style_text_color · style_aspect_ratio · style_figure_width · style_font_family · style_spines(列表)
bar参数: params_orientation · params_stacked · params_sort(按Y值排序；分组图按均值排序) · params_show_values
line参数: params_show_markers(bool) · params_marker_style(形状字符串) · params_marker_size(null=自动) · params_smooth · params_linestyle · params_line_colors(颜色列表)
scatter参数: params_alpha · params_show_regression · params_marker_style · params_marker_size(null=自动)
box参数: params_show_points("all"/"outliers"/"none"，字符串非布尔) · params_notch
heatmap参数: params_annot(bool) · params_annot_fmt(格式字符串，如".2f") · params_heatmap_value(热力值列名，null=自动)
  【宽表heatmap】data_y=行标签列名，data_x=列轴概念名（任意字符串，无需是真实列名）

【输出规则】
1. 只返回用户要求改变的字段，其余字段不输出
2. style_palette_override 只能填 morandi/nature_d/tab10/coolwarm 之一；line图自定义颜色用 params_line_colors
3. params_show_markers 是布尔(true/false)，params_marker_style 是形状字符串，不能互换
4. params_show_points 是字符串("all"/"outliers"/"none")，不是布尔值
5. 只输出 JSON，不要解释，不要 markdown 代码块"""


def _get_client() -> OpenAI:
    global _CLIENT
    if _CLIENT is None:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "未找到 DEEPSEEK_API_KEY 环境变量，请在终端中执行：\n"
                "  $env:DEEPSEEK_API_KEY='your-api-key'  (PowerShell)\n"
                "  export DEEPSEEK_API_KEY='your-api-key'  (bash)"
            )
        _CLIENT = OpenAI(api_key=api_key, base_url=_DEEPSEEK_BASE_URL)
    return _CLIENT


def _strip_markdown(text: str) -> str:
    """去除模型可能输出的 markdown 代码块标记。"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # 去掉 ```json 或 ``` 开头行
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def generate_spec(
    user_input: str,
    data_context: str,
    current_spec: dict | None = None,
) -> dict:
    """
    根据用户自然语言输入和数据摘要生成 PlotSpec 或 delta。

    Args:
        user_input:    用户的自然语言输入字符串。
        data_context:  DataLoader 生成的数据摘要字符串，注入 prompt 供模型参考。
        current_spec:  当前 PlotSpec dict；首轮为 None，修改轮传入当前值。

    Returns:
        首轮：包含所有 REQUIRED_FIELDS 的完整 PlotSpec dict。
        修改轮：仅包含变更字段的 delta dict。
        返回值已经过 JSON 解析，不是字符串。
    """
    # Plan B 实现（DeepSeek API），A线替换
    # ============================================================
    # A线实现指引（Plan B 完成后，A线按此替换函数体）
    # ============================================================
    #
    # 【第一步】模块级加载模型（只加载一次）
    #   from transformers import AutoTokenizer, AutoModelForCausalLM
    #   from peft import PeftModel
    #   import torch
    #   _BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
    #   _LORA_CKPT  = "path/to/your/lora/checkpoint"
    #   _tokenizer  = AutoTokenizer.from_pretrained(_BASE_MODEL)
    #   _model      = PeftModel.from_pretrained(
    #                     AutoModelForCausalLM.from_pretrained(
    #                         _BASE_MODEL, torch_dtype=torch.float16, device_map="auto"
    #                     ), _LORA_CKPT
    #                 ).eval()
    #
    # 【第二步】构造 prompt（格式与 _FEW_SHOT/_SYSTEM_FIRST/_SYSTEM_DELTA 一致）
    #
    # 【第三步】用 outlines 做 constrained decoding 保证合法 JSON
    #   import outlines
    #   FULL_SPEC_SCHEMA = {"type":"object","properties":{...},"required":[...]}
    #   DELTA_SCHEMA = {"type":"object"}
    #   schema = FULL_SPEC_SCHEMA if current_spec is None else DELTA_SCHEMA
    #   generator = outlines.generate.json(_model, schema)
    #   return json.loads(generator(prompt))
    # ============================================================

    client = _get_client()

    if current_spec is None:
        system_msg = _SYSTEM_FIRST
        user_msg = (
            f"{_FEW_SHOT}\n\n"
            f"数据摘要：\n{data_context}\n\n"
            f"用户需求：{user_input}\n"
            "输出："
        )
    else:
        system_msg = _SYSTEM_DELTA
        user_msg = (
            f"{_FEW_SHOT}\n\n"
            f"数据摘要：\n{data_context}\n\n"
            f"当前配置：{json.dumps(current_spec, ensure_ascii=False)}\n\n"
            f"修改需求：{user_input}\n"
            "输出："
        )

    resp = client.chat.completions.create(
        model=_DEEPSEEK_MODEL,
        max_tokens=512,
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )

    raw_text = resp.choices[0].message.content
    if DEBUG:
        round_label = "首轮" if current_spec is None else "修改轮"
        print(f"\n{'='*50}")
        print(f"[DeepSeek {round_label}] 用户输入: {user_input}")
        print(f"[DeepSeek 原始响应]:\n{raw_text}")
        print(f"{'='*50}\n")
    return json.loads(_strip_markdown(raw_text))
