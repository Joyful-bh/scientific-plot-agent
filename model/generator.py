"""
A线接口：generate_spec()
当前为 Mock 实现，A线完成后替换函数体，接口签名不变。
"""


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
    # Mock实现，A线替换
    # ============================================================
    # A线实现指引
    # ============================================================
    #
    # 【第一步】模块级加载模型（只加载一次，避免每次调用重新加载）
    #
    #   from transformers import AutoTokenizer, AutoModelForCausalLM
    #   from peft import PeftModel
    #   import torch
    #
    #   _BASE_MODEL  = "Qwen/Qwen2.5-1.5B-Instruct"
    #   _LORA_CKPT   = "path/to/your/lora/checkpoint"   # LoRA 训练产出路径
    #
    #   _tokenizer = AutoTokenizer.from_pretrained(_BASE_MODEL)
    #   _base_model = AutoModelForCausalLM.from_pretrained(
    #       _BASE_MODEL, torch_dtype=torch.float16, device_map="auto"
    #   )
    #   _model = PeftModel.from_pretrained(_base_model, _LORA_CKPT)
    #   _model.eval()
    #
    # ────────────────────────────────────────────────────────────
    # 【第二步】构造 prompt
    #
    #   首轮 prompt（current_spec is None）格式：
    #   ┌──────────────────────────────────────────────────────┐
    #   │ 你是一个科研绘图助手。根据以下数据摘要和用户需求，  │
    #   │ 输出一个 PlotSpec JSON。只输出 JSON，不要解释。      │
    #   │                                                      │
    #   │ {data_context}                                       │
    #   │                                                      │
    #   │ 用户需求：{user_input}                               │
    #   └──────────────────────────────────────────────────────┘
    #
    #   修改轮 prompt（current_spec is not None）格式：
    #   ┌──────────────────────────────────────────────────────┐
    #   │ 你是一个科研绘图助手。下面是当前图表配置和用户的     │
    #   │ 修改需求。只返回需要变更的字段，格式为 JSON。        │
    #   │                                                      │
    #   │ {data_context}                                       │
    #   │                                                      │
    #   │ 当前配置：{json.dumps(current_spec, ensure_ascii=False)} │
    #   │                                                      │
    #   │ 修改需求：{user_input}                               │
    #   └──────────────────────────────────────────────────────┘
    #
    #   示例代码：
    #
    #   import json
    #   if current_spec is None:
    #       prompt = (
    #           "你是一个科研绘图助手。根据以下数据摘要和用户需求，"
    #           "输出一个 PlotSpec JSON。只输出 JSON，不要解释。\n\n"
    #           f"{data_context}\n\n"
    #           f"用户需求：{user_input}"
    #       )
    #   else:
    #       prompt = (
    #           "你是一个科研绘图助手。下面是当前图表配置和用户的修改需求。"
    #           "只返回需要变更的字段，格式为 JSON。\n\n"
    #           f"{data_context}\n\n"
    #           f"当前配置：{json.dumps(current_spec, ensure_ascii=False)}\n\n"
    #           f"修改需求：{user_input}"
    #       )
    #
    # ────────────────────────────────────────────────────────────
    # 【第三步】用 outlines 做 constrained decoding（保证输出合法 JSON）
    #
    #   import outlines
    #   from schema import CHART_TYPES, STYLE_THEMES
    #
    #   # 首轮：约束输出必须包含所有必填字段
    #   FULL_SPEC_SCHEMA = {
    #       "type": "object",
    #       "properties": {
    #           "chart_type":  {"type": "string", "enum": CHART_TYPES},
    #           "data_source": {"type": "string"},
    #           "data_x":      {"type": "string"},   # 列名字符串，不是列的值！
    #           "data_y":      {"type": "string"},   # 列名字符串，不是列的值！
    #           "style_theme": {"type": "string", "enum": STYLE_THEMES},
    #       },
    #       "required": ["chart_type", "data_source", "data_x", "data_y", "style_theme"],
    #   }
    #
    #   # 修改轮：约束输出为任意 object（字段不定，只要是 JSON object 即可）
    #   DELTA_SCHEMA = {"type": "object"}
    #
    #   schema = FULL_SPEC_SCHEMA if current_spec is None else DELTA_SCHEMA
    #   generator = outlines.generate.json(_model, schema)
    #   result_str = generator(prompt)
    #   return json.loads(result_str)
    #
    # ────────────────────────────────────────────────────────────
    # 【Plan B：若训练效果不理想，改用 few-shot prompt 调 Claude API】
    #
    #   只需替换本函数体，其余文件一行不动。
    #
    #   import anthropic, json
    #
    #   FEW_SHOT = """
    #   示例1（首轮）：
    #   数据：method列（类别型）、accuracy列（数值型）
    #   需求：画柱状图对比各模型准确率，nature风格
    #   输出：{"chart_type":"bar","data_source":"data/example_bar.csv",
    #           "data_x":"method","data_y":"accuracy","style_theme":"nature"}
    #
    #   示例2（修改轮）：
    #   需求：换成ieee风格
    #   输出：{"style_theme":"ieee"}
    #   """
    #
    #   client = anthropic.Anthropic()
    #   if current_spec is None:
    #       user_msg = f"{FEW_SHOT}\n数据摘要：{data_context}\n需求：{user_input}\n输出："
    #   else:
    #       user_msg = (f"{FEW_SHOT}\n当前配置：{json.dumps(current_spec)}\n"
    #                   f"修改需求：{user_input}\n输出（只返回变更字段）：")
    #
    #   resp = client.messages.create(
    #       model="claude-haiku-4-5-20251001",
    #       max_tokens=512,
    #       messages=[{"role": "user", "content": user_msg}],
    #   )
    #   return json.loads(resp.content[0].text)
    #
    # ────────────────────────────────────────────────────────────
    # 【返回值约定（无论用哪种实现都必须满足）】
    #
    #   首轮（current_spec is None），返回包含所有必填字段的完整 dict，例如：
    #     {
    #       "chart_type":  "bar",                    # CHART_TYPES 中的值
    #       "data_source": "data/example_bar.csv",
    #       "data_x":      "method",                 # ✅ 列名字符串
    #       "data_y":      "accuracy",               # ✅ 列名字符串
    #       "style_theme": "nature",                 # STYLE_THEMES 中的值
    #       # 可选字段按需返回，不填由 merger.fill_defaults() 自动补充
    #       "label_title": "模型准确率对比",
    #       "data_group_by": "dataset",
    #     }
    #
    #   修改轮（current_spec is not None），只返回变更字段，例如：
    #     {"style_theme": "ieee"}
    #     {"axes_y_min": 50, "axes_y_max": 100}
    #     {"data_group_by": "dataset", "label_title": "分组对比"}
    #
    #   ⚠️  返回值必须是 dict，不能是 JSON 字符串
    #   ⚠️  data_x / data_y 填列名（如 "accuracy"），不要填列的值（如 [93.5, 94.8]）
    # ============================================================

    if current_spec is None:
        # 首轮 Mock：返回完整示例 PlotSpec
        return {
            "chart_type": "bar",
            "data_source": "data/example_bar.csv",
            "data_x": "method",
            "data_y": "accuracy",
            "style_theme": "nature",
        }
    else:
        # 修改轮 Mock：只返回变更字段
        return {
            "style_theme": "ieee",
        }
