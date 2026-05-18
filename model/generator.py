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
