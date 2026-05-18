"""
tests/test_generator.py
验证 generate_spec() 的接口契约（Mock 阶段）。
"""

import pytest

from model.generator import generate_spec
from schema import REQUIRED_FIELDS


def test_mock_first_round_contract():
    """首轮调用（current_spec=None）应返回包含所有 REQUIRED_FIELDS 的 dict。"""
    result = generate_spec("画一张柱状图", "数据摘要示例")
    assert isinstance(result, dict), "返回值应为 dict"
    for field in REQUIRED_FIELDS:
        assert field in result, f"缺少必要字段：{field}"


def test_mock_delta_round_contract():
    """修改轮调用（current_spec 有值）应返回非空 dict。"""
    current = {
        "chart_type": "bar",
        "data_source": "data/example_bar.csv",
        "data_x": "method",
        "data_y": "accuracy",
        "style_theme": "nature",
    }
    result = generate_spec("换成 ieee 风格", "数据摘要示例", current_spec=current)
    assert isinstance(result, dict), "返回值应为 dict"
    assert len(result) > 0, "delta 不应为空 dict"


def test_mock_first_round_values_are_strings():
    """首轮 Mock 的 data_x 和 data_y 必须是字符串（列名），不能是 list。"""
    result = generate_spec("任意输入", "任意摘要")
    assert isinstance(result.get("data_x"), str), "data_x 应为字符串列名"
    assert isinstance(result.get("data_y"), str), "data_y 应为字符串列名"
