"""
tests/test_validator.py
验证 SchemaValidator 的各条校验规则。
"""

import pytest

from system.validator import validate


VALID_SPEC = {
    "chart_type": "bar",
    "data_source": "data/example_bar.csv",
    "data_x": "method",
    "data_y": "accuracy",
    "style_theme": "nature",
}


def test_valid_spec():
    """合法 spec 应通过校验，ok=True。"""
    result = validate(VALID_SPEC)
    assert result.ok is True
    assert result.missing_required == []
    assert result.type_errors == []
    assert result.prompt == ""


def test_missing_required_data_x():
    """缺少 data_x 时 ok=False，missing_required 包含 'data_x'。"""
    spec = dict(VALID_SPEC)
    del spec["data_x"]
    result = validate(spec)
    assert result.ok is False
    assert "data_x" in result.missing_required


def test_missing_multiple_required():
    """缺少多个必填字段时，所有缺失字段都出现在 missing_required 中。"""
    spec = {"chart_type": "bar"}
    result = validate(spec)
    assert result.ok is False
    for field in ("data_source", "data_x", "data_y", "style_theme"):
        assert field in result.missing_required


def test_data_x_is_list():
    """data_x 传入列表时 ok=False，type_errors 非空。"""
    spec = dict(VALID_SPEC)
    spec["data_x"] = ["SST-2", "MR", "CR"]
    result = validate(spec)
    assert result.ok is False
    assert len(result.type_errors) > 0


def test_data_y_is_list():
    """data_y 传入列表时 ok=False，type_errors 包含相应说明。"""
    spec = dict(VALID_SPEC)
    spec["data_y"] = [93.5, 94.8, 91.0]
    result = validate(spec)
    assert result.ok is False
    assert any("data_y" in e for e in result.type_errors)


def test_invalid_chart_type():
    """chart_type='pie' 不在枚举内，ok=False，type_errors 非空。"""
    spec = dict(VALID_SPEC)
    spec["chart_type"] = "pie"
    result = validate(spec)
    assert result.ok is False
    assert len(result.type_errors) > 0


def test_invalid_style_theme():
    """style_theme='science' 不在枚举内，ok=False，type_errors 非空。"""
    spec = dict(VALID_SPEC)
    spec["style_theme"] = "science"
    result = validate(spec)
    assert result.ok is False
    assert len(result.type_errors) > 0


def test_axes_y_min_greater_than_max():
    """axes_y_min >= axes_y_max 时 ok=False，type_errors 非空。"""
    spec = dict(VALID_SPEC)
    spec["axes_y_min"] = 90.0
    spec["axes_y_max"] = 80.0
    result = validate(spec)
    assert result.ok is False
    assert len(result.type_errors) > 0


def test_axes_y_range_valid():
    """axes_y_min < axes_y_max 时不触发错误。"""
    spec = dict(VALID_SPEC)
    spec["axes_y_min"] = 50.0
    spec["axes_y_max"] = 100.0
    result = validate(spec)
    assert result.ok is True


def test_prompt_non_empty_on_failure():
    """校验失败时 prompt 字符串非空。"""
    spec = {}
    result = validate(spec)
    assert result.ok is False
    assert len(result.prompt) > 0
