"""
tests/test_renderer.py
验证 render_plot() 的接口契约（Mock 阶段）。
"""

from pathlib import Path

import pytest

from tools.renderer import render_plot


MOCK_SPEC = {
    "chart_type": "bar",
    "data_source": "data/example_bar.csv",
    "data_x": "method",
    "data_y": "accuracy",
    "style_theme": "nature",
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
}


def test_mock_render_returns_string():
    """Mock render_plot() 应返回字符串路径。"""
    result = render_plot(MOCK_SPEC, "data/example_bar.csv")
    assert isinstance(result, str), "render_plot 应返回字符串路径"


def test_mock_render_path_exists():
    """Mock 返回的 placeholder.png 路径应实际存在。"""
    result = render_plot(MOCK_SPEC, "data/example_bar.csv")
    assert Path(result).exists(), f"返回路径不存在：{result}"
