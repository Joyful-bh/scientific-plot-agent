"""
tests/test_format_coverage.py
覆盖长表/宽表各种常见数据格式的渲染集成测试。
每个测试用例描述数据结构、图表类型和关键渲染路径。
"""

from pathlib import Path

import pytest

from system.merger import fill_defaults
from tools.loader import load_data
from tools.renderer import render_plot


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _render(base_spec: dict, csv_path: str) -> str:
    """加载 CSV → fill_defaults → render_plot，返回输出路径。"""
    _, cache_key = load_data(csv_path)
    spec = fill_defaults({**base_spec, "data_source": cache_key})
    return render_plot(spec, cache_key)


def _assert_output(path: str) -> None:
    assert isinstance(path, str), "render_plot 应返回字符串路径"
    assert Path(path).exists(), f"输出文件不存在：{path}"


# ---------------------------------------------------------------------------
# 长表：标准分组柱状图（已有数据）
# ---------------------------------------------------------------------------

def test_long_bar_grouped():
    """长表分组柱状图：method × dataset，带误差棒，nature 风格。"""
    path = _render({
        "chart_type": "bar",
        "data_x": "method",
        "data_y": "accuracy",
        "data_group_by": "dataset",
        "data_error": "std",
        "style_theme": "nature",
        "label_title": "Grouped Bar (long)",
        "label_y": "Accuracy (%)",
        "axes_y_min": 50,
    }, "data/example_bar.csv")
    _assert_output(path)


def test_long_bar_sorted_desc():
    """无分组柱状图 + 降序排列 + 柱顶数值标注。
    使用 wide_bar.csv 单列（每个 dataset 恰好出现一次），保证 X 唯一。"""
    path = _render({
        "chart_type": "bar",
        "data_x": "dataset",
        "data_y": "BERT",
        "style_theme": "clean",
        "params_sort": "desc",
        "params_show_values": True,
        "label_title": "Sorted Bar (unique X)",
        "axes_y_min": 50,
    }, "data/wide_bar.csv")
    _assert_output(path)


def test_long_bar_stacked():
    """长表堆叠柱状图：method × dataset。"""
    path = _render({
        "chart_type": "bar",
        "data_x": "method",
        "data_y": "accuracy",
        "data_group_by": "dataset",
        "style_theme": "vivid",
        "params_stacked": True,
        "label_title": "Stacked Bar (long)",
    }, "data/example_bar.csv")
    _assert_output(path)


# ---------------------------------------------------------------------------
# 长表：多 Y 列折线图（已有数据）
# ---------------------------------------------------------------------------

def test_long_line_multi_y():
    """长表折线图：多列 Y（train_loss / val_loss / val_acc）。"""
    path = _render({
        "chart_type": "line",
        "data_x": "step",
        "data_y": ["train_loss", "val_loss"],
        "style_theme": "clean",
        "params_smooth": True,
        "label_title": "Training Curves (long multi-y)",
        "label_x": "Step",
        "label_y": "Loss",
    }, "data/example_line.csv")
    _assert_output(path)


# ---------------------------------------------------------------------------
# 长表：散点图（含分组 + 回归线）
# ---------------------------------------------------------------------------

def test_long_scatter_grouped_regression():
    """长表散点图：参数量 × 准确率，按 method 分组，显示回归线。"""
    path = _render({
        "chart_type": "scatter",
        "data_x": "param_M",
        "data_y": "accuracy",
        "data_group_by": "method",
        "style_theme": "vivid",
        "params_show_regression": True,
        "params_alpha": 0.7,
        "label_title": "Param vs Accuracy (long scatter)",
        "label_x": "Parameters (M)",
        "label_y": "Accuracy (%)",
    }, "data/long_scatter.csv")
    _assert_output(path)


# ---------------------------------------------------------------------------
# 长表：箱线图
# ---------------------------------------------------------------------------

def test_long_box_all_points():
    """长表箱线图：每模型多次测量，显示所有数据点。"""
    path = _render({
        "chart_type": "box",
        "data_x": "model",
        "data_y": "accuracy",
        "style_theme": "nature",
        "params_show_points": "all",
        "params_notch": False,
        "label_title": "Accuracy Distribution (long box)",
        "label_y": "Accuracy (%)",
        "axes_y_min": 90,
    }, "data/long_box.csv")
    _assert_output(path)


def test_long_box_outliers_only():
    """长表箱线图：仅显示离群点（默认行为）。"""
    path = _render({
        "chart_type": "box",
        "data_x": "model",
        "data_y": "accuracy",
        "style_theme": "ieee",
        "params_show_points": "outliers",
        "label_title": "Box (outliers only)",
    }, "data/long_box.csv")
    _assert_output(path)


# ---------------------------------------------------------------------------
# 长表：标准 pivot heatmap
# ---------------------------------------------------------------------------

def test_long_heatmap_explicit_value():
    """长表热力图（3列长表）：显式指定热力值列。"""
    path = _render({
        "chart_type": "heatmap",
        "data_x": "dataset",
        "data_y": "model",
        "style_theme": "ieee",
        "params_heatmap_value": "score",
        "params_annot": True,
        "params_annot_fmt": ".1f",
        "label_title": "Benchmark Heatmap (long)",
    }, "data/long_heatmap.csv")
    _assert_output(path)


def test_long_heatmap_auto_value():
    """长表热力图：未指定热力值列，自动取第一个数值列。"""
    path = _render({
        "chart_type": "heatmap",
        "data_x": "dataset",
        "data_y": "model",
        "style_theme": "morandi",
        "params_annot": True,
        "label_title": "Benchmark Heatmap (long, auto value)",
    }, "data/long_heatmap.csv")
    _assert_output(path)


# ---------------------------------------------------------------------------
# 宽表：heatmap（核心新功能）
# ---------------------------------------------------------------------------

def test_wide_heatmap_conceptual_x():
    """宽表热力图：data_x 为不存在的概念名，触发宽表路径。"""
    path = _render({
        "chart_type": "heatmap",
        "data_x": "dataset",       # 不存在于 wide_heatmap.csv 中
        "data_y": "model",
        "style_theme": "morandi",
        "params_annot": True,
        "params_annot_fmt": ".1f",
        "label_title": "Wide Heatmap",
    }, "data/wide_heatmap.csv")
    _assert_output(path)


def test_wide_heatmap_coolwarm():
    """宽表热力图：coolwarm 配色。"""
    path = _render({
        "chart_type": "heatmap",
        "data_x": "benchmark",     # 任意概念名
        "data_y": "model",
        "style_theme": "clean",
        "style_palette_override": "coolwarm",
        "params_annot": True,
        "label_title": "Wide Heatmap (coolwarm)",
    }, "data/wide_heatmap.csv")
    _assert_output(path)


# ---------------------------------------------------------------------------
# 宽表：bar（data_y 为列名列表，触发内部 melt）
# ---------------------------------------------------------------------------

def test_wide_bar_multi_y():
    """宽表柱状图：data_y 为多列列表，渲染器内部 melt 为长表。"""
    path = _render({
        "chart_type": "bar",
        "data_x": "dataset",
        "data_y": ["BERT", "RoBERTa", "XLNet", "DeBERTa", "ALBERT"],
        "style_theme": "nature",
        "label_title": "Wide Bar (multi-y melt)",
        "label_x": "Dataset",
        "label_y": "Accuracy (%)",
        "axes_y_min": 50,
    }, "data/wide_bar.csv")
    _assert_output(path)


def test_wide_bar_multi_y_sorted():
    """宽表柱状图：data_y 列表 + 排序（melt 后 sort_mode 应自动置 None，不报错）。"""
    path = _render({
        "chart_type": "bar",
        "data_x": "dataset",
        "data_y": ["BERT", "RoBERTa", "DeBERTa"],
        "style_theme": "ieee",
        "params_sort": "desc",      # melt 路径会将 sort_mode 重置为 None
        "params_show_values": True,
        "label_title": "Wide Bar sorted (sort silently skipped)",
    }, "data/wide_bar.csv")
    _assert_output(path)


def test_wide_bar_stacked_hatch():
    """宽表堆叠柱状图：带纹理序列（多分组轮换）。"""
    path = _render({
        "chart_type": "bar",
        "data_x": "dataset",
        "data_y": ["BERT", "RoBERTa", "XLNet"],
        "style_theme": "clean",
        "params_stacked": True,
        "style_hatch": ["/", "\\", "|"],
        "style_edgecolor": "black",
        "label_title": "Wide Stacked Bar with Hatch",
    }, "data/wide_bar.csv")
    _assert_output(path)


# ---------------------------------------------------------------------------
# 边界：宽表 heatmap 错误路径
# ---------------------------------------------------------------------------

def test_wide_heatmap_no_numeric_cols_raises():
    """宽表路径下 data_y 之外无数值列时应抛出 RenderError。"""
    import pandas as pd
    from tools.loader import _CACHE
    from tools.renderer import RenderError

    bad_df = pd.DataFrame({"label": ["A", "B"], "tag": ["x", "y"]})
    _CACHE["cache://bad_wide"] = bad_df

    spec = fill_defaults({
        "chart_type": "heatmap",
        "data_source": "cache://bad_wide",
        "data_x": "nonexistent",
        "data_y": "label",
        "style_theme": "clean",
    })
    with pytest.raises(RenderError):
        render_plot(spec, "cache://bad_wide")
