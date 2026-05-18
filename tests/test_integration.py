"""
tests/test_integration.py
端到端集成测试：用 Mock 实现跑通完整流水线。
"""

import pytest

from system.agent import AgentResponse, PlotAgent


def test_full_pipeline_mock_first_round():
    """首轮对话：load_data + process_input 应返回 status='ok'。"""
    agent = PlotAgent()
    agent.load_data("data/example_bar.csv")

    response = agent.process_input("画一张柱状图，对比各模型准确率，nature 风格")

    assert isinstance(response, AgentResponse)
    assert response.status == "ok", (
        f"期望 status='ok'，实际：{response.status}，"
        f"message={response.message}，question={response.question}"
    )
    assert response.image_path is not None, "ok 状态应包含 image_path"
    assert response.current_spec is not None, "ok 状态应包含 current_spec"


def test_full_pipeline_mock_delta_round():
    """两轮对话：第二轮应合并 delta 并返回 status='ok'。"""
    agent = PlotAgent()
    agent.load_data("data/example_bar.csv")

    # 首轮
    r1 = agent.process_input("画一张柱状图")
    assert r1.status == "ok"

    # 修改轮：Mock 返回 {"style_theme": "ieee"}
    r2 = agent.process_input("换成 ieee 风格")
    assert r2.status == "ok"
    assert r2.current_spec is not None
    assert r2.current_spec.get("style_theme") == "ieee"


def test_reset_clears_state():
    """reset() 后 current_spec 和 data_context 应全部清空。"""
    agent = PlotAgent()
    agent.load_data("data/example_bar.csv")
    agent.process_input("画一张柱状图")

    agent.reset()

    assert agent.current_spec is None
    assert agent.current_cache_key is None
    assert agent.data_context is None


def test_pipeline_without_data_load():
    """未加载数据时直接调用 process_input 应依然返回（Mock 不依赖数据）。"""
    agent = PlotAgent()
    response = agent.process_input("画一张折线图")
    # Mock 会返回 ok，但即使 need_input 也不应崩溃
    assert response.status in ("ok", "need_input", "error")
