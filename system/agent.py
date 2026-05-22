"""
C线核心：PlotAgent 主循环
串联 model、tools、validator、merger，为 UI 层提供统一接口。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from model.generator import generate_spec, get_backend, set_backend as _set_generator_backend
from system.merger import fill_defaults, merge_delta
from system.validator import validate
from tools.loader import DataLoadError, load_data
from tools.renderer import RenderError, render_plot


_MAX_RENDER_RETRIES = 2  # 渲染失败时最多自修正重试次数


@dataclass
class AgentResponse:
    """process_input() 的返回值结构。"""

    status: str  # "ok" | "need_input" | "error"
    image_path: str | None = None
    question: str | None = None
    message: str | None = None
    current_spec: dict | None = None


class PlotAgent:
    """
    Agent 主循环，管理多轮对话状态并协调各模块。

    UI 层通过 load_data() 和 process_input() 与 Agent 交互，
    不直接调用 model/ 或 tools/ 中的任何函数。
    """

    def __init__(self) -> None:
        self.current_spec: dict | None = None
        self.current_cache_key: str | None = None
        self.data_context: str | None = None
        self.pending_user_input: str | None = None   # ask_user 触发时暂存原始请求

    def load_data(self, source: str) -> str:
        """
        加载数据文件，更新内部状态。

        Args:
            source: CSV 文件路径。

        Returns:
            DataContext 字符串（展示给用户确认）。

        Raises:
            DataLoadError: 文件不存在或格式错误时。
        """
        data_context, cache_key = load_data(source)
        self.data_context = data_context
        self.current_cache_key = cache_key
        return data_context

    def process_input(self, user_input: str, output_format: str = "png") -> AgentResponse:
        """
        处理用户自然语言输入，驱动完整的绘图流水线。

        流程：
        1. 调用 generate_spec() 生成 spec 或 delta。
        2. 若 current_spec 存在则 merge_delta，否则直接用新 spec。
        3. fill_defaults() 填充 optional 默认值。
        4. validate() 校验。
        5. 校验不通过 → 返回 need_input + question。
        6. 校验通过 → 调用 render_plot()。
        7. 返回 ok + image_path。

        Args:
            user_input: 用户的自然语言描述。

        Returns:
            AgentResponse dataclass。
        """
        # ── C线扩展提示 ────────────────────────────────────────────
        # 可在此处添加前置守卫，提升用户体验，例如：
        #
        # 1. 未加载数据时给出友好提示（而非让模型对空 data_context 乱猜）：
        #    if not self.current_cache_key:
        #        return AgentResponse(
        #            status="need_input",
        #            question="请先上传数据文件，我才能帮你绘图。",
        #        )
        #
        # 2. 空输入检查（UI 层已有，但 agent 层双重保障更健壮）：
        #    if not user_input.strip():
        #        return AgentResponse(status="need_input", question="请输入绘图需求。")
        # ──────────────────────────────────────────────────────────
        # 若 ask_user 已触发过，将原始请求与用户补充合并后再推理
        if self.pending_user_input is not None:
            effective_input = f"{self.pending_user_input}（补充信息：{user_input}）"
            self.pending_user_input = None
        else:
            effective_input = user_input

        data_context = self.data_context or ""
        data_source = self.current_cache_key or ""

        try:
            raw = generate_spec(effective_input, data_context, self.current_spec)
        except Exception as exc:
            return AgentResponse(
                status="error",
                message=f"模型推理失败：{exc}",
                current_spec=self.current_spec,
            )

        # ask_user 工具：模型主动要求澄清，暂存当前有效输入等待用户回答
        if raw.get("__ask_user__"):
            self.pending_user_input = effective_input
            return AgentResponse(
                status="need_input",
                question=raw["question"],
                current_spec=self.current_spec,
            )

        if self.current_spec is not None:
            merged = merge_delta(self.current_spec, raw)
        else:
            merged = raw

        full_spec = fill_defaults(merged)

        result = validate(full_spec)
        if not result.ok:
            # 保留当前 spec 状态，等待用户补充信息后继续
            self.current_spec = merged
            return AgentResponse(
                status="need_input",
                question=result.prompt,
                current_spec=merged,
            )

        self.current_spec = full_spec
        full_spec["output_format"] = output_format

        # data_source 优先使用缓存键，若无则从 spec 中取
        effective_source = data_source or full_spec.get("data_source", "")

        image_path: str | None = None
        last_error: str | None = None
        for attempt in range(_MAX_RENDER_RETRIES + 1):
            try:
                image_path = render_plot(full_spec, effective_source)
                break
            except (RenderError, Exception) as exc:
                last_error = str(exc)
                if attempt >= _MAX_RENDER_RETRIES:
                    break
                # 把渲染错误反馈给模型，以 delta 模式自修正 spec
                try:
                    correction = generate_spec(
                        f"渲染报错，请修正PlotSpec：{exc}",
                        data_context,
                        full_spec,
                    )
                    # 模型在修正阶段触发 ask_user 无法继续修正，直接放弃重试
                    if correction.get("__ask_user__"):
                        break
                    candidate = fill_defaults(merge_delta(full_spec, correction))
                    if validate(candidate).ok:
                        full_spec = candidate
                        self.current_spec = full_spec
                except Exception:
                    break  # 修正推理本身失败，不再重试

        if image_path is None:
            return AgentResponse(
                status="error",
                message=f"渲染失败（已重试 {_MAX_RENDER_RETRIES} 次）：{last_error}",
                current_spec=full_spec,
            )

        return AgentResponse(
            status="ok",
            image_path=image_path,
            current_spec=full_spec,
        )

    def render_from_spec(self, spec_json: str, output_format: str = "png") -> AgentResponse:
        """
        用手动编辑的 PlotSpec JSON 字符串直接渲染，跳过 generate_spec 步骤。
        渲染成功后同步更新 current_spec，后续 LLM 轮次的 delta 以此为基础。

        Args:
            spec_json: 用户在 UI 中手动编辑的 PlotSpec JSON 字符串。

        Returns:
            AgentResponse dataclass。
        """
        try:
            spec = json.loads(spec_json)
        except json.JSONDecodeError as exc:
            return AgentResponse(
                status="error",
                message=f"JSON 格式错误：{exc}",
                current_spec=self.current_spec,
            )

        full_spec = fill_defaults(spec)
        result = validate(full_spec)
        if not result.ok:
            return AgentResponse(
                status="need_input",
                question=result.prompt,
                current_spec=spec,
            )

        full_spec["output_format"] = output_format
        effective_source = self.current_cache_key or full_spec.get("data_source", "")
        try:
            image_path = render_plot(full_spec, effective_source)
        except (RenderError, Exception) as exc:
            return AgentResponse(
                status="error",
                message=f"渲染失败：{exc}",
                current_spec=full_spec,
            )

        self.current_spec = full_spec
        return AgentResponse(
            status="ok",
            image_path=image_path,
            current_spec=full_spec,
        )

    def set_backend(self, backend: str) -> None:
        """切换推理后端（"plan_a" 或 "plan_b"）。"""
        _set_generator_backend(backend)

    def get_backend(self) -> str:
        """返回当前推理后端名称。"""
        return get_backend()

    def reset(self) -> None:
        """清空当前状态，开始新的绘图任务。"""
        self.current_spec = None
        self.current_cache_key = None
        self.data_context = None
        self.pending_user_input = None
