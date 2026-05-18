"""
C线模块：SchemaValidator
校验 PlotSpec dict 的合法性，返回 ValidationResult，不抛出异常。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from schema import (
    CHART_TYPES,
    OPTIONAL_DEFAULTS,
    REQUIRED_FIELDS,
    STYLE_THEMES,
)


class ValidationError(Exception):
    """外部调用者需要将校验失败作为异常处理时使用（validate() 本身不抛出）。"""


@dataclass
class ValidationResult:
    """validate() 的返回值结构。"""

    ok: bool
    missing_required: list[str] = field(default_factory=list)
    type_errors: list[str] = field(default_factory=list)
    prompt: str = ""


def validate(spec: dict) -> ValidationResult:
    """
    校验 PlotSpec 合法性。

    校验规则：
    1. REQUIRED_FIELDS 中的字段全部存在且非空。
    2. chart_type 在 CHART_TYPES 枚举内。
    3. style_theme 在 STYLE_THEMES 枚举内。
    4. data_x / data_y 必须是字符串（列名），不能是 list。
    5. axes_y_min / axes_y_max 同时存在时 min < max。

    Returns:
        ValidationResult（ok=True 表示校验通过）。
    """
    missing: list[str] = []
    errors: list[str] = []

    # 规则 1：必填字段
    for field_name in REQUIRED_FIELDS:
        val = spec.get(field_name)
        if val is None or val == "":
            missing.append(field_name)

    # 规则 2：chart_type 枚举
    chart_type = spec.get("chart_type")
    if chart_type and chart_type not in CHART_TYPES:
        errors.append(
            f"chart_type='{chart_type}' 不合法，合法值：{CHART_TYPES}"
        )

    # 规则 3：style_theme 枚举
    style_theme = spec.get("style_theme")
    if style_theme and style_theme not in STYLE_THEMES:
        errors.append(
            f"style_theme='{style_theme}' 不合法，合法值：{STYLE_THEMES}"
        )

    # 规则 4：data_x / data_y 必须是字符串，不能是 list
    for key in ("data_x", "data_y"):
        val = spec.get(key)
        if val is not None and isinstance(val, list):
            errors.append(
                f"{key}应为列名字符串，不是值列表"
                "（模型可能将列的值填入了该字段，请检查 PlotSpec）"
            )
        elif val is not None and not isinstance(val, str):
            errors.append(f"{key} 应为字符串列名，收到类型：{type(val).__name__}")

    # 规则 5：axes_y_min < axes_y_max
    y_min = spec.get("axes_y_min")
    y_max = spec.get("axes_y_max")
    if y_min is not None and y_max is not None:
        try:
            if float(y_min) >= float(y_max):
                errors.append(
                    f"axes_y_min({y_min}) 必须小于 axes_y_max({y_max})"
                )
        except (TypeError, ValueError):
            errors.append("axes_y_min / axes_y_max 必须是数值类型")

    ok = not missing and not errors
    prompt = _build_prompt(missing, errors) if not ok else ""

    return ValidationResult(ok=ok, missing_required=missing, type_errors=errors, prompt=prompt)


def _build_prompt(missing: list[str], errors: list[str]) -> str:
    """生成友好的回问语句。"""
    parts: list[str] = []
    if missing:
        fields_str = "、".join(missing)
        parts.append(f"以下必填信息还缺少，请补充：{fields_str}")
    if errors:
        parts.append("另外发现以下问题：\n" + "\n".join(f"  - {e}" for e in errors))
    return "\n".join(parts)
