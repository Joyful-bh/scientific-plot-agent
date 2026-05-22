"""
C线 UI：Gradio 界面
所有回调通过共享的 PlotAgent 实例交互，不直接调用 model/ 或 tools/。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# 将项目根目录加入 sys.path，使 `python ui/app.py` 和 `python -m ui.app` 均可用
sys.path.insert(0, str(Path(__file__).parent.parent))

# 必须在 import gradio 之前设置，否则 Gradio 已使用默认 /tmp/gradio 初始化。
# 共享服务器上 /tmp/gradio 通常没有写权限，改为项目内 output/tmp。
_GRADIO_TMP = Path(__file__).parent.parent / "output" / "tmp"
_GRADIO_TMP.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("GRADIO_TEMP_DIR", str(_GRADIO_TMP))

import gradio as gr

from system.agent import PlotAgent

# 模块级唯一 Agent 实例，所有回调共享
agent = PlotAgent()


# ---------------------------------------------------------------------------
# 回调函数
# ---------------------------------------------------------------------------

def on_upload(file) -> str:
    """上传文件后自动加载数据，返回 DataContext 摘要。"""
    if file is None:
        return ""
    try:
        context = agent.load_data(file.name)
        return context
    except Exception as exc:
        return f"[错误] 数据加载失败：{exc}"


def on_path_input(path: str) -> str:
    """直接输入服务器端文件路径后加载数据，返回 DataContext 摘要。"""
    path = path.strip() if path else ""
    if not path:
        return ""
    try:
        context = agent.load_data(path)
        return context
    except Exception as exc:
        return f"[错误] 数据加载失败：{exc}"


def on_submit(user_input: str, fmt: str) -> tuple[str | None, str, str]:
    """
    发送按钮回调。

    Returns:
        (image_path_or_None, status_message, spec_json)
    """
    if not user_input.strip():
        return None, "请输入绘图需求", _spec_json()

    output_format = fmt.lower()
    response = agent.process_input(user_input, output_format=output_format)

    spec_text = _spec_json(response.current_spec)

    if response.status == "ok":
        return _image_and_status(response.image_path, "绘图完成") + (spec_text,)
    elif response.status == "need_input":
        question = response.question or ""
        # ask_user 触发时（agent 有 pending 输入），提示用户在输入框回复
        prefix = "需要确认（请在输入框回复）：\n" if agent.pending_user_input else ""
        return None, prefix + question, spec_text
    else:
        return None, f"[错误] {response.message}", spec_text


def on_rerender(spec_json: str, fmt: str) -> tuple[str | None, str, str]:
    """
    重新渲染回调：用 spec_display 中的 JSON 直接渲染，跳过 LLM。

    Returns:
        (image_path_or_None, status_message, spec_json)
    """
    if not spec_json or not spec_json.strip():
        return None, "PlotSpec 为空，请先生成图表", ""
    output_format = fmt.lower()
    response = agent.render_from_spec(spec_json, output_format=output_format)
    spec_text = _spec_json(response.current_spec)
    if response.status == "ok":
        return _image_and_status(response.image_path, "重新渲染完成") + (spec_text,)
    elif response.status == "need_input":
        return None, response.question or "", spec_text
    else:
        return None, f"[错误] {response.message}", spec_text


def on_backend_change(choice: str) -> str:
    """切换推理后端回调。"""
    backend = "plan_a" if "A" in choice else "plan_b"
    try:
        agent.set_backend(backend)
        return f"已切换到 {choice}"
    except Exception as exc:
        return f"[错误] 后端切换失败：{exc}"


def on_reset() -> tuple[None, str, str, str]:
    """重置按钮回调，清空所有状态。"""
    agent.reset()
    return None, "", "", ""


def _image_and_status(image_path: str | None, base_status: str) -> tuple[str | None, str]:
    """PNG 直接返回路径供 gr.Image 显示；PDF 返回 None 并在状态栏提示文件名。"""
    if image_path and not image_path.lower().endswith(".png"):
        filename = Path(image_path).name
        return None, f"{base_status}（{filename}，请使用导出按钮下载）"
    return image_path, base_status


def _spec_json(spec: dict | None = None) -> str:
    """将 PlotSpec 格式化为可读 JSON 字符串。"""
    if spec is None:
        return ""
    return json.dumps(spec, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 界面布局
# ---------------------------------------------------------------------------

with gr.Blocks(title="Scientific Plot Agent") as demo:
    gr.Markdown("# Scientific Plot Agent\n自然语言驱动的科研绘图系统")

    with gr.Row():
        # 左侧列：数据上传 + 数据摘要
        with gr.Column(scale=1):
            file_upload = gr.File(
                label="上传数据文件（CSV，本地文件）",
                file_types=[".csv"],
            )
            server_path = gr.Textbox(
                label="服务器文件路径（输入后按 Enter）",
                placeholder="/mnt/data/xxx.csv  或相对路径 data/train/xxx.csv",
                lines=1,
            )
            data_summary = gr.Textbox(
                label="数据摘要",
                lines=8,
                interactive=False,
                placeholder="上传文件或输入路径后自动显示数据摘要…",
            )
        # 右侧列：图表预览 + 输入区
        with gr.Column(scale=2):
            image_output = gr.Image(
                label="图表预览",
                height=400,
            )
            with gr.Row():
                user_input = gr.Textbox(
                    label="绘图需求",
                    placeholder="例：画一张柱状图，对比各模型在不同数据集上的准确率，使用 nature 风格",
                    lines=3,
                    scale=5,
                )
                submit_btn = gr.Button("发送", variant="primary", scale=1)
            status_box = gr.Textbox(
                label="状态 / 回问",
                interactive=False,
                lines=3,
            )

    # 底部：PlotSpec 编辑区 + 操作按钮
    with gr.Row():
        spec_display = gr.Code(
            label="当前 PlotSpec（可直接编辑后点击「重新渲染」）",
            language="json",
            interactive=True,
        )

    with gr.Row():
        backend_selector = gr.Radio(
            choices=["Plan A（本地 LoRA）", "Plan B（DeepSeek API）"],
            value="Plan A（本地 LoRA）",
            label="推理后端",
            scale=2,
        )
        fmt_selector = gr.Radio(
            choices=["PNG", "PDF"],
            value="PNG",
            label="导出格式",
            scale=1,
        )
        rerender_btn = gr.Button("重新渲染", variant="primary", scale=2)
        export_btn = gr.DownloadButton(label="导出文件", visible=True, scale=1)
        reset_btn = gr.Button("重置", variant="secondary", scale=1)

    # ---------------------------------------------------------------------------
    # 事件绑定
    # ---------------------------------------------------------------------------

    backend_selector.change(
        fn=on_backend_change,
        inputs=[backend_selector],
        outputs=[status_box],
    )

    file_upload.change(
        fn=on_upload,
        inputs=[file_upload],
        outputs=[data_summary],
    )

    server_path.submit(
        fn=on_path_input,
        inputs=[server_path],
        outputs=[data_summary],
    )

    submit_btn.click(
        fn=on_submit,
        inputs=[user_input, fmt_selector],
        outputs=[image_output, status_box, spec_display],
    )

    rerender_btn.click(
        fn=on_rerender,
        inputs=[spec_display, fmt_selector],
        outputs=[image_output, status_box, spec_display],
    )

    reset_btn.click(
        fn=on_reset,
        inputs=[],
        outputs=[image_output, data_summary, status_box, spec_display],
    )


if __name__ == "__main__":
    demo.launch()
