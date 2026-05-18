"""
C线 UI：Gradio 界面
所有回调通过共享的 PlotAgent 实例交互，不直接调用 model/ 或 tools/。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# 将项目根目录加入 sys.path，使 `python ui/app.py` 和 `python -m ui.app` 均可用
sys.path.insert(0, str(Path(__file__).parent.parent))

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


def on_submit(user_input: str) -> tuple[str | None, str, str]:
    """
    发送按钮回调。

    Returns:
        (image_path_or_None, status_message, spec_json)
    """
    if not user_input.strip():
        return None, "请输入绘图需求", _spec_json()

    response = agent.process_input(user_input)

    spec_text = _spec_json(response.current_spec)

    if response.status == "ok":
        return response.image_path, "绘图完成", spec_text
    elif response.status == "need_input":
        return None, response.question or "", spec_text
    else:
        return None, f"[错误] {response.message}", spec_text

    # ── C线扩展提示：发送后清空输入框 ─────────────────────────────
    # 在 submit_btn.click 的 outputs 列表末尾加上 user_input 组件，
    # 并在此函数的返回值元组末尾追加 ""，即可在每次发送后自动清空：
    #
    #   submit_btn.click(
    #       fn=on_submit,
    #       inputs=[user_input],
    #       outputs=[image_output, status_box, spec_display, user_input],  # 追加
    #   )
    #
    #   def on_submit(...) -> tuple[...]:
    #       ...
    #       return ..., ""   # 追加空字符串清空输入框
    # ──────────────────────────────────────────────────────────────


def on_reset() -> tuple[None, str, str, str]:
    """重置按钮回调，清空所有状态。"""
    agent.reset()
    return None, "", "", ""


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
                label="上传数据文件（CSV）",
                file_types=[".csv"],
            )
            data_summary = gr.Textbox(
                label="数据摘要",
                lines=10,
                interactive=False,
                placeholder="上传文件后自动显示数据摘要…",
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

    # 底部：PlotSpec 展示 + 操作按钮
    with gr.Row():
        spec_display = gr.Code(
            label="当前 PlotSpec",
            language="json",
            interactive=False,
        )

    with gr.Row():
        # ── C线扩展提示：导出按钮绑定实际文件路径 ─────────────────
        # 当前 export_btn 未绑定事件，点击无效。
        # 需要用一个 gr.State 存储最新的 image_path，再绑定 DownloadButton：
        #
        #   current_image = gr.State(value=None)   # 在 gr.Blocks 内模块级声明
        #
        #   # on_submit 返回 image_path 时同步更新 State：
        #   submit_btn.click(fn=on_submit, ..., outputs=[image_output, status_box, spec_display, current_image])
        #
        #   # DownloadButton 通过 value 参数绑定文件路径：
        #   export_btn = gr.DownloadButton(label="导出 PNG", value=lambda: current_image.value)
        #   # 或者用 .click 事件触发一个返回文件路径的函数
        # ──────────────────────────────────────────────────────────
        export_btn = gr.DownloadButton(label="导出 PNG", visible=True)
        reset_btn = gr.Button("重置", variant="secondary")

    # ---------------------------------------------------------------------------
    # 事件绑定
    # ---------------------------------------------------------------------------

    file_upload.change(
        fn=on_upload,
        inputs=[file_upload],
        outputs=[data_summary],
    )

    submit_btn.click(
        fn=on_submit,
        inputs=[user_input],
        outputs=[image_output, status_box, spec_display],
    )

    reset_btn.click(
        fn=on_reset,
        inputs=[],
        outputs=[image_output, data_summary, status_box, spec_display],
    )


if __name__ == "__main__":
    demo.launch()
