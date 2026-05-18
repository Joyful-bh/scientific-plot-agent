"""model 包：A线，1.5B 模型推理，输入用户意图+数据摘要，输出 PlotSpec JSON。"""

from model.generator import generate_spec

__all__ = ["generate_spec"]
