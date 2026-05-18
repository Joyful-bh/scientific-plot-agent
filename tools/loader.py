"""
B线工具：DataLoader
解析数据文件，生成 DataContext 摘要字符串，并维护内存缓存供 renderer 使用。
"""

import uuid
from pathlib import Path

import pandas as pd


class DataLoadError(Exception):
    """数据加载失败时抛出。"""


# 模块级内存缓存：cache_key -> DataFrame
_CACHE: dict[str, pd.DataFrame] = {}


def load_data(source: str) -> tuple[str, str]:
    """
    加载 CSV 文件，生成 DataContext 摘要并存入内存缓存。

    Args:
        source: CSV 文件路径（初版只支持 CSV）。

    Returns:
        (data_context, cache_key) 二元组。
        data_context 是可注入 prompt 的可读摘要字符串。
        cache_key 格式为 "cache://xxxx"。

    Raises:
        DataLoadError: 文件不存在或格式无法解析时。
    """
    path = Path(source)
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise DataLoadError(f"文件不存在：{source}")
    except pd.errors.ParserError:
        raise DataLoadError(f"文件格式无法解析，请确认是合法的 CSV：{source}")
    except Exception as exc:
        raise DataLoadError(f"加载文件时出错：{exc}") from exc

    cache_key = f"cache://{uuid.uuid4().hex[:8]}"
    _CACHE[cache_key] = df

    data_context = _build_context(df, cache_key)
    return data_context, cache_key


def get_dataframe(cache_key: str) -> pd.DataFrame:
    """
    根据 cache_key 从内存缓存取回完整 DataFrame。

    Args:
        cache_key: load_data() 返回的缓存键。

    Returns:
        对应的 DataFrame。

    Raises:
        DataLoadError: cache_key 不存在时。
    """
    if cache_key not in _CACHE:
        raise DataLoadError(f"缓存键不存在：{cache_key}，请重新加载数据文件。")
    return _CACHE[cache_key]


def _infer_column_type(series: pd.Series) -> str:
    """
    推断列类型。

    Returns:
        "类别型" | "数值型" | "时间型"
    """
    if pd.api.types.is_numeric_dtype(series):
        return "数值型"
    try:
        pd.to_datetime(series, format="mixed", dayfirst=False)
        return "时间型"
    except (ValueError, TypeError):
        pass
    return "类别型"


def _build_context(df: pd.DataFrame, cache_key: str) -> str:
    """
    构建可读的 DataContext 摘要字符串。

    ⚠️  B线若修改此函数的输出格式，必须提前告知 A线。
    A线的微调训练数据（instruction 字段）使用与此处完全相同的格式。
    两者格式不一致会导致推理时模型无法正确理解数据摘要。

    当前输出格式示例（以 example_bar.csv 为例）：

        数据摘要：
        - 形状：24行 × 4列
        - 列信息：
          · method（类别型，唯一值6个）：BERT-base, RoBERTa-base, XLNet-base, ALBERT-base
          · accuracy（数值型，范围55.1~95.1）
          · std（数值型，范围0.3~1.1）
        - 前2行预览：[["BERT-base","SST-2",93.5,0.3],["BERT-base","MR",87.3,0.4]]
        - 缓存key：cache://a1b2c3d4

    验证方式：python -c "from tools.loader import load_data; ctx,_ = load_data('data/example_bar.csv'); print(ctx)"
    """
    rows, cols = df.shape
    lines: list[str] = [
        "数据摘要：",
        f"- 形状：{rows}行 × {cols}列",
        "- 列信息：",
    ]

    for col in df.columns:
        col_type = _infer_column_type(df[col])
        if col_type == "类别型":
            unique_vals = df[col].dropna().unique()
            unique_count = len(unique_vals)
            sample = ", ".join(str(v) for v in unique_vals[:4])
            lines.append(f"  · {col}（{col_type}，唯一值{unique_count}个）：{sample}")
        elif col_type == "数值型":
            lo = df[col].min()
            hi = df[col].max()
            lines.append(f"  · {col}（{col_type}，范围{lo}~{hi}）")
        else:
            lines.append(f"  · {col}（{col_type}）")

    preview = df.head(2).values.tolist()
    lines.append(f"- 前2行预览：{preview}")
    lines.append(f"- 缓存key：{cache_key}")

    return "\n".join(lines)
