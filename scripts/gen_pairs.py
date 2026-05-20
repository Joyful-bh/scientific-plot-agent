"""
scripts/gen_pairs.py

调用 DeepSeek API，为 data/train/ 下的每个 CSV 生成 N 个 (user_input, plotspec) 配对。
输出到 data/pairs/raw_pairs.jsonl，每行一条记录。

用法：
    # 测试模式：只处理前 2 个 CSV
    python scripts/gen_pairs.py --limit 2

    # 完整运行
    python scripts/gen_pairs.py

环境变量：
    DEEPSEEK_API_KEY   DeepSeek API 密钥（必须）
    DEEPSEEK_MODEL     模型名称（默认 deepseek-chat，可改为 deepseek-v4-pro 等）
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

# 本项目模块
# sys.path 调整：scripts/ 在项目根下，直接 import 根目录模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.prompts import SYSTEM_FIRST_FINETUNE
from tools.loader import load_data

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).parent.parent / ".env")

TRAIN_DIR = Path("data/train")
PAIRS_DIR = Path("data/pairs")
PAIRS_DIR.mkdir(parents=True, exist_ok=True)

RAW_PAIRS_PATH = PAIRS_DIR / "raw_pairs.jsonl"

N_PAIRS_PER_CSV = 5         # 每个 CSV 生成的配对数量
MAX_RETRIES = 3              # API 调用失败时的最大重试次数
RETRY_DELAY = 5.0            # 重试间隔（秒）

_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
_DEFAULT_MODEL = "deepseek-chat"


# ---------------------------------------------------------------------------
# 合成 Prompt
# ---------------------------------------------------------------------------

def _build_synthesis_system_prompt() -> str:
    """构建用于生成训练数据的 DeepSeek system prompt。"""
    return f"""\
你是科研绘图训练数据生成专家，任务是为一个科研绘图 AI 模型构建监督微调数据集。

给定一个科研数据 CSV 的摘要，你需要生成 {N_PAIRS_PER_CSV} 个多样化的 (user_input, plotspec) 配对样本。
这些样本将用于训练一个 1.7B 参数的语言模型，让它学会根据自然语言请求生成 PlotSpec JSON。

=== PlotSpec 规则（严格遵守）===

{SYSTEM_FIRST_FINETUNE}

=== 额外约束 ===
- plotspec 中绝对不要包含 data_source 字段（系统自动注入，模型不需要输出）
- data_x 和 data_y 只能填数据摘要中出现的真实列名，不能填数据值或自己编造的名字
- 所有字段平铺在顶层，不允许嵌套 dict

=== 多样性要求（{N_PAIRS_PER_CSV} 个样本必须整体满足）===

1. 图表类型多样：尽量覆盖多种 chart_type（以数据特征为准）
   · 时序/连续数据适合 line；分组对比适合 bar；分布适合 box；相关性适合 scatter/heatmap

2. 风格主题覆盖至少 3 种不同的 style_theme

3. 语气/详细程度必须覆盖以下三类：
   a. 口语随意（占 1-2 条）：如"帮我画个图看哪个模型好"、"对比一下这几个结果"
   b. 学术正式详细（占 1-2 条）：明确指定 chart_type、style_theme、label、分组、误差棒等
   c. 中等（占 1-2 条）：指定图表类型但不指定全部参数

4. 可选字段使用多样：
   · 至少 1 条使用 data_group_by（若数据有分类列）
   · 至少 1 条指定 label_title 和 label_y
   · 至少 1 条使用坐标轴参数（axes_y_min 或 axes_x_tick_rotation 等）
   · 适当使用 params_sort / params_show_values / params_smooth 等图表专属参数

5. user_input 必须用中文，长度 15-120 字，禁止用相同的句式重复

=== 输出格式 ===
只输出合法 JSON 数组，不要任何解释文字、markdown 代码块或其他内容：
[
  {{
    "user_input": "用户的中文绘图请求",
    "plotspec": {{"chart_type": "...", "data_x": "...", "data_y": "...", "style_theme": "...", ...}}
  }},
  ...
]"""


def _build_synthesis_user_message(csv_path: Path, data_context: str) -> str:
    return (
        f"数据文件：{csv_path.name}\n"
        f"数据摘要：\n{data_context}\n\n"
        f"请生成 {N_PAIRS_PER_CSV} 个多样化的配对样本。"
    )


# ---------------------------------------------------------------------------
# API 调用
# ---------------------------------------------------------------------------

def _get_client(model: str) -> tuple[OpenAI, str]:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未找到 DEEPSEEK_API_KEY 环境变量。请设置：\n"
            "  $env:DEEPSEEK_API_KEY='your-key'  (PowerShell)\n"
            "  export DEEPSEEK_API_KEY='your-key'  (bash)"
        )
    client = OpenAI(api_key=api_key, base_url=_DEEPSEEK_BASE_URL)
    return client, model


def _call_api(client: OpenAI, model: str, system: str, user: str) -> list[dict]:
    """调用 DeepSeek API，返回解析后的 pair 列表。失败时抛出异常。"""
    resp = client.chat.completions.create(
        model=model,
        temperature=0.8,      # 较高温度增加多样性
        max_tokens=4096,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    raw = resp.choices[0].message.content.strip()

    # DeepSeek json_object 模式返回 object，但我们要 array；
    # 模型可能包装成 {"pairs": [...]} 或直接输出 [...]
    parsed = json.loads(raw)
    if isinstance(parsed, list):
        return parsed
    # 尝试从顶层 object 中找到 list 值
    for v in parsed.values():
        if isinstance(v, list):
            return v
    raise ValueError(f"无法从响应中提取 list，原始内容：{raw[:200]}")


def _call_with_retry(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
) -> list[dict] | None:
    """带重试的 API 调用。全部失败返回 None。"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return _call_api(client, model, system, user)
        except RateLimitError:
            wait = RETRY_DELAY * attempt
            print(f"    [限速] 等待 {wait:.0f}s 后重试（{attempt}/{MAX_RETRIES}）...")
            time.sleep(wait)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"    [解析失败 {attempt}/{MAX_RETRIES}] {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"    [API错误 {attempt}/{MAX_RETRIES}] {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    return None


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------

def _validate_pair_structure(pair: object) -> str | None:
    """
    快速结构检查（不做渲染校验，那是 validate_pairs.py 的工作）。
    返回 None 表示通过，返回错误描述字符串表示失败。
    """
    if not isinstance(pair, dict):
        return "pair 不是 dict"
    if "user_input" not in pair or "plotspec" not in pair:
        return "缺少 user_input 或 plotspec 字段"
    spec = pair["plotspec"]
    if not isinstance(spec, dict):
        return "plotspec 不是 dict"
    if "data_source" in spec:
        return "plotspec 含有 data_source（禁止）"
    for required in ("chart_type", "data_x", "data_y", "style_theme"):
        if required not in spec:
            return f"plotspec 缺少必填字段：{required}"
    # 检查嵌套 dict（禁止）
    for k, v in spec.items():
        if isinstance(v, dict):
            return f"plotspec 含嵌套 dict（字段 {k!r}）"
    return None


def generate_pairs(
    csv_paths: list[Path],
    model: str,
    append: bool = False,
) -> None:
    """
    为给定的 CSV 文件列表生成配对，追加写入 RAW_PAIRS_PATH。

    Args:
        csv_paths: 待处理的 CSV 文件路径列表。
        model:     DeepSeek 模型名称。
        append:    True=追加写入，False=覆盖写入（默认）。
    """
    client, model = _get_client(model)
    system_prompt = _build_synthesis_system_prompt()

    mode = "a" if append else "w"
    total_written = 0
    total_skipped = 0

    with RAW_PAIRS_PATH.open(mode, encoding="utf-8") as fout:
        for i, csv_path in enumerate(csv_paths, 1):
            print(f"[{i}/{len(csv_paths)}] {csv_path.name}")
            try:
                data_context, _ = load_data(str(csv_path))
            except Exception as e:
                print(f"  ✗ load_data 失败：{e}")
                continue

            user_msg = _build_synthesis_user_message(csv_path, data_context)
            pairs = _call_with_retry(client, model, system_prompt, user_msg)

            if pairs is None:
                print(f"  ✗ API 调用全部失败，跳过")
                continue

            written = 0
            for j, pair in enumerate(pairs):
                err = _validate_pair_structure(pair)
                if err:
                    print(f"  ⚠ pair[{j}] 结构检查失败：{err}")
                    total_skipped += 1
                    continue
                record = {
                    "id": f"{csv_path.stem}_{j}",
                    "csv_path": str(csv_path),
                    "user_input": pair["user_input"],
                    "plotspec": pair["plotspec"],
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

            total_written += written
            print(f"  ✓ 写入 {written}/{len(pairs)} 条（跳过 {len(pairs)-written} 条）")

    print(f"\n完成：总写入 {total_written} 条，结构检查跳过 {total_skipped} 条。")
    print(f"输出文件：{RAW_PAIRS_PATH}")


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="为训练 CSV 生成 (user_input, plotspec) 配对")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="只处理前 N 个 CSV（用于快速测试，默认处理全部）",
    )
    parser.add_argument(
        "--model", type=str,
        default=os.environ.get("DEEPSEEK_MODEL", _DEFAULT_MODEL),
        help=f"DeepSeek 模型名称（默认：{_DEFAULT_MODEL}）",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="追加到已有的 raw_pairs.jsonl，而非覆盖",
    )
    args = parser.parse_args()

    csv_paths = sorted(TRAIN_DIR.glob("*.csv"))
    if not csv_paths:
        print(f"✗ {TRAIN_DIR} 下没有找到 CSV 文件，请先运行 gen_csv.py")
        return

    if args.limit:
        csv_paths = csv_paths[: args.limit]
        print(f"测试模式：只处理前 {args.limit} 个 CSV\n")

    print(f"模型：{args.model}")
    print(f"CSV 数量：{len(csv_paths)}")
    print(f"每个 CSV 生成：{N_PAIRS_PER_CSV} 条配对")
    print(f"输出：{RAW_PAIRS_PATH}\n")

    generate_pairs(csv_paths, model=args.model, append=args.append)


if __name__ == "__main__":
    main()
