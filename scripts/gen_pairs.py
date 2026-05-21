"""
scripts/gen_pairs.py

为 data/train/ 下的每个 CSV 调用 DeepSeek API 生成首轮 (user_input, plotspec) 配对。

基础模式（默认）：
  每个 CSV 生成 N 条配对，写入 data/pairs/raw_pairs.jsonl，
  后续由 validate_pairs.py 过滤为 valid_pairs.jsonl。

补充模式（--supplement）：
  针对低频主题（P1: rococo/macaron）或从未出现的字段（P2）进行有针对性的生成，
  内置四级校验，直接追加到 data/pairs/valid_pairs.jsonl，无需再经 validate_pairs.py。

用法：
    # 基础模式
    python scripts/gen_pairs.py
    python scripts/gen_pairs.py --workers 8      # 8 个并发线程（加速，默认 1）
    python scripts/gen_pairs.py --limit 2        # 只处理前 2 个 CSV（测试）
    python scripts/gen_pairs.py --append         # 追加而非覆盖 raw_pairs.jsonl

    # 补充模式
    python scripts/gen_pairs.py --supplement             # P1 + P2 全补
    python scripts/gen_pairs.py --supplement --p1-only  # 只补 P1（低频主题）
    python scripts/gen_pairs.py --supplement --p2-only  # 只补 P2（缺失字段）
    python scripts/gen_pairs.py --supplement --dry-run  # 只打印场景，不调 LLM
    python scripts/gen_pairs.py --supplement --no-render # 跳过渲染校验（快速）

环境变量：
    DEEPSEEK_API_KEY   DeepSeek API 密钥（必须）
    DEEPSEEK_MODEL     模型名称（默认 deepseek-v4-pro）
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

from model.prompts import SYSTEM_FIRST_FINETUNE
from tools.loader import DataLoadError, load_data
from scripts.validate_pairs import validate_one

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).parent.parent / ".env")

TRAIN_DIR = Path("data/train")
PAIRS_DIR = Path("data/pairs")
PAIRS_DIR.mkdir(parents=True, exist_ok=True)

RAW_PAIRS_PATH   = PAIRS_DIR / "raw_pairs.jsonl"
VALID_PAIRS_PATH = PAIRS_DIR / "valid_pairs.jsonl"   # 补充模式写入目标

N_PAIRS_PER_CSV = 5
MAX_RETRIES = 3
RETRY_DELAY = 5.0

_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
_DEFAULT_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")

_SUPP_RNG = random.Random(99)   # 补充模式随机源（固定种子保证可复现）


# ---------------------------------------------------------------------------
# 合成 Prompt（基础模式）
# ---------------------------------------------------------------------------

def _build_synthesis_system_prompt() -> str:
    """构建用于批量生成训练数据的 system prompt。"""
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

【颜色字段使用规则——极易出错，务必遵守】
主题配色已自动为不同分组分配不同颜色，绝大多数情况不需要额外指定颜色字段。
只在以下场景才输出颜色字段，其余一律不输出：

  场景A：用户明确给出了具体十六进制颜色列表 → 用 style_custom_palette（适用所有图表类型）
    ✓ "颜色依次用 #E64B35、#4DBBD5、#00A087"（值为列表：["#E64B35","#4DBBD5","#00A087"]）
    ✗ "每个模型用不同颜色" ← 主题自动处理，禁止输出 style_custom_palette
    ✗ "第一条线用红色" ← 描述性颜色名而非十六进制列表，也不输出

  场景B：用户要求切换整套预设配色方案 → 用 style_palette_override
    ✓ "换成 tab10 配色"、"用莫兰迪色系"、"换一套更鲜艳的颜色"
    ✗ "用不同颜色区分" ← 不属于切换配色方案，禁止输出 style_palette_override

  既未指定具体十六进制颜色、也未要求换配色方案 → 两个字段都不输出
    ✗ "不同颜色的线"、"颜色区分开"、"每组一种颜色" ← 全部不需要输出颜色字段

【字段组合约束——以下组合语义无效，绝对不能出现】

  ① scatter 图 + 类别型 data_x + params_show_regression: true
    回归线需要数值型 X 轴，类别型列（如模型名、方法名、数据集名）无法计算。
    ✓ data_x="epoch"/"lr"/"threshold"（数值型）→ 可用 params_show_regression
    ✗ data_x="model"/"method"/"dataset"（类别型）→ 禁止 params_show_regression

  ② box 图 + data_y 为列表
    box 图的 data_y 必须是单个列名字符串，多指标对比用 data_group_by。
    ✗ "data_y": ["BLEU", "ROUGE-L"]（box图禁止列表）
    ✓ "data_y": "score", "data_group_by": "metric"（正确做法）

  ③ line 图 + 类别型 data_x + params_smooth: true
    平滑插值需要数值型 X 轴，类别字符串无法插值。
    ✗ data_x="model" + params_smooth: true

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
   · 至少 1 条使用坐标轴参数（axes_y_min 或 axes_x_rotate_labels 等）
   · 至少 1 条使用 data_filter 按类别或数值条件筛选数据行（若数据有合适的分类/数值列）
     示例：data_filter="benchmark == 'MMLU'"  data_filter="epoch >= 10"
     列名含连字符/空格等特殊字符时必须用反引号：data_filter="`SST-2` > 93"
   · 适当使用 params_sort / params_show_values / params_smooth / params_stacked 等图表专属参数

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
# API 调用工具
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
    """调用 DeepSeek API，返回解析后的 pair 列表（基础模式：期望 JSON 数组）。"""
    resp = client.chat.completions.create(
        model=model,
        temperature=0.8,
        max_tokens=4096,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    raw = resp.choices[0].message.content.strip()
    parsed = json.loads(raw)
    if isinstance(parsed, list):
        return parsed
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
    """带重试的批量 API 调用（基础模式）。全部失败返回 None。"""
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


def _call_llm_one(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
) -> dict | None:
    """调用 LLM，返回单条 {user_input, plotspec} dict（补充模式）。失败返回 None。"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.85,
                max_tokens=1500,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            )
            raw = resp.choices[0].message.content.strip()
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and "user_input" in parsed and "plotspec" in parsed:
                return parsed
            print(f"    [格式错误] 响应缺少 user_input 或 plotspec 字段")
            return None
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
# 基础模式：generate_pairs
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
    for k, v in spec.items():
        if isinstance(v, dict):
            return f"plotspec 含嵌套 dict（字段 {k!r}）"
    return None


def _process_csv_file(
    csv_path: Path,
    idx: int,
    total: int,
    client: OpenAI,
    model: str,
    system_prompt: str,
) -> tuple[list[dict], int]:
    """
    处理单个 CSV：加载数据 → 调用 API → 结构校验。
    返回 (有效记录列表, 跳过数量)。
    线程安全：client 本身是线程安全的，无共享可变状态。
    """
    label = f"[{idx}/{total}] {csv_path.name}"
    try:
        data_context, _ = load_data(str(csv_path))
    except Exception as e:
        print(f"{label}  ✗ load_data 失败：{e}")
        return [], 0

    user_msg = _build_synthesis_user_message(csv_path, data_context)
    pairs = _call_with_retry(client, model, system_prompt, user_msg)

    if pairs is None:
        print(f"{label}  ✗ API 调用全部失败，跳过")
        return [], 0

    records: list[dict] = []
    skipped = 0
    for j, pair in enumerate(pairs):
        err = _validate_pair_structure(pair)
        if err:
            print(f"{label}  ⚠ pair[{j}] 结构检查失败：{err}")
            skipped += 1
            continue
        records.append({
            "id": f"{csv_path.stem}_{j}",
            "csv_path": str(csv_path),
            "user_input": pair["user_input"],
            "plotspec": pair["plotspec"],
        })

    print(f"{label}  ✓ {len(records)}/{len(pairs)} 条有效（跳过 {skipped} 条）")
    return records, skipped


def generate_pairs(
    csv_paths: list[Path],
    model: str,
    append: bool = False,
    workers: int = 1,
) -> None:
    """
    为给定的 CSV 文件列表生成配对，写入 RAW_PAIRS_PATH。

    Args:
        csv_paths: 待处理的 CSV 文件路径列表。
        model:     DeepSeek 模型名称。
        append:    True=追加写入，False=覆盖写入。
        workers:   并发线程数（默认 1=顺序执行）。
                   建议值：4~8（受 DeepSeek API 并发限制）。
    """
    client, _ = _get_client(model)
    system_prompt = _build_synthesis_system_prompt()
    total = len(csv_paths)

    mode = "a" if append else "w"
    total_written = 0
    total_skipped = 0

    with RAW_PAIRS_PATH.open(mode, encoding="utf-8") as fout:
        if workers <= 1:
            # ── 顺序模式（原有行为）──────────────────────────────────────
            for i, csv_path in enumerate(csv_paths, 1):
                records, skipped = _process_csv_file(
                    csv_path, i, total, client, model, system_prompt
                )
                total_skipped += skipped
                for r in records:
                    fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                total_written += len(records)
        else:
            # ── 并发模式 ─────────────────────────────────────────────────
            # fout 的写入操作通过 write_lock 串行化，其余（API 调用、解析）并发执行
            write_lock = threading.Lock()
            print(f"并发模式：{workers} 个线程同时处理 {total} 个 CSV\n")

            futures: dict = {}
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for i, csv_path in enumerate(csv_paths, 1):
                    fut = executor.submit(
                        _process_csv_file,
                        csv_path, i, total, client, model, system_prompt,
                    )
                    futures[fut] = csv_path

                for fut in as_completed(futures):
                    try:
                        records, skipped = fut.result()
                    except Exception as exc:
                        csv_path = futures[fut]
                        print(f"  ✗ {csv_path.name} 线程异常：{exc}")
                        continue

                    with write_lock:
                        for r in records:
                            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                        total_written += len(records)
                        total_skipped += skipped

    print(f"\n完成：总写入 {total_written} 条，结构检查跳过 {total_skipped} 条。")
    print(f"输出文件：{RAW_PAIRS_PATH}")


# ---------------------------------------------------------------------------
# 补充模式：supplement_pairs
# ---------------------------------------------------------------------------

_THEME_DESC: dict[str, str] = {
    "rococo":  "洛可可低对比淡雅色调，清淡柔和",
    "macaron": "马卡龙轻快糖果色，明亮活泼",
}


@dataclass
class P1Scenario:
    """低频主题补充场景（P1）。"""
    theme: str
    chart_type: str
    n_target: int


@dataclass
class P2Scenario:
    """缺失字段补充场景（P2）。"""
    field: str
    chart_types: list[str]
    n_target: int
    field_desc: str
    value_example: str


P1_SCENARIOS: list[P1Scenario] = [
    P1Scenario("rococo", "bar",     5),
    P1Scenario("rococo", "line",    4),
    P1Scenario("rococo", "scatter", 3),
    P1Scenario("rococo", "heatmap", 3),
    P1Scenario("macaron", "bar",     5),
    P1Scenario("macaron", "heatmap", 5),
    P1Scenario("macaron", "box",     4),
    P1Scenario("macaron", "line",    4),
    P1Scenario("macaron", "scatter", 3),
]

P2_SCENARIOS: list[P2Scenario] = [
    P2Scenario(
        field="style_hatch",
        chart_types=["bar"],
        n_target=8,
        field_desc="柱子纹理填充，字符串如\"/\"或列表[\"/\",\"\\\\\",\"|\"]，黑白打印时用于区分不同分组",
        value_example='示例值："/" 或 ["/", "\\\\", "|"]（注意JSON中反斜杠需写成"\\\\"）',
    ),
    P2Scenario(
        field="style_font_size",
        chart_types=["bar", "line", "scatter", "box", "heatmap"],
        n_target=6,
        field_desc="字体大小（整数磅），覆盖主题默认字号，常见值12或14",
        value_example="示例值：14",
    ),
    P2Scenario(
        field="style_figure_width",
        chart_types=["bar", "line", "scatter", "box", "heatmap"],
        n_target=5,
        field_desc="图幅最小宽度（英寸），适合论文中精确控制图幅大小",
        value_example="示例值：7.0 或 8.0",
    ),
    P2Scenario(
        field="style_spines",
        chart_types=["bar", "line", "scatter", "box"],
        n_target=5,
        field_desc="保留的轴脊列表，去掉上边框和右边框只保留左轴和下轴",
        value_example='示例值：["left","bottom"]',
    ),
    P2Scenario(
        field="style_line_width",
        chart_types=["line", "scatter", "box"],
        n_target=5,
        field_desc="线宽（磅），加粗线条用于投影展示或期刊要求",
        value_example="示例值：2.5",
    ),
    P2Scenario(
        field="params_marker_size",
        chart_types=["line", "scatter"],
        n_target=5,
        field_desc="数据点标记大小（数值），null时按数据密度自动计算",
        value_example="示例值：4 或 6",
    ),
    P2Scenario(
        field="style_dpi",
        chart_types=["bar", "line", "scatter", "box", "heatmap"],
        n_target=4,
        field_desc="输出分辨率（dpi），期刊通常要求300 dpi",
        value_example="示例值：300",
    ),
]


def _build_p1_system(theme: str, chart_type: str) -> str:
    theme_desc = _THEME_DESC.get(theme, theme)
    return (
        f"{SYSTEM_FIRST_FINETUNE}\n\n"
        f"⚠️ 本次生成的额外约束（优先级高于上述规则）：\n"
        f'- chart_type 必须是："{chart_type}"\n'
        f'- style_theme 必须是："{theme}"（{theme_desc}）\n'
        f"- user_input 要自然地提到希望使用该风格，长度 15-70 字，用中文\n"
        f'- 输出格式：{{"user_input": "...", "plotspec": {{...完整PlotSpec，不含data_source...}}}}\n'
        f"- 只输出 JSON 对象，无任何解释文字"
    )


def _build_p2_system(scenario: P2Scenario, chart_type: str) -> str:
    return (
        f"{SYSTEM_FIRST_FINETUNE}\n\n"
        f"⚠️ 本次生成的额外约束（优先级高于上述规则）：\n"
        f'- chart_type 选择："{chart_type}"\n'
        f'- plotspec 中必须包含字段 "{scenario.field}"（{scenario.field_desc}）\n'
        f"- {scenario.value_example}\n"
        f"- user_input 要自然地描述需要 {scenario.field} 的使用场景，长度 20-80 字，用中文\n"
        f'- 输出格式：{{"user_input": "...", "plotspec": {{...完整PlotSpec，不含data_source...}}}}\n'
        f"- 只输出 JSON 对象，无任何解释文字"
    )


def _build_supp_user_message(data_context: str) -> str:
    return (
        f"{data_context}\n\n"
        "请根据上述数据生成一条 (user_input, plotspec) 配对样本。\n"
        "输出：/no_think"
    )


def _supp_get_csvs() -> list[Path]:
    csvs = sorted(TRAIN_DIR.glob("*.csv"))
    _SUPP_RNG.shuffle(csvs)
    return csvs


def _try_generate_one_supp(
    client: OpenAI,
    model: str,
    system: str,
    csv_path: Path,
    do_render: bool,
    scenario_id: str,
    required_field: Optional[str] = None,
) -> dict | None:
    """
    对给定 CSV 调用 LLM 生成一条样本，经四级校验后返回 enriched 记录。
    若指定 required_field，则验证该字段确实出现在生成的 plotspec 中。
    """
    try:
        data_context, _ = load_data(str(csv_path))
    except DataLoadError as e:
        print(f"      load_data 失败：{e}")
        return None

    user_msg = _build_supp_user_message(data_context)
    result = _call_llm_one(client, model, system, user_msg)
    if result is None:
        return None

    user_input = result.get("user_input", "").strip()
    plotspec = result.get("plotspec", {})
    if not user_input or not isinstance(plotspec, dict):
        print(f"      LLM 输出格式无效")
        return None

    if required_field and required_field not in plotspec:
        print(f"      LLM 未输出目标字段 {required_field}，跳过")
        return None

    record = {
        "id": scenario_id,
        "record_type": "first",
        "csv_path": str(csv_path),
        "user_input": user_input,
        "plotspec": plotspec,
    }

    ok, reason, enriched = validate_one(record, do_render=do_render)
    if not ok:
        print(f"      校验失败：{reason}")
        return None

    return enriched


def _run_supp_scenario(
    client: OpenAI,
    model: str,
    system: str,
    n_target: int,
    scenario_label: str,
    fout,
    do_render: bool,
    required_field: Optional[str] = None,
) -> int:
    """
    对一个场景反复尝试不同 CSV，直到写出 n_target 条有效样本。
    返回实际写出条数。
    """
    csvs = _supp_get_csvs()
    written = 0
    tried = 0

    for csv_path in csvs:
        if written >= n_target:
            break
        tried += 1
        scenario_id = f"supp_{scenario_label}_{tried}"
        print(f"    尝试 {csv_path.name} ...")
        enriched = _try_generate_one_supp(
            client, model, system, csv_path, do_render, scenario_id,
            required_field=required_field,
        )
        if enriched is not None:
            fout.write(json.dumps(enriched, ensure_ascii=False) + "\n")
            fout.flush()
            written += 1
            print(f"    ✓ 写入第 {written}/{n_target} 条")

    if written < n_target:
        print(f"    ⚠ CSV 已用完，只写出 {written}/{n_target} 条")

    return written


def supplement_pairs(
    do_p1: bool,
    do_p2: bool,
    dry_run: bool,
    model: str,
    do_render: bool,
) -> None:
    """
    补充模式：针对低频主题（P1）和缺失字段（P2）生成有针对性的样本，
    内置四级校验，直接追加到 valid_pairs.jsonl。
    """
    if dry_run:
        print("=== dry-run 模式，只打印场景列表 ===\n")
        if do_p1:
            print("P1 场景：")
            for s in P1_SCENARIOS:
                print(f"  theme={s.theme} chart_type={s.chart_type} target={s.n_target}")
        if do_p2:
            print("\nP2 场景：")
            for s in P2_SCENARIOS:
                print(f"  field={s.field} charts={s.chart_types} target={s.n_target}")
        return

    client, model = _get_client(model)

    total_written = 0
    with VALID_PAIRS_PATH.open("a", encoding="utf-8") as fout:

        # ── P1：低频主题补充 ─────────────────────────────────────────────
        if do_p1:
            print("\n===== P1：低频主题补充（rococo / macaron）=====\n")
            for s in P1_SCENARIOS:
                print(f"  场景：theme={s.theme}  chart_type={s.chart_type}  target={s.n_target}")
                system = _build_p1_system(s.theme, s.chart_type)
                label = f"p1_{s.theme}_{s.chart_type}"
                n = _run_supp_scenario(client, model, system, s.n_target, label, fout, do_render)
                total_written += n
                print()

        # ── P2：缺失字段补充 ─────────────────────────────────────────────
        if do_p2:
            print("\n===== P2：缺失字段补充 =====\n")
            for s in P2_SCENARIOS:
                chart_type = _SUPP_RNG.choice(s.chart_types)
                print(f"  场景：field={s.field}  chart_type={chart_type}  target={s.n_target}")
                system = _build_p2_system(s, chart_type)
                label = f"p2_{s.field}"
                n = _run_supp_scenario(
                    client, model, system, s.n_target, label, fout, do_render,
                    required_field=s.field,
                )
                total_written += n
                print()

    print(f"\n补充完成：共写入 {total_written} 条新样本 → {VALID_PAIRS_PATH}")
    print("请运行 python scripts/pack_finetune.py 重建 train/val.jsonl")


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成首轮 (user_input, plotspec) 配对数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # 基础模式参数
    parser.add_argument(
        "--limit", type=int, default=None,
        help="只处理前 N 个 CSV（快速测试用，默认处理全部）",
    )
    parser.add_argument(
        "--model", type=str, default=_DEFAULT_MODEL,
        help=f"DeepSeek 模型名称（默认：{_DEFAULT_MODEL}）",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="追加到已有的 raw_pairs.jsonl，而非覆盖（仅基础模式有效）",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="并发线程数（默认 1=顺序；建议 4~8，受 DeepSeek API 并发限制）",
    )

    # 补充模式参数
    parser.add_argument(
        "--supplement", action="store_true",
        help="补充模式：为低频主题（P1）和缺失字段（P2）生成有针对性的样本",
    )
    parser.add_argument(
        "--p1-only", action="store_true",
        help="补充模式：只运行 P1（低频主题 rococo/macaron）",
    )
    parser.add_argument(
        "--p2-only", action="store_true",
        help="补充模式：只运行 P2（缺失字段补充）",
    )
    parser.add_argument(
        "--no-render", action="store_true",
        help="补充模式：跳过渲染校验（快速模式，不保证可出图）",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="补充模式：只打印场景列表，不调用 LLM",
    )

    args = parser.parse_args()

    if args.supplement:
        do_p1 = not args.p2_only
        do_p2 = not args.p1_only
        supplement_pairs(
            do_p1=do_p1,
            do_p2=do_p2,
            dry_run=args.dry_run,
            model=args.model,
            do_render=not args.no_render,
        )
    else:
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

        generate_pairs(csv_paths, model=args.model, append=args.append, workers=args.workers)


if __name__ == "__main__":
    main()
