"""
临时脚本：将 reject_log.jsonl 里的 17 条记录手动修正后追加到 manual_pairs.jsonl。

修正策略：
  - 10条宽表 heatmap：validate_pairs bug 误拒，spec 本身正确，仅补 data_context
  - s19_rct_2 / s20a_physical_temperature_4：data_x 列名错误（大小写/中文），修正为实际列名
  - s18_hardware_latency_3：style_theme='nature_d' 不合法 → style_theme='nature' + palette_override
  - s15_random_search_3：heatmap 有重复 index → 改为 scatter
  - s38_tsf_benchmark_1：(model,benchmark) 重复 → 加 data_filter="horizon == 96"
  - s46_hparam_search_4：(optimizer,architecture) 重复 → 加 data_filter 固定超参
  - s25_wide_model_dataset_4：data_filter 列名含连字符未加反引号 → 修正
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.loader import load_data

MANUAL_PAIRS_PATH = Path("data/pairs/manual_pairs.jsonl")


def _ctx(csv_path: str) -> str:
    data_context, _ = load_data(csv_path)
    return data_context


# ---------------------------------------------------------------------------
# 17 条修正记录
# ---------------------------------------------------------------------------

def _build_records() -> list[dict]:
    records = []

    # ── 1. s02c_ablation_mt_2：宽表 heatmap，spec 正确 ──────────────────────
    csv = "data/train/s02c_ablation_mt.csv"
    records.append({
        "id": "rej_fix_s02c_ablation_mt_2",
        "record_type": "first",
        "csv_path": csv,
        "user_input": "画个热力图，看看这四个配置在BLEU和ROUGE-L上的表现。",
        "plotspec": {
            "chart_type": "heatmap",
            "data_x": "metric",
            "data_y": "configuration",
            "style_theme": "morandi",
            "params_annot": True,
            "label_title": "Ablation Study Heatmap",
        },
        "data_context": _ctx(csv),
    })

    # ── 2. s06a_image_cls_4：宽表 heatmap ───────────────────────────────────
    csv = "data/train/s06a_image_cls.csv"
    records.append({
        "id": "rej_fix_s06a_image_cls_4",
        "record_type": "first",
        "csv_path": csv,
        "user_input": "画个热力图看看所有模型在各个指标上的表现，行是模型名字，列是指标。",
        "plotspec": {
            "chart_type": "heatmap",
            "data_y": "algorithm",
            "data_x": "metric",
            "style_theme": "nature",
            "params_annot": True,
        },
        "data_context": _ctx(csv),
    })

    # ── 3. s07_detection_map_3：宽表 heatmap ────────────────────────────────
    csv = "data/train/s07_detection_map.csv"
    records.append({
        "id": "rej_fix_s07_detection_map_3",
        "record_type": "first",
        "csv_path": csv,
        "user_input": "用热力图展示这六个检测器在mAP_50、mAP_75、mAP_50_95三个指标上的表现，格子里显示数值，用科学期刊配色。",
        "plotspec": {
            "chart_type": "heatmap",
            "data_x": "metric",
            "data_y": "detector",
            "style_theme": "science",
            "params_annot": True,
        },
        "data_context": _ctx(csv),
    })

    # ── 4. s08_segmentation_iou_2：宽表 heatmap ─────────────────────────────
    csv = "data/train/s08_segmentation_iou.csv"
    records.append({
        "id": "rej_fix_s08_segmentation_iou_2",
        "record_type": "first",
        "csv_path": csv,
        "user_input": "用热力图展示所有模型在所有类别上的IoU得分，看看哪个模型在哪个类别表现好。",
        "plotspec": {
            "chart_type": "heatmap",
            "data_x": "class",
            "data_y": "system",
            "style_theme": "earth",
            "params_annot": True,
            "params_annot_fmt": ".1f",
            "label_title": "分割模型IoU热力图",
        },
        "data_context": _ctx(csv),
    })

    # ── 5. s15_random_search_3：改为 scatter（heatmap 有重复 index）──────────
    # 原始意图：看 learning_rate 和 hidden_size 对 accuracy 的影响
    # 修正：scatter，按 hidden_size 分组着色，保留所有 150 条记录
    csv = "data/train/s15_random_search.csv"
    records.append({
        "id": "rej_fix_s15_random_search_3",
        "record_type": "first",
        "csv_path": csv,
        "user_input": "用散点图展示learning rate和准确率的关系，不同hidden size用不同颜色区分，看看各规模模型在不同学习率下的表现。",
        "plotspec": {
            "chart_type": "scatter",
            "data_x": "learning_rate",
            "data_y": "val_accuracy",
            "data_group_by": "hidden_size",
            "style_theme": "macaron",
            "params_alpha": 0.7,
        },
        "data_context": _ctx(csv),
    })

    # ── 6. s18_hardware_latency_3：style_theme='nature_d' 不合法 ────────────
    # nature_d 是 palette_override 的值，不是 style_theme 的值
    # 修正：style_theme='nature' + style_palette_override='nature_d'
    # 补充：params_heatmap_value='latency_ms'（避免自动选到 batch_size）
    csv = "data/train/s18_hardware_latency.csv"
    records.append({
        "id": "rej_fix_s18_hardware_latency_3",
        "record_type": "first",
        "csv_path": csv,
        "user_input": "用热力图展示config和batch_size对延迟的影响",
        "plotspec": {
            "chart_type": "heatmap",
            "data_x": "batch_size",
            "data_y": "config",
            "style_theme": "nature",
            "style_palette_override": "nature_d",
            "params_annot": True,
            "params_heatmap_value": "latency_ms",
        },
        "data_context": _ctx(csv),
    })

    # ── 7. s19_rct_2：data_x 大小写错误（"Timepoint" → "timepoint"）──────────
    # 原始数据是长格式：condition, timepoint, response_rate
    # data_x 必须与实际列名一致；补 params_heatmap_value 指定值列
    csv = "data/train/s19_rct.csv"
    records.append({
        "id": "rej_fix_s19_rct_2",
        "record_type": "first",
        "csv_path": csv,
        "user_input": "我想看一个热力图，行是condition，列是timepoint，颜色深浅代表response_rate，用自然主题，格子里显示数字。",
        "plotspec": {
            "chart_type": "heatmap",
            "data_y": "condition",
            "data_x": "timepoint",
            "style_theme": "nature",
            "params_annot": True,
            "params_annot_fmt": ".1f",
            "params_heatmap_value": "response_rate",
        },
        "data_context": _ctx(csv),
    })

    # ── 8. s20a_physical_temperature_4：data_x 填了中文（"温度" → "temperature_C"）
    # 原始数据列名是 temperature_C，不是 "温度"
    # 修正：data_x='temperature_C'，补 params_heatmap_value='reaction_rate'
    csv = "data/train/s20a_physical_temperature.csv"
    records.append({
        "id": "rej_fix_s20a_physical_temperature_4",
        "record_type": "first",
        "csv_path": csv,
        "user_input": "用热力图展示不同催化剂在不同温度下的反应速率，横轴是温度，纵轴是催化剂，颜色深浅表示速率高低。",
        "plotspec": {
            "chart_type": "heatmap",
            "data_x": "temperature_C",
            "data_y": "catalyst",
            "style_theme": "earth",
            "params_heatmap_value": "reaction_rate",
        },
        "data_context": _ctx(csv),
    })

    # ── 9. s25_wide_model_dataset_0：宽表 heatmap ───────────────────────────
    csv = "data/train/s25_wide_model_dataset.csv"
    records.append({
        "id": "rej_fix_s25_wide_model_dataset_0",
        "record_type": "first",
        "csv_path": csv,
        "user_input": "帮我画个热力图看看这些模型在各个数据集上的效果怎么样",
        "plotspec": {
            "chart_type": "heatmap",
            "data_x": "dataset",
            "data_y": "model",
            "style_theme": "science",
        },
        "data_context": _ctx(csv),
    })

    # ── 10. s25_wide_model_dataset_4：data_filter 含连字符列名未加反引号 ──────
    # "SST-2 > 93" → "`SST-2` > 93"
    records.append({
        "id": "rej_fix_s25_wide_model_dataset_4",
        "record_type": "first",
        "csv_path": csv,
        "user_input": "绘制模型在RTE和MRPC两个推理任务上的堆叠柱状图，仅显示SST-2高于93的模型，用自定义颜色[#E64B35,#4DBBD5]，图表背景色为#f0f0f0，去除图例边框，使用莫兰迪主题",
        "plotspec": {
            "chart_type": "bar",
            "data_x": "model",
            "data_y": ["RTE", "MRPC"],
            "data_filter": "`SST-2` > 93",
            "style_theme": "morandi",
            "style_custom_palette": ["#E64B35", "#4DBBD5"],
            "style_bg_color": "#f0f0f0",
            "style_legend_frameon": False,
            "params_stacked": True,
        },
        "data_context": _ctx(csv),
    })

    # ── 11. s27_wide_time_category_3：宽表 heatmap ──────────────────────────
    csv = "data/train/s27_wide_time_category.csv"
    records.append({
        "id": "rej_fix_s27_wide_time_category_3",
        "record_type": "first",
        "csv_path": csv,
        "user_input": "用热力图展示这些领域在不同年份的发表数量，颜色越深代表越多，science风格，每个格子里显示数字。",
        "plotspec": {
            "chart_type": "heatmap",
            "data_x": "Field",
            "data_y": "year",
            "style_theme": "science",
            "params_annot": True,
            "params_annot_fmt": ".0f",
        },
        "data_context": _ctx(csv),
    })

    # ── 12. s22_likert_2：宽表 heatmap ──────────────────────────────────────
    csv = "data/train/s22_likert.csv"
    records.append({
        "id": "rej_fix_s22_likert_2",
        "record_type": "first",
        "csv_path": csv,
        "user_input": "用热力图展示各个问题在五个评价等级上的百分比分布，要显示数值，用science配色。",
        "plotspec": {
            "chart_type": "heatmap",
            "data_x": "response",
            "data_y": "question",
            "style_theme": "science",
            "params_annot": True,
            "params_annot_fmt": ".1f",
            "label_title": "Likert Scale Heatmap",
        },
        "data_context": _ctx(csv),
    })

    # ── 13. s26_wide_method_metric_1：宽表 heatmap ──────────────────────────
    csv = "data/train/s26_wide_method_metric.csv"
    records.append({
        "id": "rej_fix_s26_wide_method_metric_1",
        "record_type": "first",
        "csv_path": csv,
        "user_input": "绘制一个热力图，直观对比各方法在各项评价指标上的表现，使用Science期刊风格，并标注具体数值。",
        "plotspec": {
            "chart_type": "heatmap",
            "data_x": "指标",
            "data_y": "method",
            "style_theme": "science",
            "params_annot": True,
            "params_annot_fmt": ".2f",
            "label_title": "方法性能热力图",
        },
        "data_context": _ctx(csv),
    })

    # ── 14. s24_regional_comparison_2：宽表 heatmap ─────────────────────────
    csv = "data/train/s24_regional_comparison.csv"
    records.append({
        "id": "rej_fix_s24_regional_comparison_2",
        "record_type": "first",
        "csv_path": csv,
        "user_input": "我想用热力图展示不同区域在GDP、HDI和互联网渗透率上的差异，用大地色系，每个格子里显示数值",
        "plotspec": {
            "chart_type": "heatmap",
            "data_x": "指标",
            "data_y": "region",
            "style_theme": "earth",
            "params_annot": True,
            "label_title": "区域发展指标对比",
        },
        "data_context": _ctx(csv),
    })

    # ── 15. s38_tsf_benchmark_1：加 data_filter 消除重复 index ──────────────
    # (model, benchmark) 各有 4 个 horizon → 过滤到 horizon==96 得唯一组合
    csv = "data/train/s38_tsf_benchmark.csv"
    records.append({
        "id": "rej_fix_s38_tsf_benchmark_1",
        "record_type": "first",
        "csv_path": csv,
        "user_input": "请绘制分组柱状图，横轴为模型名称，纵轴为MAE，按benchmark分组，仅展示96步预测结果，采用Nature期刊风格，标题为'Model MAE Comparison across Benchmarks'，Y轴范围0到1",
        "plotspec": {
            "chart_type": "bar",
            "data_x": "model",
            "data_y": "MAE",
            "data_group_by": "benchmark",
            "data_filter": "horizon == 96",
            "style_theme": "nature",
            "label_title": "Model MAE Comparison across Benchmarks",
            "axes_y_min": 0,
            "axes_y_max": 1,
            "params_show_values": True,
        },
        "data_context": _ctx(csv),
    })

    # ── 16. s43b_clinical_ai_auc_wide_0：宽表 heatmap ───────────────────────
    csv = "data/train/s43b_clinical_ai_auc_wide.csv"
    records.append({
        "id": "rej_fix_s43b_clinical_ai_auc_wide_0",
        "record_type": "first",
        "csv_path": csv,
        "user_input": "请绘制临床AI模型各算法在不同疾病上AUC性能的热力图，使用科学期刊风格，添加标题'Clinical AI AUC Performance'，显示数值并保留三位小数。",
        "plotspec": {
            "chart_type": "heatmap",
            "data_x": "Disease",
            "data_y": "algorithm",
            "style_theme": "science",
            "label_title": "Clinical AI AUC Performance",
            "params_annot": True,
            "params_annot_fmt": ".3f",
            "style_dpi": 300,
        },
        "data_context": _ctx(csv),
    })

    # ── 17. s46_hparam_search_4：加 data_filter 固定超参，保证唯一 (optimizer,architecture) ──
    # 原始数据每个 (optimizer, architecture) 有 24 行（4lr×3bs×2wd）
    # 过滤 learning_rate=2e-5, batch_size=32, weight_decay=0.0 → 每对唯一
    csv = "data/train/s46_hparam_search.csv"
    records.append({
        "id": "rej_fix_s46_hparam_search_4",
        "record_type": "first",
        "csv_path": csv,
        "user_input": "比较不同优化器下各个模型架构的验证准确率（learning_rate=2e-5，batch_size=32，weight_decay=0），用分组柱状图展示，按准确率降序排列，柱子顶部显示数值",
        "plotspec": {
            "chart_type": "bar",
            "data_x": "optimizer",
            "data_y": "val_acc",
            "data_group_by": "architecture",
            "data_filter": "learning_rate == 2e-05 and batch_size == 32 and weight_decay == 0.0",
            "style_theme": "earth",
            "label_title": "不同优化器与架构的准确率对比",
            "label_y": "验证准确率 (%)",
            "params_sort": "desc",
            "params_show_values": True,
        },
        "data_context": _ctx(csv),
    })

    return records


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------

def main() -> None:
    records = _build_records()

    # 读取已有 ID，避免重复写入
    existing_ids: set[str] = set()
    if MANUAL_PAIRS_PATH.exists():
        with MANUAL_PAIRS_PATH.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        existing_ids.add(json.loads(line).get("id", ""))
                    except json.JSONDecodeError:
                        pass

    written = 0
    skipped = 0
    with MANUAL_PAIRS_PATH.open("a", encoding="utf-8") as f:
        for rec in records:
            if rec["id"] in existing_ids:
                print(f"  skip (already exists): {rec['id']}")
                skipped += 1
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"  wrote: {rec['id']}")
            written += 1

    print(f"\n完成：写入 {written} 条，跳过 {skipped} 条")
    print(f"输出：{MANUAL_PAIRS_PATH}")


if __name__ == "__main__":
    main()
