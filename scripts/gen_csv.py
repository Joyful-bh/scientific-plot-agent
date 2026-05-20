"""
scripts/gen_csv.py

读取 scripts/需要合成的场景.md，生成所有场景对应的训练数据 CSV。
输出到 data/train/ 目录，每个场景生成 2-4 个变体实例。

运行：  python scripts/gen_csv.py
"""

from __future__ import annotations

import itertools
import random
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
OUT_DIR = Path("data/train")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _save(df: pd.DataFrame, name: str) -> None:
    path = OUT_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"  写入 {path}  ({df.shape[0]}行 × {df.shape[1]}列)")


def _jitter(base: float, lo: float, hi: float, n: int) -> np.ndarray:
    """在 base 附近加均匀噪声，clip 到 [lo, hi]。"""
    return np.clip(base + RNG.uniform(-1.5, 1.5, n), lo, hi)


def _smooth_loss(n: int, start: float = 2.4, end: float = 0.15) -> np.ndarray:
    """指数衰减 loss 曲线，加小噪声。"""
    t = np.linspace(0, 1, n)
    base = start * np.exp(-4 * t) + end
    noise = RNG.normal(0, 0.02, n)
    return np.clip(base + noise, 0.05, None)


# ---------------------------------------------------------------------------
# 场景 1：NLP 模型×数据集基准测试（长表）
# ---------------------------------------------------------------------------

def gen_s01() -> None:
    """4-8 个模型 × 3-6 个数据集，1-2 个指标，含/不含 std。"""

    # 变体 A：4 模型 × 4 数据集，accuracy + std，model 列
    models_a = ["BERT-base", "RoBERTa-base", "XLNet-base", "DeBERTa-base"]
    datasets_a = ["SST-2", "CoLA", "RTE", "QNLI"]
    base_acc = {"SST-2": 93.0, "CoLA": 62.0, "RTE": 83.0, "QNLI": 91.5}
    model_delta = {"BERT-base": 0, "RoBERTa-base": 1.5, "XLNet-base": 0.8, "DeBERTa-base": 2.5}
    rows = []
    for m in models_a:
        for d in datasets_a:
            acc = float(np.clip(base_acc[d] + model_delta[m] + RNG.uniform(-0.5, 0.5), 55, 99))
            std = float(RNG.uniform(0.2, 0.8))
            rows.append({"model": m, "dataset": d, "accuracy": round(acc, 2), "std": round(std, 2)})
    _save(pd.DataFrame(rows), "s01a_nlp_benchmark_long")

    # 变体 B：6 模型 × 5 数据集，F1 指标，method 列名
    methods_b = ["BERT", "RoBERTa", "XLNet", "DeBERTa", "ALBERT", "ELECTRA"]
    datasets_b = ["CoNLL-03", "OntoNotes", "WNUT", "BC5CDR", "NCBI"]
    base_f1 = {"CoNLL-03": 91.5, "OntoNotes": 88.0, "WNUT": 57.0, "BC5CDR": 88.5, "NCBI": 86.0}
    m_delta_b = {"BERT": 0, "RoBERTa": 2.0, "XLNet": 1.2, "DeBERTa": 3.0, "ALBERT": -1.0, "ELECTRA": 1.8}
    rows = []
    for m in methods_b:
        for d in datasets_b:
            f1 = float(np.clip(base_f1[d] + m_delta_b[m] + RNG.uniform(-0.8, 0.8), 50, 99))
            rows.append({"method": m, "dataset": d, "F1": round(f1, 2)})
    _save(pd.DataFrame(rows), "s01b_nlp_benchmark_long")

    # 变体 C：5 模型 × 3 数据集，BLEU + ROUGE-L（机器翻译）
    systems_c = ["mBART", "mT5-base", "mT5-large", "OPUS-MT", "M2M-100"]
    datasets_c = ["WMT-EN-DE", "WMT-EN-ZH", "WMT-EN-FR"]
    base_bleu = {"WMT-EN-DE": 29.0, "WMT-EN-ZH": 22.0, "WMT-EN-FR": 38.0}
    base_rouge = {"WMT-EN-DE": 54.0, "WMT-EN-ZH": 47.0, "WMT-EN-FR": 60.0}
    s_delta = {"mBART": 0, "mT5-base": -1.5, "mT5-large": 1.8, "OPUS-MT": -0.5, "M2M-100": 2.2}
    rows = []
    for s in systems_c:
        for d in datasets_c:
            bleu = float(np.clip(base_bleu[d] + s_delta[s] + RNG.uniform(-0.5, 0.5), 15, 50))
            rouge = float(np.clip(base_rouge[d] + s_delta[s] * 1.5 + RNG.uniform(-0.8, 0.8), 35, 70))
            rows.append({"system": s, "dataset": d, "BLEU": round(bleu, 2), "ROUGE-L": round(rouge, 2)})
    _save(pd.DataFrame(rows), "s01c_nlp_mt_long")

    # 变体 D：8 模型 × 6 数据集，score + params_M
    approaches = ["GPT2-small", "GPT2-medium", "GPT2-large", "GPT2-XL",
                  "GPT-Neo-1.3B", "GPT-Neo-2.7B", "OPT-1.3B", "OPT-2.7B"]
    datasets_d = ["HellaSwag", "WinoGrande", "ARC-Easy", "ARC-Challenge", "PIQA", "OpenBookQA"]
    base_score = {"HellaSwag": 55, "WinoGrande": 60, "ARC-Easy": 65, "ARC-Challenge": 40, "PIQA": 70, "OpenBookQA": 55}
    params_map = {"GPT2-small": 117, "GPT2-medium": 345, "GPT2-large": 774, "GPT2-XL": 1558,
                  "GPT-Neo-1.3B": 1300, "GPT-Neo-2.7B": 2700, "OPT-1.3B": 1300, "OPT-2.7B": 2700}
    rows = []
    for a in approaches:
        p = params_map[a]
        scale_bonus = np.log10(p / 100) * 5
        for d in datasets_d:
            sc = float(np.clip(base_score[d] + scale_bonus + RNG.uniform(-1.5, 1.5), 30, 95))
            rows.append({"approach": a, "dataset": d, "score": round(sc, 2), "params_M": p})
    _save(pd.DataFrame(rows), "s01d_nlp_lm_benchmark_long")


# ---------------------------------------------------------------------------
# 场景 2：消融实验
# ---------------------------------------------------------------------------

def gen_s02() -> None:
    """基础模型 + 3-6 个变体，1-3 个指标。"""

    # 变体 A：6 个 variant，accuracy + F1
    variants_a = ["Full model", "w/o attention", "w/o position emb", "w/o layer norm", "w/o residual", "w/o pre-training"]
    base_acc, base_f1 = 93.2, 92.5
    drops_acc = [0, -3.8, -1.4, -2.1, -4.5, -7.2]
    drops_f1  = [0, -3.5, -1.2, -2.0, -4.2, -7.0]
    rows = []
    for v, da, df_ in zip(variants_a, drops_acc, drops_f1):
        rows.append({
            "variant": v,
            "accuracy": round(base_acc + da + float(RNG.uniform(-0.3, 0.3)), 2),
            "F1": round(base_f1 + df_ + float(RNG.uniform(-0.3, 0.3)), 2),
        })
    _save(pd.DataFrame(rows), "s02a_ablation")

    # 变体 B：5 个 setting，单一 mAP 指标，目标检测消融
    settings_b = ["Full", "-FPN", "-data aug", "-multi-scale", "-deformable conv"]
    base_map = 48.3
    drops_map = [0, -2.1, -3.5, -1.8, -4.2]
    rows = []
    for s, d in zip(settings_b, drops_map):
        rows.append({"setting": s, "mAP": round(base_map + d + float(RNG.uniform(-0.2, 0.2)), 2)})
    _save(pd.DataFrame(rows), "s02b_ablation_detection")

    # 变体 C：4 个 configuration，3 个指标，含推理速度
    configs_c = ["Full", "No cross-attn", "No self-attn", "No FFN"]
    base_bleu, base_rouge, base_speed = 34.2, 58.4, 120.0
    d_bleu  = [0, -1.8, -2.5, -0.9]
    d_rouge = [0, -1.2, -2.0, -0.7]
    d_speed = [0, 15, 22, 8]
    rows = []
    for c, db, dr, ds in zip(configs_c, d_bleu, d_rouge, d_speed):
        rows.append({
            "configuration": c,
            "BLEU": round(base_bleu + db + float(RNG.uniform(-0.2, 0.2)), 2),
            "ROUGE-L": round(base_rouge + dr + float(RNG.uniform(-0.3, 0.3)), 2),
            "speed_ms": round(base_speed - ds + float(RNG.uniform(-2, 2)), 1),
        })
    _save(pd.DataFrame(rows), "s02c_ablation_mt")


# ---------------------------------------------------------------------------
# 场景 3：不同训练数据量的学习曲线（横截面）
# ---------------------------------------------------------------------------

def gen_s03() -> None:
    """5-8 个数据量节点，2-3 种方法对比。"""

    # 变体 A：7 个节点，3 种方法
    sizes_a = [100, 500, 1000, 5000, 10000, 50000, 100000]
    methods_a = ["Supervised", "Semi-supervised", "Pre-trained+fine-tune"]
    base_perf = {
        "Supervised": lambda n: 40 + 18 * np.log10(n / 100),
        "Semi-supervised": lambda n: 52 + 14 * np.log10(n / 100),
        "Pre-trained+fine-tune": lambda n: 72 + 10 * np.log10(n / 100),
    }
    rows = []
    for m in methods_a:
        for n in sizes_a:
            acc = float(np.clip(base_perf[m](n) + RNG.uniform(-1.5, 1.5), 30, 98))
            rows.append({"method": m, "train_size": n, "accuracy": round(acc, 2)})
    _save(pd.DataFrame(rows), "s03a_data_scaling")

    # 变体 B：6 个节点，2 种方法，F1 指标
    sizes_b = [500, 1000, 2000, 5000, 10000, 20000]
    rows = []
    for method in ["Baseline", "Our method"]:
        base_bonus = 0 if method == "Baseline" else 8
        for n in sizes_b:
            f1 = float(np.clip(55 + base_bonus + 12 * np.log10(n / 500) + RNG.uniform(-1, 1), 40, 95))
            rows.append({"method": method, "n_train": n, "F1": round(f1, 2)})
    _save(pd.DataFrame(rows), "s03b_data_scaling")


# ---------------------------------------------------------------------------
# 场景 4：Few-shot 性能（k-shot 对比）
# ---------------------------------------------------------------------------

def gen_s04() -> None:
    """k = 0/1/2/4/8/16/32，2-4 种方法。"""

    k_vals = [0, 1, 2, 4, 8, 16, 32]

    # 变体 A：4 种方法
    methods_a = ["GPT-3", "PET", "LM-BFF", "In-Context"]
    zero_shot = {"GPT-3": 65, "PET": 58, "LM-BFF": 55, "In-Context": 63}
    saturate  = {"GPT-3": 87, "PET": 90, "LM-BFF": 91, "In-Context": 85}
    rows = []
    for m in methods_a:
        z, s = zero_shot[m], saturate[m]
        for k in k_vals:
            if k == 0:
                acc = z + float(RNG.uniform(-1, 1))
            else:
                gain = (s - z) * (1 - np.exp(-k / 8))
                acc = z + gain + float(RNG.uniform(-1, 1))
            rows.append({"method": m, "k_shot": k, "accuracy": round(float(np.clip(acc, 40, 99)), 2)})
    _save(pd.DataFrame(rows), "s04a_fewshot")

    # 变体 B：3 种方法，accuracy + F1
    methods_b = ["Prompt-tuning", "Adapter", "Full fine-tune"]
    rows = []
    for m in methods_b:
        zs = {"Prompt-tuning": 60, "Adapter": 52, "Full fine-tune": 48}[m]
        sat = {"Prompt-tuning": 85, "Adapter": 88, "Full fine-tune": 92}[m]
        for k in k_vals:
            if k == 0:
                acc = zs
            else:
                acc = zs + (sat - zs) * (1 - np.exp(-k / 10))
            acc += float(RNG.uniform(-1.2, 1.2))
            f1 = acc - float(RNG.uniform(0.5, 2.0))
            rows.append({"method": m, "k_shot": k,
                         "accuracy": round(float(np.clip(acc, 35, 99)), 2),
                         "F1": round(float(np.clip(f1, 33, 99)), 2)})
    _save(pd.DataFrame(rows), "s04b_fewshot")


# ---------------------------------------------------------------------------
# 场景 5：多语言评测
# ---------------------------------------------------------------------------

def gen_s05() -> None:
    """3-6 种语言，1-2 个模型，2-3 个指标。"""

    langs = ["English", "German", "French", "Chinese", "Arabic", "Swahili", "Thai"]
    lang_base = {"English": 91, "German": 83, "French": 82, "Chinese": 80, "Arabic": 73, "Swahili": 62, "Thai": 60}

    # 变体 A：6 种语言，2 个模型，accuracy
    rows = []
    for lang in list(lang_base.keys())[:6]:
        for model in ["mBERT", "XLM-R"]:
            bonus = 0 if model == "mBERT" else 4
            acc = float(np.clip(lang_base[lang] + bonus + RNG.uniform(-1, 1), 50, 99))
            rows.append({"model": model, "language": lang, "accuracy": round(acc, 2)})
    _save(pd.DataFrame(rows), "s05a_multilingual")

    # 变体 B：7 种语言，1 个模型，F1 + EM（QA 任务）
    rows = []
    for lang in langs:
        f1 = float(np.clip(lang_base[lang] - 2 + RNG.uniform(-1.5, 1.5), 45, 96))
        em = float(np.clip(f1 - float(RNG.uniform(3, 8)), 40, 92))
        rows.append({"language": lang, "F1": round(f1, 2), "EM": round(em, 2)})
    _save(pd.DataFrame(rows), "s05b_multilingual_qa")


# ---------------------------------------------------------------------------
# 场景 6：图像分类基准（含计算成本）
# ---------------------------------------------------------------------------

def gen_s06() -> None:
    """5-8 个模型，top1/top5，params_M，FLOPs_G，img/s。"""

    # 变体 A：8 个模型
    models = ["MobileNetV3", "EfficientNet-B0", "ResNet-50", "EfficientNet-B4",
              "ViT-S/16", "ResNet-101", "EfficientNet-B7", "ViT-L/16"]
    top1   = [75.2, 77.1, 76.4, 82.9, 81.4, 77.5, 84.3, 86.1]
    top5   = [92.5, 93.4, 93.0, 96.4, 95.8, 93.6, 97.0, 97.8]
    params = [5.5,  5.3, 25.6, 19.3, 22.1, 44.5, 66.4, 307.0]
    flops  = [0.22, 0.39, 4.1, 4.2, 4.6, 7.9, 37.0, 61.6]
    speed  = [3200, 2800, 1050, 420, 640, 520, 95, 180]
    rows = []
    for i, m in enumerate(models):
        rows.append({
            "model": m,
            "top1_acc": round(top1[i] + float(RNG.uniform(-0.2, 0.2)), 2),
            "top5_acc": round(top5[i] + float(RNG.uniform(-0.1, 0.1)), 2),
            "params_M": params[i],
            "FLOPs_G": flops[i],
            "img_per_sec": round(speed[i] * float(RNG.uniform(0.95, 1.05))),
        })
    _save(pd.DataFrame(rows), "s06a_image_cls")

    # 变体 B：5 个模型，简化版
    rows_b = []
    for model, t1, p, sp in zip(
        ["ResNet-18", "ResNet-34", "ResNet-50", "ResNet-101", "ResNet-152"],
        [69.8, 73.3, 76.1, 77.4, 78.3],
        [11.7, 21.8, 25.6, 44.5, 60.2],
        [5800, 3200, 1050, 520, 340],
    ):
        rows_b.append({
            "model": model,
            "top1_acc": round(t1 + float(RNG.uniform(-0.2, 0.2)), 2),
            "params_M": p,
            "img_per_sec": round(sp * float(RNG.uniform(0.95, 1.05))),
        })
    _save(pd.DataFrame(rows_b), "s06b_resnet_scaling")


# ---------------------------------------------------------------------------
# 场景 7：目标检测（mAP 多阈值）
# ---------------------------------------------------------------------------

def gen_s07() -> None:
    """4-6 个检测器，mAP@0.5 / @0.75 / @0.5:0.95，推理时间。"""

    detectors = ["Faster-RCNN", "RetinaNet", "FCOS", "DETR", "DINO", "Co-DETR"]
    map50  = [65.2, 63.8, 64.5, 61.3, 72.4, 75.1]
    map75  = [47.8, 46.2, 47.0, 44.1, 53.6, 56.2]
    mapall = [41.5, 39.8, 40.5, 37.8, 48.3, 51.0]
    lat_ms = [72,   58,   54,   92,   85,   120]
    rows = []
    for i, d in enumerate(detectors):
        rows.append({
            "detector": d,
            "mAP_50": round(map50[i] + float(RNG.uniform(-0.3, 0.3)), 2),
            "mAP_75": round(map75[i] + float(RNG.uniform(-0.3, 0.3)), 2),
            "mAP_50_95": round(mapall[i] + float(RNG.uniform(-0.3, 0.3)), 2),
            "latency_ms": round(lat_ms[i] * float(RNG.uniform(0.95, 1.05)), 1),
        })
    _save(pd.DataFrame(rows), "s07_detection_map")


# ---------------------------------------------------------------------------
# 场景 8：分割任务（多类别 IoU）
# ---------------------------------------------------------------------------

def gen_s08() -> None:
    """3-5 个方法，overall mIoU + 4-8 个类别 IoU。"""

    categories = ["background", "person", "car", "bicycle", "motorcycle", "bus", "truck", "sky"]
    base_iou = {"background": 92, "person": 78, "car": 82, "bicycle": 65, "motorcycle": 62,
                "bus": 73, "truck": 68, "sky": 90}
    methods = ["FCN", "DeepLabV3+", "Segformer", "Mask2Former", "SETR"]
    m_bonus = {"FCN": -5, "DeepLabV3+": 0, "Segformer": 2, "Mask2Former": 4, "SETR": 3}
    rows = []
    for m in methods:
        row: dict = {"method": m}
        class_ious = []
        for c in categories:
            iou = float(np.clip(base_iou[c] + m_bonus[m] + RNG.uniform(-2, 2), 25, 99))
            row[f"IoU_{c}"] = round(iou, 2)
            class_ious.append(iou)
        row["mIoU"] = round(float(np.mean(class_ious)), 2)
        rows.append(row)
    _save(pd.DataFrame(rows), "s08_segmentation_iou")


# ---------------------------------------------------------------------------
# 场景 9：数据增强消融
# ---------------------------------------------------------------------------

def gen_s09() -> None:
    """基础 + 4-6 种增强策略，1-2 个指标。"""

    strategies = [
        "Baseline",
        "+RandomCrop",
        "+ColorJitter",
        "+Mixup",
        "+CutMix",
        "+RandomCrop+ColorJitter",
        "+all augmentations",
    ]
    base_top1 = 74.2
    increments = [0, 1.8, 1.5, 2.2, 2.4, 2.9, 4.1]
    rows = []
    for s, inc in zip(strategies, increments):
        acc = float(np.clip(base_top1 + inc + RNG.uniform(-0.2, 0.2), 70, 82))
        rows.append({"strategy": s, "top1_acc": round(acc, 2)})
    _save(pd.DataFrame(rows), "s09_augmentation_ablation")


# ---------------------------------------------------------------------------
# 场景 10：单模型训练曲线
# ---------------------------------------------------------------------------

def gen_s10() -> None:
    """epoch 为 X 轴，train_loss + val_loss（必有），可能含 val_acc / lr。"""

    # 变体 A：100 epoch，含 val_acc
    n = 100
    epochs = np.arange(1, n + 1)
    train_loss = _smooth_loss(n, 2.5, 0.12)
    val_loss = train_loss + np.clip(RNG.uniform(-0.02, 0.05, n) + np.linspace(0, 0.08, n), 0, None)
    val_acc = np.clip(98 - 60 * np.exp(-epochs / 15) + RNG.normal(0, 0.3, n), 30, 99)
    df = pd.DataFrame({
        "epoch": epochs.tolist(),
        "train_loss": [round(float(x), 4) for x in train_loss],
        "val_loss": [round(float(x), 4) for x in val_loss],
        "val_acc": [round(float(x), 2) for x in val_acc],
    })
    _save(df, "s10a_training_curve")

    # 变体 B：200 step（以千为单位），含 lr（余弦衰减）
    n2 = 40
    steps = [i * 5 for i in range(1, n2 + 1)]
    t_loss2 = _smooth_loss(n2, 3.1, 0.18)
    v_loss2 = t_loss2 + np.clip(RNG.uniform(-0.01, 0.04, n2) + np.linspace(0, 0.12, n2), 0, None)
    lr = 3e-4 * (1 + np.cos(np.pi * np.arange(n2) / n2)) / 2
    df2 = pd.DataFrame({
        "step": steps,
        "train_loss": [round(float(x), 4) for x in t_loss2],
        "val_loss": [round(float(x), 4) for x in v_loss2],
        "lr": [float(f"{x:.2e}") for x in lr],
    })
    _save(df2, "s10b_training_curve_step")


# ---------------------------------------------------------------------------
# 场景 11：多模型训练曲线对比
# ---------------------------------------------------------------------------

def gen_s11() -> None:
    """3-5 个模型，同一指标对比收敛速度和最终性能。"""

    n = 50
    epochs = list(range(1, n + 1))
    models = {
        "ResNet-18": (2.8, 0.28, 12),
        "ResNet-50": (2.6, 0.21, 10),
        "ViT-S":     (3.2, 0.19, 20),
        "EfficientNet-B0": (2.5, 0.23, 11),
        "MobileNetV3":     (2.9, 0.35, 14),
    }
    rows = []
    for m, (start, end, speed) in models.items():
        t = np.linspace(0, 1, n)
        loss = start * np.exp(-speed * t) + end + RNG.normal(0, 0.015, n)
        for e, l in zip(epochs, loss):
            rows.append({"model": m, "epoch": e, "val_loss": round(float(np.clip(l, 0.1, None)), 4)})
    _save(pd.DataFrame(rows), "s11_multi_model_training")


# ---------------------------------------------------------------------------
# 场景 12：不同学习率/批大小的训练曲线
# ---------------------------------------------------------------------------

def gen_s12() -> None:
    """4-6 个超参数配置，val_loss 随 epoch 变化。"""

    n = 40
    epochs = list(range(1, n + 1))
    configs = {
        "lr=1e-2 (too large)": {"start": 3.0, "end": 0.8, "noise": 0.15, "speed": 3},
        "lr=1e-3 (optimal)":   {"start": 2.8, "end": 0.22, "noise": 0.02, "speed": 8},
        "lr=1e-4 (slow)":      {"start": 2.8, "end": 0.60, "noise": 0.01, "speed": 4},
        "lr=1e-5 (too slow)":  {"start": 2.8, "end": 1.80, "noise": 0.005, "speed": 2},
        "lr=5e-3 (unstable)":  {"start": 2.9, "end": 0.45, "noise": 0.12, "speed": 5},
    }
    rows = []
    for cfg, params in configs.items():
        t = np.linspace(0, 1, n)
        base = params["start"] * np.exp(-params["speed"] * t) + params["end"]
        noise_arr = RNG.normal(0, params["noise"], n)
        for e, b, ns in zip(epochs, base, noise_arr):
            rows.append({"config": cfg, "epoch": e,
                         "val_loss": round(float(np.clip(b + ns, 0.1, 4.0)), 4)})
    _save(pd.DataFrame(rows), "s12_lr_ablation_curves")


# ---------------------------------------------------------------------------
# 场景 13：双因子网格搜索（热力图场景）
# ---------------------------------------------------------------------------

def gen_s13() -> None:
    """学习率 × 批大小，或正则化 × dropout，5×4 矩阵。"""

    # 变体 A：LR × batch_size
    lrs = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
    batch_sizes = [16, 32, 64, 128]
    opt_lr, opt_bs = 1e-3, 64
    rows = []
    for lr in lrs:
        for bs in batch_sizes:
            lr_dist = abs(np.log10(lr) - np.log10(opt_lr))
            bs_dist = abs(np.log2(bs) - np.log2(opt_bs))
            perf = float(np.clip(92 - 3 * lr_dist - 1.5 * bs_dist + RNG.uniform(-0.5, 0.5), 60, 95))
            rows.append({"learning_rate": lr, "batch_size": bs, "val_accuracy": round(perf, 2)})
    _save(pd.DataFrame(rows), "s13a_grid_search_heatmap")

    # 变体 B：weight_decay × dropout（长表，热力图用）
    wds = [0.0, 0.01, 0.05, 0.1, 0.2]
    dropouts = [0.0, 0.1, 0.2, 0.3]
    opt_wd, opt_do = 0.01, 0.1
    rows = []
    for wd in wds:
        for do in dropouts:
            wd_dist = abs(np.log10(max(wd, 1e-4)) - np.log10(opt_wd))
            do_dist = abs(do - opt_do)
            perf = float(np.clip(88 - 2.5 * wd_dist - 8 * do_dist + RNG.uniform(-0.4, 0.4), 60, 92))
            rows.append({"weight_decay": wd, "dropout": do, "accuracy": round(perf, 2)})
    _save(pd.DataFrame(rows), "s13b_grid_search_heatmap")


# ---------------------------------------------------------------------------
# 场景 14：单因子扫描（折线图）
# ---------------------------------------------------------------------------

def gen_s14() -> None:
    """1 个超参数，4-8 个取值，1-2 个指标。"""

    # 变体 A：dropout 扫描（倒 U 形）
    dropouts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    acc_peak = 91.2
    rows = []
    for d in dropouts:
        acc = float(np.clip(acc_peak - 20 * (d - 0.2)**2 + RNG.uniform(-0.3, 0.3), 70, 95))
        rows.append({"dropout": d, "val_accuracy": round(acc, 2)})
    _save(pd.DataFrame(rows), "s14a_dropout_sweep")

    # 变体 B：hidden_size 扫描（单调递增后平台）
    sizes = [64, 128, 256, 512, 768, 1024]
    rows = []
    for sz in sizes:
        acc = float(np.clip(85 + 5 * (1 - np.exp(-sz / 300)) + RNG.uniform(-0.3, 0.3), 80, 95))
        f1  = float(np.clip(acc - float(RNG.uniform(0.5, 2.0)), 78, 94))
        rows.append({"hidden_size": sz, "accuracy": round(acc, 2), "F1": round(f1, 2)})
    _save(pd.DataFrame(rows), "s14b_hidden_size_sweep")

    # 变体 C：num_layers 扫描
    layers = [2, 4, 6, 8, 12, 16, 24]
    rows = []
    for nl in layers:
        acc = float(np.clip(82 + 6 * (1 - np.exp(-nl / 8)) + RNG.uniform(-0.4, 0.4), 78, 93))
        rows.append({"num_layers": nl, "val_acc": round(acc, 2)})
    _save(pd.DataFrame(rows), "s14c_num_layers_sweep")


# ---------------------------------------------------------------------------
# 场景 15：随机搜索结果（散点图场景）
# ---------------------------------------------------------------------------

def gen_s15() -> None:
    """50-200 次随机配置，每次记录超参数值 + 验证集性能。"""

    n = 150
    lr = np.power(10, RNG.uniform(-5, -1, n))
    dropout = RNG.uniform(0, 0.7, n)
    hidden  = RNG.choice([64, 128, 256, 512, 768], n)
    # 性能与 LR、dropout 相关（有曲线关系）
    lr_score   = -5 * (np.log10(lr) + 3)**2
    do_score   = -15 * (dropout - 0.2)**2
    hid_score  = 4 * np.log2(hidden / 64)
    val_acc = np.clip(82 + lr_score + do_score + hid_score + RNG.normal(0, 1.5, n), 40, 95)
    df = pd.DataFrame({
        "learning_rate": [float(f"{x:.2e}") for x in lr],
        "dropout": [round(float(x), 3) for x in dropout],
        "hidden_size": hidden.tolist(),
        "val_accuracy": [round(float(x), 2) for x in val_acc],
    })
    _save(df, "s15_random_search")


# ---------------------------------------------------------------------------
# 场景 16：推理速度-精度权衡（散点图）
# ---------------------------------------------------------------------------

def gen_s16() -> None:
    """8-15 个模型，params_M 或 FLOPs vs accuracy，散点。"""

    model_data = [
        ("MobileNetV2",     3.4,  72.0),
        ("MobileNetV3-S",   2.5,  67.4),
        ("MobileNetV3-L",   5.5,  75.2),
        ("EfficientNet-B0", 5.3,  77.1),
        ("EfficientNet-B1", 7.8,  79.1),
        ("EfficientNet-B2", 9.2,  80.1),
        ("EfficientNet-B3", 12.0, 81.6),
        ("EfficientNet-B4", 19.3, 82.9),
        ("ResNet-18",       11.7, 69.8),
        ("ResNet-34",       21.8, 73.3),
        ("ResNet-50",       25.6, 76.1),
        ("ResNet-101",      44.5, 77.4),
        ("ViT-S/32",        22.9, 75.9),
        ("ViT-B/32",        88.2, 80.7),
        ("ViT-L/32",        307.0, 83.5),
    ]
    rows = []
    for name, params, acc in model_data:
        rows.append({
            "model": name,
            "params_M": params,
            "top1_acc": round(acc + float(RNG.uniform(-0.3, 0.3)), 2),
        })
    _save(pd.DataFrame(rows), "s16_speed_accuracy_tradeoff")


# ---------------------------------------------------------------------------
# 场景 17：量化/剪枝实验
# ---------------------------------------------------------------------------

def gen_s17() -> None:
    """原始模型 + 4-6 个压缩比例，精度 vs 模型大小。"""

    # 变体 A：剪枝
    sparsity = [0, 0.1, 0.25, 0.5, 0.65, 0.75, 0.9]
    size_mb  = [178, 162, 134, 90, 63, 45, 18]
    rows = []
    for s, sz in zip(sparsity, size_mb):
        # 非线性精度下降
        if s < 0.5:
            drop = s * 3
        else:
            drop = 1.5 + (s - 0.5) * 20
        acc = float(np.clip(76.1 - drop + RNG.uniform(-0.3, 0.3), 30, 78))
        rows.append({
            "sparsity": s,
            "model_size_MB": sz,
            "top1_acc": round(acc, 2),
        })
    _save(pd.DataFrame(rows), "s17a_pruning")

    # 变体 B：量化（int8 / int4 等）
    schemes = ["FP32", "FP16", "INT8", "INT4", "INT3", "INT2"]
    bits    = [32, 16, 8, 4, 3, 2]
    size_ratio = [1.0, 0.5, 0.25, 0.125, 0.094, 0.063]
    acc_drop = [0, 0.1, 0.3, 1.8, 4.5, 12.0]
    rows = []
    for sc, b, sr, ad in zip(schemes, bits, size_ratio, acc_drop):
        acc = float(np.clip(76.1 - ad + RNG.uniform(-0.2, 0.2), 30, 78))
        rows.append({
            "scheme": sc,
            "bits": b,
            "size_ratio": sr,
            "top1_acc": round(acc, 2),
        })
    _save(pd.DataFrame(rows), "s17b_quantization")


# ---------------------------------------------------------------------------
# 场景 18：不同硬件/框架的推理延迟
# ---------------------------------------------------------------------------

def gen_s18() -> None:
    """3-5 种配置，batch_size 从 1 到 128。"""

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    # latency_ms 特性：CPU 线性，GPU 在大 batch 效率高
    configs = {
        "CPU-PyTorch": lambda bs: 85 * bs + float(RNG.uniform(-2, 2)),
        "GPU-PyTorch": lambda bs: 2.1 * bs + 5 + float(RNG.uniform(-0.5, 0.5)),
        "GPU-TensorRT": lambda bs: 1.3 * bs + 3 + float(RNG.uniform(-0.3, 0.3)),
        "GPU-ONNX": lambda bs: 1.6 * bs + 4 + float(RNG.uniform(-0.4, 0.4)),
    }
    rows = []
    for cfg, fn in configs.items():
        for bs in batch_sizes:
            rows.append({"config": cfg, "batch_size": bs,
                         "latency_ms": round(fn(bs), 2)})
    _save(pd.DataFrame(rows), "s18_hardware_latency")


# ---------------------------------------------------------------------------
# 场景 19：药物/干预实验（RCT 风格）
# ---------------------------------------------------------------------------

def gen_s19() -> None:
    """对照组 + 2-4 处理组，2-4 个时间点，response_rate + se + n。"""

    groups = ["Control", "Drug-A 10mg", "Drug-A 20mg", "Drug-B 15mg"]
    timepoints = ["baseline", "week4", "week8", "week12"]
    base_rate = {"Control": 20, "Drug-A 10mg": 20, "Drug-A 20mg": 20, "Drug-B 15mg": 20}
    growth = {
        "Control":      [0,  2,  3,  4],
        "Drug-A 10mg":  [0, 15, 22, 28],
        "Drug-A 20mg":  [0, 22, 35, 42],
        "Drug-B 15mg":  [0, 18, 30, 38],
    }
    n_per_group = {"Control": 120, "Drug-A 10mg": 115, "Drug-A 20mg": 118, "Drug-B 15mg": 112}
    rows = []
    for g in groups:
        n = n_per_group[g]
        for i, tp in enumerate(timepoints):
            rate = float(np.clip(base_rate[g] + growth[g][i] + RNG.uniform(-1.5, 1.5), 5, 85))
            se = float(np.clip(np.sqrt(rate * (100 - rate) / n) + RNG.uniform(-0.2, 0.2), 0.5, 8))
            rows.append({"group": g, "timepoint": tp, "response_rate": round(rate, 1),
                         "se": round(se, 2), "n": n})
    _save(pd.DataFrame(rows), "s19_rct")


# ---------------------------------------------------------------------------
# 场景 20：物理/化学测量
# ---------------------------------------------------------------------------

def gen_s20() -> None:
    """X 轴物理量，Y 轴响应变量，多条件对比。"""

    # 变体 A：温度 vs 反应速率（阿伦尼乌斯）
    temps = np.array([20, 30, 40, 50, 60, 70, 80, 90])
    catalysts = {
        "Catalyst-A": (1.2, 1.08),
        "Catalyst-B": (0.8, 1.06),
        "No catalyst": (0.3, 1.12),
    }
    rows = []
    for cat, (k0, a_fac) in catalysts.items():
        for T in temps:
            rate = k0 * (a_fac ** ((T - 20) / 10))
            rate += float(RNG.normal(0, rate * 0.03))
            rows.append({"catalyst": cat, "temperature_C": int(T),
                         "reaction_rate": round(float(rate), 4)})
    _save(pd.DataFrame(rows), "s20a_physical_temperature")

    # 变体 B：浓度 vs 吸光度（Beer-Lambert，线性）
    concentrations = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
    dyes = {
        "Dye-A": 0.042,
        "Dye-B": 0.031,
        "Dye-C": 0.055,
    }
    rows = []
    for dye, eps in dyes.items():
        for c in concentrations:
            abs_val = eps * c + float(RNG.normal(0, 0.001))
            rows.append({"dye": dye, "concentration_mM": float(c),
                         "absorbance": round(float(np.clip(abs_val, 0, None)), 4)})
    _save(pd.DataFrame(rows), "s20b_physical_absorbance")


# ---------------------------------------------------------------------------
# 场景 21：生物学重复实验
# ---------------------------------------------------------------------------

def gen_s21() -> None:
    """3-5 个处理组，每组 3-6 次生物学重复，均值 + SEM。"""

    treatments = ["Control", "Treatment-A", "Treatment-B", "Treatment-C", "Treatment-D"]
    true_means = [100.0, 142.0, 118.0, 165.0, 95.0]
    n_reps = [6, 5, 6, 4, 5]
    rows = []
    for trt, mu, nr in zip(treatments, true_means, n_reps):
        # 组内方差不齐
        sigma = {"Control": 8, "Treatment-A": 12, "Treatment-B": 9,
                 "Treatment-C": 18, "Treatment-D": 7}[trt]
        vals = RNG.normal(mu, sigma, nr)
        rows.append({
            "treatment": trt,
            "mean": round(float(np.mean(vals)), 2),
            "sem": round(float(np.std(vals, ddof=1) / np.sqrt(nr)), 2),
            "n": nr,
        })
    _save(pd.DataFrame(rows), "s21_bio_replicates")


# ---------------------------------------------------------------------------
# 场景 22：问卷 Likert 量表结果
# ---------------------------------------------------------------------------

def gen_s22() -> None:
    """5-8 个题目，每题 5 个选项的百分比。"""

    questions = [
        "Q1: Ease of use",
        "Q2: Feature richness",
        "Q3: Performance",
        "Q4: Documentation",
        "Q5: Support quality",
        "Q6: Overall satisfaction",
    ]
    # 每行加和 = 100
    distributions = [
        [5, 10, 20, 35, 30],   # 偏正面
        [8, 18, 30, 28, 16],   # 中性偏正
        [12, 22, 26, 28, 12],  # 中性
        [20, 30, 25, 18, 7],   # 偏负面
        [6, 14, 22, 32, 26],   # 偏正面
        [4, 8, 18, 38, 32],    # 强正面
    ]
    rows = []
    for q, dist in zip(questions, distributions):
        dist_arr = np.array(dist, dtype=float)
        noise = RNG.uniform(-1, 1, 5)
        dist_arr = dist_arr + noise
        dist_arr = np.clip(dist_arr, 1, None)
        dist_arr = dist_arr / dist_arr.sum() * 100
        rows.append({
            "question": q,
            "strongly_disagree": round(dist_arr[0], 1),
            "disagree": round(dist_arr[1], 1),
            "neutral": round(dist_arr[2], 1),
            "agree": round(dist_arr[3], 1),
            "strongly_agree": round(dist_arr[4], 1),
        })
    _save(pd.DataFrame(rows), "s22_likert")


# ---------------------------------------------------------------------------
# 场景 23：年度趋势数据
# ---------------------------------------------------------------------------

def gen_s23() -> None:
    """5-10 年，3-5 个类别，数量或比例随时间变化。"""

    years = list(range(2016, 2025))
    categories = {
        "Transformer": [0.5, 1.2, 5.8, 22.0, 45.0, 68.0, 82.0, 89.0, 93.0],
        "CNN":         [65.0, 68.0, 62.0, 55.0, 42.0, 28.0, 18.0, 12.0, 8.0],
        "RNN/LSTM":    [30.0, 28.0, 28.0, 20.0, 12.0, 8.0, 5.0, 3.5, 2.5],
        "Other":       [4.5, 2.8, 4.2, 3.0, 1.0, 4.0, 5.0, 5.5, 6.5],
    }
    rows = []
    for cat, vals in categories.items():
        for yr, val in zip(years, vals):
            noisy = float(np.clip(val + RNG.uniform(-0.8, 0.8), 0, 100))
            rows.append({"year": yr, "category": cat, "proportion_pct": round(noisy, 1)})
    _save(pd.DataFrame(rows), "s23_annual_trend")


# ---------------------------------------------------------------------------
# 场景 24：地区/群体对比
# ---------------------------------------------------------------------------

def gen_s24() -> None:
    """6-12 个地区，2-4 个指标。"""

    regions = ["North America", "Europe", "East Asia", "South Asia",
               "Latin America", "Middle East", "Sub-Saharan Africa", "Southeast Asia"]
    gdp_pc  = [55000, 42000, 38000, 6000, 12000, 22000, 3500, 11000]
    hdi     = [0.926, 0.905, 0.882, 0.644, 0.762, 0.720, 0.541, 0.723]
    internet= [88, 85, 89, 42, 68, 72, 28, 64]
    rows = []
    for r, g, h, i in zip(regions, gdp_pc, hdi, internet):
        rows.append({
            "region": r,
            "gdp_per_capita": round(g * float(RNG.uniform(0.97, 1.03))),
            "hdi": round(h + float(RNG.uniform(-0.008, 0.008)), 3),
            "internet_pct": round(i + float(RNG.uniform(-1.5, 1.5)), 1),
        })
    _save(pd.DataFrame(rows), "s24_regional_comparison")


# ---------------------------------------------------------------------------
# 场景 25：模型×数据集宽表
# ---------------------------------------------------------------------------

def gen_s25() -> None:
    """行是模型，列是数据集名称（宽表）。"""

    models = ["BERT", "RoBERTa", "XLNet", "DeBERTa", "ALBERT", "ELECTRA"]
    datasets = ["SST-2", "CoLA", "RTE", "MRPC", "QNLI", "STS-B"]
    base = {"SST-2": 93, "CoLA": 61, "RTE": 82, "MRPC": 88, "QNLI": 91, "STS-B": 89}
    m_bonus = {"BERT": 0, "RoBERTa": 1.5, "XLNet": 0.8, "DeBERTa": 2.8, "ALBERT": -0.8, "ELECTRA": 1.2}
    rows = []
    for m in models:
        row: dict = {"model": m}
        for d in datasets:
            row[d] = round(float(np.clip(base[d] + m_bonus[m] + RNG.uniform(-0.5, 0.5), 50, 99)), 2)
        rows.append(row)
    _save(pd.DataFrame(rows), "s25_wide_model_dataset")


# ---------------------------------------------------------------------------
# 场景 26：方法×指标宽表
# ---------------------------------------------------------------------------

def gen_s26() -> None:
    """行是方法，列是指标（宽表）。"""

    methods = ["Ours", "Baseline-A", "Baseline-B", "State-of-the-art", "Simplified"]
    metrics_base = {"accuracy": 88, "F1": 86, "BLEU": 32, "ROUGE-L": 55, "EM": 72}
    m_bonus = {"Ours": 2, "Baseline-A": 0, "Baseline-B": -2, "State-of-the-art": 3, "Simplified": -4}
    rows = []
    for m in methods:
        row: dict = {"method": m}
        for metric, base_val in metrics_base.items():
            row[metric] = round(float(np.clip(base_val + m_bonus[m] + RNG.uniform(-0.5, 0.5), 10, 99)), 2)
        rows.append(row)
    _save(pd.DataFrame(rows), "s26_wide_method_metric")


# ---------------------------------------------------------------------------
# 场景 27：时间×类别宽表
# ---------------------------------------------------------------------------

def gen_s27() -> None:
    """行是年份，列是类别（宽表）。"""

    years = list(range(2018, 2025))
    categories = ["NLP", "CV", "RL", "Multimodal", "Other"]
    trends = {
        "NLP": [100, 130, 170, 250, 380, 510, 640],
        "CV":  [200, 220, 235, 260, 290, 310, 330],
        "RL":  [80, 95, 110, 130, 155, 175, 195],
        "Multimodal": [20, 35, 55, 100, 180, 290, 420],
        "Other": [150, 160, 170, 185, 200, 215, 230],
    }
    rows = []
    for i, yr in enumerate(years):
        row: dict = {"year": yr}
        for cat in categories:
            row[cat] = round(trends[cat][i] * float(RNG.uniform(0.97, 1.03)))
        rows.append(row)
    _save(pd.DataFrame(rows), "s27_wide_time_category")


# ---------------------------------------------------------------------------
# 场景 28：多次随机种子实验结果（box 图）
# ---------------------------------------------------------------------------

def gen_s28() -> None:
    """同一方法用 5-10 个不同随机种子，记录最终性能。"""

    methods = ["BERT", "RoBERTa", "DeBERTa", "ELECTRA"]
    true_means = {"BERT": 93.0, "RoBERTa": 94.5, "DeBERTa": 95.2, "ELECTRA": 93.8}
    stds = {"BERT": 0.4, "RoBERTa": 0.3, "DeBERTa": 0.5, "ELECTRA": 0.6}
    seeds = [42, 123, 456, 789, 1024, 2048, 3141, 9999]
    rows = []
    for m in methods:
        for seed in seeds:
            rng = np.random.default_rng(seed + hash(m) % 1000)
            acc = float(np.clip(rng.normal(true_means[m], stds[m]), 90, 98))
            f1  = float(np.clip(acc - rng.uniform(0.3, 1.2), 89, 97))
            rows.append({"method": m, "seed": seed,
                         "accuracy": round(acc, 2), "F1": round(f1, 2)})
    _save(pd.DataFrame(rows), "s28_multi_seed")


# ---------------------------------------------------------------------------
# 场景 29：k 折交叉验证结果
# ---------------------------------------------------------------------------

def gen_s29() -> None:
    """5-10 折，3-5 个方法，记录每折性能。"""

    methods = ["SVM", "Random Forest", "BERT-ft", "XGBoost", "MLP"]
    true_accs = {"SVM": 82.0, "Random Forest": 84.5, "BERT-ft": 91.0, "XGBoost": 85.0, "MLP": 83.5}
    fold_stds = {"SVM": 1.8, "Random Forest": 1.2, "BERT-ft": 0.8, "XGBoost": 1.5, "MLP": 2.0}
    n_folds = 10
    rows = []
    for m in methods:
        for fold in range(1, n_folds + 1):
            acc = float(np.clip(RNG.normal(true_accs[m], fold_stds[m]), 70, 98))
            rows.append({"method": m, "fold": f"fold{fold}", "accuracy": round(acc, 2)})
    _save(pd.DataFrame(rows), "s29_cross_validation")


# ---------------------------------------------------------------------------
# 场景 30：系统延迟/响应时间分布（长尾）
# ---------------------------------------------------------------------------

def gen_s30() -> None:
    """4-6 个系统，每个系统 100-200 次测量，长尾分布。"""

    systems = {
        "SystemA": {"median": 12, "tail_prob": 0.03, "tail_mult": 8},
        "SystemB": {"median": 18, "tail_prob": 0.08, "tail_mult": 12},
        "SystemC": {"median": 8,  "tail_prob": 0.05, "tail_mult": 6},
        "SystemD": {"median": 25, "tail_prob": 0.02, "tail_mult": 5},
    }
    rows = []
    req_id = 1
    for sys_name, params in systems.items():
        n = 150
        latencies = RNG.exponential(params["median"] * 0.6, n) + params["median"] * 0.5
        tail_mask = RNG.random(n) < params["tail_prob"]
        latencies[tail_mask] *= params["tail_mult"]
        batch_sizes = RNG.choice([1, 4, 8, 16, 32], n)
        latencies = latencies + batch_sizes * 0.05
        for i, (lat, bs) in enumerate(zip(latencies, batch_sizes)):
            rows.append({
                "system": sys_name,
                "request_id": req_id,
                "latency_ms": round(float(np.clip(lat, 1, None)), 2),
                "batch_size": int(bs),
            })
            req_id += 1
    _save(pd.DataFrame(rows), "s30_latency_distribution")


# ---------------------------------------------------------------------------
# 场景 31：多数据集上的性能分布（box 图）
# ---------------------------------------------------------------------------

def gen_s31() -> None:
    """1-3 个方法，在 15-30 个数据集上各跑一次。"""

    n_datasets = 25
    methods = ["BERT-base", "RoBERTa-base", "DeBERTa-base"]
    method_mu = {"BERT-base": 82, "RoBERTa-base": 85, "DeBERTa-base": 87}
    method_sigma = {"BERT-base": 8, "RoBERTa-base": 7, "DeBERTa-base": 6}
    rows = []
    for m in methods:
        for i in range(1, n_datasets + 1):
            acc = float(np.clip(RNG.normal(method_mu[m], method_sigma[m]), 50, 99))
            rows.append({"method": m, "dataset": f"dataset_{i:02d}", "accuracy": round(acc, 2)})
    _save(pd.DataFrame(rows), "s31_multi_dataset_distribution")


# ---------------------------------------------------------------------------
# 场景 32：模型×数据集性能矩阵（heatmap 设计）
# ---------------------------------------------------------------------------

def gen_s32() -> None:
    """行是方法，列是数据集，值是性能分数。"""

    # 变体 A：6×6 矩阵（长表）
    methods = ["BERT", "RoBERTa", "XLNet", "DeBERTa", "ALBERT", "ELECTRA"]
    datasets = ["SST-2", "CoLA", "RTE", "MRPC", "QNLI", "STS-B"]
    base = {"SST-2": 93, "CoLA": 61, "RTE": 82, "MRPC": 88, "QNLI": 91, "STS-B": 89}
    m_bonus = {"BERT": 0, "RoBERTa": 1.5, "XLNet": 0.8, "DeBERTa": 2.8, "ALBERT": -0.8, "ELECTRA": 1.2}
    rows = []
    for m in methods:
        for d in datasets:
            score = float(np.clip(base[d] + m_bonus[m] + RNG.uniform(-0.8, 0.8), 50, 99))
            rows.append({"model": m, "dataset": d, "score": round(score, 2)})
    _save(pd.DataFrame(rows), "s32a_perf_matrix_long")

    # 变体 B：8×5 矩阵（长表，CV 任务）
    detectors = ["Faster-RCNN", "RetinaNet", "FCOS", "DETR", "DINO", "Co-DETR", "YOLOv8", "RT-DETR"]
    benchmarks = ["COCO-val", "COCO-test", "VOC-07", "OpenImages", "LVIS"]
    base_det = {"COCO-val": 45, "COCO-test": 44, "VOC-07": 82, "OpenImages": 55, "LVIS": 38}
    d_bonus = {"Faster-RCNN": -2, "RetinaNet": -3, "FCOS": -2, "DETR": -4, "DINO": 3,
               "Co-DETR": 5, "YOLOv8": 1, "RT-DETR": 4}
    rows = []
    for det in detectors:
        for bm in benchmarks:
            score = float(np.clip(base_det[bm] + d_bonus[det] + RNG.uniform(-0.5, 0.5), 20, 99))
            rows.append({"detector": det, "benchmark": bm, "mAP": round(score, 2)})
    _save(pd.DataFrame(rows), "s32b_detection_matrix_long")


# ---------------------------------------------------------------------------
# 场景 33：相关性矩阵
# ---------------------------------------------------------------------------

def gen_s33() -> None:
    """5-8 个指标两两相关系数，值域 -1 到 1。"""

    metrics = ["accuracy", "F1", "BLEU", "ROUGE-L", "METEOR", "CIDEr", "SPICE"]
    n = len(metrics)
    # 构造一个真实的相关矩阵（正定）
    A = RNG.uniform(0.2, 0.8, (n, n))
    A = (A + A.T) / 2
    np.fill_diagonal(A, 1.0)
    # 保证正定
    eigvals = np.linalg.eigvalsh(A)
    if eigvals.min() < 0:
        A += np.eye(n) * (-eigvals.min() + 0.01)
    A = A / np.sqrt(np.outer(np.diag(A), np.diag(A)))
    np.clip(A, -1, 1, out=A)
    rows = []
    for i, m1 in enumerate(metrics):
        for j, m2 in enumerate(metrics):
            rows.append({"metric_x": m1, "metric_y": m2, "correlation": round(float(A[i, j]), 3)})
    _save(pd.DataFrame(rows), "s33_correlation_matrix_long")

    # 宽表版本
    df_wide = pd.DataFrame(A, index=metrics, columns=metrics).round(3)
    df_wide.insert(0, "metric", metrics)
    _save(df_wide, "s33_correlation_matrix_wide")


# ---------------------------------------------------------------------------
# 场景 34：Scaling Law（散点图，含 log 轴）
# ---------------------------------------------------------------------------

def gen_s34() -> None:
    """模型参数量（log scale）vs 性能，3-5 条曲线，每条 10-20 个点。"""

    # 参数量（M）：从 10M 到 100B
    param_sizes = np.array([10, 30, 70, 125, 350, 750, 1300, 3000, 7000, 13000,
                             30000, 70000, 130000, 350000, 700000]) * 1e6

    tasks = {
        "HellaSwag": {"slope": 5.0, "intercept": 45.0},
        "WinoGrande": {"slope": 4.0, "intercept": 50.0},
        "ARC-Challenge": {"slope": 4.5, "intercept": 30.0},
        "MMLU": {"slope": 6.0, "intercept": 25.0},
        "HumanEval": {"slope": 7.0, "intercept": 5.0},
    }
    rows = []
    for task, params in tasks.items():
        for p in param_sizes:
            log_p = np.log10(p / 1e9)  # 相对于 1B 参数的对数
            perf = float(np.clip(
                params["intercept"] + params["slope"] * (log_p + 9 * np.log10(10)) / np.log10(10) * 0.5
                + RNG.normal(0, 1.5),
                10, 99
            ))
            rows.append({
                "task": task,
                "params_B": round(p / 1e9, 4),
                "performance": round(perf, 2),
            })
    _save(pd.DataFrame(rows), "s34_scaling_law")


# ---------------------------------------------------------------------------
# 列名多样化：对生成的 CSV 随机替换语义等价的列名
# ---------------------------------------------------------------------------

# 可替换的列名别名表（语义等价，真实数据中都会出现）
COL_ALIASES: dict[str, list[str]] = {
    # 模型/方法身份
    "model":        ["method", "system", "approach", "algorithm", "baseline"],
    "method":       ["model", "approach", "system", "algorithm"],
    "system":       ["model", "method", "approach"],
    "approach":     ["model", "method", "system", "algorithm"],
    "algorithm":    ["model", "method", "approach"],
    # 数据集/基准
    "dataset":      ["benchmark", "task", "corpus", "evaluation_set"],
    "benchmark":    ["dataset", "task"],
    "task":         ["dataset", "benchmark"],
    # 性能指标
    "accuracy":     ["acc", "score", "performance", "准确率", "Accuracy"],
    "score":        ["accuracy", "acc", "performance", "result"],
    "F1":           ["f1", "f1_score", "macro_f1", "F1_score"],
    "BLEU":         ["bleu", "bleu_score", "BLEU_score"],
    "mIoU":         ["mean_iou", "miou", "IoU"],
    # 误差/标准差
    "std":          ["se", "sd", "stderr", "std_dev", "error_bar"],
    "se":           ["std", "sd", "sem", "stderr"],
    # 训练过程指标
    "train_loss":   ["training_loss", "loss_train", "tr_loss"],
    "val_loss":     ["validation_loss", "dev_loss", "valid_loss", "eval_loss"],
    "val_acc":      ["validation_acc", "dev_acc", "valid_acc", "eval_acc"],
    "val_accuracy": ["validation_accuracy", "dev_accuracy", "eval_accuracy"],
    # 延迟/速度
    "latency_ms":   ["latency", "time_ms", "delay_ms", "response_time"],
    "speed_ms":     ["latency_ms", "inference_time", "time_ms"],
    # 语言
    "language":     ["lang", "locale"],
    # 分组/处理
    "group":        ["condition", "arm", "treatment_group"],
    "treatment":    ["group", "condition", "arm"],
    "variant":      ["ablation", "setting", "configuration"],
    "setting":      ["configuration", "variant", "setup"],
    # 类别
    "category":     ["class", "type", "label"],
    "strategy":     ["method", "approach", "technique"],
    # 计算成本
    "params_M":     ["num_params_M", "parameters_M", "model_size_M"],
}

# 不参与随机替换的列名（有严格语义或是数值轴）
_KEEP_FIXED: frozenset[str] = frozenset({
    "epoch", "step", "k_shot", "sparsity", "bits", "fold", "seed",
    "year", "region", "temperature_C", "concentration_mM",
    "request_id", "batch_size", "learning_rate", "dropout",
    "hidden_size", "num_layers", "weight_decay", "top1_acc", "top5_acc",
    "mAP", "mAP_50", "mAP_75", "mAP_50_95", "params_B",
    "FLOPs_G", "img_per_sec", "model_size_MB", "size_ratio",
    "n", "mean", "sem", "timepoint", "gdp_per_capita", "hdi", "internet_pct",
    "metric", "metric_x", "metric_y", "correlation",
    "strongly_disagree", "disagree", "neutral", "agree", "strongly_agree",
    "proportion_pct", "performance", "reaction_rate", "absorbance",
    "response_rate",
})


def _apply_column_aliases() -> None:
    """对 data/train/ 下的所有 CSV 随机重命名部分列名，增加训练数据的列名多样性。"""
    renamed_count = 0
    for csv_path in sorted(OUT_DIR.glob("*.csv")):
        df = pd.read_csv(csv_path)
        existing_cols = set(df.columns)
        rename_map: dict[str, str] = {}
        for col in list(df.columns):
            if col in _KEEP_FIXED or col not in COL_ALIASES:
                continue
            candidates = [col] * 3 + COL_ALIASES[col]  # 原名权重 3:1，保持多数 CSV 用原名
            new_name = str(RNG.choice(candidates))
            if new_name != col and new_name not in existing_cols:
                rename_map[col] = new_name
                existing_cols.add(new_name)
                existing_cols.discard(col)
        if rename_map:
            df = df.rename(columns=rename_map)
            df.to_csv(csv_path, index=False)
            renames_str = ", ".join(f"{k}→{v}" for k, v in rename_map.items())
            print(f"    列名替换 [{csv_path.name}]: {renames_str}")
            renamed_count += 1
    print(f"  列名多样化完成，共修改 {renamed_count} 个文件。")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

GENERATORS = [
    ("场景01", gen_s01),
    ("场景02", gen_s02),
    ("场景03", gen_s03),
    ("场景04", gen_s04),
    ("场景05", gen_s05),
    ("场景06", gen_s06),
    ("场景07", gen_s07),
    ("场景08", gen_s08),
    ("场景09", gen_s09),
    ("场景10", gen_s10),
    ("场景11", gen_s11),
    ("场景12", gen_s12),
    ("场景13", gen_s13),
    ("场景14", gen_s14),
    ("场景15", gen_s15),
    ("场景16", gen_s16),
    ("场景17", gen_s17),
    ("场景18", gen_s18),
    ("场景19", gen_s19),
    ("场景20", gen_s20),
    ("场景21", gen_s21),
    ("场景22", gen_s22),
    ("场景23", gen_s23),
    ("场景24", gen_s24),
    ("场景25", gen_s25),
    ("场景26", gen_s26),
    ("场景27", gen_s27),
    ("场景28", gen_s28),
    ("场景29", gen_s29),
    ("场景30", gen_s30),
    ("场景31", gen_s31),
    ("场景32", gen_s32),
    ("场景33", gen_s33),
    ("场景34", gen_s34),
]


if __name__ == "__main__":
    print(f"输出目录：{OUT_DIR.resolve()}\n")
    for name, fn in GENERATORS:
        print(f"[{name}]")
        fn()
    total = len(list(OUT_DIR.glob("*.csv")))
    print(f"\n共生成 {total} 个 CSV 文件。")
    print("\n[列名多样化]")
    _apply_column_aliases()
    print(f"\n完成！")
