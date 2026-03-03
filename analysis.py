"""analysis.py — P0-b 结果分析。

产出：
- Table 1: CED 公式消融（JS / CED / KL / cosine / 纯熵）各指标的 AUC-ROC
- Table 2: 跨任务一致性（existence / spatial / attribute / counting）
- Figure 1: 四组行为的 CED 分布（violin/box plot）
- Figure 2: ROC 曲线（correct_positive vs hallucination）
- Figure 3: 跨层 JS 散度对比
- Figure 4: λ_e 扫描曲线

通过标准：correct_positive vs hallucination 的 AUC > 0.85
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

try:
    from sklearn.metrics import roc_auc_score, roc_curve
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


def load_raw_results(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_auc(
    records: List[Dict],
    metric_key: str,
    pos_label: str = "correct_positive",
    neg_label: str = "hallucination",
) -> Dict[str, Any]:
    """计算指定 metric 在 pos vs neg 两组间的 AUC-ROC。
    
    正例（pos_label）应该有更高的 metric 值（CED/JS 高 = 真的看了图）。
    """
    scores = []
    labels = []

    for r in records:
        behavior = r.get("behavior")
        val = r.get(metric_key)
        if val is None or behavior not in (pos_label, neg_label):
            continue
        scores.append(val)
        labels.append(1 if behavior == pos_label else 0)

    if len(set(labels)) < 2 or len(labels) < 10:
        return {"auc": None, "n_pos": labels.count(1), "n_neg": labels.count(0),
                "error": "insufficient_data"}

    if not HAS_SKLEARN:
        return {"auc": None, "error": "sklearn not available"}

    auc = roc_auc_score(labels, scores)
    fpr, tpr, thresholds = roc_curve(labels, scores)

    return {
        "auc": round(auc, 4),
        "n_pos": labels.count(1),
        "n_neg": labels.count(0),
        "fpr": fpr.tolist() if len(fpr) <= 200 else fpr[::max(1, len(fpr)//200)].tolist(),
        "tpr": tpr.tolist() if len(tpr) <= 200 else tpr[::max(1, len(tpr)//200)].tolist(),
    }


def group_stats(records: List[Dict], metric_key: str) -> Dict[str, Dict]:
    """按 behavior 分组计算统计量。"""
    groups = defaultdict(list)
    for r in records:
        b = r.get("behavior")
        v = r.get(metric_key)
        if b and v is not None:
            groups[b].append(v)

    stats = {}
    for b, vals in groups.items():
        arr = np.array(vals)
        stats[b] = {
            "count": len(vals),
            "mean": round(float(arr.mean()), 6),
            "std": round(float(arr.std()), 6),
            "median": round(float(np.median(arr)), 6),
            "q25": round(float(np.percentile(arr, 25)), 6),
            "q75": round(float(np.percentile(arr, 75)), 6),
        }
    return stats


def analyze_formula_ablation(records: List[Dict], lambda_values: List[float]) -> Dict:
    """公式消融：比较不同 metric 的 AUC。
    
    对比：裸 JS / CED (各种 λ) / KL / cosine / 纯熵惩罚
    """
    metrics = {}

    # 裸 JS
    metrics["js_only"] = compute_auc(records, "logits_js")

    # CED（各 lambda）
    for lam in lambda_values:
        key = f"ced_lambda_{lam:.2f}"
        metrics[f"ced_lambda_{lam}"] = compute_auc(records, key)

    # KL
    metrics["kl_only"] = compute_auc(records, "logits_kl")

    # Cosine distance
    metrics["cosine_only"] = compute_auc(records, "logits_cosine_dist")

    # 纯熵惩罚
    metrics["entropy_penalty_only"] = compute_auc(records, "entropy_penalty_only")

    return metrics


def analyze_cross_task(records: List[Dict], metric_key: str = "logits_js") -> Dict:
    """跨任务一致性：每类任务的 AUC。"""
    task_groups = defaultdict(list)
    for r in records:
        task_groups[r.get("task_type", "unknown")].append(r)

    results = {}
    for task, task_records in task_groups.items():
        auc_info = compute_auc(task_records, metric_key)
        stats = group_stats(task_records, metric_key)
        results[task] = {
            "auc": auc_info,
            "group_stats": stats,
            "n_samples": len(task_records),
        }
    return results


def analyze_cross_layer(records: List[Dict], layer_indices: List[int]) -> Dict:
    """跨层分析：每层的 JS 散度 AUC。"""
    results = {}

    # Logits 层
    results["logits"] = compute_auc(records, "logits_js")

    # 中间层
    for idx in layer_indices:
        key = f"layer_{idx}_js"
        results[f"layer_{idx}"] = compute_auc(records, key)

    return results


# ─────────────────── 绘图 ───────────────────

def plot_behavior_distribution(records, metric_key, output_path):
    """Figure 1: 四组行为的 CED 分布（box plot）。"""
    if not HAS_PLOT:
        print("[WARN] matplotlib not available, skipping plot")
        return

    groups = defaultdict(list)
    for r in records:
        b = r.get("behavior")
        v = r.get(metric_key)
        if b and v is not None:
            groups[b].append(v)

    if not groups:
        return

    order = ["correct_positive", "hallucination", "correct_negative", "miss"]
    present = [g for g in order if g in groups]
    data = [groups[g] for g in present]
    colors = {
        "correct_positive": "#2ecc71",
        "hallucination": "#e74c3c",
        "correct_negative": "#3498db",
        "miss": "#f39c12",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(data, labels=present, patch_artist=True, showmeans=True)
    for patch, group in zip(bp["boxes"], present):
        patch.set_facecolor(colors.get(group, "#95a5a6"))
        patch.set_alpha(0.7)

    ax.set_ylabel(metric_key)
    ax.set_title(f"CED Distribution by Behavior Group\n({metric_key})")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  [PLOT] {output_path}")


def plot_roc_curve(records, metric_key, output_path):
    """Figure 2: ROC 曲线。"""
    if not HAS_PLOT or not HAS_SKLEARN:
        return

    scores, labels = [], []
    for r in records:
        b = r.get("behavior")
        v = r.get(metric_key)
        if v is None or b not in ("correct_positive", "hallucination"):
            continue
        scores.append(v)
        labels.append(1 if b == "correct_positive" else 0)

    if len(set(labels)) < 2:
        return

    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, "b-", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC: correct_positive vs hallucination\n({metric_key})")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  [PLOT] {output_path}")


def plot_cross_layer(layer_aucs, output_path):
    """Figure 3: 跨层 AUC 对比。"""
    if not HAS_PLOT:
        return

    layers = sorted(layer_aucs.keys(), key=lambda x: int(x.split("_")[-1]) if "layer" in x else 999)
    auc_vals = [layer_aucs[l].get("auc", 0) or 0 for l in layers]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(layers)), auc_vals, color="#3498db", alpha=0.8)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=45)
    ax.set_ylabel("AUC-ROC")
    ax.set_title("CED Discrimination (AUC) Across Layers")
    ax.axhline(y=0.85, color="red", linestyle="--", alpha=0.5, label="Target: 0.85")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  [PLOT] {output_path}")


def plot_lambda_sweep(ablation_results, output_path):
    """Figure 4: λ_e 扫描曲线。"""
    if not HAS_PLOT:
        return

    ced_items = [(k, v) for k, v in ablation_results.items() if k.startswith("ced_lambda_")]
    if not ced_items:
        return

    lambdas = []
    aucs = []
    for k, v in sorted(ced_items):
        lam = float(k.split("_")[-1])
        auc = v.get("auc", 0) or 0
        lambdas.append(lam)
        aucs.append(auc)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lambdas, aucs, "bo-", lw=2, markersize=8)
    ax.set_xlabel("λ_e (entropy penalty weight)")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("CED AUC vs λ_e")
    ax.axhline(y=0.85, color="red", linestyle="--", alpha=0.5, label="Target: 0.85")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  [PLOT] {output_path}")


# ─────────────────── 主函数 ───────────────────

def main():
    parser = argparse.ArgumentParser(description="P0-b Analysis")
    parser.add_argument("--raw_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    print("=" * 60)
    print("  P0-b Analysis")
    print("=" * 60)

    records = load_raw_results(args.raw_file)
    print(f"  Loaded {len(records)} records")

    if not records:
        print("[FATAL] No records to analyze")
        return

    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # ---- 1. 行为分组统计 ----
    print("\n--- Behavior Group Stats (logits_js) ---")
    behavior_stats = group_stats(records, "logits_js")
    for b, s in behavior_stats.items():
        print(f"  {b:20s}  n={s['count']:5d}  mean={s['mean']:.6f}  "
              f"median={s['median']:.6f}  std={s['std']:.6f}")

    # ---- 2. 主 AUC (correct_positive vs hallucination) ----
    print("\n--- Main AUC: correct_positive vs hallucination ---")
    main_auc = compute_auc(records, "logits_js")
    print(f"  AUC (logits_js): {main_auc.get('auc')}")
    print(f"  n_pos={main_auc.get('n_pos')}, n_neg={main_auc.get('n_neg')}")

    passed = (main_auc.get("auc") or 0) >= 0.85
    print(f"  {'✓ PASSED' if passed else '✗ FAILED'} (threshold: 0.85)")

    # ---- 3. 公式消融 ----
    print("\n--- Formula Ablation ---")
    # 检测存在的 lambda 值
    lambda_values = set()
    for r in records:
        for k in r.keys():
            if k.startswith("ced_lambda_"):
                lam = float(k.replace("ced_lambda_", ""))
                lambda_values.add(lam)
    lambda_values = sorted(lambda_values)

    ablation = analyze_formula_ablation(records, lambda_values)
    print(f"  {'Metric':<30s}  {'AUC':>8s}")
    print(f"  {'-'*30}  {'-'*8}")
    best_metric = None
    best_auc = 0
    for name, info in ablation.items():
        auc = info.get("auc")
        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        print(f"  {name:<30s}  {auc_str:>8s}")
        if auc is not None and auc > best_auc:
            best_auc = auc
            best_metric = name
    if best_metric:
        print(f"\n  Best metric: {best_metric} (AUC={best_auc:.4f})")

    # ---- 4. 跨任务一致性 ----
    print("\n--- Cross-Task Consistency ---")
    cross_task = analyze_cross_task(records, "logits_js")
    for task, info in cross_task.items():
        auc = info["auc"].get("auc")
        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        print(f"  {task:<15s}  AUC={auc_str}  n={info['n_samples']}")

    # ---- 5. 跨层分析 ----
    print("\n--- Cross-Layer Analysis ---")
    # 检测存在的层
    layer_indices = set()
    for r in records:
        for k in r.keys():
            if k.startswith("layer_") and k.endswith("_js"):
                idx = int(k.replace("layer_", "").replace("_js", ""))
                layer_indices.add(idx)
    layer_indices = sorted(layer_indices)

    cross_layer = analyze_cross_layer(records, layer_indices)
    best_layer = None
    best_layer_auc = 0
    for name, info in sorted(cross_layer.items()):
        auc = info.get("auc")
        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        print(f"  {name:<15s}  AUC={auc_str}")
        if auc is not None and auc > best_layer_auc:
            best_layer_auc = auc
            best_layer = name
    if best_layer:
        print(f"\n  Best layer: {best_layer} (AUC={best_layer_auc:.4f})")

    # ---- 6. 绘图 ----
    print("\n--- Generating Plots ---")
    plot_behavior_distribution(records, "logits_js", os.path.join(fig_dir, "fig1_behavior_dist.png"))
    plot_roc_curve(records, "logits_js", os.path.join(fig_dir, "fig2_roc_curve.png"))
    plot_cross_layer(cross_layer, os.path.join(fig_dir, "fig3_cross_layer.png"))
    plot_lambda_sweep(ablation, os.path.join(fig_dir, "fig4_lambda_sweep.png"))

    # 如果有最优 CED 指标，也画一下
    if best_metric and best_metric != "js_only":
        best_key = None
        for r in records:
            for k in r.keys():
                if best_metric.replace("ced_lambda_", "ced_lambda_") in k:
                    best_key = k
                    break
            if best_key:
                break
        if best_key:
            plot_behavior_distribution(
                records, best_key,
                os.path.join(fig_dir, f"fig1b_behavior_dist_{best_metric}.png")
            )
            plot_roc_curve(
                records, best_key,
                os.path.join(fig_dir, f"fig2b_roc_{best_metric}.png")
            )

    # ---- 7. 汇总报告 ----
    report = {
        "pass_threshold": 0.85,
        "passed": passed,
        "main_auc_logits_js": main_auc,
        "best_metric": best_metric,
        "best_auc": best_auc,
        "best_layer": best_layer,
        "best_layer_auc": best_layer_auc,
        "behavior_stats": behavior_stats,
        "formula_ablation": {k: {"auc": v.get("auc"), "n_pos": v.get("n_pos"), "n_neg": v.get("n_neg")}
                             for k, v in ablation.items()},
        "cross_task": {k: {"auc": v["auc"].get("auc"), "n": v["n_samples"]}
                       for k, v in cross_task.items()},
        "cross_layer": {k: {"auc": v.get("auc")} for k, v in cross_layer.items()},
    }

    report_path = os.path.join(args.output_dir, "p0b_analysis_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  Analysis Complete.")
    print(f"  Main AUC (logits JS): {main_auc.get('auc')}")
    print(f"  Best metric: {best_metric} (AUC={best_auc:.4f})")
    print(f"  Best layer: {best_layer} (AUC={best_layer_auc:.4f})")
    print(f"  Result: {'✓ PASSED' if passed else '✗ FAILED'}")
    print(f"  Report: {report_path}")
    print(f"  Figures: {fig_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
