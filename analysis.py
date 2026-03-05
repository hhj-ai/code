"""P0-b 结果分析：AUC-ROC / 公式消融 / 跨任务 / 跨层 / 图表。通过标准 AUC>0.85。"""

import argparse, json, os
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    PLT = True
except ImportError:
    PLT = False


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def auc(recs, key, pos="correct_positive", neg="hallucination"):
    s, lb = [], []
    for r in recs:
        if r.get("behavior") not in (pos, neg) or r.get(key) is None: continue
        s.append(r[key]); lb.append(1 if r["behavior"] == pos else 0)
    if len(set(lb)) < 2 or len(lb) < 10:
        return {"auc": None, "n_pos": lb.count(1), "n_neg": lb.count(0)}
    a = roc_auc_score(lb, s); fpr, tpr, _ = roc_curve(lb, s)
    return {"auc": round(a, 4), "n_pos": lb.count(1), "n_neg": lb.count(0),
            "fpr": fpr.tolist(), "tpr": tpr.tolist()}


def grp_stats(recs, key):
    g = defaultdict(list)
    for r in recs:
        b, v = r.get("behavior"), r.get(key)
        if b and v is not None: g[b].append(v)
    return {b: {"n": len(v), "mean": round(float(np.mean(v)), 6),
                "std": round(float(np.std(v)), 6), "med": round(float(np.median(v)), 6)}
            for b, v in g.items()}


# ─── 绘图 ───

def plot_box(recs, key, path):
    if not PLT: return
    g = defaultdict(list)
    for r in recs:
        b, v = r.get("behavior"), r.get(key)
        if b and v is not None: g[b].append(v)
    order = [x for x in ["correct_positive","hallucination","correct_negative","miss"] if x in g]
    if not order: return
    colors = {"correct_positive":"#2ecc71","hallucination":"#e74c3c",
              "correct_negative":"#3498db","miss":"#f39c12"}
    fig, ax = plt.subplots(figsize=(10,6))
    bp = ax.boxplot([g[o] for o in order], labels=order, patch_artist=True, showmeans=True)
    for p, o in zip(bp["boxes"], order): p.set_facecolor(colors.get(o,"#999")); p.set_alpha(.7)
    ax.set_ylabel(key); ax.set_title(f"Behavior Distribution ({key})")
    ax.grid(axis="y", alpha=.3); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def plot_roc(recs, key, path):
    if not PLT: return
    s, lb = [], []
    for r in recs:
        b, v = r.get("behavior"), r.get(key)
        if v is None or b not in ("correct_positive","hallucination"): continue
        s.append(v); lb.append(1 if b=="correct_positive" else 0)
    if len(set(lb)) < 2: return
    fpr, tpr, _ = roc_curve(lb, s); a = roc_auc_score(lb, s)
    fig, ax = plt.subplots(figsize=(8,8))
    ax.plot(fpr, tpr, "b-", lw=2, label=f"AUC={a:.4f}")
    ax.plot([0,1],[0,1],"k--",alpha=.3); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"ROC ({key})"); ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def plot_layers(la, path):
    if not PLT: return
    names = sorted(la, key=lambda x: int(x.split("_")[-1]) if "layer" in x else 999)
    vals = [la[n].get("auc") or 0 for n in names]
    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar(range(len(names)), vals, color="#3498db", alpha=.8)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45)
    ax.set_ylabel("AUC"); ax.axhline(.85, color="red", ls="--", alpha=.5, label="0.85")
    ax.legend(); ax.grid(axis="y", alpha=.3); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def plot_lam(ab, path):
    if not PLT: return
    items = sorted([(k,v) for k,v in ab.items() if k.startswith("ced_lambda_")])
    if not items: return
    lams = [float(k.split("_")[-1]) for k,_ in items]
    aucs = [v.get("auc") or 0 for _,v in items]
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(lams, aucs, "bo-", lw=2, ms=8); ax.set_xlabel("λ_e"); ax.set_ylabel("AUC")
    ax.axhline(.85, color="red", ls="--", alpha=.5, label="0.85"); ax.legend()
    ax.grid(alpha=.3); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


# ─── 主分析 ───

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_file", required=True)
    ap.add_argument("--output_dir", default="results")
    args = ap.parse_args()

    recs = load(args.raw_file)
    print(f"加载 {len(recs)} 条"); assert recs

    fdir = f"{args.output_dir}/figures"; os.makedirs(fdir, exist_ok=True)

    # 1. 行为统计
    print("\n--- 行为分组 (logits_js) ---")
    bs = grp_stats(recs, "logits_js")
    for b, s in bs.items():
        print(f"  {b:20s}  n={s['n']:5d}  mean={s['mean']:.6f}  med={s['med']:.6f}")

    # 2. 主 AUC
    print("\n--- 主 AUC ---")
    ma = auc(recs, "logits_js")
    print(f"  logits_js AUC={ma.get('auc')}  pos={ma.get('n_pos')} neg={ma.get('n_neg')}")
    passed = (ma.get("auc") or 0) >= 0.85
    print(f"  {'✓ 通过' if passed else '✗ 未通过'} (阈值 0.85)")

    # 3. 公式消融
    print("\n--- 公式消融 ---")
    lams = sorted({float(k.replace("ced_lambda_","")) for r in recs for k in r if k.startswith("ced_lambda_")})
    ab = {"js_only": auc(recs, "logits_js")}
    for l in lams: ab[f"ced_lambda_{l}"] = auc(recs, f"ced_lambda_{l:.2f}")
    ab["kl_only"] = auc(recs, "logits_kl")
    ab["cosine_only"] = auc(recs, "logits_cosine_dist")
    ab["entropy_only"] = auc(recs, "entropy_penalty_only")
    bm, ba = None, 0
    for n, v in ab.items():
        a = v.get("auc"); print(f"  {n:<30s}  {a or 'N/A'}")
        if a and a > ba: ba, bm = a, n
    if bm: print(f"  最优: {bm} ({ba:.4f})")

    # 4. 跨任务
    print("\n--- 跨任务 ---")
    tg = defaultdict(list)
    for r in recs: tg[r.get("task_type","?")].append(r)
    ct = {}
    for t, tr in tg.items():
        a = auc(tr, "logits_js"); ct[t] = {"auc": a.get("auc"), "n": len(tr)}
        print(f"  {t:<15s}  AUC={a.get('auc') or 'N/A'}  n={len(tr)}")

    # 5. 跨层
    print("\n--- 跨层 ---")
    lids = sorted({int(k.replace("layer_","").replace("_js",""))
                    for r in recs for k in r if k.startswith("layer_") and k.endswith("_js")})
    cl = {"logits": auc(recs, "logits_js")}
    for i in lids: cl[f"layer_{i}"] = auc(recs, f"layer_{i}_js")
    bl, bla = None, 0
    for n, v in sorted(cl.items()):
        a = v.get("auc"); print(f"  {n:<15s}  AUC={a or 'N/A'}")
        if a and a > bla: bla, bl = a, n
    if bl: print(f"  最优层: {bl} ({bla:.4f})")

    # 6. 绘图
    print("\n--- 生成图表 ---")
    plot_box(recs, "logits_js", f"{fdir}/fig1_behavior.png")
    plot_roc(recs, "logits_js", f"{fdir}/fig2_roc.png")
    plot_layers(cl, f"{fdir}/fig3_layers.png")
    plot_lam(ab, f"{fdir}/fig4_lambda.png")

    # 7. 报告
    rpt = {"passed": passed, "main_auc": ma.get("auc"), "best_metric": bm, "best_auc": ba,
           "best_layer": bl, "best_layer_auc": bla, "behavior_stats": bs,
           "ablation": {k: {"auc": v.get("auc")} for k,v in ab.items()},
           "cross_task": ct, "cross_layer": {k: {"auc": v.get("auc")} for k,v in cl.items()}}
    rpt_path = f"{args.output_dir}/p0b_analysis_report.json"
    json.dump(rpt, open(rpt_path, "w"), indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  主AUC={ma.get('auc')}  最优={bm}({ba:.4f})  最优层={bl}({bla:.4f})")
    print(f"  {'✓ 通过' if passed else '✗ 未通过'}  报告: {rpt_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
