import json, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "gulf_scores.csv"))
BTH = json.load(open(os.path.join(HERE, "..", "..", "models", "browse_head_v1.json")))["threshold"]

def boot(d, n=2000, seed=0):
    rng = np.random.default_rng(seed); by = {p: g for p, g in d.groupby("product")}; ps = list(by)
    out = {"browse": [], "current": [], "diff": []}
    for _ in range(n):
        s = pd.concat([by[p] for p in rng.choice(ps, len(ps))])
        if s.label.nunique() < 2: continue
        b, c = roc_auc_score(s.label, s.browse_score), roc_auc_score(s.label, s.current_score)
        out["browse"].append(b); out["current"].append(c); out["diff"].append(b - c)
    return {k: [round(float(x), 3) for x in np.percentile(v, [2.5, 97.5])] for k, v in out.items()}

summary = {}
def report(d, name):
    if d.label.nunique() < 2: return
    b, c = roc_auc_score(d.label, d.browse_score), roc_auc_score(d.label, d.current_score); ci = boot(d)
    oil, neg = d[d.label == 1], d[d.label == 0]
    r = dict(n_oil=len(oil), n_neg=len(neg), n_scenes=d["product"].nunique(), browse_auc=round(b, 3), browse_ci=ci["browse"],
             current_auc=round(c, 3), current_ci=ci["current"], diff_ci=ci["diff"],
             browse_recall=round(float((oil.browse_score > BTH).mean()), 3), browse_far=round(float((neg.browse_score > BTH).mean()), 3),
             current_recall=round(float((oil.current_score > 0.75).mean()), 3), current_far=round(float((neg.current_score > 0.75).mean()), 3))
    summary[name] = r
    print(f"{name}: oil {r['n_oil']} vs neg {r['n_neg']} ({r['n_scenes']} scenes)\n"
          f"   browse  AUC {r['browse_auc']} {r['browse_ci']}  recall {r['browse_recall']}  false alarms {r['browse_far']}\n"
          f"   current AUC {r['current_auc']} {r['current_ci']}  recall {r['current_recall']}  false alarms {r['current_far']}\n"
          f"   difference CI {r['diff_ci']}")

bg = df[df.kind == "background"]
report(df[df.kind.isin(["oil", "background"])], "oil vs ordinary sea")
if (df.kind == "lookalike").any():
    report(df[df.kind.isin(["oil", "lookalike"])], "oil vs look-alikes")
    report(df, "oil vs all negatives")
print("\nBy slick length (bounding box long side), vs ordinary sea: browse / current AUC, browse recall")
summary["by_length"] = {}
for lo, hi, lab in [(0, 1, "<1 km"), (1, 2, "1-2 km"), (2, 5, "2-5 km"), (5, 10, "5-10 km"), (10, 1e9, ">=10 km")]:
    o = df[(df.kind == "oil") & (df.length_km >= lo) & (df.length_km < hi)]
    if len(o) < 10: continue
    d = pd.concat([o, bg])
    r = dict(n=len(o), browse_auc=round(roc_auc_score(d.label, d.browse_score), 3),
             current_auc=round(roc_auc_score(d.label, d.current_score), 3), browse_recall=round(float((o.browse_score > BTH).mean()), 3))
    summary["by_length"][lab] = r
    print(f"  {lab:8s} n={r['n']:4d}  {r['browse_auc']} / {r['current_auc']}   {r['browse_recall']}")
json.dump(summary, open(os.path.join(HERE, "gulf_summary.json"), "w"), indent=2)
