"""Paired bootstrap of the AUC change between shifted and centred crops (Mediterranean
browse-trained head; Gulf mean-of-seeds score), resampling days (Med) or scenes (Gulf).
Ordinary-sea negatives are unshifted and identical at every shift. Output: offcentre_diff.json"""
import json, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__))
U = pd.read_csv(os.path.join(HERE, "offcentre_scores.csv")); U = U[U.paired]
bgm = pd.read_csv(os.path.join(HERE, "..", "browse_eval", "oof_scores_v2.csv")); bgm = bgm[bgm.kind == "background"]
out = {}
for region, grp in (("med", "day"), ("gulf", "product")):
    R = U[U.region == region]
    if region == "med":
        B = pd.DataFrame(dict(uid="bg" + bgm.index.astype(str), label=0, g=bgm.day, s=bgm.browse_score))
    else:
        B = None  # Gulf background scores per seed are not stored; use oil vs shifted look-alikes
    wide = {s: R[R.shift_km == s].set_index("uid") for s in (0.0, 2.0, 5.0, 10.0)}
    keys = R[R.shift_km == 0.0][grp].unique(); rng = np.random.default_rng(0)
    base = wide[0.0]
    by = {k: base.index[base[grp] == k] for k in keys}
    bg_by = {k: B.index[B.g == k] for k in keys} if B is not None else None
    res = {}
    for s in (2.0, 5.0, 10.0):
        diffs = []
        for _ in range(2000):
            pick = rng.choice(keys, len(keys))
            ids = np.concatenate([by[k] for k in pick])
            l = base.loc[ids, "label"].values
            a0, a1 = base.loc[ids, "browse"].values, wide[s].loc[ids, "browse"].values
            if B is not None:
                bi = np.concatenate([bg_by[k] for k in pick if k in bg_by])
                l = np.r_[l, np.zeros(len(bi))]; a0 = np.r_[a0, B.loc[bi, "s"].values]; a1 = np.r_[a1, B.loc[bi, "s"].values]
            if len(set(l)) < 2: continue
            diffs.append(roc_auc_score(l, a1) - roc_auc_score(l, a0))
        res[str(s)] = [round(float(x), 3) for x in np.percentile(diffs, [2.5, 50, 97.5])]
    out[region + (" (oil vs look-alikes)" if B is None else " (oil vs all negatives)")] = res
print(json.dumps(out, indent=1)); json.dump(out, open(os.path.join(HERE, "offcentre_diff.json"), "w"), indent=1)
