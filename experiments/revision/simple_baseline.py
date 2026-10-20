"""Is a deep network needed? A simple baseline on the same 25 km browse crops.

Each crop (the exact pixels the CNN sees, before resizing) is described by 24 hand-made
intensity and texture statistics: mean, standard deviation, percentiles, the share of dark
pixels relative to the crop median (the physical signature of oil damping), the size of the
largest dark connected region, the share of bright point targets (ships), gradient and local
contrast, and entropy. Two classifiers are trained on them: logistic regression and gradient
boosting (HistGradientBoosting). Protocols are identical to the browse-trained heads:
  Mediterranean: 5-fold CV grouped by acquisition day, same folds; threshold at 5% false
                 alarms on 20% held-out training days.
  Gulf:          train on 2022, test on 2021, validation scenes from 2022 (5 seeds).
Output: simple_baseline.json, simple_baseline_features.csv
"""
import json, math, os
import cv2, numpy as np, pandas as pd
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__)); EXP = os.path.dirname(HERE)
MED, GULF = os.path.join(EXP, "browse_eval"), os.path.join(EXP, "gulf_eval")
CROP_KM = 25.0


def km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(a))


def bbox(geom):
    pts = np.array(geom["coordinates"][0], dtype=float)
    return pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()


def crop_at(g, geom, lat, lon):
    h, w = g.shape
    lon0, lon1, lat0, lat1 = bbox(geom)
    half = int(round(CROP_KM / (km(lat0, lon0, lat0, lon1) / w) / 2))
    x = (lon - lon0) / (lon1 - lon0) * w; y = (lat1 - lat) / (lat1 - lat0) * h
    x0, y0 = int(round(x)) - half, int(round(y)) - half
    if x0 < 0 or y0 < 0 or x0 + 2 * half > w or y0 + 2 * half > h:
        return None
    c = g[y0:y0 + 2 * half, x0:x0 + 2 * half]
    return None if np.mean(c < 3) > 0.2 else c


NAMES = ["mean", "std", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "iqr_rel",
         "dark50", "dark70", "dark85", "largest_dark70", "n_dark70_regions", "bright150", "bright200",
         "grad_mean", "grad_std", "local_std_mean", "local_std_std", "entropy", "cv", "skew"]


def stats(c):
    v = c[c >= 3].astype(np.float32)          # ignore no-data pixels
    m = float(np.median(v)) + 1e-3
    p = np.percentile(v, [5, 10, 25, 50, 75, 90, 95])
    f = c.astype(np.float32)
    blur = cv2.GaussianBlur(f, (3, 3), 0)
    dark = (blur < 0.7 * m) & (c >= 3)
    lab, n = ndimage.label(dark)
    largest = (np.bincount(lab.ravel())[1:].max() / dark.size) if n else 0.0
    gx, gy = cv2.Sobel(f, cv2.CV_32F, 1, 0), cv2.Sobel(f, cv2.CV_32F, 0, 1)
    gm = np.hypot(gx, gy)
    mu = cv2.blur(f, (5, 5)); ls = np.sqrt(np.maximum(cv2.blur(f * f, (5, 5)) - mu * mu, 0))
    hist = np.bincount(c.ravel(), minlength=256) / c.size; hist = hist[hist > 0]
    sd = float(v.std()) + 1e-3
    return [float(v.mean()), sd, *p, (p[4] - p[2]) / m,
            float(np.mean(v < 0.5 * m)), float(np.mean(v < 0.7 * m)), float(np.mean(v < 0.85 * m)),
            float(largest), float(n), float(np.mean(v > 1.5 * m)), float(np.mean(v > 2.0 * m)),
            float(gm.mean()), float(gm.std()), float(ls.mean()), float(ls.std()),
            float(-(hist * np.log2(hist)).sum()), sd / m, float(((v - v.mean()) ** 3).mean() / sd ** 3)]


# ---- units: exactly the crops used for the CNN heads
oc = pd.read_csv(os.path.join(HERE, "offcentre_units.csv"))
oc = oc[oc.shift_km == 0]
med_c = pd.read_csv(os.path.join(MED, "crops_25km.csv"))
med_c = med_c.merge(oc[oc.region == "med"][["uid", "clat", "clon"]], left_on="jpg", right_on="uid")
med_c["kind"] = np.where(med_c.label == 1, "oil", "lookalike")
med_b = pd.read_csv(os.path.join(MED, "background_25km.csv")).assign(label=0, kind="background")
med = pd.concat([med_c[["product", "day", "label", "kind", "clat", "clon"]],
                 med_b[["product", "day", "label", "kind", "clat", "clon"]]], ignore_index=True)
assert len(med) == 3114 + 1627, len(med)
gul = pd.read_csv(os.path.join(GULF, "gulf_scores.csv")).rename(columns=dict(lat="clat", lon="clon"))
gul["year"] = gul["product"].str[17:21].astype(int)


def featurise(df, browse_dir, fp):
    rows, cur, img = [], None, None
    for r in df.itertuples():
        if r.product != cur:
            cur = r.product; img = cv2.imread(os.path.join(browse_dir, r.product + ".jpg"), cv2.IMREAD_GRAYSCALE)
        c = crop_at(img, fp[r.product]["geometry"], r.clat, r.clon)
        rows.append(stats(c) if c is not None else [np.nan] * len(NAMES))
    return np.array(rows, dtype=np.float64)


med = med.sort_values("product", kind="stable")
Xm = featurise(med, os.path.join(MED, "browse"), json.load(open(os.path.join(MED, "footprints.json"))))
gul = gul.sort_values("product", kind="stable")
Xg = featurise(gul, os.path.join(GULF, "data", "browse"), json.load(open(os.path.join(GULF, "data", "footprints.json"))))
print("NaN crops", int(np.isnan(Xm).any(1).sum()), int(np.isnan(Xg).any(1).sum()), flush=True)
pd.concat([med.assign(region="med"), gul.assign(region="gulf")], ignore_index=True).join(
    pd.DataFrame(np.vstack([Xm, Xg]), columns=NAMES)).to_csv(os.path.join(HERE, "simple_baseline_features.csv"), index=False)

MODELS = {"logistic": lambda s: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
          "boosting": lambda s: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced",
                                                               random_state=s)}


def th_at(y, s, far=0.05):
    fpr, tpr, th = roc_curve(y, s)
    return float(th[np.where(fpr <= far)[0][-1]])


def boot(y, s, g, n=2000):
    rng = np.random.default_rng(0); df = pd.DataFrame(dict(y=y, s=s, g=g)); by = {k: v for k, v in df.groupby("g")}
    keys = list(by); a = []
    for _ in range(n):
        x = pd.concat([by[k] for k in rng.choice(keys, len(keys))])
        if x.y.nunique() == 2:
            a.append(roc_auc_score(x.y, x.s))
    return [round(float(v), 3) for v in np.percentile(a, [2.5, 97.5])]


out = {}
# ---- Mediterranean: same folds as train_compare_v2 (GroupKFold over the meta order there)
crops = pd.read_csv(os.path.join(MED, "crops_25km.csv")); bgm = pd.read_csv(os.path.join(MED, "background_25km.csv"))
days_meta = np.concatenate([crops.day.values, bgm.day.values]); y_meta = np.concatenate([crops.label.values, np.zeros(len(bgm))])
fold_of_day = {}
for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(days_meta, y_meta, days_meta)):
    for d in set(days_meta[te]):
        fold_of_day[d] = k
ym, gm_ = med.label.values, med.day.values
ok = ~np.isnan(Xm).any(1)
for name, mk in MODELS.items():
    oof = np.full(len(ym), np.nan); flag = np.zeros(len(ym), bool)
    for k in range(5):
        te = np.array([fold_of_day[d] == k for d in gm_]) & ok
        tr = ~te & ok
        days = np.unique(gm_[tr]); rng = np.random.default_rng(42 + k)
        val_days = rng.choice(days, size=max(1, len(days) // 5), replace=False)
        va = tr & np.isin(gm_, val_days); fit = tr & ~np.isin(gm_, val_days)
        m = mk(42 + k).fit(Xm[fit], ym[fit])
        th = th_at(ym[va], m.predict_proba(Xm[va])[:, 1])
        oof[te] = m.predict_proba(Xm[te])[:, 1]; flag[te] = oof[te] > th
    sel = ok
    kinds = med.kind.values
    res = dict(auc_all=round(float(roc_auc_score(ym[sel], oof[sel])), 3), ci=boot(ym[sel], oof[sel], gm_[sel]),
               recall=round(float(flag[sel & (ym == 1)].mean()), 3), false_alarms=round(float(flag[sel & (ym == 0)].mean()), 3))
    for neg in ("lookalike", "background"):
        s2 = sel & ((kinds == "oil") | (kinds == neg))
        res[f"auc_vs_{neg}"] = round(float(roc_auc_score(ym[s2], oof[s2])), 3)
    if name == "boosting":
        # paired comparison with the CNN browse head (same crops, same folds)
        cnn = pd.read_csv(os.path.join(MED, "oof_scores_v2.csv")).browse_score.values[med.index.values]
        rng = np.random.default_rng(1); by = {d: np.where(gm_ == d)[0] for d in np.unique(gm_)}; keys = list(by); dif = []
        for _ in range(2000):
            ii = np.concatenate([by[k] for k in rng.choice(keys, len(keys))])
            if len(set(ym[ii])) == 2:
                dif.append(roc_auc_score(ym[ii], cnn[ii]) - roc_auc_score(ym[ii], oof[ii]))
        res["cnn_auc_same_crops"] = round(float(roc_auc_score(ym, cnn)), 3)
        res["cnn_minus_baseline_ci"] = [round(float(v), 3) for v in np.percentile(dif, [2.5, 50, 97.5])]
    out[f"med_{name}"] = res
    print("med", name, res, flush=True)

# ---- Gulf: train 2022, test 2021 (multi_region protocol)
okg = ~np.isnan(Xg).any(1)
tr_all = (gul.year == 2022).values & okg; te = (gul.year == 2021).values & okg
yg, grp = gul.label.values, gul["product"].values
for name, mk in MODELS.items():
    aucs, recs, fars, scores = [], [], [], []
    for seed in range(5):
        rng = np.random.default_rng(100 + seed); ug = np.unique(grp[tr_all])
        val = tr_all & np.isin(grp, rng.choice(ug, max(1, len(ug) // 5), replace=False)); fit = tr_all & ~val
        m = mk(100 + seed).fit(Xg[fit], yg[fit])
        th = th_at(yg[val], m.predict_proba(Xg[val])[:, 1])
        s = m.predict_proba(Xg[te])[:, 1]; scores.append(s)
        aucs.append(roc_auc_score(yg[te], s)); recs.append((s[yg[te] == 1] > th).mean()); fars.append((s[yg[te] == 0] > th).mean())
    res = dict(auc_all=round(float(np.mean(aucs)), 3), auc_sd=round(float(np.std(aucs)), 3),
               ci=boot(yg[te], np.mean(scores, axis=0), grp[te]),
               recall=round(float(np.mean(recs)), 3), false_alarms=round(float(np.mean(fars)), 3))
    out[f"gulf2021_{name}"] = res
    print("gulf", name, res, flush=True)

# feature importance proxy: single-feature AUCs on the Mediterranean set
single = {n: round(float(max(roc_auc_score(ym[ok], Xm[ok, i]), 1 - roc_auc_score(ym[ok], Xm[ok, i]))), 3)
          for i, n in enumerate(NAMES)}
out["med_single_feature_auc_top5"] = dict(sorted(single.items(), key=lambda kv: -kv[1])[:5])
out["n"] = dict(med=int(ok.sum()), gulf_test=int(te.sum()), gulf_train=int(tr_all.sum()))
json.dump(out, open(os.path.join(HERE, "simple_baseline.json"), "w"), indent=1)
print(json.dumps(out["med_single_feature_auc_top5"]))
