"""Score the off-centre crops with the same heads and protocols as the paper.

Mediterranean: the browse-trained head of Table 12 (train_compare_v2.py protocol:
5-fold CV grouped by day, same seeds; thresholds at 5% false alarms on validation
days). Each shifted crop is scored by the fold head that never saw its day.
Gulf: heads trained on Gulf 2022 (multi_region.py protocol, 5 seeds), tested on 2021.
The CSIRO-trained head (threshold 0.75) is scored on the same features.
Heads are trained on the original, centred crops only; only the test crops move.
Negatives at each shift: shifted look-alike crops plus the unshifted ordinary-sea crops.
To keep shifts comparable, only crops valid at every shift are used (paired set).
Confidence intervals: 2,000 bootstrap resamples of days (Med) or scenes (Gulf).
Output: offcentre_results.json, offcentre_scores.csv
"""
import json, os
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__)); EXP = os.path.dirname(HERE)
MED, GULF = os.path.join(EXP, "browse_eval"), os.path.join(EXP, "gulf_eval")
SHIFTS = [0.0, 2.0, 5.0, 10.0]

U = pd.read_csv(os.path.join(HERE, "offcentre_units.csv"))
FU = np.load(os.path.join(HERE, "offcentre_features.npz"))["feats"].astype(np.float32)
valid_all = U.groupby("uid").shift_km.nunique() == len(SHIFTS)
U["paired"] = U.uid.map(valid_all)

import sys
REPO = os.path.dirname(EXP); sys.path.insert(0, REPO)
w = tf.keras.Sequential  # noqa
L = tf.keras.layers


def make_head():
    m = tf.keras.Sequential([tf.keras.Input((1792,)), L.BatchNormalization(), L.Dropout(0.4),
                             L.Dense(512, activation="relu"), L.BatchNormalization(), L.Dropout(0.5),
                             L.Dense(1, activation="sigmoid")])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
              metrics=[tf.keras.metrics.AUC(name="auc")])
    return m


def th_at(labels, scores, far=0.05):
    fpr, tpr, th = roc_curve(labels, scores)
    return float(th[np.where(fpr <= far)[0][-1]])


# CSIRO head = layers[2:] of the deployed CSIRO model; rebuild from the stored per-crop
# scores is not possible for new crops, so load the model head.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras.src.applications import efficientnet as eff  # noqa: E402
eff.file_utils.get_file = lambda *a, **k: os.path.join(MED, "fake_notop.weights.h5")
os.chdir(REPO)
import config  # noqa: E402
config.MODEL_KIND = "csiro"
import ml_logic  # noqa: E402
csiro_head = tf.keras.Sequential(ml_logic.load_ai_model().layers[2:])
U["csiro"] = csiro_head.predict(FU, verbose=0, batch_size=256).ravel()
U["browse"] = np.nan; U["flag"] = False

# ---------------- Mediterranean (train_compare_v2 protocol)
crops = pd.read_csv(os.path.join(MED, "crops_25km.csv")); zc = np.load(os.path.join(MED, "features_25km.npz"))
bg = pd.read_csv(os.path.join(MED, "background_25km.csv")); zb = np.load(os.path.join(MED, "features_background.npz"))
crops["kind"] = np.where(crops.label == 1, "oil", "lookalike")
bg = bg.assign(label=0, kind="background", subset="bg", max_obj_km=0.0)
meta = pd.concat([crops, bg], ignore_index=True)
X = np.concatenate([zc["feats"], zb["feats"]]); Xf = np.concatenate([zc["feats_flip"], zb["feats"]])
y = meta.label.values.astype(np.float32); groups = meta.day.values
SEED = 42
med_bg = meta[meta.kind == "background"].copy(); med_bg["browse"] = np.nan; med_bg["flag"] = False
check = np.full(len(y), np.nan)
mU = U.region == "med"
for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, groups)):
    tf.keras.utils.set_random_seed(SEED + k)
    days = np.unique(groups[tr]); rng = np.random.default_rng(SEED + k)
    val_days = rng.choice(days, size=max(1, len(days) // 5), replace=False)
    va = tr[np.isin(groups[tr], val_days)]; fit = tr[~np.isin(groups[tr], val_days)]
    Xtr = np.concatenate([X[fit], Xf[fit]]); ytr = np.concatenate([y[fit], y[fit]])
    n1 = ytr.sum(); n0 = len(ytr) - n1
    h = make_head()
    h.fit(Xtr, ytr, validation_data=(X[va], y[va]), epochs=60, batch_size=64, verbose=0,
          class_weight={0: len(ytr) / (2 * n0), 1: len(ytr) / (2 * n1)},
          callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=8, restore_best_weights=True)])
    th = th_at(y[va], h.predict(X[va], verbose=0).ravel())
    check[te] = h.predict(X[te], verbose=0).ravel()
    test_days = set(groups[te])
    sel = mU & U.day.isin(test_days)
    s = h.predict(FU[sel.values], verbose=0, batch_size=256).ravel()
    U.loc[sel, "browse"] = s; U.loc[sel, "flag"] = s > th
    bsel = meta.index[te][(meta.kind.values[te] == "background")]
    med_bg.loc[bsel, "browse"] = check[bsel]; med_bg.loc[bsel, "flag"] = check[bsel] > th
    print("med fold", k, "threshold", round(th, 3), flush=True)
ref = pd.read_csv(os.path.join(MED, "oof_scores_v2.csv")).browse_score.values
med_check = dict(rerun_auc_all=float(roc_auc_score(y, check)), stored_auc_all=float(roc_auc_score(y, ref)),
                 score_corr=float(np.corrcoef(check, ref)[0, 1]))
print("med check", med_check, flush=True)

# ---------------- Gulf (multi_region protocol, gulf2022 head, 5 seeds)
g = pd.read_csv(os.path.join(GULF, "gulf_scores.csv")); Fg = np.load(os.path.join(GULF, "gulf_features.npz"))["feats"]
g["year"] = g["product"].str[17:21].astype(int)
trn = (g.year == 2022).values
Xg, yg, gg = Fg[trn], g.label.values[trn].astype(np.float32), g["product"].values[trn]
g_bg = g[(g.year == 2021) & (g.kind == "background")].copy()
gU = (U.region == "gulf").values
gul_scores, gul_flags, bg_scores, bg_flags = [], [], [], []
for seed in range(5):
    tf.keras.utils.set_random_seed(100 + seed); rng = np.random.default_rng(100 + seed)
    ug = np.unique(gg); val = np.isin(gg, rng.choice(ug, max(1, len(ug) // 5), replace=False))
    n1 = yg[~val].sum(); n0 = (~val).sum() - n1
    h = make_head()
    h.fit(Xg[~val], yg[~val], validation_data=(Xg[val], yg[val]), epochs=60, batch_size=64, verbose=0,
          class_weight={0: len(yg) / (2 * n0), 1: len(yg) / (2 * n1)},
          callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=8, restore_best_weights=True)])
    th = th_at(yg[val], h.predict(Xg[val], verbose=0).ravel())
    s = h.predict(FU[gU], verbose=0, batch_size=256).ravel(); gul_scores.append(s); gul_flags.append(s > th)
    b = h.predict(Fg[g_bg.index.values], verbose=0).ravel(); bg_scores.append(b); bg_flags.append(b > th)
    print("gulf seed", seed, "threshold", round(th, 3), flush=True)
U.loc[gU, "browse"] = np.mean(gul_scores, axis=0)        # for the score file only
U.loc[gU, "flag"] = np.mean(gul_flags, axis=0) >= 0.5
U.to_csv(os.path.join(HERE, "offcentre_scores.csv"), index=False)


def boot_auc(lab, sc, grp, n=2000, seed=0):
    rng = np.random.default_rng(seed); df = pd.DataFrame(dict(l=lab, s=sc, g=grp)); by = {k: v for k, v in df.groupby("g")}
    keys = list(by); out = []
    for _ in range(n):
        x = pd.concat([by[k] for k in rng.choice(keys, len(keys))])
        if x.l.nunique() == 2:
            out.append(roc_auc_score(x.l, x.s))
    return [round(float(v), 3) for v in np.percentile(out, [2.5, 97.5])]


res = {"med_check": med_check}
P = U[U.paired]
for region in ("med", "gulf"):
    R = P[P.region == region]; res[region] = {}
    for sft in SHIFTS:
        S = R[R.shift_km == sft]
        if region == "med":
            B = med_bg
            lab = np.r_[S.label.values, np.zeros(len(B))]; grp = np.r_[S.day.values, B.day.values]
            sb = np.r_[S.browse.values, B.browse.values]; sc = np.r_[S.csiro.values, B.current_score.values]
            fl = np.r_[S.flag.values, B.flag.values]
            auc_b = float(roc_auc_score(lab, sb)); rec = float(fl[lab == 1].mean()); far = float(fl[lab == 0].mean())
            ci = boot_auc(lab, sb, grp)
        else:
            sel = np.where(gU)[0]; idx = {u: i for i, u in enumerate(U.loc[gU, "uid"].values) if True}
            pos = [U.index.get_loc(i) for i in S.index]
            gpos = np.searchsorted(sel, pos)
            lab = np.r_[S.label.values, np.zeros(len(g_bg))]; grp = np.r_[S["product"].values, g_bg["product"].values]
            aucs, recs, fars = [], [], []
            for k in range(5):
                sbk = np.r_[gul_scores[k][gpos], bg_scores[k]]; flk = np.r_[gul_flags[k][gpos], bg_flags[k]]
                aucs.append(roc_auc_score(lab, sbk)); recs.append(flk[lab == 1].mean()); fars.append(flk[lab == 0].mean())
            auc_b, rec, far = float(np.mean(aucs)), float(np.mean(recs)), float(np.mean(fars))
            sb = np.r_[np.mean(gul_scores, axis=0)[gpos], np.mean(bg_scores, axis=0)]
            ci = boot_auc(lab, sb, grp)
            sc = np.r_[S.csiro.values, g_bg.current_score.values]
        res[region][str(sft)] = dict(n_oil=int(S.label.sum()), n_lookalike=int((S.label == 0).sum()),
                                     browse_auc=round(auc_b, 3), browse_auc_ci=ci, recall=round(rec, 3),
                                     false_alarms=round(far, 3), csiro_auc=round(float(roc_auc_score(lab, sc)), 3),
                                     csiro_recall=round(float((sc[lab == 1] > 0.75).mean()), 3))
        print(region, sft, res[region][str(sft)], flush=True)
res["n_units"] = int(U.uid.nunique()); res["n_paired_units"] = int(P.uid.nunique())
res["paired_by_region"] = P.groupby("region").uid.nunique().to_dict()
json.dump(res, open(os.path.join(HERE, "offcentre_results.json"), "w"), indent=1)
