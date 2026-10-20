"""Browse-domain head trained with BOTH kinds of negatives, compared with the current head.

v1 (train_compare.py) used only ESSD look-alikes as negatives and learned partly
"look-alike vs ordinary sea" (background false alarms 24.5%). Here the negatives
are look-alikes + ordinary-sea background crops (extract_background.py).

Same protocol as v1: identical frozen backbone and crops for both heads; 5-fold CV
grouped by acquisition day; within each training fold 20% of days are held out
for early stopping and to set the threshold at 5% false alarms (on that fold's
mixed negatives); each test fold is used once.

Outputs: oof_scores_v2.csv, comparison_v2.json
"""
import json, os
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
SEED, FOLDS, TARGET_FAR = 42, 5, 0.05

crops = pd.read_csv(os.path.join(HERE, "crops_25km.csv"))
zc = np.load(os.path.join(HERE, "features_25km.npz"))
bg = pd.read_csv(os.path.join(HERE, "background_25km.csv"))
zb = np.load(os.path.join(HERE, "features_background.npz"))

crops["kind"] = np.where(crops.label == 1, "oil", "lookalike")
bg = bg.assign(label=0, kind="background", subset="bg", max_obj_km=0.0)
meta = pd.concat([crops, bg], ignore_index=True)
X = np.concatenate([zc["feats"], zb["feats"]])
# Background crops have no flipped view; flip augmentation is applied to crops only.
Xf = np.concatenate([zc["feats_flip"], zb["feats"]])
y = meta.label.values.astype(np.float32)
groups = meta.day.values


def make_head():
    L = tf.keras.layers
    m = tf.keras.Sequential([tf.keras.Input((X.shape[1],)), L.BatchNormalization(), L.Dropout(0.4),
                             L.Dense(512, activation="relu"), L.BatchNormalization(), L.Dropout(0.5),
                             L.Dense(1, activation="sigmoid")])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
              metrics=[tf.keras.metrics.AUC(name="auc")])
    return m


def threshold_at_far(labels, scores, far=TARGET_FAR):
    fpr, tpr, th = roc_curve(labels, scores)
    return float(th[np.where(fpr <= far)[0][-1]])


oof = np.full(len(y), np.nan)
flag = np.zeros(len(y), bool)
for k, (tr, te) in enumerate(GroupKFold(n_splits=FOLDS).split(X, y, groups)):
    tf.keras.utils.set_random_seed(SEED + k)
    days = np.unique(groups[tr]); rng = np.random.default_rng(SEED + k)
    val_days = rng.choice(days, size=max(1, len(days) // 5), replace=False)
    va = tr[np.isin(groups[tr], val_days)]; fit = tr[~np.isin(groups[tr], val_days)]
    Xtr = np.concatenate([X[fit], Xf[fit]]); ytr = np.concatenate([y[fit], y[fit]])
    n1 = ytr.sum(); n0 = len(ytr) - n1
    head = make_head()
    head.fit(Xtr, ytr, validation_data=(X[va], y[va]), epochs=60, batch_size=64, verbose=0,
             class_weight={0: len(ytr) / (2 * n0), 1: len(ytr) / (2 * n1)},
             callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=8,
                                                         restore_best_weights=True)])
    th = threshold_at_far(y[va], head.predict(X[va], verbose=0).ravel())
    oof[te] = head.predict(X[te], verbose=0).ravel()
    flag[te] = oof[te] > th
    print(f"fold {k}: threshold {th:.3f} | test AUC {roc_auc_score(y[te], oof[te]):.3f}", flush=True)

meta["browse_score"] = oof
meta["browse_flag"] = flag
meta.to_csv(os.path.join(HERE, "oof_scores_v2.csv"), index=False)


def boot(df, col_a, col_b, n=2000, seed=0):
    rng = np.random.default_rng(seed); by = {d: g for d, g in df.groupby("day")}; days = list(by)
    a, b, dif = [], [], []
    for _ in range(n):
        s = pd.concat([by[d] for d in rng.choice(days, len(days))])
        if s.label.nunique() < 2:
            continue
        va, vb = roc_auc_score(s.label, s[col_a]), roc_auc_score(s.label, s[col_b])
        a.append(va); b.append(vb); dif.append(va - vb)
    q = lambda v: [round(float(x), 3) for x in np.percentile(v, [2.5, 97.5])]
    return q(a), q(b), q(dif)


out = {}
oil = meta[meta.kind == "oil"]
print()
for name, negs in [("oil vs look-alikes", ["lookalike"]), ("oil vs ordinary sea", ["background"]),
                   ("oil vs all negatives", ["lookalike", "background"])]:
    df = pd.concat([oil, meta[meta.kind.isin(negs)]])
    ab, ac = roc_auc_score(df.label, df.browse_score), roc_auc_score(df.label, df.current_score)
    ci_b, ci_c, ci_d = boot(df, "browse_score", "current_score")
    out[name] = dict(browse_auc=round(ab, 3), browse_ci=ci_b, current_auc=round(ac, 3), current_ci=ci_c, diff_ci=ci_d)
    print(f"{name:22s} browse AUC {ab:.3f} {ci_b} | current AUC {ac:.3f} {ci_c} | diff CI {ci_d}")

print("\nAt operating thresholds (browse: 5% false alarms on validation days; current: 0.75):")
rates = {}
for kind in ["oil", "lookalike", "background"]:
    g = meta[meta.kind == kind]
    rates[kind] = dict(n=len(g), browse_flag_rate=round(float(g.browse_flag.mean()), 3),
                       current_flag_rate=round(float((g.current_score > 0.75).mean()), 3))
    what = "recall" if kind == "oil" else "false alarms"
    print(f"  {kind:10s} n={len(g):4d}  {what}: browse {rates[kind]['browse_flag_rate']} | current {rates[kind]['current_flag_rate']}")
out["rates"] = rates

print("\nBy slick size (vs all negatives): browse AUC / current AUC / browse recall")
negs = meta[meta.label == 0]; out["by_size"] = {}
for lo, hi, lab in [(0, 1, "<1 km"), (1, 2, "1-2 km"), (2, 5, "2-5 km"), (5, 10, "5-10 km"), (10, 1e9, ">=10 km")]:
    pos = oil[(oil.max_obj_km >= lo) & (oil.max_obj_km < hi)]; both = pd.concat([pos, negs])
    r = dict(n=len(pos), browse_auc=round(roc_auc_score(both.label, both.browse_score), 3),
             current_auc=round(roc_auc_score(both.label, both.current_score), 3),
             browse_recall=round(float(pos.browse_flag.mean()), 3))
    out["by_size"][lab] = r
    print(f"  {lab:8s} n={r['n']:4d}  {r['browse_auc']:.3f} / {r['current_auc']:.3f} / {r['browse_recall']}")

print("\nOpen water vs coastal (oil vs all negatives in that setting):")
for name, oil_sets, neg_sets in [("open water", ["ow"], ["nw", "bg"]), ("coastal", ["oc"], ["nc"])]:
    df = meta[(meta.kind == "oil") & meta.subset.isin(oil_sets) | (meta.label == 0) & meta.subset.isin(neg_sets)]
    out[name] = dict(browse_auc=round(roc_auc_score(df.label, df.browse_score), 3),
                     current_auc=round(roc_auc_score(df.label, df.current_score), 3))
    print(f"  {name:10s} browse {out[name]['browse_auc']} | current {out[name]['current_auc']}")
json.dump(out, open(os.path.join(HERE, "comparison_v2.json"), "w"), indent=2)
