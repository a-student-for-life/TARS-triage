"""B0 versus B4 on real browse crops (Eastern Mediterranean, ESSD labels).

Same crops, same head, same protocol as browse_eval/train_compare_v2.py
(5-fold CV grouped by acquisition day; threshold at 5% false alarms on held-out
training days), repeated for 3 seeds; only the frozen backbone differs.
Needs features_25km{,_b0}.npz and features_background{,_b0}.npz.

  python browse_b0_vs_b4.py  -> browse_b0_vs_b4.json
"""
import json, os
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
BE = os.path.join(HERE, "..", "browse_eval")
FOLDS, TARGET_FAR, SEEDS = 5, 0.05, [42, 43, 44]


def load(sfx):
    crops = pd.read_csv(os.path.join(BE, f"crops_25km{sfx}.csv"))
    bg = pd.read_csv(os.path.join(BE, f"background_25km{sfx}.csv"))
    zc = np.load(os.path.join(BE, f"features_25km{sfx}.npz"))
    zb = np.load(os.path.join(BE, f"features_background{sfx}.npz"))
    crops["kind"] = np.where(crops.label == 1, "oil", "lookalike")
    bg = bg.assign(label=0, kind="background")
    meta = pd.concat([crops, bg], ignore_index=True)
    return meta, np.concatenate([zc["feats"], zb["feats"]]), np.concatenate([zc["feats_flip"], zb["feats"]])


def make_head(d):
    L = tf.keras.layers
    m = tf.keras.Sequential([tf.keras.Input((d,)), L.BatchNormalization(), L.Dropout(0.4),
                             L.Dense(512, activation="relu"), L.BatchNormalization(), L.Dropout(0.5),
                             L.Dense(1, activation="sigmoid")])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
              metrics=[tf.keras.metrics.AUC(name="auc")])
    return m


def cv(meta, X, Xf, seed):
    y = meta.label.values.astype(np.float32); groups = meta.day.values
    oof = np.full(len(y), np.nan); flag = np.zeros(len(y), bool)
    for k, (tr, te) in enumerate(GroupKFold(n_splits=FOLDS).split(X, y, groups)):
        tf.keras.utils.set_random_seed(seed + k)
        days = np.unique(groups[tr]); rng = np.random.default_rng(seed + k)
        val_days = rng.choice(days, size=max(1, len(days) // 5), replace=False)
        va = tr[np.isin(groups[tr], val_days)]; fit = tr[~np.isin(groups[tr], val_days)]
        Xtr = np.concatenate([X[fit], Xf[fit]]); ytr = np.concatenate([y[fit], y[fit]])
        n1 = ytr.sum(); n0 = len(ytr) - n1
        head = make_head(X.shape[1])
        head.fit(Xtr, ytr, validation_data=(X[va], y[va]), epochs=60, batch_size=64, verbose=0,
                 class_weight={0: len(ytr) / (2 * n0), 1: len(ytr) / (2 * n1)},
                 callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=8,
                                                             restore_best_weights=True)])
        fpr, tpr, th = roc_curve(y[va], head.predict(X[va], verbose=0).ravel())
        t = float(th[np.where(fpr <= TARGET_FAR)[0][-1]])
        oof[te] = head.predict(X[te], verbose=0).ravel(); flag[te] = oof[te] > t
    return oof, flag


out = {}
ref = None
for arch, sfx in [("b4", ""), ("b0", "_b0")]:
    meta, X, Xf = load(sfx)
    if ref is None:
        ref = meta[["jpg", "product", "day"]].fillna("").astype(str).values
    else:
        assert (meta[["jpg", "product", "day"]].fillna("").astype(str).values == ref).all(), "crop sets differ"
    y = meta.label.values; kind = meta.kind.values
    per = []
    for seed in SEEDS:
        s, f = cv(meta, X, Xf, seed)
        neg_la, neg_bg = kind == "lookalike", kind == "background"
        m = dict(auc_all=roc_auc_score(y, s),
                 auc_lookalike=roc_auc_score(y[(y == 1) | neg_la], s[(y == 1) | neg_la]),
                 auc_sea=roc_auc_score(y[(y == 1) | neg_bg], s[(y == 1) | neg_bg]),
                 recall=f[y == 1].mean(), fa_lookalike=f[neg_la].mean(), fa_sea=f[neg_bg].mean())
        tp = int((f & (y == 1)).sum()); fp = int((f & (y == 0)).sum()); fn = int((~f & (y == 1)).sum())
        m.update(precision=tp / max(tp + fp, 1), f1=2 * tp / (2 * tp + fp + fn), fnr=fn / (tp + fn))
        per.append({k: float(v) for k, v in m.items()})
        print(arch, seed, {k: round(v, 3) for k, v in per[-1].items()}, flush=True)
    out[arch] = {k: [float(np.mean([p[k] for p in per])), float(np.std([p[k] for p in per]))] for k in per[0]}
    out[arch]["n"] = dict(oil=int((y == 1).sum()), lookalike=int((kind == "lookalike").sum()),
                          sea=int((kind == "background").sum()))
json.dump(out, open(os.path.join(HERE, "browse_b0_vs_b4.json"), "w"), indent=1)
print("DONE")
