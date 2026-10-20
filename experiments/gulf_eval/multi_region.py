"""Does training on more than one region fix the regional drop?

Train heads on frozen-backbone features and test on Gulf 2021 (never used for
training; GulfOilSAR's own split uses 2022 for training and 2021 for validation):
  current      : deployed CSIRO head (unchanged)
  med_only     : browse head trained on Eastern Mediterranean 2019 only (= browse_head_v1 recipe)
  gulf2022     : head trained on Gulf 2022 only
  med+gulf2022 : head trained on both
Epochs and thresholds are chosen on a validation subset of TRAINING scenes only.
Each training run is repeated with 5 seeds; mean and spread are reported.
"""
import json, os
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.metrics import roc_auc_score, roc_curve

HERE = os.path.dirname(os.path.abspath(__file__)); MED = os.path.join(HERE, "..", "browse_eval")
g = pd.read_csv(os.path.join(HERE, "gulf_scores.csv")); Fg = np.load(os.path.join(HERE, "gulf_features.npz"))["feats"]
g["year"] = g["product"].str[17:21].astype(int); g["group"] = g["product"]
mc = pd.read_csv(os.path.join(MED, "crops_25km.csv")); mb = pd.read_csv(os.path.join(MED, "background_25km.csv"))
Fm = np.concatenate([np.load(os.path.join(MED, "features_25km.npz"))["feats"], np.load(os.path.join(MED, "features_background.npz"))["feats"]])
ym = np.concatenate([mc.label.values, np.zeros(len(mb))]); gm = np.concatenate([mc["product"].values, mb["product"].values])

test = (g.year == 2021).values
Xte, yte = Fg[test], g.label.values[test]; te = g[test].copy()
train_g = (g.year == 2022).values

def head():
    L = tf.keras.layers
    m = tf.keras.Sequential([tf.keras.Input((1792,)), L.BatchNormalization(), L.Dropout(0.4), L.Dense(512, activation="relu"),
                             L.BatchNormalization(), L.Dropout(0.5), L.Dense(1, activation="sigmoid")])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC(name="auc")])
    return m

def train(X, y, groups, seed):
    tf.keras.utils.set_random_seed(seed); rng = np.random.default_rng(seed)
    ug = np.unique(groups); val = np.isin(groups, rng.choice(ug, max(1, len(ug) // 5), replace=False))
    n1 = y[~val].sum(); n0 = (~val).sum() - n1
    m = head()
    m.fit(X[~val], y[~val], validation_data=(X[val], y[val]), epochs=60, batch_size=64, verbose=0,
          class_weight={0: len(y) / (2 * n0), 1: len(y) / (2 * n1)},
          callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=8, restore_best_weights=True)])
    fpr, tpr, th = roc_curve(y[val], m.predict(X[val], verbose=0).ravel())
    return m, float(th[np.where(fpr <= 0.05)[0][-1]])

def evaluate(scores, th):
    neg = yte == 0; bg = (te.kind == "background").values; la = (te.kind == "lookalike").values; oil = yte == 1
    return dict(auc_all=roc_auc_score(yte, scores),
                auc_vs_sea=roc_auc_score(yte[oil | bg], scores[oil | bg]),
                auc_vs_lookalike=roc_auc_score(yte[oil | la], scores[oil | la]),
                recall=float((scores[oil] > th).mean()), far=float((scores[neg] > th).mean()))

configs = {
    "med_only": (Fm, ym, gm),
    "gulf2022": (Fg[train_g], g.label.values[train_g], g.group.values[train_g]),
    "med+gulf2022": (np.concatenate([Fm, Fg[train_g]]), np.concatenate([ym, g.label.values[train_g]]),
                     np.concatenate([gm, g.group.values[train_g]])),
}
results = {"current": evaluate(te.current_score.values, 0.75)}
for name, (X, y, grp) in configs.items():
    runs = []
    for seed in range(5):
        m, th = train(X, y.astype(np.float32), grp, 100 + seed)
        runs.append(evaluate(m.predict(Xte, verbose=0).ravel(), th))
    results[name] = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
    results[name].update({k + "_sd": float(np.std([r[k] for r in runs])) for k in ("auc_all",)})
print(f"Test: Gulf 2021 — oil {int(yte.sum())}, look-alikes {int((te.kind=='lookalike').sum())}, ordinary sea {int((te.kind=='background').sum())}, scenes {te['product'].nunique()}")
print(f"{'model':14s} {'AUC all':>9s} {'vs sea':>7s} {'vs LA':>7s} {'recall':>7s} {'false al.':>9s}")
for k, r in results.items():
    sd = f"±{r['auc_all_sd']:.3f}" if "auc_all_sd" in r else ""
    print(f"{k:14s} {r['auc_all']:.3f}{sd:>7s} {r['auc_vs_sea']:.3f}  {r['auc_vs_lookalike']:.3f}  {r['recall']:.3f}  {r['far']:.3f}")
json.dump(results, open(os.path.join(HERE, "multi_region.json"), "w"), indent=2)
