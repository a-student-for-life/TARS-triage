"""CSIRO validation without chip-level leakage, and B0 versus B4.

CSIRO chips are cut from larger source images on a 200-px grid with 400-px
windows, so neighbouring chips from one source image share half their pixels
(file name: <chip>_<x>_<y>_img_<source-image-id>_<region>_cls_<label>.jpg).
A chip-level random split therefore places overlapping pixels in both
training and validation.

Three validation schemes, identical head and training for every scheme:
  random   5-fold stratified by class only (as in the original split)
  image    5-fold, all chips of a source image in the same fold
  region   leave-region-out: GBR, SIN, JAV, and all other regions pooled

Head = deployed TARS head (BN, Dropout 0.4, Dense 512 ReLU, BN, Dropout 0.5,
sigmoid) on frozen ImageNet features; class-balanced weights; horizontal-flip
augmentation; early stopping on 15% of the training source images; the
decision threshold is the one that maximises F1 on those held-out images
(the rule used to choose 0.75 for the deployed model). Each chip is scored
once by a model that never saw its source image (image/region schemes).

Class-0 chips with more than half their pixels above 250 are excluded, as in
the original training notebook.

  python csiro_cv.py   -> csiro_cv.json, csiro_oof.csv
"""
import json, os
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [0, 1, 2]


def load(arch):
    z = np.load(os.path.join(HERE, f"csiro_features_{arch}.npz"))
    keep = ~((z["label"] == 0) & (z["white_frac"] > 0.5))
    meta = pd.DataFrame({k: z[k][keep] for k in ["fname", "label", "region", "image_id", "x", "y"]})
    return meta, z["feats"][keep], z["feats_flip"][keep], (z["deployed_score"][keep] if len(z["deployed_score"]) else None)


def make_head(d, seed):
    tf.keras.utils.set_random_seed(seed)
    L = tf.keras.layers
    m = tf.keras.Sequential([tf.keras.Input((d,)), L.BatchNormalization(), L.Dropout(0.4),
                             L.Dense(512, activation="relu"), L.BatchNormalization(), L.Dropout(0.5),
                             L.Dense(1, activation="sigmoid")])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
              metrics=[tf.keras.metrics.AUC(name="auc")])
    return m


def best_f1_threshold(y, s):
    p, r, th = precision_recall_curve(y, s)
    f = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-9)
    return float(th[np.argmax(f)])


def splits(scheme, meta, seed):
    y, g = meta.label.values, meta.image_id.values
    if scheme == "random":
        return list(StratifiedKFold(5, shuffle=True, random_state=seed).split(meta, y))
    if scheme == "image":
        return list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(meta, y, g))
    reg = np.where(meta.region.isin(["GBR", "SIN", "JAV"]), meta.region, "OTHER")
    return [(np.where(reg != r)[0], np.where(reg == r)[0]) for r in ["GBR", "SIN", "JAV", "OTHER"]]


def run(arch, scheme, seed, meta, X, Xf):
    y = meta.label.values.astype(np.float32)
    g = meta.image_id.values
    oof, flag, fold_of = np.full(len(y), np.nan), np.zeros(len(y), bool), np.full(len(y), -1)
    for k, (tr, te) in enumerate(splits(scheme, meta, seed)):
        rng = np.random.default_rng(seed * 100 + k)
        imgs = np.unique(g[tr])
        va_imgs = rng.choice(imgs, size=max(1, int(0.15 * len(imgs))), replace=False)
        va, fit = tr[np.isin(g[tr], va_imgs)], tr[~np.isin(g[tr], va_imgs)]
        Xtr, ytr = np.concatenate([X[fit], Xf[fit]]), np.concatenate([y[fit], y[fit]])
        n1 = ytr.sum(); n0 = len(ytr) - n1
        head = make_head(X.shape[1], seed * 100 + k)
        head.fit(Xtr, ytr, validation_data=(X[va], y[va]), epochs=40, batch_size=64, verbose=0,
                 class_weight={0: len(ytr) / (2 * n0), 1: len(ytr) / (2 * n1)},
                 callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=5,
                                                             restore_best_weights=True)])
        th = best_f1_threshold(y[va], head.predict(X[va], verbose=0).ravel())
        oof[te] = head.predict(X[te], verbose=0).ravel()
        flag[te] = oof[te] > th
        fold_of[te] = k
    return oof, flag, fold_of


def metrics(y, s, f):
    tn, fp, fn, tp = confusion_matrix(y, f).ravel()
    return dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), auc=float(roc_auc_score(y, s)),
                accuracy=float((tp + tn) / len(y)), precision=float(tp / (tp + fp)),
                recall=float(tp / (tp + fn)), fnr=float(fn / (tp + fn)), f1=float(f1_score(y, f)))


def boot(meta, s, f, n=1000, seed=0):
    """95% intervals resampling whole source images."""
    rng = np.random.default_rng(seed)
    y = meta.label.values
    idx = pd.Series(np.arange(len(meta))).groupby(meta.image_id.values).apply(np.array).values
    a, f1 = [], []
    for _ in range(n):
        pick = np.concatenate(idx[rng.integers(0, len(idx), len(idx))])
        if y[pick].min() == y[pick].max():
            continue
        a.append(roc_auc_score(y[pick], s[pick])); f1.append(f1_score(y[pick], f[pick]))
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))], \
           [float(np.percentile(f1, 2.5)), float(np.percentile(f1, 97.5))]


out, rows = {}, []
for arch in ["b4", "b0"]:
    meta, X, Xf, dep = load(arch)
    y = meta.label.values
    if arch == "b4":
        sizes = meta.groupby("image_id").size()
        sib = meta.image_id.map(sizes).values > 1
        # expected share of validation chips that overlap a training chip under a random 80/20 split
        rng = np.random.default_rng(0); share = []
        for _ in range(200):
            val = rng.random(len(meta)) < 0.2
            tr_imgs = set(meta.image_id.values[~val])
            share.append(np.mean([i in tr_imgs for i in meta.image_id.values[val]]))
        out["data"] = dict(n_chips=int(len(meta)), n_oil=int(y.sum()), n_source_images=int(meta.image_id.nunique()),
                           chips_with_overlapping_sibling=int(sib.sum()),
                           random_split_val_chips_sharing_source_image=float(np.mean(share)),
                           deployed_model_auc_all_chips=float(roc_auc_score(y, dep)),
                           excluded_white=int(5630 - len(meta)))
    out[arch] = {}
    for scheme in ["random", "image", "region"]:
        seeds = SEEDS
        per = []
        for seed in seeds:
            s, f, fold = run(arch, scheme, seed, meta, X, Xf)
            m = metrics(y, s, f)
            per.append(m)
            print(arch, scheme, seed, {k: round(v, 4) for k, v in m.items()}, flush=True)
            if seed == SEEDS[0]:
                ci_auc, ci_f1 = boot(meta, s, f)
                m0 = dict(m, auc_ci=ci_auc, f1_ci=ci_f1)
                rows.append(pd.DataFrame({"arch": arch, "scheme": scheme, "fname": meta.fname, "label": y,
                                          "region": meta.region, "fold": fold, "score": s, "flag": f}))
                if scheme == "region":
                    reg = np.where(meta.region.isin(["GBR", "SIN", "JAV"]), meta.region, "OTHER")
                    m0["by_heldout_region"] = {r: metrics(y[reg == r], s[reg == r], f[reg == r])
                                               for r in ["GBR", "SIN", "JAV", "OTHER"]}
        summary = {k: [float(np.mean([p[k] for p in per])), float(np.std([p[k] for p in per]))]
                   for k in ["auc", "accuracy", "precision", "recall", "fnr", "f1"]}
        out[arch][scheme] = dict(seed0=m0, mean_sd_over_seeds=summary, n_seeds=len(per))
        json.dump(out, open(os.path.join(HERE, "csiro_cv.json"), "w"), indent=1)
pd.concat(rows).to_csv(os.path.join(HERE, "csiro_oof.csv"), index=False)
print("DONE")
