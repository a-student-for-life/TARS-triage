"""Train a browse-domain head and compare it with the current (CSIRO-trained) head.

Both heads sit on the identical frozen EfficientNet-B4 backbone and see the
identical 25 km crops (features from extract_features.py); only the head's
training data differs.

Evaluation: 5-fold cross-validation grouped by acquisition DAY, so a test fold
never shares a day (and therefore a slick seen in an overlapping scene) with
training. Within each training fold, 20% of days are held out for early
stopping and for choosing the operating threshold; the test fold is used once.
Every crop gets exactly one out-of-fold score.

Outputs: oof_scores.csv, comparison.json, and a printed summary.
"""
import json, os, random
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
SEED, FOLDS, TARGET_FAR = 42, 5, 0.05

meta = pd.read_csv(os.path.join(HERE, "crops_25km.csv"))
z = np.load(os.path.join(HERE, "features_25km.npz"))
X, Xf = z["feats"], z["feats_flip"]
y = meta.label.values.astype(np.float32)
groups = meta.day.values


def make_head():
    """Same head as the deployed model (see ml_logic.build_model_architecture)."""
    L = tf.keras.layers
    m = tf.keras.Sequential([tf.keras.Input((X.shape[1],)), L.BatchNormalization(), L.Dropout(0.4),
                             L.Dense(512, activation="relu"), L.BatchNormalization(), L.Dropout(0.5),
                             L.Dense(1, activation="sigmoid")])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
              metrics=[tf.keras.metrics.AUC(name="auc")])
    return m


def threshold_at_far(labels, scores, far=TARGET_FAR):
    """Highest-recall threshold whose false-alarm rate on these labels is <= far."""
    fpr, tpr, th = roc_curve(labels, scores)
    ok = np.where(fpr <= far)[0]
    return float(th[ok[-1]])


oof = np.full(len(y), np.nan)
fold_of = np.full(len(y), -1)
fold_thresholds = {}
heads = {}
day_fold = {}
for k, (tr, te) in enumerate(GroupKFold(n_splits=FOLDS).split(X, y, groups)):
    tf.keras.utils.set_random_seed(SEED + k); random.seed(SEED + k)
    days = np.unique(groups[tr]); rng = np.random.default_rng(SEED + k)
    val_days = set(rng.choice(days, size=max(1, len(days) // 5), replace=False))
    va = tr[np.isin(groups[tr], list(val_days))]
    fit = tr[~np.isin(groups[tr], list(val_days))]
    Xtr = np.concatenate([X[fit], Xf[fit]]); ytr = np.concatenate([y[fit], y[fit]])   # flip augmentation
    n1 = ytr.sum(); n0 = len(ytr) - n1
    cw = {0: len(ytr) / (2 * n0), 1: len(ytr) / (2 * n1)}
    head = make_head()
    head.fit(Xtr, ytr, validation_data=(X[va], y[va]), epochs=60, batch_size=64, class_weight=cw, verbose=0,
             callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=8,
                                                         restore_best_weights=True)])
    fold_thresholds[k] = threshold_at_far(y[va], head.predict(X[va], verbose=0).ravel())
    oof[te] = head.predict(X[te], verbose=0).ravel()
    heads[k] = head
    day_fold.update({dd: k for dd in np.unique(groups[te])})
    fold_of[te] = k
    print(f"fold {k}: train {len(fit)} val {len(va)} test {len(te)} | test AUC "
          f"{roc_auc_score(y[te], oof[te]):.3f} (current head {roc_auc_score(y[te], meta.current_score.values[te]):.3f})",
          flush=True)

meta["browse_score"] = oof
meta["fold"] = fold_of
meta["browse_flag"] = [s > fold_thresholds[f] for s, f in zip(oof, fold_of)]
meta.to_csv(os.path.join(HERE, "oof_scores.csv"), index=False)


def boot_auc_diff(df, n=2000, seed=0):
    rng = np.random.default_rng(seed); days = df.day.unique(); by = {d: g for d, g in df.groupby("day")}
    diffs, a_b, a_c = [], [], []
    for _ in range(n):
        s = pd.concat([by[d] for d in rng.choice(days, len(days))])
        if s.label.nunique() < 2:
            continue
        b, c = roc_auc_score(s.label, s.browse_score), roc_auc_score(s.label, s.current_score)
        a_b.append(b); a_c.append(c); diffs.append(b - c)
    q = lambda v: [round(float(x), 3) for x in np.percentile(v, [2.5, 97.5])]
    return q(a_b), q(a_c), q(diffs)


def summary(df, name):
    if df.label.nunique() < 2:
        return None
    ab, ac = roc_auc_score(df.label, df.browse_score), roc_auc_score(df.label, df.current_score)
    ci_b, ci_c, ci_d = boot_auc_diff(df)
    neg, pos = df[df.label == 0], df[df.label == 1]
    res = dict(n=len(df), n_oil=int(df.label.sum()),
               browse_auc=round(ab, 3), browse_auc_ci=ci_b, current_auc=round(ac, 3), current_auc_ci=ci_c,
               auc_diff_ci=ci_d,
               browse_recall=round(float(pos.browse_flag.mean()), 3), browse_far=round(float(neg.browse_flag.mean()), 3),
               current_recall_075=round(float((pos.current_score > 0.75).mean()), 3),
               current_far_075=round(float((neg.current_score > 0.75).mean()), 3))
    print(f"\n{name}: n={res['n']} oil={res['n_oil']}\n"
          f"  browse-trained head AUC {res['browse_auc']} {ci_b} | recall {res['browse_recall']} at false-alarm {res['browse_far']}"
          f" (threshold set on validation days for {TARGET_FAR:.0%})\n"
          f"  current head       AUC {res['current_auc']} {ci_c} | recall {res['current_recall_075']} at false-alarm {res['current_far_075']} (threshold 0.75)\n"
          f"  AUC difference (browse - current) 95% CI {ci_d}")
    return res


out = {"all": summary(meta, "ALL CROPS"),
       "open_water": summary(meta[meta.subset.isin(["ow", "nw"])], "OPEN WATER"),
       "coastal": summary(meta[meta.subset.isin(["oc", "nc"])], "COASTAL")}
neg = meta[meta.label == 0]
out["by_size"] = {}
print("\nBy largest slick (each size group vs all look-alikes): browse AUC / current AUC")
for lo, hi, lab in [(0, 1, "<1 km"), (1, 2, "1-2 km"), (2, 5, "2-5 km"), (5, 10, "5-10 km"), (10, 1e9, ">=10 km")]:
    pos = meta[(meta.label == 1) & (meta.max_obj_km >= lo) & (meta.max_obj_km < hi)]
    both = pd.concat([pos, neg])
    r = dict(n=len(pos), browse_auc=round(roc_auc_score(both.label, both.browse_score), 3),
             current_auc=round(roc_auc_score(both.label, both.current_score), 3),
             browse_recall=round(float(pos.browse_flag.mean()), 3))
    out["by_size"][lab] = r
    print(f"  {lab:8s} n={r['n']:4d}  {r['browse_auc']:.3f} / {r['current_auc']:.3f}   browse recall {r['browse_recall']}")

# --- Confound check: ordinary-sea background crops ------------------------------
bg_path = os.path.join(HERE, "features_background.npz")
if os.path.exists(bg_path):
    bg = pd.read_csv(os.path.join(HERE, "background_25km.csv"))
    Fb = np.load(bg_path)["feats"]
    # Score each background crop with the head whose training never saw its day.
    k_of = np.array([day_fold.get(dd, 0) for dd in bg.day])
    bg["browse_score"] = np.concatenate([heads[k].predict(Fb[k_of == k], verbose=0).ravel()
                                         if (k_of == k).any() else np.array([]) for k in range(FOLDS)])[
        np.argsort(np.argsort(k_of, kind="stable"), kind="stable")]
    bg["browse_flag"] = [s > fold_thresholds[k] for s, k in zip(bg.browse_score, k_of)]
    bg.to_csv(os.path.join(HERE, "background_scores.csv"), index=False)
    oil = meta[meta.label == 1]
    both = pd.concat([oil.assign(label=1), bg.assign(label=0)])
    res = dict(n_background=len(bg),
               browse_far=round(float(bg.browse_flag.mean()), 3), current_far_075=round(float((bg.current_score > 0.75).mean()), 3),
               browse_auc_oil_vs_background=round(roc_auc_score(both.label, both.browse_score), 3),
               current_auc_oil_vs_background=round(roc_auc_score(both.label, both.current_score), 3),
               browse_auc_lookalike_vs_background=round(roc_auc_score(
                   np.r_[np.ones((meta.label == 0).sum()), np.zeros(len(bg))],
                   np.r_[1 - meta[meta.label == 0].browse_score, 1 - bg.browse_score]), 3))
    out["background"] = res
    print(f"\nBACKGROUND (ordinary sea, n={len(bg)}):\n"
          f"  false alarms: browse head {res['browse_far']} | current head {res['current_far_075']}\n"
          f"  oil vs background AUC: browse {res['browse_auc_oil_vs_background']} | current {res['current_auc_oil_vs_background']}\n"
          f"  look-alike vs background separability (browse head): AUC {res['browse_auc_lookalike_vs_background']}")
json.dump(out, open(os.path.join(HERE, "comparison.json"), "w"), indent=2)
