"""Out-of-fold scores of the gradient-boosting baseline on the Mediterranean crops, using the
saved features (simple_baseline_features.csv) and the same folds, seeds and threshold rule as
simple_baseline.py. Output: baseline_oof_med.csv"""
import os, numpy as np, pandas as pd, os
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
E_ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = pd.read_csv(f"{E_}/revision/simple_baseline_features.csv"); med = F[F.region == "med"].reset_index(drop=True)
NAMES = list(F.columns[F.columns.get_loc("mean"):])
Xm = med[NAMES].values; ym = med.label.values; gm_ = med.day.values
crops = pd.read_csv(f"{E_}/browse_eval/crops_25km.csv"); bgm = pd.read_csv(f"{E_}/browse_eval/background_25km.csv")
days_meta = np.concatenate([crops.day.values, bgm.day.values]); y_meta = np.concatenate([crops.label.values, np.zeros(len(bgm))])
fold_of_day = {}
for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(days_meta, y_meta, days_meta)):
    for d in set(days_meta[te]): fold_of_day[d] = k
def th_at(y, s, far=0.05):
    fpr, tpr, th = roc_curve(y, s); return float(th[np.where(fpr <= far)[0][-1]])
ok = ~np.isnan(Xm).any(1); oof = np.full(len(ym), np.nan); flag = np.zeros(len(ym), bool)
for k in range(5):
    te = np.array([fold_of_day[d] == k for d in gm_]) & ok; tr = ~te & ok
    days = np.unique(gm_[tr]); rng = np.random.default_rng(42 + k)
    val_days = rng.choice(days, size=max(1, len(days) // 5), replace=False)
    va = tr & np.isin(gm_, val_days); fit = tr & ~np.isin(gm_, val_days)
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=42 + k).fit(Xm[fit], ym[fit])
    th = th_at(ym[va], m.predict_proba(Xm[va])[:, 1]); oof[te] = m.predict_proba(Xm[te])[:, 1]; flag[te] = oof[te] > th
print("AUC", roc_auc_score(ym[ok], oof[ok]), "recall", flag[ok & (ym == 1)].mean(), "FA", flag[ok & (ym == 0)].mean())
med.assign(score=oof, flag=flag)[["product", "day", "label", "kind", "score", "flag"]].to_csv(os.path.join(E_, "revision", "baseline_oof_med.csv"), index=False)
