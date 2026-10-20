import sys, pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
import os; r = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval_results.csv'))
TH = 0.75
def block(df, title):
    print(f"\n== {title}: n={len(df)} (oil {int(df.label.sum())}, no-oil {int((df.label==0).sum())})")
    for path in ['raw', 'rad', 'sti']:
        y, s = df.label, df[path]; p = s > TH
        tp, fp = int((p & (y==1)).sum()), int((p & (y==0)).sum())
        fn, tn = int((~p & (y==1)).sum()), int((~p & (y==0)).sum())
        auc = roc_auc_score(y, s) if y.nunique() == 2 else float('nan')
        print(f"  {path:3s}  AUC {auc:.3f} | recall {tp/max(tp+fn,1):.3f}  false-alarm rate {fp/max(fp+tn,1):.3f}  precision {tp/max(tp+fp,1):.3f} | TP {tp} FP {fp} FN {fn} TN {tn}")
def by_size(df, title):
    bins = [0, 1, 2, 5, 10, 1000]; labels = ['<1 km', '1-2', '2-5', '5-10', '>=10 km']
    oil = df[df.label == 1].copy(); oil['size'] = pd.cut(oil.max_obj_km, bins, labels=labels, right=False)
    neg = df[df.label == 0]
    print(f"\n   {title}: AUC of each oil size group vs all no-oil units (n_neg={len(neg)})")
    for g, grp in oil.groupby('size', observed=True):
        row = [f"{g:8s} n={len(grp):4d}"]
        for path in ['raw', 'rad', 'sti']:
            both = pd.concat([grp, neg]); row.append(f"{path} AUC {roc_auc_score(both.label, both[path]):.3f} rec {np.mean(grp[path] > TH):.2f}")
        print('   ' + ' | '.join(row))
s = r[r.unit == 'scene']; c = r[r.unit == 'crop']
block(s, 'WHOLE SCENES'); by_size(s, 'scenes by largest slick in scene')
if len(c):
    block(c, 'CROPS 25 km'); by_size(c, 'crops by largest slick in patch')
    block(c[c.subset.isin(['ow','nw'])], 'CROPS, open water only'); block(c[c.subset.isin(['oc','nc'])], 'CROPS, coastal only')
