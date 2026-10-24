"""ROC curves: browse-trained vs CSIRO head; Mediterranean (CV) and Gulf (no Gulf training)."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
import os
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out'); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.linewidth': 0.6})
med = pd.read_csv(f'{R}/browse_eval/oof_scores_v2.csv'); gulf = pd.read_csv(f'{R}/gulf_eval/gulf_scores.csv')
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)
for ax, df, title in [(axes[0], med, '(a) E. Mediterranean, 5-fold CV by day'),
                      (axes[1], gulf, '(b) Gulf, no Gulf training data')]:
    for col, lab, sty in [('browse_score', 'Browse-trained head (Med.)', ('-', '#1f5fa8')),
                          ('current_score', 'CSIRO-trained head', ('--', '#c0392b'))]:
        f, t, _ = roc_curve(df.label, df[col]); a = roc_auc_score(df.label, df[col])
        ax.plot(f, t, sty[0], color=sty[1], lw=1.3, label=f'{lab} (AUC {a:.3f})')
    ax.plot([0, 1], [0, 1], color='#bbbbbb', lw=0.6)
    ax.set_title(title, fontsize=8); ax.set_xlabel('False-alarm rate'); ax.legend(fontsize=6.5, frameon=False, loc='lower right')
    ax.grid(lw=0.3, alpha=0.5)
axes[0].set_ylabel('Recall')
fig.tight_layout(pad=0.4); fig.savefig(f'{OUT}/fig_roc.pdf'); fig.savefig(f'{OUT}/fig_roc.png', dpi=200)
