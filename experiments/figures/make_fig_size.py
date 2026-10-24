"""Figure: AUC vs largest slick size (Eastern Mediterranean, ESSD labels)."""
import json, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
import os
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out'); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.linewidth': 0.6})
v2 = json.load(open(f'{R}/browse_eval/comparison_v2.json'))
sizes = list(v2['by_size'])
b = [v2['by_size'][s]['browse_auc'] for s in sizes]; c = [v2['by_size'][s]['current_auc'] for s in sizes]
n = [v2['by_size'][s]['n'] for s in sizes]
e = pd.read_csv(f'{R}/browse_eval/eval_results.csv'); sc = e[e.unit == 'scene']; neg = sc[sc.label == 0]; scene = []
for lo, hi in [(0, 1), (1, 2), (2, 5), (5, 10), (10, 1e9)]:
    pos = sc[(sc.label == 1) & (sc.max_obj_km >= lo) & (sc.max_obj_km < hi)]
    d = pd.concat([pos, neg]); scene.append(roc_auc_score(d.label, d.raw))
fig, ax = plt.subplots(figsize=(3.45, 2.15)); x = np.arange(len(sizes))
ax.plot(x, b, 'o-', color='#1f5fa8', lw=1.4, ms=4, label='Browse-trained head, 25 km crop')
ax.plot(x, c, 's--', color='#c0392b', lw=1.2, ms=3.5, label='CSIRO-trained head, 25 km crop')
ax.plot(x, scene, '^:', color='#555555', lw=1.0, ms=3.5, label='CSIRO-trained head, whole scene')
ax.axhline(0.5, color='#999999', lw=0.5)
ax.set_xticks(x); ax.set_xticklabels([f"{s}\n(n={k})" for s, k in zip(['<1', '1–2', '2–5', '5–10', '≥10'], n)], fontsize=6.5)
ax.set_xlabel('Largest slick in patch (km)', fontsize=7); ax.set_ylabel('AUC', fontsize=7); ax.set_ylim(0.45, 1.12)
ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.legend(fontsize=6, frameon=False, loc='upper left', ncol=1, borderaxespad=0.2)
ax.grid(axis='y', lw=0.3, alpha=0.5)
fig.tight_layout(pad=0.3); fig.savefig(f'{OUT}/fig_size.pdf'); fig.savefig(f'{OUT}/fig_size.png', dpi=200)
