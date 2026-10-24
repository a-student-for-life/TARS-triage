"""Off-centre crops: AUC (with 95% CI) and recall at the deployed thresholds vs. shift of the
labelled object from the crop centre. Source: experiments/revision/offcentre_results.json."""
import os, json, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out'); os.makedirs(OUT, exist_ok=True)
R = f'{EXP}/revision/offcentre_results.json'
d = json.load(open(R))
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.linewidth': 0.6})
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
sty = {'med': ('o-', '#1f5fa8', 'E. Mediterranean (n = 2,913)'), 'gulf': ('s--', '#c0392b', 'Gulf 2021 (n = 2,462)')}
for r, (m, c, lab) in sty.items():
    xs = [float(k) for k in d[r]]; v = list(d[r].values())
    auc = [x['browse_auc'] for x in v]; lo = [x['browse_auc'] - x['browse_auc_ci'][0] for x in v]; hi = [x['browse_auc_ci'][1] - x['browse_auc'] for x in v]
    off = -0.12 if r == 'med' else 0.12
    ax[0].errorbar([x + off for x in xs], auc, yerr=[lo, hi], fmt=m, color=c, ms=3.5, lw=1.1, capsize=2, label=lab)
    ax[1].plot(xs, [100 * x['recall'] for x in v], m, color=c, ms=3.5, lw=1.1, label=lab)
    ax[1].plot(xs, [100 * x['false_alarms'] for x in v], m.replace('-', ':').replace('::', ':'), color=c, ms=3, lw=0.8, alpha=0.6)
ax[0].set_ylabel('AUC (oil vs. all negatives)'); ax[0].set_ylim(0.78, 0.95)
ax[1].set_ylabel('% of crops flagged'); ax[1].set_ylim(0, 70)
ax[1].text(9.8, 63, 'recall', ha='right', fontsize=7); ax[1].text(9.8, 9, 'false alarms', ha='right', fontsize=7)
for a, t in zip(ax, ['(a) Discrimination', '(b) Operating point (deployed thresholds)']):
    a.set_xlabel('Distance of labelled object from crop centre (km)'); a.set_xticks([0, 2, 5, 10]); a.set_title(t, fontsize=8)
    a.grid(lw=0.3, alpha=0.5)
ax[0].legend(fontsize=6.5, frameon=False, loc='lower left')
fig.tight_layout(pad=0.4); fig.savefig(f'{OUT}/fig_offcentre.pdf'); fig.savefig(f'{OUT}/fig_offcentre.png', dpi=200)
print('ok')
