"""Four additional result figures. Sources (experiments/revision unless noted):
fig_csiro_cv      csiro_oof.csv (B4, seed 0, out-of-fold scores of Table 7)
fig_pr_browse     browse_eval/oof_scores_v2.csv and revision/baseline_oof_med.csv (revision/baseline_oof.py)
fig_layer3        layer3_sensitivity.json, layer3_national.json, layer3_national_budget.json
fig_latency       revision/latency_waits.csv (revision/latency_waits.py) and revision/publication_lag.csv"""
import os, json, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, f1_score
E = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out'); os.makedirs(OUT, exist_ok=True)
R = E + '/revision'
BLUE, RED, TEAL, GREY = '#1f5fa8', '#c0392b', '#2a9d8f', '#7f7f7f'
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.linewidth': 0.6})
def finish(fig, ax, name, titles):
    for a, t in zip(ax, titles):
        a.set_title(t, fontsize=8); a.grid(lw=0.3, alpha=0.5)
    fig.tight_layout(pad=0.4); fig.savefig(f'{OUT}/{name}.pdf'); fig.savefig(f'{OUT}/{name}.png', dpi=200)

# ---------- A: CSIRO re-validation
d = pd.read_csv(R + '/csiro_oof.csv'); d = d[d.arch == 'b4']
sch = [('random', 'Chip-level', BLUE, '-'), ('image', 'Grouped by source image', TEAL, '--'), ('region', 'Region held out', RED, '-.')]
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
th = np.linspace(0.01, 0.99, 99)
for s, lab, c, ls in sch:
    g = d[d.scheme == s]; fpr, tpr, _ = roc_curve(g.label, g.score)
    ax[0].plot(fpr, tpr, ls, color=c, lw=1.2, label=f'{lab} (AUC {roc_auc_score(g.label, g.score):.3f})')
    f1 = [f1_score(g.label, g.score > t) for t in th]
    ax[1].plot(th, f1, ls, color=c, lw=1.2, label=lab)
ax[0].plot([0, 1], [0, 1], ':', color=GREY, lw=0.7)
ax[0].set_xlim(0, 0.3); ax[0].set_ylim(0.6, 1.0); ax[0].set_xlabel('False-positive rate (axis cut at 0.3)'); ax[0].set_ylabel('True-positive rate (recall)')
ax[0].legend(fontsize=6.5, frameon=False, loc='lower right')
ax[1].axvline(0.75, color=GREY, lw=0.8, ls=':'); ax[1].text(0.74, 0.62, r'$\theta = 0.75$', ha='right', fontsize=7)
ax[1].set_xlim(0, 1); ax[1].set_ylim(0.6, 1.0); ax[1].set_xlabel('Decision threshold'); ax[1].set_ylabel('F1-score')
finish(fig, ax, 'fig_csiro_cv', ['(a) ROC, out-of-fold scores', '(b) F1 against threshold'])

# ---------- B: browse crops, Mediterranean
o = pd.read_csv(E + '/browse_eval/oof_scores_v2.csv'); b = pd.read_csv(R + '/baseline_oof_med.csv').dropna(subset=['score'])
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
curves = [('Browse-trained head', o.label, o.browse_score, o.browse_flag, BLUE, '-'),
          ('Gradient boosting (24 statistics)', b.label, b.score, b.flag, TEAL, '--'),
          ('CSIRO-trained head', o.label, o.current_score, o.current_score > 0.75, RED, '-.')]
for lab, y, s, f, c, ls in curves:
    p, r, _ = precision_recall_curve(y, s); ax[0].plot(r, p, ls, color=c, lw=1.2, label=lab)
    tp = (f & (y == 1)).sum(); ax[0].plot(tp / (y == 1).sum(), tp / max(f.sum(), 1), 'o', color=c, ms=4.5, mec='white', mew=0.8)
ax[0].axhline((o.label == 1).mean(), color=GREY, lw=0.7, ls=':')
ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1.02); ax[0].set_xlabel('Recall'); ax[0].set_ylabel('Precision')
ax[0].legend(fontsize=6.5, frameon=False, loc='lower left')
bins = np.linspace(0, 1, 26)
for k, lab, c, ls in [('oil', 'Oil', BLUE, '-'), ('lookalike', 'Look-alike', RED, '-.'), ('background', 'Open sea', TEAL, '--')]:
    s = o[o.kind == k].browse_score
    ax[1].hist(s, bins=bins, histtype='step', color=c, lw=1.2, ls=ls, weights=np.full(len(s), 100 / len(s)), label=f'{lab} (n = {len(s):,})')
ax[1].axvspan(0.70, 0.87, color=GREY, alpha=0.15, lw=0); ax[1].text(0.785, 22, 'fold\nthresholds', ha='center', fontsize=6.5)
ax[1].set_xlabel('Score of the browse-trained head'); ax[1].set_ylabel('% of crops in class'); ax[1].set_xlim(0, 1)
ax[1].legend(fontsize=6.5, frameon=False, loc='upper left', bbox_to_anchor=(0.1, 1.0))
finish(fig, ax, 'fig_pr_browse', ['(a) Precision and recall, E. Mediterranean', '(b) Score distribution by class'])

# ---------- C: Layer 3 on the US fleet
sens = json.load(open(R + '/layer3_sensitivity.json'))['settings']
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
cont = ['auto', '0.05', '0.02', '0.01']; xs = np.arange(4)
for z, c, ls, m in [('3.5', BLUE, '-', 'o'), ('10.0', RED, '--', 's')]:
    v = [sens[f'{k}|{z}']['mean_pct'] for k in cont]
    lo = [v[i] - sens[f'{k}|{z}']['range_pct'][0] for i, k in enumerate(cont)]; hi = [sens[f'{k}|{z}']['range_pct'][1] - v[i] for i, k in enumerate(cont)]
    ax[0].errorbar(xs, v, yerr=[lo, hi], fmt=m + ls, color=c, ms=3.5, lw=1.1, capsize=2, label=f'robust $z > {float(z):g}$')
ax[0].set_xticks(xs); ax[0].set_xticklabels(['automatic', '0.05', '0.02', '0.01\n(TARS)']); ax[0].set_xlabel('Isolation Forest contamination')
ax[0].set_ylabel('% of vessels flagged by Layer 3'); ax[0].set_ylim(0, 9.5); ax[0].legend(fontsize=6.5, frameon=False)
cat = lambda r: 'Rendezvous' if 'RENDEZVOUS' in r else ('Loitering' if 'LOITERING' in r else 'Layer 3 only')
comp = {}
for f, lab in [('layer3_national.json', 'auto'), ('layer3_national_budget.json', '0.01')]:
    for s in json.load(open(f'{R}/{f}'))['native_1min']:
        c = {'Layer 3 only': 0, 'Loitering': 0, 'Rendezvous': 0}
        for r, n in s['reasons'].items(): c[cat(r)] += n
        comp[(lab, s['time'][11:13])] = {k: 100 * v / s['vessels'] for k, v in c.items()}
hours = ['03', '09', '15', '21']; pos = []
for j, lab in enumerate(['auto', '0.01']):
    for i, h in enumerate(hours):
        x = j * 5 + i; pos.append(x); bottom = 0
        for k, c, hatch in [('Layer 3 only', BLUE, None), ('Loitering', TEAL, None), ('Rendezvous', RED, None)]:
            v = comp[(lab, h)][k]
            ax[1].bar(x, v, 0.8, bottom=bottom, color=c, edgecolor='white', lw=0.5, label=k if (j, i) == (0, 0) else None); bottom += v
ax[1].set_xticks(pos); ax[1].set_xticklabels(hours * 2, fontsize=6.5)
ax[1].text(1.5, 10.9, 'automatic', ha='center', fontsize=7); ax[1].text(6.5, 3.4, '0.01 (TARS)', ha='center', fontsize=7)
ax[1].set_xlabel('Cycle time (UTC) for each contamination setting'); ax[1].set_ylabel('% of vessels escalated'); ax[1].set_ylim(0, 12)
ax[1].legend(fontsize=6.5, frameon=False, loc='upper right')
finish(fig, ax, 'fig_layer3', ['(a) Layer 3 flags outside dense areas', '(b) Escalations per cycle by reason'])

# ---------- D: latency
w = pd.read_csv(R + '/latency_waits.csv'); lag = pd.read_csv(R + '/publication_lag.csv')
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
for reg, lab, c, ls in [('eastern_med', 'E. Mediterranean', BLUE, '-'), ('arabian_gulf', 'Gulf', RED, '--')]:
    x = np.sort(w[w.region == reg].wait_h.values); ax[0].plot(x, np.arange(1, len(x) + 1) / len(x) * 100, ls, color=c, lw=1.2, label=f'{lab} (median {np.median(x):.1f} h)')
    x = np.sort(lag[lag.region == reg].lag_h.values); ax[1].step(x, np.arange(1, len(x) + 1) / len(x) * 100, ls, where='post', color=c, lw=1.2, label=f'{lab} (median {np.median(x):.1f} h)')
for t in (24, 48): ax[0].axvline(t, color=GREY, lw=0.7, ls=':'); ax[0].text(t + 1, 92, f'{t} h', fontsize=6.5)
ax[0].set_xlim(0, 170); ax[0].set_xlabel('Wait from anomaly to next acquisition (h)'); ax[0].set_ylabel('% of anomalies')
ax[1].set_xlim(0, 95); ax[1].set_xlabel('Acquisition to browse image on ASF (h)'); ax[1].set_ylabel('% of products')
for a in ax: a.set_ylim(0, 100); a.legend(fontsize=6.5, frameon=False, loc='lower right')
finish(fig, ax, 'fig_latency', ['(a) Satellite revisit (2,000 points per region)', '(b) Publication lag (60 products per region)'])
print('ok')
