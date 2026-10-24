"""Example 25 km browse crops with the out-of-fold score of the browse-trained Mediterranean head.
Two crops per outcome (correct detection, missed slick, false alarm, correct rejection), drawn at
random with a fixed seed from experiments/browse_eval/oof_scores_v2.csv, so they are not selected."""
import os, json, math, cv2, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
E = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out'); os.makedirs(OUT, exist_ok=True)
o = pd.read_csv(f'{E}/browse_eval/oof_scores_v2.csv')
u = pd.read_csv(f'{E}/revision/offcentre_units.csv'); u = u[(u.region == 'med') & (u.shift_km == 0)].set_index('uid')
o['lat'] = np.where(o.jpg.notna(), o.jpg.map(u.clat), o.clat); o['lon'] = np.where(o.jpg.notna(), o.jpg.map(u.clon), o.clon)
fp = json.load(open(f'{E}/browse_eval/footprints.json'))
def km(a, b, c, d):
    p1, p2 = math.radians(a), math.radians(c); h = math.sin((p2-p1)/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(math.radians(d-b)/2)**2
    return 2*6371*math.asin(math.sqrt(h))
def crop(prod, lat, lon):
    g = cv2.imread(f'{E}/browse_eval/browse/{prod}.jpg', cv2.IMREAD_GRAYSCALE); h, w = g.shape
    pts = np.array(fp[prod]['geometry']['coordinates'][0], float); lon0, lon1, lat0, lat1 = pts[:,0].min(), pts[:,0].max(), pts[:,1].min(), pts[:,1].max()
    half = int(round(25.0 / (km(lat0, lon0, lat0, lon1) / w) / 2))
    x = int(round((lon - lon0)/(lon1 - lon0)*w)); y = int(round((lat1 - lat)/(lat1 - lat0)*h))
    return g[y-half:y+half, x-half:x+half]
o = o.dropna(subset=['lat'])
groups = [('Correct detection', (o.label == 1) & o.browse_flag), ('Missed slick', (o.label == 1) & ~o.browse_flag),
          ('False alarm', (o.label == 0) & o.browse_flag), ('Correct rejection', (o.label == 0) & ~o.browse_flag)]
name = {'oil': 'oil', 'lookalike': 'look-alike', 'background': 'open sea'}
plt.rcParams.update({'font.family': 'serif', 'font.size': 7.5})
fig, ax = plt.subplots(2, 4, figsize=(6.6, 3.9))
picked = []
for j, (title, m) in enumerate(groups):
    s = o[m].sample(2, random_state=7)
    for i, r in enumerate(s.itertuples()):
        a = ax[i, j]; a.imshow(crop(r.product, r.lat, r.lon), cmap='gray', vmin=0, vmax=255)
        a.set_xticks([]); a.set_yticks([])
        a.set_xlabel(f'{name[r.kind]}, score {r.browse_score:.2f}', fontsize=7)
        if i == 0: a.set_title(title, fontsize=8)
        picked.append(dict(group=title, kind=r.kind, score=round(r.browse_score, 3), product=r.product, day=r.day))
fig.tight_layout(pad=0.3, h_pad=1.6); fig.savefig(f'{OUT}/fig_crop_examples.pdf'); fig.savefig(f'{OUT}/fig_crop_examples.png', dpi=200)
pd.DataFrame(picked).to_csv(f'{OUT}/fig_crop_examples_selection.csv', index=False); print(pd.DataFrame(picked))
