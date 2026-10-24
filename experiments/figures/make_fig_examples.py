"""Figure: randomly drawn 25 km browse crops (fixed seed), both regions.
Oil examples are drawn among slicks >= 5 km so the slick can be visible at all."""
import os, json, math, sys, cv2, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out'); os.makedirs(OUT, exist_ok=True); sys.path.insert(0, f'{R}/browse_eval')
from geo import bbox
def km(a, b, c, d):
    p1, p2 = math.radians(a), math.radians(c)
    h = math.sin((p2-p1)/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(math.radians(d-b)/2)**2
    return 2*6371*math.asin(math.sqrt(h))
def crop(img_dir, fp, prod, lat, lon, ckm=25.0):
    g = cv2.imread(f'{img_dir}/{prod}.jpg', 0); h, w = g.shape; geom = fp[prod]['geometry']
    lon0, lon1, lat0, lat1 = bbox(geom); half = int(round(ckm / (km(lat0, lon0, lat0, lon1) / w) / 2))
    x = (lon-lon0)/(lon1-lon0)*w; y = (lat1-lat)/(lat1-lat0)*h
    return g[int(y)-half:int(y)+half, int(x)-half:int(x)+half]
rng = np.random.default_rng(3)
# Mediterranean
mfp = json.load(open(f'{R}/browse_eval/footprints.json')); mc = pd.read_csv(f'{R}/browse_eval/oof_scores_v2.csv')
d = pd.read_csv(f'{R}/browse_eval/essd_table.csv'); d['plat']=(d.pul_lat+d.pbr_lat)/2; d['plon']=(d.pul_lon+d.pbr_lon)/2
cen = d.groupby('jpg')[['plat','plon']].first()
def med_pick(kind):
    s = mc[mc.kind==kind]; s = s[s.max_obj_km>=5] if kind=='oil' else s
    r = s.iloc[rng.integers(len(s))]
    lat, lon = (r.clat, r.clon) if kind=='background' else tuple(cen.loc[r.jpg])
    return crop(f'{R}/browse_eval/browse', mfp, r['product'], lat, lon), r.browse_score, r.current_score
# Gulf
gfp = json.load(open(f'{R}/gulf_eval/data/footprints.json')); gs = pd.read_csv(f'{R}/gulf_eval/gulf_scores.csv')
def gulf_pick(kind):
    s = gs[gs.kind==kind]; s = s[s.length_km>=5] if kind=='oil' else s
    r = s.iloc[rng.integers(len(s))]
    return crop(f'{R}/gulf_eval/data/browse', gfp, r['product'], r.lat, r.lon), None, r.current_score
plt.rcParams.update({'font.family':'serif','font.size':7})
fig, ax = plt.subplots(2, 3, figsize=(3.45, 2.45))
for i, (region, pick) in enumerate([('E. Med.', med_pick), ('Gulf', gulf_pick)]):
    for j, (kind, title) in enumerate([('oil','Oil (≥5 km)'),('lookalike','Look-alike'),('background','Random sea')]):
        c, b, cur = pick(kind); a = ax[i, j]
        a.imshow(c, cmap='gray', vmin=0, vmax=np.percentile(c, 99.5)); a.set_xticks([]); a.set_yticks([])
        if i == 0: a.set_title(title, fontsize=7)
        if j == 0: a.set_ylabel(region, fontsize=7)
fig.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.2); fig.savefig(f'{OUT}/fig_examples.pdf'); fig.savefig(f'{OUT}/fig_examples.png', dpi=200)
