"""Do oil crops simply contain more ships? Count bright point targets per crop
and test whether that count alone separates oil from negatives, and whether
the browse head's score depends on it within each class."""
import json, math, os, sys
import cv2, numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo import bbox, bbox_latlon_to_pixel

def km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2-p1)/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(math.radians(lon2-lon1)/2)**2
    return 2*6371*math.asin(math.sqrt(a))

def bright_points(crop):
    """Small, very bright blobs: ship echoes (or platforms)."""
    c = crop.astype(np.float32); med = np.median(c); mad = np.median(np.abs(c - med)) + 1
    mask = (c > med + 8 * mad).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    return int(sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] <= 30))

fp = json.load(open(os.path.join(HERE, "footprints.json")))
d = pd.read_csv(os.path.join(HERE, "essd_table.csv")); d["plat"] = (d.pul_lat + d.pbr_lat) / 2; d["plon"] = (d.pul_lon + d.pbr_lon) / 2
centre = d.groupby("jpg")[["plat", "plon"]].first()
m = pd.read_csv(os.path.join(HERE, "oof_scores_v2.csv"))
counts = []
cache = {}
for r in m.itertuples():
    if r.product not in cache:
        cache.clear(); cache[r.product] = cv2.imread(os.path.join(HERE, "browse", r.product + ".jpg"), cv2.IMREAD_GRAYSCALE)
    g = cache[r.product]; h, w = g.shape; geom = fp[r.product]["geometry"]
    lat, lon = (r.clat, r.clon) if r.kind == "background" else tuple(centre.loc[r.jpg])
    x, y = bbox_latlon_to_pixel(geom, lat, lon, h, w)
    lon0, lon1, lat0, lat1 = bbox(geom); half = int(round(25 / (km(lat0, lon0, lat0, lon1) / w) / 2))
    x0, y0 = int(round(x)) - half, int(round(y)) - half
    counts.append(bright_points(g[y0:y0 + 2 * half, x0:x0 + 2 * half]))
m["bright_points"] = counts
print(m.groupby("kind").bright_points.describe()[["mean", "50%", "75%"]].round(2))
for negs in (["background"], ["lookalike"], ["lookalike", "background"]):
    df = m[m.kind.isin(["oil"] + negs)]
    print(f"bright-point count alone, oil vs {'+'.join(negs)}: AUC {roc_auc_score(df.label, df.bright_points):.3f}")
for kind in ("oil", "lookalike", "background"):
    g = m[m.kind == kind]; rho, p = spearmanr(g.bright_points, g.browse_score)
    print(f"within {kind:10s}: Spearman(bright points, browse score) = {rho:.2f} (p={p:.1g})")
# Browse AUC among crops with similar ship counts (stratified): does oil still separate?
m["bp_bin"] = pd.cut(m.bright_points, [-1, 0, 2, 5, 1000], labels=["0", "1-2", "3-5", "6+"])
print("\nOil vs all negatives, within bright-point strata (browse / current AUC):")
for b, g in m.groupby("bp_bin", observed=True):
    if g.label.nunique() == 2 and g.label.sum() >= 20:
        print(f"  {b:4s} n={len(g):4d} oil={int(g.label.sum()):4d}  {roc_auc_score(g.label, g.browse_score):.3f} / {roc_auc_score(g.label, g.current_score):.3f}")
m.to_csv(os.path.join(HERE, "oof_scores_v2_with_ships.csv"), index=False)
