"""Off-centre crops: how much does detection depend on the slick being at the crop centre?

The browse evaluation centres each 25 km crop on the labelled object, but in deployment
the crop is centred on the AIS-flagged vessel, and a slick may lie some distance away.
Here every labelled crop of the two test sets is re-cut with its centre moved by
d = 2, 5 and 10 km along one random bearing per crop (fixed seed; the same bearing for
every d, so the object moves outward along one line). At d = 10 km the object centre is
2.5 km inside the crop edge. Crop selection rules are the same as in the original
extraction (inside the image, at most 20% no-data; Mediterranean look-alike crops with
labelled oil within 18.75 km of the crop centre are dropped). Background (ordinary-sea)
crops are not shifted: their position is arbitrary already.

  Mediterranean: oil and look-alike crops of browse_eval/crops_25km.csv
  Gulf:          2021 (held-out year) oil and look-alike units of gulf_eval/gulf_scores.csv

Features come from the same frozen EfficientNet-B4 backbone as before.
Output: offcentre_features.npz, offcentre_units.csv
"""
import json, math, os, sys
import cv2, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE); REPO = os.path.dirname(EXP)
MED, GULF = os.path.join(EXP, "browse_eval"), os.path.join(EXP, "gulf_eval")
CROP_KM, IMG, SHIFTS = 25.0, 380, (0.0, 2.0, 5.0, 10.0)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
if os.getenv("TARS_OFFLINE_IMAGENET") == "1":
    from keras.src.applications import efficientnet as eff
    fake = os.path.join(MED, "fake_notop.weights.h5")
    eff.file_utils.get_file = lambda *a, **k: fake
sys.path.insert(0, REPO); os.chdir(REPO)
import config  # noqa: E402
config.MODEL_KIND = "csiro"
import ml_logic  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow.keras.applications.efficientnet import preprocess_input  # noqa: E402

model = ml_logic.load_ai_model()
backbone = tf.keras.Sequential(model.layers[:2])


def km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(a))


def bbox(geom):
    pts = np.array(geom["coordinates"][0], dtype=float)
    return pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()


def crop_at(g, geom, lat, lon):
    """Identical to the original extraction: lat/lon bounding-box georeferencing."""
    h, w = g.shape
    lon0, lon1, lat0, lat1 = bbox(geom)
    half = int(round(CROP_KM / (km(lat0, lon0, lat0, lon1) / w) / 2))
    x = (lon - lon0) / (lon1 - lon0) * w; y = (lat1 - lat) / (lat1 - lat0) * h
    x0, y0 = int(round(x)) - half, int(round(y)) - half
    if x0 < 0 or y0 < 0 or x0 + 2 * half > w or y0 + 2 * half > h:
        return None
    c = g[y0:y0 + 2 * half, x0:x0 + 2 * half]
    return None if np.mean(c < 3) > 0.2 else c


def move(lat, lon, d_km, bearing):
    dlat = d_km * math.cos(bearing) / 111.32
    dlon = d_km * math.sin(bearing) / (111.32 * math.cos(math.radians(lat)))
    return lat + dlat, lon + dlon


# ---- Mediterranean: rebuild crop centres exactly as extract_features.py did
d = pd.read_csv(os.path.join(MED, "essd_table.csv"))
d["product"] = d.sid.str.split(";").str[0].str.replace(".SAFE", "", regex=False)
d["is_oil"] = d.set.isin(["ow", "oc"])
d["plat"] = (d.pul_lat + d.pbr_lat) / 2; d["plon"] = (d.pul_lon + d.pbr_lon) / 2
d = d[~d.sid.str.contains(";")]
centre = d.groupby("jpg").agg(plat=("plat", "first"), plon=("plon", "first"))
oil_pts = d[d.is_oil][["product", "plat", "plon"]]
med = pd.read_csv(os.path.join(MED, "crops_25km.csv")).join(centre, on="jpg")
med = med.assign(region="med", kind=np.where(med.label == 1, "oil", "lookalike"), lat=med.plat, lon=med.plon,
                 browse_dir=os.path.join(MED, "browse"), uid=med.jpg)
fp_med = json.load(open(os.path.join(MED, "footprints.json")))

# ---- Gulf 2021
g = pd.read_csv(os.path.join(GULF, "gulf_scores.csv"))
g["year"] = g["product"].str[17:21].astype(int)
g = g[(g.year == 2021) & g.kind.isin(["oil", "lookalike"])].reset_index()
g = g.assign(region="gulf", browse_dir=os.path.join(GULF, "data", "browse"), uid="gulf_" + g["index"].astype(str),
             day=g["product"].str[17:25])
fp_gulf = json.load(open(os.path.join(GULF, "data", "footprints.json")))

units = pd.concat([med[["uid", "region", "kind", "label", "product", "day", "lat", "lon", "browse_dir"]],
                   g[["uid", "region", "kind", "label", "product", "day", "lat", "lon", "browse_dir"]]],
                  ignore_index=True)
rng = np.random.default_rng(2026)
units["bearing"] = rng.uniform(0, 2 * math.pi, len(units))
units = units.sort_values(["region", "product"]).reset_index(drop=True)

rows, feats, batch = [], [], []


def flush(force=False):
    while len(batch) >= 32 or (force and batch):
        b, batch[:] = batch[:32], batch[32:]
        feats.append(backbone.predict(preprocess_input(np.stack(b).astype(np.float32)), verbose=0))


cur_prod, img = None, None
for i, u in enumerate(units.itertuples()):
    if u.product != cur_prod:
        cur_prod = u.product
        img = cv2.imread(os.path.join(u.browse_dir, u.product + ".jpg"), cv2.IMREAD_GRAYSCALE)
    fp = fp_med if u.region == "med" else fp_gulf
    if img is None or u.product not in fp:
        continue
    geom = fp[u.product]["geometry"]
    for s in SHIFTS:
        lat, lon = move(u.lat, u.lon, s, u.bearing)
        if u.region == "med" and u.kind == "lookalike":
            same = oil_pts[oil_pts["product"] == u.product]
            if any(km(lat, lon, a, b) < CROP_KM * 0.75 for a, b in zip(same.plat, same.plon)):
                continue
        c = crop_at(img, geom, lat, lon)
        if c is None:
            continue
        batch.append(cv2.cvtColor(cv2.resize(c, (IMG, IMG)), cv2.COLOR_GRAY2RGB))
        rows.append(dict(uid=u.uid, region=u.region, kind=u.kind, label=u.label, product=u.product, day=u.day,
                         shift_km=s, bearing=round(u.bearing, 4), clat=lat, clon=lon))
    flush()
    if i % 500 == 0:
        print(i, len(rows), flush=True)
flush(force=True)
np.savez_compressed(os.path.join(HERE, "offcentre_features.npz"), feats=np.concatenate(feats))
pd.DataFrame(rows).to_csv(os.path.join(HERE, "offcentre_units.csv"), index=False)
print("DONE", len(rows), flush=True)
