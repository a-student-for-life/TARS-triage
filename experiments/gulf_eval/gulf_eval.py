"""External test on a different sea: GulfOilSAR (Arabian/Persian Gulf, 2021-2022).

Culot, Wang & Hanert (2026), GulfOilSAR, Zenodo, https://doi.org/10.5281/zenodo.23261453 (CC-BY 4.0).

Nothing here is trained. Both heads are applied as they are:
  current : the deployed CSIRO-trained head
  browse  : browse_head_v1, trained only on Eastern Mediterranean 2019 crops
Units (25 km crops of the Sentinel-1 browse preview, same as the Mediterranean study):
  oil        centred on each GulfOilSAR oil polygon
  lookalike  centred on each GulfOilSAR look-alike polygon (if the layers are present)
  background ordinary sea in the same scenes, >30 km from any labelled polygon, <5% land
Scores are computed from the shared frozen backbone features, so both heads see
exactly the same input.

Usage: python gulf_eval.py   (expects ./data/*.gpkg, ./data/footprints.json, ./data/browse/)
Output: gulf_scores.csv, gulf_summary.json
"""
import glob, json, math, os, sqlite3, struct, sys
import cv2, numpy as np, pandas as pd
from global_land_mask import globe
from shapely import wkb

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CROP_KM, IMG, BG_PER_SCENE, MIN_SEP_KM, MAX_PER_SCENE = 25.0, 380, 3, 30.0, 40

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
if os.getenv("TARS_OFFLINE_IMAGENET") == "1":
    from keras.src.applications import efficientnet as eff
    fake = os.path.join(REPO, "experiments", "browse_eval", "fake_notop.weights.h5")
    eff.file_utils.get_file = lambda *a, **k: fake
sys.path.insert(0, REPO); os.chdir(REPO)
import config  # noqa: E402
config.MODEL_KIND = "csiro"            # load the original model; the browse head is applied separately
import ml_logic  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow.keras.applications.efficientnet import preprocess_input  # noqa: E402

model = ml_logic.load_ai_model()
backbone = tf.keras.Sequential(model.layers[:2])
current_head = tf.keras.Sequential(model.layers[2:])
L = tf.keras.layers
browse_head = tf.keras.Sequential([tf.keras.Input((1792,)), L.BatchNormalization(), L.Dropout(0.4),
                                   L.Dense(512, activation="relu"), L.BatchNormalization(), L.Dropout(0.5),
                                   L.Dense(1, activation="sigmoid")])
browse_head.load_weights(os.path.join(REPO, "models", "browse_head_v1.weights.h5"))
BROWSE_TH = json.load(open(os.path.join(REPO, "models", "browse_head_v1.json")))["threshold"]


def gpkg_geom(blob):
    """GeoPackage binary -> shapely geometry."""
    flags = blob[3]
    env = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[(flags >> 1) & 0b111]
    return wkb.loads(bytes(blob[8 + env:]))


def km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(a))


def load(kind):
    rows = []
    for f in sorted(glob.glob(os.path.join(DATA, f"*_{kind}.gpkg"))):
        table = os.path.basename(f)[:-5]
        con = sqlite3.connect(f)
        for blob, source, area, wind, coast in con.execute(
                f'select geom, source, area, mean_wind_speed, mean_coast from "{table}"'):
            g = gpkg_geom(blob)
            minx, miny, maxx, maxy = g.bounds
            length = max(km(miny, minx, miny, maxx), km(miny, minx, maxy, minx))
            c = g.centroid
            rows.append(dict(product=source, lat=c.y, lon=c.x, length_km=length, area=area, wind=wind,
                             coast_m=coast, year=2021 if "2021" in table else 2022))
    return pd.DataFrame(rows)


oil = load("oil_spill").assign(kind="oil")
la = load("look_alike").assign(kind="lookalike") if glob.glob(os.path.join(DATA, "*_look_alike.gpkg")) else pd.DataFrame()
fp = json.load(open(os.path.join(DATA, "footprints.json")))


def bbox(geom):
    pts = np.array(geom["coordinates"][0], dtype=float)
    return pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()


def crop_at(g, geom, lat, lon):
    h, w = g.shape
    lon0, lon1, lat0, lat1 = bbox(geom)
    half = int(round(CROP_KM / (km(lat0, lon0, lat0, lon1) / w) / 2))
    x = (lon - lon0) / (lon1 - lon0) * w; y = (lat1 - lat) / (lat1 - lat0) * h
    x0, y0 = int(round(x)) - half, int(round(y)) - half
    if x0 < 0 or y0 < 0 or x0 + 2 * half > w or y0 + 2 * half > h:
        return None
    c = g[y0:y0 + 2 * half, x0:x0 + 2 * half]
    return None if np.mean(c < 3) > 0.2 else c


rng = np.random.default_rng(11)
units, imgs, feats = [], [], []


def flush(force=False):
    """Run the backbone on pending crops in batches so images aren't all held in memory."""
    while len(imgs) >= 32 or (force and imgs):
        batch, imgs[:] = imgs[:32], imgs[32:]
        feats.append(backbone.predict(preprocess_input(np.stack(batch).astype(np.float32)), verbose=0))

labelled = pd.concat([oil, la], ignore_index=True)
for prod, info in fp.items():
    path = os.path.join(DATA, "browse", prod + ".jpg")
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        continue
    geom = info["geometry"]
    here = labelled[labelled["product"] == prod]
    for kind in ("oil", "lookalike"):
        sub = here[here.kind == kind]
        if len(sub) > MAX_PER_SCENE:      # cap so a few huge scenes don't dominate
            sub = sub.sample(MAX_PER_SCENE, random_state=0)
        for r in sub.itertuples():
            c = crop_at(g, geom, r.lat, r.lon)
            if c is not None:
                units.append(dict(kind=kind, product=prod, lat=r.lat, lon=r.lon, length_km=r.length_km,
                                  area=r.area, wind=r.wind, coast_m=r.coast_m))
                imgs.append(cv2.cvtColor(cv2.resize(c, (IMG, IMG)), cv2.COLOR_GRAY2RGB))
    lon0, lon1, lat0, lat1 = bbox(geom); got = 0
    for _ in range(300):
        if got >= BG_PER_SCENE:
            break
        lat, lon = rng.uniform(lat0, lat1), rng.uniform(lon0, lon1)
        if any(km(lat, lon, a, b) < MIN_SEP_KM for a, b in zip(here.lat, here.lon)):
            continue
        c = crop_at(g, geom, lat, lon)
        if c is None:
            continue
        d = CROP_KM / 111 / 2
        LAT, LON = np.meshgrid(np.linspace(lat - d, lat + d, 15), np.linspace(lon - d, lon + d, 15))
        if globe.is_land(LAT, LON).mean() > 0.05:
            continue
        units.append(dict(kind="background", product=prod, lat=lat, lon=lon, length_km=0.0, area=0.0,
                          wind=np.nan, coast_m=np.nan))
        imgs.append(cv2.cvtColor(cv2.resize(c, (IMG, IMG)), cv2.COLOR_GRAY2RGB)); got += 1
    flush()

flush(force=True)
feats = np.concatenate(feats)
df = pd.DataFrame(units)
df["current_score"] = current_head.predict(feats, verbose=0).ravel()
df["browse_score"] = browse_head.predict(feats, verbose=0).ravel()
df["label"] = (df.kind == "oil").astype(int)
df.to_csv(os.path.join(HERE, "gulf_scores.csv"), index=False)
np.savez_compressed(os.path.join(HERE, "gulf_features.npz"), feats=feats)
print(df.groupby("kind").size().to_dict(), flush=True)
