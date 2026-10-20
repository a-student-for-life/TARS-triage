"""Background (ordinary sea) crops, to test for a look-alike confound.

ESSD's no-oil set contains only look-alikes (dark oceanic features), never
ordinary sea. A head trained on it could learn "ordinary sea = oil". Most
AIS-flagged ships sit on ordinary sea, so this must be checked.

Background crops: 25 km windows inside the oil-annotation area (30-36 E,
31-34.7 N, where every oil slick was labelled), in open water (<5% land), at
least 30 km from any labelled oil or look-alike patch in the same product.
They are presumed oil-free (no slick was labelled there), with the same caveat
as the dataset itself.
Output: features_background.npz + background_25km.csv
"""
import json, math, os, sys
import cv2, numpy as np, pandas as pd
from global_land_mask import globe

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CROP_KM, IMG, PER_PRODUCT, MIN_SEP_KM = 25.0, 380, 2, 30.0
sys.path.insert(0, HERE)
from geo import bbox  # noqa: E402

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
if os.getenv("TARS_OFFLINE_IMAGENET") == "1":
    from keras.src.applications import efficientnet as eff
    fake = os.path.join(HERE, "fake_notop.weights.h5")
    eff.file_utils.get_file = lambda *a, **k: fake
sys.path.insert(0, REPO); os.chdir(REPO)
import ml_logic  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow.keras.applications.efficientnet import preprocess_input  # noqa: E402
model = ml_logic.load_ai_model()
backbone = tf.keras.Sequential(model.layers[:2]); head = tf.keras.Sequential(model.layers[2:])

# TARS_BACKBONE=b0 swaps in a frozen ImageNet EfficientNet-B0 at 224 px (reviewer
# comparison); crops are selected identically and outputs get a _b0 suffix.
BACKBONE = os.getenv("TARS_BACKBONE", "b4")
SUFFIX = "" if BACKBONE == "b4" else "_" + BACKBONE
if BACKBONE == "b0":
    IMG = 224
    backbone = tf.keras.applications.EfficientNetB0(weights="imagenet", include_top=False, pooling="avg",
                                                    input_shape=(IMG, IMG, 3))
    head = None


def km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(a))


fp = json.load(open(os.path.join(HERE, "footprints.json")))
d = pd.read_csv(os.path.join(HERE, "essd_table.csv"))
d["product"] = d.sid.str.split(";").str[0].str.replace(".SAFE", "", regex=False)
d["plat"] = (d.pul_lat + d.pbr_lat) / 2
d["plon"] = (d.pul_lon + d.pbr_lon) / 2
day_of = d.groupby("product").start.first().str[:10].to_dict()
rng = np.random.default_rng(7)
rows, imgs = [], []
for prod in sorted(set(d["product"])):
    if prod not in fp:
        continue
    path = os.path.join(HERE, "browse", prod + ".jpg")
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        continue
    h, w = g.shape
    lon0, lon1, lat0, lat1 = bbox(fp[prod]["geometry"])
    half = int(round(CROP_KM / (km(lat0, lon0, lat0, lon1) / w) / 2))
    pts = d[d["product"] == prod][["plat", "plon"]].values
    got = 0
    for _ in range(200):
        if got >= PER_PRODUCT:
            break
        x0 = int(rng.integers(0, w - 2 * half)); y0 = int(rng.integers(0, h - 2 * half))
        clon = lon0 + (x0 + half) / w * (lon1 - lon0); clat = lat1 - (y0 + half) / h * (lat1 - lat0)
        if not (30 <= clon <= 36 and 31 <= clat <= 34.7):
            continue
        if any(km(clat, clon, a, b) < MIN_SEP_KM for a, b in pts):
            continue
        crop = g[y0:y0 + 2 * half, x0:x0 + 2 * half]
        if np.mean(crop < 3) > 0.2:
            continue
        lats = np.linspace(lat1 - y0 / h * (lat1 - lat0), lat1 - (y0 + 2 * half) / h * (lat1 - lat0), 15)
        lons = np.linspace(lon0 + x0 / w * (lon1 - lon0), lon0 + (x0 + 2 * half) / w * (lon1 - lon0), 15)
        LON, LAT = np.meshgrid(lons, lats)
        if globe.is_land(LAT, LON).mean() > 0.05:
            continue
        imgs.append(cv2.cvtColor(cv2.resize(crop, (IMG, IMG)), cv2.COLOR_GRAY2RGB))
        rows.append(dict(product=prod, day=day_of[prod], clat=round(clat, 4), clon=round(clon, 4)))
        got += 1

feats = []
for i in range(0, len(imgs), 32):
    feats.append(backbone.predict(preprocess_input(np.stack(imgs[i:i + 32]).astype(np.float32)), verbose=0))
feats = np.concatenate(feats)
cur = head.predict(feats, verbose=0).ravel() if head is not None else np.zeros(len(feats))
np.savez_compressed(os.path.join(HERE, f"features_background{SUFFIX}.npz"), feats=feats, current_score=cur)
pd.DataFrame(rows).assign(current_score=cur).to_csv(os.path.join(HERE, f"background_25km{SUFFIX}.csv"), index=False)
print("DONE", len(rows), flush=True)
