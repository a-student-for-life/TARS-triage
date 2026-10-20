"""Extract frozen-backbone features for 25 km browse crops (ESSD 2019 labels).

The TARS model is a frozen ImageNet EfficientNet-B4 plus a trained head. Because
the backbone was frozen, both the current (CSIRO-trained) head and a new
browse-trained head can share exactly the same features. This script runs the
backbone once per crop and stores:
  - the 1,792-d pooled features (original view, and a horizontally flipped view
    used only as training augmentation)
  - the current model's score for the original view (its head applied to the
    same features), so the comparison is controlled: same backbone, same crop,
    same preprocessing, only the head differs.
Output: features_25km.npz + crops_25km.csv
"""
import csv, json, math, os, sys
import cv2, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CROP_KM, IMG = 25.0, 380
sys.path.insert(0, HERE)
from geo import bbox, bbox_latlon_to_pixel  # noqa: E402

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
if os.getenv("TARS_OFFLINE_IMAGENET") == "1":
    from keras.src.applications import efficientnet as eff
    from keras import applications
    fake = os.path.join(HERE, "fake_notop.weights.h5")
    if not os.path.exists(fake):
        applications.EfficientNetB4(weights=None, include_top=False, input_shape=(IMG, IMG, 3)).save_weights(fake)
    eff.file_utils.get_file = lambda *a, **k: fake
sys.path.insert(0, REPO)
os.chdir(REPO)
import ml_logic  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow.keras.applications.efficientnet import preprocess_input  # noqa: E402

model = ml_logic.load_ai_model()
backbone = tf.keras.Sequential(model.layers[:2])          # EfficientNet-B4 + GlobalAveragePooling
head = tf.keras.Sequential(model.layers[2:])              # the trained CSIRO head

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
d["is_oil"] = d.set.isin(["ow", "oc"])
d["obj_km"] = [max(km(r.oul_lat, r.oul_lon, r.our_lat, r.our_lon), km(r.oul_lat, r.oul_lon, r.obl_lat, r.obl_lon))
               if r.is_oil else 0.0 for r in d.itertuples()]
d["plat"] = (d.pul_lat + d.pbr_lat) / 2
d["plon"] = (d.pul_lon + d.pbr_lon) / 2
d = d[~d.sid.str.contains(";")]
patches = d.groupby("jpg").agg(product=("product", "first"), label=("is_oil", "first"), subset=("set", "first"),
                               plat=("plat", "first"), plon=("plon", "first"), start=("start", "first"),
                               max_obj_km=("obj_km", "max")).reset_index().sort_values("product")
oil_pts = d[d.is_oil][["product", "plat", "plon"]]

rows, feats, feats_flip, cur = [], [], [], []
cache = {}
batch_imgs, batch_meta = [], []


def flush():
    if not batch_imgs:
        return
    x = preprocess_input(np.stack(batch_imgs).astype(np.float32))
    f = backbone.predict(x, verbose=0)
    n = len(batch_meta)
    feats.extend(f[:n]); feats_flip.extend(f[n:])
    cur.extend(head.predict(f[:n], verbose=0).ravel() if head is not None else np.zeros(n))
    rows.extend(batch_meta)
    batch_imgs.clear(); batch_meta.clear()


pending_flip = []
for p in patches.itertuples():
    if p.product not in fp:
        continue
    if p.product not in cache:
        cache.clear()
        path = os.path.join(HERE, "browse", p.product + ".jpg")
        cache[p.product] = cv2.imread(path, cv2.IMREAD_GRAYSCALE) if os.path.exists(path) else None
    g = cache[p.product]
    if g is None:
        continue
    if not p.label:
        same = oil_pts[oil_pts["product"] == p.product]
        if any(km(p.plat, p.plon, a, b) < CROP_KM * 0.75 for a, b in zip(same.plat, same.plon)):
            continue
    h, w = g.shape
    geom = fp[p.product]["geometry"]
    x, y = bbox_latlon_to_pixel(geom, p.plat, p.plon, h, w)
    lon0, lon1, lat0, lat1 = bbox(geom)
    half = int(round(CROP_KM / (km(lat0, lon0, lat0, lon1) / w) / 2))
    x0, y0 = int(round(x)) - half, int(round(y)) - half
    if x0 < 0 or y0 < 0 or x0 + 2 * half > w or y0 + 2 * half > h:
        continue
    crop = g[y0:y0 + 2 * half, x0:x0 + 2 * half]
    if np.mean(crop < 3) > 0.2:
        continue
    rgb = cv2.cvtColor(cv2.resize(crop, (IMG, IMG)), cv2.COLOR_GRAY2RGB)
    batch_meta.append(dict(jpg=p.jpg, product=p.product, label=int(p.label), subset=p.subset,
                           day=str(p.start)[:10], max_obj_km=round(p.max_obj_km, 3)))
    batch_imgs.insert(len(batch_meta) - 1, rgb)
    pending_flip.append(rgb[:, ::-1])
    if len(batch_meta) == 16:
        batch_imgs.extend(pending_flip); pending_flip.clear(); flush()
        if len(rows) % 320 == 0:
            print(len(rows), flush=True)
batch_imgs.extend(pending_flip); pending_flip.clear(); flush()

np.savez_compressed(os.path.join(HERE, f"features_25km{SUFFIX}.npz"), feats=np.array(feats), feats_flip=np.array(feats_flip),
                    current_score=np.array(cur))
pd.DataFrame(rows).assign(current_score=cur).to_csv(os.path.join(HERE, f"crops_25km{SUFFIX}.csv"), index=False)
print("DONE", len(rows), flush=True)
