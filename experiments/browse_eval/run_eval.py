"""Evaluate the deployed TARS classifier on real Sentinel-1 browse images,
labelled by the ESSD 2025 Eastern Mediterranean dataset (Yang et al., PANGAEA.980773).

Units
  scene : whole browse image (what the live system classifies today)
  crop  : window of CROP_KM centred on a labelled patch (oil or look-alike),
          located with the north-up bounding-box georeferencing of the browse image
Paths
  raw   : grayscale -> resize to 380
  rad   : STI with sigma=0 (blur, CLAHE, +70) - radiometric alignment only
  sti   : full STI, sigma=0.5, mean of N_DRAWS seeded draws
Output: one CSV row per (unit, path) with the model score.
"""
import csv, hashlib, json, math, os, sys
import cv2, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE                                   # data files live next to this script
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "eval_results.csv")
CROP_KMS = [25.0]
MAX_NEG_CROPS = 1000  # fixed random sample of look-alike patches
N_DRAWS = 4
IMG = 380
SHIFT = 70

sys.path.insert(0, HERE)
from geo import bbox, bbox_latlon_to_pixel  # noqa: E402

# --- model: exactly ml_logic's build. If the ImageNet file can't be downloaded
# (offline machine), substitute a same-shaped file: those values are overwritten
# by the TARS weights anyway; only the architecture flag matters.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
if os.getenv("TARS_OFFLINE_IMAGENET") == "1":
    from keras.src.applications import efficientnet as eff
    from keras import applications
    fake = os.path.join(ROOT, "fake_notop.weights.h5")
    if not os.path.exists(fake):
        applications.EfficientNetB4(weights=None, include_top=False, input_shape=(IMG, IMG, 3)).save_weights(fake)
    eff.file_utils.get_file = lambda *a, **k: fake
sys.path.insert(0, REPO)
os.chdir(REPO)
import ml_logic  # noqa: E402
from tensorflow.keras.applications.efficientnet import preprocess_input  # noqa: E402
model = ml_logic.load_ai_model()
assert model is not None


def to_rgb(g):
    return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)


def raw(gray):
    return to_rgb(cv2.resize(gray, (IMG, IMG)))


def sti(gray, sigma, rng):
    """Same stages and order as ml_logic.preprocess_style."""
    img = cv2.resize(gray, (IMG, IMG))
    if sigma > 0:
        f = img.astype(np.float32)
        img = np.clip(f + f * rng.normal(0, sigma, img.shape), 0, 255).astype("uint8")
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
    img = np.clip(img.astype(np.int16) + SHIFT, 0, 255).astype(np.uint8)
    return to_rgb(img)


def score(batch):
    return model.predict(preprocess_input(np.stack(batch).astype(np.float32)), verbose=0).ravel()


def seed(key):
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")


def score_unit(gray, key):
    imgs = [raw(gray), sti(gray, 0.0, None)]
    rng = np.random.default_rng(seed(key))
    imgs += [sti(gray, 0.5, rng) for _ in range(N_DRAWS)]
    s = score(imgs)
    return s[0], s[1], float(s[2:].mean()), float(s[2:].std())


def km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(a))


fp = json.load(open(os.path.join(ROOT, "footprints.json")))
d = pd.read_csv(os.path.join(ROOT, "essd_table.csv"))
d["product"] = d.sid.str.split(";").str[0].str.replace(".SAFE", "", regex=False)
d["multi"] = d.sid.str.contains(";")
d["is_oil"] = d.set.isin(["ow", "oc"])
d["obj_km"] = [max(km(r.oul_lat, r.oul_lon, r.our_lat, r.our_lon), km(r.oul_lat, r.oul_lon, r.obl_lat, r.obl_lon))
               if r.is_oil else 0.0 for r in d.itertuples()]
d["plat"] = (d.pul_lat + d.pbr_lat) / 2
d["plon"] = (d.pul_lon + d.pbr_lon) / 2

done = set()
if os.path.exists(OUT):
    done = {(r["unit"], r["key"]) for r in csv.DictReader(open(OUT))}
f = open(OUT, "a", newline="")
w = csv.writer(f)
if not done:
    w.writerow(["unit", "key", "product", "label", "subset", "max_obj_km", "n_objects",
                "black_frac", "land_frac", "raw", "rad", "sti", "sti_std"])

images = {}


def browse(product):
    if product not in images:
        p = os.path.join(ROOT, "browse", product + ".jpg")
        images.clear()
        images[product] = cv2.imread(p, cv2.IMREAD_GRAYSCALE) if os.path.exists(p) else None
    return images[product]


# --- scenes ---------------------------------------------------------------
oil_products = set(d[d.is_oil]["product"])
# Negatives: look-alike-only products inside the oil annotation area, so an
# absence of oil labels means "inspected, no oil" rather than "not inspected".
neg_products = set(d[~d.is_oil]["product"]) - oil_products
scene_rows = []
for prod in sorted(oil_products | neg_products):
    if prod not in fp:
        continue
    lon0, lon1, lat0, lat1 = bbox(fp[prod]["geometry"])
    clat, clon = (lat0 + lat1) / 2, (lon0 + lon1) / 2
    if prod in neg_products and not (30 <= clon <= 36 and 31 <= clat <= 34.7):
        continue
    sub = d[(d["product"] == prod) & d.is_oil]
    scene_rows.append((prod, int(prod in oil_products), sub.obj_km.max() if len(sub) else 0.0, len(sub)))

n = 0
for prod, label, maxkm, nobj in scene_rows:
    if ("scene", prod) in done:
        continue
    g = browse(prod)
    if g is None:
        continue
    r, rad, s, ss = score_unit(g, "scene:" + prod)
    w.writerow(["scene", prod, prod, label, "", f"{maxkm:.3f}", nobj, f"{np.mean(g < 3):.3f}", "", f"{r:.6f}", f"{rad:.6f}", f"{s:.6f}", f"{ss:.6f}"])
    n += 1
    if n % 50 == 0:
        f.flush(); print("scenes", n, flush=True)
f.flush()

# --- crops ----------------------------------------------------------------
from global_land_mask import globe  # noqa: E402

patches = d[~d.multi].groupby("jpg").agg(product=("product", "first"), label=("is_oil", "first"),
                                          subset=("set", "first"), plat=("plat", "first"), plon=("plon", "first"),
                                          max_obj_km=("obj_km", "max"), n_objects=("is_oil", "sum")).reset_index()
oil_pts = d[d.is_oil][["product", "plat", "plon"]]
neg = patches[~patches.label]
patches = pd.concat([patches[patches.label], neg.sample(n=min(MAX_NEG_CROPS, len(neg)), random_state=42)])
n = 0
for crop_km in CROP_KMS:
    for p in patches.sort_values("product").itertuples():
        key = f"{crop_km:g}km:{p.jpg}"
        if ("crop", key) in done:
            continue
        g = browse(p.product)
        if g is None or p.product not in fp:
            continue
        if not p.label:
            # Skip look-alike crops that would also contain a labelled oil patch.
            same = oil_pts[oil_pts["product"] == p.product]
            if any(km(p.plat, p.plon, a, b) < crop_km * 0.75 for a, b in zip(same.plat, same.plon)):
                continue
        h, wd = g.shape
        geom = fp[p.product]["geometry"]
        x, y = bbox_latlon_to_pixel(geom, p.plat, p.plon, h, wd)
        lon0, lon1, lat0, lat1 = bbox(geom)
        px_km = km(lat0, lon0, lat0, lon1) / wd
        half = int(round(crop_km / px_km / 2))
        x0, y0 = int(round(x)) - half, int(round(y)) - half
        if x0 < 0 or y0 < 0 or x0 + 2 * half > wd or y0 + 2 * half > h:
            continue
        crop = g[y0:y0 + 2 * half, x0:x0 + 2 * half]
        black = float(np.mean(crop < 3))
        if black > 0.2:
            continue
        lats = np.linspace(lat1 - (y0 / h) * (lat1 - lat0), lat1 - ((y0 + 2 * half) / h) * (lat1 - lat0), 20)
        lons = np.linspace(lon0 + (x0 / wd) * (lon1 - lon0), lon0 + ((x0 + 2 * half) / wd) * (lon1 - lon0), 20)
        LON, LAT = np.meshgrid(lons, lats)
        land = float(globe.is_land(LAT, LON).mean())
        r, rad, s, ss = score_unit(crop, "crop:" + key)
        w.writerow(["crop", key, p.product, int(p.label), p.subset, f"{p.max_obj_km:.3f}", int(p.n_objects),
                    f"{black:.3f}", f"{land:.3f}", f"{r:.6f}", f"{rad:.6f}", f"{s:.6f}", f"{ss:.6f}"])
        n += 1
        if n % 200 == 0:
            f.flush(); print("crops", n, flush=True)
f.close()
print("DONE", flush=True)
