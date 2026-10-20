"""EfficientNet-B4 oil spill classifier and the STI (Statistical Texture
Injection) preprocessing applied to compressed browse previews.

Changes from the original version:
  * STI's speckle step is random, so one draw gives a different score each run.
    analyze_image() now averages STI_DRAWS draws with a seed derived from the
    image file name, so the same image always gets the same score, and it
    reports the spread across draws.
  * The backbone must still be built with weights="imagenet" even though the
    saved weights overwrite every layer: Keras's EfficientNet adds an extra
    input Rescaling layer only in that case, and the model was trained with it.
    (Building with weights=None silently changes every prediction.) The ImageNet
    file is downloaded once and cached in ~/.keras/models.
  * No debug image is written unless TARS_SAVE_DEBUG_IMAGES=1.
  * TensorFlow is imported lazily so preprocessing can be tested without it.
"""
import hashlib
import json
import os

import cv2
import numpy as np

import config

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # CPU deployment

WEIGHTS_PATH = config.WEIGHTS_PATH
IMG_SIZE = config.IMG_SIZE
OPTIMAL_THRESHOLD = config.OPTIMAL_THRESHOLD

_model = None


def build_model_architecture():
    """Rebuild the training architecture (see notebooks/TARS.ipynb).
    The EfficientNet-B4 backbone was frozen during training; only the head
    (BatchNorm, Dropout 0.4, Dense 512, BatchNorm, Dropout 0.5, sigmoid) was trained."""
    from tensorflow.keras import applications, layers, models

    # weights="imagenet" is required for the architecture, not the values:
    # see the module docstring.
    base_model = applications.EfficientNetB4(
        weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(512, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def load_ai_model():
    """Build the architecture once and load the original (CSIRO) weights.
    Regional heads are separate small models on top of this backbone."""
    global _model
    if _model is None:
        try:
            print("[ML] Building EfficientNet-B4...", flush=True)
            model = build_model_architecture()
            print(f"[ML] Loading weights from {WEIGHTS_PATH}...", flush=True)
            model.load_weights(WEIGHTS_PATH)
            _model = model
            print("[ML] Model ready.", flush=True)
        except Exception as e:
            print(f"[ML] Critical error loading model: {e}", flush=True)
            return None
    return _model


# --- Regional heads ------------------------------------------------------------
_registry = None
_heads = {}
_backbone = None


def registry():
    global _registry
    if _registry is None:
        with open(config.REGIONAL_HEADS) as f:
            _registry = json.load(f)
    return _registry


def region_for(lat, lon):
    """Name of the regional head whose bbox contains (lat, lon), or None."""
    if lat is None or lon is None:
        return None
    for h in registry()["heads"]:
        lon0, lon1, lat0, lat1 = h["bbox"]
        if lon0 <= lon <= lon1 and lat0 <= lat <= lat1:
            return h["name"]
    return None


def _head(name):
    """(keras head, threshold) for a regional head, or the CSIRO fallback."""
    if name in _heads:
        return _heads[name]
    model = load_ai_model()
    if name == registry()["fallback"]["name"]:
        from tensorflow.keras import Sequential
        _heads[name] = (Sequential(model.layers[2:]), float(registry()["fallback"]["threshold"]))
        return _heads[name]
    from tensorflow.keras import Input, Sequential, layers
    spec = next(h for h in registry()["heads"] if h["name"] == name)
    head = Sequential([
        Input((model.layers[1].output.shape[-1],)),
        layers.BatchNormalization(), layers.Dropout(0.4), layers.Dense(512, activation="relu"),
        layers.BatchNormalization(), layers.Dropout(0.5), layers.Dense(1, activation="sigmoid"),
    ])
    head.load_weights(spec["weights"])
    with open(spec["info"]) as f:
        _heads[name] = (head, float(json.load(f)["threshold"]))
    return _heads[name]


def _features(batch):
    """Frozen-backbone features (EfficientNet-B4 + global average pooling)."""
    global _backbone
    if _backbone is None:
        from tensorflow.keras import Sequential
        _backbone = Sequential(load_ai_model().layers[:2])
    from tensorflow.keras.applications.efficientnet import preprocess_input
    return _backbone.predict(preprocess_input(batch), verbose=0)


def choose_head(lat=None, lon=None):
    """Head name for this vessel under the configured MODEL_KIND."""
    if config.MODEL_KIND == "browse_v1":
        return "eastern_med_v1"
    return region_for(lat, lon) or registry()["fallback"]["name"]


def preprocess_raw(img_path, target_size=(IMG_SIZE, IMG_SIZE)):
    """Grayscale + resize only (no adaptation). Used for comparisons."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, target_size)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def preprocess_style(img_path, target_size=(IMG_SIZE, IMG_SIZE), sigma=None, rng=None):
    """STI pipeline: multiplicative speckle -> 3x3 Gaussian blur -> CLAHE ->
    brightness shift. sigma=0 skips the speckle noise (the blur still runs).

    rng: a numpy Generator for reproducible draws. If None, the global
    np.random state is used (kept so existing scripts that patch
    numpy.random.normal still work)."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, target_size)
    noise_sigma = config.STI_SIGMA if sigma is None else sigma

    # Stage 1: multiplicative speckle, I * (1 + N(0, sigma^2)). Dark (oil-like)
    # pixels stay dark because the noise scales with the signal.
    if noise_sigma > 0:
        img_float = img.astype(np.float32)
        if rng is None:
            noise = np.random.normal(0, noise_sigma, img.shape)
        else:
            noise = rng.normal(0, noise_sigma, img.shape)
        img = np.clip(img_float + img_float * noise, 0, 255).astype("uint8")
    # The blur is applied even at sigma=0, matching the original pipeline and the
    # sigma=0 row of the published sensitivity table.
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Stage 2: local contrast.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # Stage 3: brightness alignment toward the training corpus mean.
    img = np.clip(img.astype(np.int16) + config.STI_BRIGHTNESS_SHIFT, 0, 255).astype(np.uint8)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if config.SAVE_DEBUG_IMAGES:
        cv2.imwrite("debug_ai_input_v3.jpg", img_rgb)
    return img_rgb


def image_seed(image_path):
    """Stable seed from the file name, so re-running gives identical draws."""
    name = os.path.basename(image_path).encode()
    return int.from_bytes(hashlib.sha256(name).digest()[:8], "little")


def sti_batch(image_path, n_draws=None, sigma=None):
    """Stack of n_draws STI-processed images (float32, before preprocess_input)."""
    noise_sigma = config.STI_SIGMA if sigma is None else sigma
    draws = 1 if noise_sigma == 0 else (n_draws or config.STI_DRAWS)
    rng = np.random.default_rng(image_seed(image_path))
    images = []
    for _ in range(draws):
        processed = preprocess_style(image_path, sigma=noise_sigma, rng=rng)
        if processed is None:
            return None
        images.append(processed)
    return np.stack(images).astype(np.float32)


def analyze_image(image_path, lat=None, lon=None, n_draws=None, sigma=None):
    """Classify one crop (or preview). lat/lon select the regional head.
    Returns None on failure."""
    try:
        if config.MODEL_KIND == "csiro":
            return _analyze_csiro_sti(image_path, n_draws, sigma)
        raw = preprocess_raw(image_path)
        if raw is None:
            return None
        name = choose_head(lat, lon)
        head, th = _head(name)
        score = float(head.predict(_features(raw[None].astype(np.float32)), verbose=0).ravel()[0])
        is_oil = score > th
        return {
            "is_oil": is_oil,
            "confidence": round((score if is_oil else 1.0 - score) * 100, 2),
            "raw_score": score,
            "score_std": 0.0,
            "scores": [round(score, 6)],
            "threshold_used": th,
            "model": name,
            "sigma": None,
            "n_draws": 1,
        }
    except Exception as e:
        print(f"[ML] Prediction error: {e}", flush=True)
        return None


def _analyze_csiro_sti(image_path, n_draws=None, sigma=None):
    """Original behaviour: full CSIRO model on STI-processed draws."""
    model = load_ai_model()
    if not model:
        return None
    from tensorflow.keras.applications.efficientnet import preprocess_input

    batch = sti_batch(image_path, n_draws=n_draws, sigma=sigma)
    if batch is None:
        return None
    scores = model.predict(preprocess_input(batch), verbose=0).ravel().astype(float)
    raw_score = float(np.mean(scores))
    is_oil = raw_score > OPTIMAL_THRESHOLD
    return {
        "is_oil": is_oil,
        "confidence": round((raw_score if is_oil else 1.0 - raw_score) * 100, 2),
        "raw_score": raw_score,
        "score_std": float(np.std(scores)),
        "scores": [round(s, 6) for s in scores.tolist()],
        "threshold_used": OPTIMAL_THRESHOLD,
        "model": "csiro_sti",
        "sigma": config.STI_SIGMA if sigma is None else sigma,
        "n_draws": len(scores),
    }
