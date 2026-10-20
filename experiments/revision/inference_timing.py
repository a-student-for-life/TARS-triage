"""Single-image inference time and size of the full B4 and B0 classifiers (CPU).

Each model is backbone + the TARS head; one image per call (as in the
pipeline), 10 warm-up calls, then 100 timed calls. Run on an otherwise idle machine.

  python inference_timing.py  -> inference_timing.json
"""
import json, os, platform, time
import numpy as np, tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
HERE = os.path.dirname(os.path.abspath(__file__))
L = tf.keras.layers


def full(base_fn, size):
    base = base_fn(weights="imagenet", include_top=False, input_shape=(size, size, 3))
    return tf.keras.Sequential([base, L.GlobalAveragePooling2D(), L.BatchNormalization(), L.Dropout(0.4),
                                L.Dense(512, activation="relu"), L.BatchNormalization(), L.Dropout(0.5),
                                L.Dense(1, activation="sigmoid")]), base


out = {"cpu": platform.processor() or platform.machine(), "threads": os.cpu_count(), "tf": tf.__version__}
for name, fn, size in [("b4", tf.keras.applications.EfficientNetB4, 380), ("b0", tf.keras.applications.EfficientNetB0, 224)]:
    model, base = full(fn, size)
    x = np.random.default_rng(0).uniform(0, 255, (1, size, size, 3)).astype(np.float32)
    f = tf.function(lambda t: model(t, training=False))
    for _ in range(10):
        f(x)
    ts = []
    for _ in range(100):
        t = time.perf_counter(); f(x).numpy(); ts.append(time.perf_counter() - t)
    ts = np.array(ts) * 1000
    out[name] = dict(input_px=size, params_total=int(model.count_params()), params_backbone=int(base.count_params()),
                     ms_median=float(np.median(ts)), ms_p90=float(np.percentile(ts, 90)))
    print(name, out[name], flush=True)
json.dump(out, open(os.path.join(HERE, "inference_timing.json"), "w"), indent=1)
print("DONE")
