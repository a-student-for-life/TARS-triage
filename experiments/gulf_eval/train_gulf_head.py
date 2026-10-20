"""Deployable Gulf head: trained on ALL GulfOilSAR crops (2021 + 2022).

Evaluation numbers come from multi_region.py (trained on 2022, tested on 2021).
This script only produces weights for the pipeline, choosing epochs and the 5%
false-alarm threshold by 5-fold CV grouped by scene, as for browse_head_v1.
Output: ../../models/gulf_head_v1.weights.h5 and gulf_head_v1.json
"""
import json, os
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.metrics import roc_curve
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
g = pd.read_csv(os.path.join(HERE, "gulf_scores.csv")); X = np.load(os.path.join(HERE, "gulf_features.npz"))["feats"]
y = g.label.values.astype(np.float32); groups = g["product"].values

def head():
    L = tf.keras.layers
    m = tf.keras.Sequential([tf.keras.Input((1792,)), L.BatchNormalization(), L.Dropout(0.4), L.Dense(512, activation="relu"),
                             L.BatchNormalization(), L.Dropout(0.5), L.Dense(1, activation="sigmoid")])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC(name="auc")])
    return m

def cw(yy):
    n1 = yy.sum(); n0 = len(yy) - n1
    return {0: len(yy) / (2 * n0), 1: len(yy) / (2 * n1)}

epochs, ths = [], []
for k, (tr, va) in enumerate(GroupKFold(5).split(X, y, groups)):
    tf.keras.utils.set_random_seed(42 + k); m = head()
    h = m.fit(X[tr], y[tr], validation_data=(X[va], y[va]), epochs=60, batch_size=64, verbose=0, class_weight=cw(y[tr]),
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=8, restore_best_weights=True)])
    epochs.append(int(np.argmax(h.history["val_auc"])) + 1)
    fpr, tpr, th = roc_curve(y[va], m.predict(X[va], verbose=0).ravel()); ths.append(float(th[np.where(fpr <= 0.05)[0][-1]]))
tf.keras.utils.set_random_seed(42); m = head()
m.fit(X, y, epochs=int(np.median(epochs)), batch_size=64, verbose=0, class_weight=cw(y))
m.save_weights(os.path.join(REPO, "models", "gulf_head_v1.weights.h5"))
info = dict(threshold=round(float(np.median(ths)), 3), epochs=int(np.median(epochs)), cv_thresholds=[round(t, 3) for t in ths],
            n_train=int(len(y)), n_oil=int(y.sum()), preprocessing="raw (grayscale, resize to 380; no STI)", crop_km=25.0,
            data="GulfOilSAR 2021-2022 (doi:10.5281/zenodo.23261453) + ordinary-sea crops",
            evaluation="see experiments/gulf_eval/README.md (trained 2022, tested 2021)")
json.dump(info, open(os.path.join(REPO, "models", "gulf_head_v1.json"), "w"), indent=2)
print(info)
