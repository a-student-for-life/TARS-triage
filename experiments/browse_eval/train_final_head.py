"""Train the deployable browse-domain head on ALL crops (oil, look-alikes, ordinary sea).

Evaluation numbers come from train_compare_v2.py (cross-validated). This script
only produces the weights for the live pipeline, using the median epoch count
and median threshold found across those CV folds.
Output: ../../models/browse_head_v1.weights.h5 and browse_head_v1.json
"""
import json, os
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_curve

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
crops = pd.read_csv(os.path.join(HERE, "crops_25km.csv")); zc = np.load(os.path.join(HERE, "features_25km.npz"))
bg = pd.read_csv(os.path.join(HERE, "background_25km.csv")); zb = np.load(os.path.join(HERE, "features_background.npz"))
X = np.concatenate([zc["feats"], zb["feats"]]); Xf = np.concatenate([zc["feats_flip"], zb["feats"]])
y = np.concatenate([crops.label.values, np.zeros(len(bg))]).astype(np.float32)
groups = np.concatenate([crops.day.values, bg.day.values])

def make_head():
    L = tf.keras.layers
    m = tf.keras.Sequential([tf.keras.Input((X.shape[1],)), L.BatchNormalization(), L.Dropout(0.4),
                             L.Dense(512, activation="relu"), L.BatchNormalization(), L.Dropout(0.5),
                             L.Dense(1, activation="sigmoid")])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC(name="auc")])
    return m

def fit(idx_fit, idx_val=None, epochs=60):
    Xtr = np.concatenate([X[idx_fit], Xf[idx_fit]]); ytr = np.concatenate([y[idx_fit], y[idx_fit]])
    n1 = ytr.sum(); n0 = len(ytr) - n1; head = make_head()
    cb = [tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=8, restore_best_weights=True)] if idx_val is not None else []
    h = head.fit(Xtr, ytr, validation_data=(X[idx_val], y[idx_val]) if idx_val is not None else None, epochs=epochs,
                 batch_size=64, verbose=0, class_weight={0: len(ytr) / (2 * n0), 1: len(ytr) / (2 * n1)}, callbacks=cb)
    return head, h

best_epochs, thresholds = [], []
for k, (tr, te) in enumerate(GroupKFold(5).split(X, y, groups)):
    tf.keras.utils.set_random_seed(42 + k)
    days = np.unique(groups[tr]); val_days = np.random.default_rng(42 + k).choice(days, len(days) // 5, replace=False)
    va = tr[np.isin(groups[tr], val_days)]; ft = tr[~np.isin(groups[tr], val_days)]
    head, h = fit(ft, va)
    best_epochs.append(int(np.argmax(h.history["val_auc"])) + 1)
    fpr, tpr, th = roc_curve(y[va], head.predict(X[va], verbose=0).ravel())
    thresholds.append(float(th[np.where(fpr <= 0.05)[0][-1]]))
epochs, threshold = int(np.median(best_epochs)), float(np.median(thresholds))
tf.keras.utils.set_random_seed(42)
head, _ = fit(np.arange(len(y)), None, epochs=epochs)
os.makedirs(os.path.join(REPO, "models"), exist_ok=True)
head.save_weights(os.path.join(REPO, "models", "browse_head_v1.weights.h5"))
info = dict(threshold=round(threshold, 3), epochs=epochs, cv_best_epochs=best_epochs, cv_thresholds=[round(t, 3) for t in thresholds],
            n_train=int(len(y)), n_oil=int(y.sum()), preprocessing="raw (grayscale, resize to 380; no STI)",
            crop_km=25.0, data="ESSD 2019 Eastern Mediterranean (PANGAEA.980773) + ordinary-sea crops",
            evaluation="see experiments/browse_eval/comparison_v2.json (5-fold CV grouped by day)")
json.dump(info, open(os.path.join(REPO, "models", "browse_head_v1.json"), "w"), indent=2)
print(json.dumps(info, indent=2))
