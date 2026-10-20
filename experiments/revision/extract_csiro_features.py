"""Frozen-backbone features for every CSIRO chip, for EfficientNet-B4 and -B0.

Used to answer reviewer questions on (i) acquisition-level leakage in the CSIRO
validation split and (ii) the B0-versus-B4 choice. Both backbones are frozen
ImageNet networks, as in the deployed TARS model, so a head trained on these
features reproduces the deployed training setup without augmentation of the
backbone input other than a horizontal flip.

  python extract_csiro_features.py b4   -> csiro_features_b4.npz
  python extract_csiro_features.py b0   -> csiro_features_b0.npz

B4 uses the deployed model's own backbone (layers[:2] of the saved model) at
380 px; B0 uses ImageNet weights at its native 224 px. Each file holds the
pooled features of the original chip and of its horizontal flip, the chip
metadata, and (B4 only) the deployed model's score for the original chip.
"""
import csv, glob, os, re, sys, time
import cv2, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
arch = sys.argv[1]
sys.path.insert(0, REPO)
os.chdir(REPO)
import tensorflow as tf  # noqa: E402

files = sorted(glob.glob("archive/data/S1SAR_UnBalanced_400by400_Class_*/**/*.jpg", recursive=True))

if arch == "b4":
    import ml_logic
    model = ml_logic.load_ai_model()
    backbone = tf.keras.Sequential(model.layers[:2])
    head = tf.keras.Sequential(model.layers[2:])
    size = 380
else:
    base = tf.keras.applications.EfficientNetB0(weights="imagenet", include_top=False,
                                                pooling="avg", input_shape=(224, 224, 3))
    backbone, head, size = base, None, 224

meta, F, Ff, S = [], [], [], []
t = time.time()
for i in range(0, len(files), 16):
    chunk = files[i:i + 16]
    imgs = []
    for f in chunk:
        g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        imgs.append(cv2.cvtColor(cv2.resize(g, (size, size)), cv2.COLOR_GRAY2BGR))
        name = os.path.basename(f)
        p = name.split("_")
        meta.append((name, int(re.search(r"Class_([01])", f).group(1)),
                     re.search(r"_([A-Za-z]+)_cls_", name).group(1), p[4], int(p[1]), int(p[2]),
                     float(np.mean(g > 250))))
    x = np.stack(imgs).astype(np.float32)          # EfficientNet rescales internally
    x = np.concatenate([x, x[:, :, ::-1]])
    f = backbone.predict(x, verbose=0)
    n = len(chunk)
    F.append(f[:n]); Ff.append(f[n:])
    if head is not None:
        S.append(head.predict(f[:n], verbose=0).ravel())
    if i % 640 == 0:
        print(arch, i, len(files), f"{time.time() - t:.0f}s", flush=True)

m = np.array(meta, dtype=object)
np.savez_compressed(os.path.join(HERE, f"csiro_features_{arch}.npz"),
                    feats=np.concatenate(F), feats_flip=np.concatenate(Ff),
                    deployed_score=np.concatenate(S) if S else np.array([]),
                    fname=m[:, 0].astype(str), label=m[:, 1].astype(int), region=m[:, 2].astype(str),
                    image_id=m[:, 3].astype(str), x=m[:, 4].astype(int), y=m[:, 5].astype(int),
                    white_frac=m[:, 6].astype(float))
print("DONE", arch, f"{time.time() - t:.0f}s", flush=True)
