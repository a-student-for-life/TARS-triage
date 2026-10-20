"""
evaluate_model.py
-----------------
Runs the CSIRO dataset through both preprocessing paths and computes a
full set of classification metrics.

# NOTE: Full-corpus evaluation used as validation split indices
# were not persisted during training. True held-out performance
# is reported via training logs (val_auc=0.9913 at epoch 11).

Expected folder layout (actual on this machine):
    VAL_DIR/
        S1SAR_UnBalanced_400by400_Class_0/
            0/      <-- Class 0: Clean Sea  (images here)
        S1SAR_UnBalanced_400by400_Class_1/
            1/      <-- Class 1: Oil Spill  (images here)

discover_images() walks this two-level structure: it descends into each
top-level subfolder, then into each numeric subfolder, and infers the
label from the numeric folder name (0 or 1).
Set VAL_DIR below to point at your data root before running.

Outputs
-------
    confusion_matrix_raw.png      -- CM for the RAW (no preprocessing) path
    confusion_matrix_tars.png     -- CM for the TARS (full speckle pipeline) path
    roc_curve_comparison.png      -- Both ROC curves on one plot
    evaluation_results.csv        -- Per-image scores and binary predictions
"""

import os
# Force CPU – avoids GPU driver issues during batch evaluation
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import sys
# ml_logic.py uses emoji in print() calls; reconfigure stdout to UTF-8
# before importing it so Windows cp1252 consoles don't crash.
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import csv
import time

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend; safe on any machine
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from tensorflow.keras.applications.efficientnet import preprocess_input

# ---------------------------------------------------------------------------
# CONFIGURATION – update VAL_DIR before running
# ---------------------------------------------------------------------------

# TODO: Update VAL_DIR to your local data directory before running.
VAL_DIR = os.getenv("TARS_CSIRO_DIR", "path/to/csiro/data")

# Must match ml_logic.py
IMG_SIZE  = 380
THRESHOLD = 0.75

# Reproducible speckle noise across runs
SEED = 42
np.random.seed(SEED)

SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}

# ---------------------------------------------------------------------------
# Model loading – reuses ml_logic's lazy singleton so weights load once
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_logic import load_ai_model


# ---------------------------------------------------------------------------
# Preprocessing paths
# ---------------------------------------------------------------------------

def preprocess_raw(img_path: str) -> np.ndarray | None:
    """
    RAW (control) path: load as BGR colour, resize, apply EfficientNet
    channel normalisation.  No domain adaptation whatsoever.
    Matches the 'Stage 1 Baseline' in run_abalation.py.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = preprocess_input(img.astype(np.float32))
    return img


def preprocess_tars(img_path: str) -> np.ndarray | None:
    """
    TARS V3 path: multiplicative speckle injection + Gaussian blur +
    CLAHE + brightness shift, then EfficientNet normalisation.
    Mirrors preprocess_style() in ml_logic.py exactly, but without
    the debug-file side-effect so it is safe to call 1 000+ times.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Step 1 – Multiplicative speckle
    img_float = img.astype(np.float32)
    noise     = np.random.normal(0, 0.5, img.shape)
    img       = np.clip(img_float + (img_float * noise), 0, 255).astype(np.uint8)

    # Step 2 – Gentle Gaussian blur
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Step 3 – CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img   = clahe.apply(img)

    # Step 4 – Brightness alignment (+70)
    img = np.clip(img.astype(np.int16) + 70, 0, 255).astype(np.uint8)

    # Convert to 3-channel and normalise for EfficientNet
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_rgb = preprocess_input(img_rgb.astype(np.float32))
    return img_rgb


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

def infer_label(folder_name: str) -> int | None:
    """
    Map a subfolder name to an integer class label.
      0 = Clean Sea
      1 = Oil Spill
    Returns None when the folder name is unrecognisable.
    """
    name = folder_name.lower().strip()

    # Numeric convention used by CSIRO split
    if name == '0':
        return 0
    if name == '1':
        return 1

    # Named-folder conventions
    oil_keywords   = {'oil', 'spill', 'leak', 'slick', 'contamina'}
    clean_keywords = {'clean', 'no_spill', 'nospill', 'sea', 'water',
                      'clear', 'background', 'negative'}

    if any(k in name for k in oil_keywords):
        return 1
    if any(k in name for k in clean_keywords):
        return 0

    return None


def discover_images(val_dir: str) -> list[tuple[str, int]]:
    """
    Walk the two-level CSIRO structure:
        VAL_DIR/
            S1SAR_UnBalanced_400by400_Class_0/   (or any name)
                0/                               ← images live here, label = 0
            S1SAR_UnBalanced_400by400_Class_1/
                1/                               ← images live here, label = 1

    Algorithm:
      1. Iterate top-level subdirs of VAL_DIR  (the "Class_X" folders).
      2. Within each, iterate their subdirs     (the "0" / "1" folders).
      3. Infer the label from the inner folder name via infer_label().
      4. Collect all supported image files from the inner folder.

    Returns a list of (absolute_image_path, label) tuples.
    """
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(
            f"Validation directory not found: {val_dir}\n"
            "Please update VAL_DIR at the top of evaluate_model.py."
        )

    samples: list[tuple[str, int]] = []
    skipped_inner: list[str] = []

    # Level 1 – e.g. S1SAR_UnBalanced_400by400_Class_0
    for top_entry in sorted(os.scandir(val_dir), key=lambda e: e.name):
        if not top_entry.is_dir():
            continue

        # Level 2 – e.g. "0" or "1"
        for inner_entry in sorted(os.scandir(top_entry.path), key=lambda e: e.name):
            if not inner_entry.is_dir():
                continue

            label = infer_label(inner_entry.name)
            if label is None:
                skipped_inner.append(f"{top_entry.name}/{inner_entry.name}")
                continue

            for fname in sorted(os.listdir(inner_entry.path)):
                if os.path.splitext(fname)[1].lower() in SUPPORTED_EXTS:
                    samples.append((os.path.join(inner_entry.path, fname), label))

    if skipped_inner:
        print(f"[WARN] Skipped unrecognised inner folders: {skipped_inner}")

    return samples


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, preprocessed: np.ndarray) -> float:
    """Return the raw sigmoid score (0.0–1.0) for a single preprocessed image."""
    batch = np.expand_dims(preprocessed, axis=0)
    return float(model.predict(batch, verbose=0)[0][0])


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def print_metrics(path_name: str, y_true, y_pred_bin, y_scores) -> None:
    print(f"\n{'='*60}")
    print(f"  {path_name} PATH METRICS")
    print(f"{'='*60}")

    auc_score = roc_auc_score(y_true, y_scores)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_bin, average='binary', zero_division=0
    )

    print(f"  AUC-ROC   : {auc_score:.4f}")
    print(f"  Precision : {p:.4f}")
    print(f"  Recall    : {r:.4f}")
    print(f"  F1-score  : {f1:.4f}")
    print()
    print(classification_report(
        y_true, y_pred_bin,
        target_names=['Clean Sea (0)', 'Oil Spill (1)'],
        zero_division=0
    ))


def save_confusion_matrix(y_true, y_pred_bin, path_name: str, out_path: str) -> None:
    cm  = confusion_matrix(y_true, y_pred_bin)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Clean Sea', 'Oil Spill']
    )
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(
        f"Confusion Matrix – {path_name} path\n"
        f"(threshold = {THRESHOLD}  |  n = {len(y_true)})",
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def save_roc_comparison(y_true, raw_scores, tars_scores, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))

    for scores, label, colour in [
        (raw_scores,  'RAW (no adaptation)',      '#e74c3c'),
        (tars_scores, 'TARS (speckle pipeline)', '#2ecc71'),
    ]:
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc_val = roc_auc_score(y_true, scores)
        ax.plot(fpr, tpr, color=colour, lw=2,
                label=f'{label}  (AUC = {auc_val:.4f})')

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.6, label='Random classifier')

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(
        f'ROC Curve Comparison – CSIRO Validation Split\n'
        f'(n = {len(y_true)}, threshold = {THRESHOLD})',
        fontsize=12
    )
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

CSV_HEADER = [
    'filename',
    'true_label',
    'true_class',
    'raw_pred_score',
    'raw_pred_binary',
    'raw_pred_class',
    'tars_pred_score',
    'tars_pred_binary',
    'tars_pred_class',
]

CLASS_NAMES = {0: 'Clean Sea', 1: 'Oil Spill'}


def save_csv(records: list[dict], out_path: str) -> None:
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(records)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "="*60)
    print("  OilSpillWatchdog – Model Evaluation")
    print("  CSIRO Validation Split")
    print("="*60)

    # 1. Discover images ---------------------------------------------------
    print(f"\n[1/5] Scanning validation directory:\n      {VAL_DIR}")
    samples = discover_images(VAL_DIR)
    if not samples:
        print("ERROR: No images found.  Check VAL_DIR and folder structure.")
        return

    class_counts = {0: sum(1 for _, l in samples if l == 0),
                    1: sum(1 for _, l in samples if l == 1)}
    print(f"      Found {len(samples)} images  "
          f"(Clean Sea: {class_counts[0]}, Oil Spill: {class_counts[1]})")

    # 2. Load model once ---------------------------------------------------
    print("\n[2/5] Loading EfficientNetB4 model …")
    model = load_ai_model()
    if model is None:
        print("ERROR: Model failed to load. Check weights file path in ml_logic.py.")
        return
    print("      Model ready.")

    # 3. Run inference on every image --------------------------------------
    print(f"\n[3/5] Running inference ({len(samples)} images × 2 paths) …")

    y_true:      list[int]   = []
    raw_scores:  list[float] = []
    tars_scores: list[float] = []
    records:     list[dict]  = []

    t_start = time.time()
    for idx, (img_path, label) in enumerate(samples, 1):

        # RAW path
        raw_arr = preprocess_raw(img_path)
        if raw_arr is None:
            print(f"  [SKIP] Could not load (RAW): {img_path}")
            continue
        raw_score = run_inference(model, raw_arr)

        # TARS path (fresh random seed per image keeps speckle meaningful
        # while still being deterministic across full evaluation runs)
        np.random.seed(SEED + idx)
        tars_arr = preprocess_tars(img_path)
        if tars_arr is None:
            print(f"  [SKIP] Could not load (TARS): {img_path}")
            continue
        tars_score = run_inference(model, tars_arr)

        y_true.append(label)
        raw_scores.append(raw_score)
        tars_scores.append(tars_score)

        raw_bin  = int(raw_score  > THRESHOLD)
        tars_bin = int(tars_score > THRESHOLD)

        records.append({
            'filename'        : os.path.basename(img_path),
            'true_label'      : label,
            'true_class'      : CLASS_NAMES[label],
            'raw_pred_score'  : f'{raw_score:.6f}',
            'raw_pred_binary' : raw_bin,
            'raw_pred_class'  : CLASS_NAMES[raw_bin],
            'tars_pred_score' : f'{tars_score:.6f}',
            'tars_pred_binary': tars_bin,
            'tars_pred_class' : CLASS_NAMES[tars_bin],
        })

        if idx % 100 == 0 or idx == len(samples):
            elapsed = time.time() - t_start
            print(f"      {idx}/{len(samples)}  ({elapsed:.0f}s elapsed)")

    print(f"      Done. Processed {len(y_true)} images successfully.")

    y_true_arr      = np.array(y_true)
    raw_scores_arr  = np.array(raw_scores)
    tars_scores_arr = np.array(tars_scores)
    raw_bin_arr     = (raw_scores_arr  > THRESHOLD).astype(int)
    tars_bin_arr    = (tars_scores_arr > THRESHOLD).astype(int)

    # 4. Print metrics to console -----------------------------------------
    print("\n[4/5] Computing metrics …")
    print_metrics("RAW",  y_true_arr, raw_bin_arr,  raw_scores_arr)
    print_metrics("TARS", y_true_arr, tars_bin_arr, tars_scores_arr)

    # Ablation delta summary
    raw_auc  = roc_auc_score(y_true_arr, raw_scores_arr)
    tars_auc = roc_auc_score(y_true_arr, tars_scores_arr)
    _, _, raw_f1,  _ = precision_recall_fscore_support(
        y_true_arr, raw_bin_arr,  average='binary', zero_division=0)
    _, _, tars_f1, _ = precision_recall_fscore_support(
        y_true_arr, tars_bin_arr, average='binary', zero_division=0)

    print(f"\n{'='*60}")
    print("  ABLATION SUMMARY  (TARS vs RAW)")
    print(f"{'='*60}")
    print(f"  AUC-ROC delta : {tars_auc - raw_auc:+.4f}  "
          f"({raw_auc:.4f} → {tars_auc:.4f})")
    print(f"  F1-score delta: {tars_f1 - raw_f1:+.4f}  "
          f"({raw_f1:.4f} → {tars_f1:.4f})")

    fp_prevented = int(np.sum((raw_bin_arr == 1) & (tars_bin_arr == 0) & (y_true_arr == 0)))
    fn_introduced = int(np.sum((raw_bin_arr == 0) & (tars_bin_arr == 1) & (y_true_arr == 1)))
    print(f"  False positives prevented by TARS : {fp_prevented}")
    print(f"  False negatives introduced by TARS: {fn_introduced}")
    print(f"{'='*60}\n")

    # 5. Save artefacts ----------------------------------------------------
    print("[5/5] Saving output files …")

    out_dir = os.path.dirname(os.path.abspath(__file__))

    save_confusion_matrix(
        y_true_arr, raw_bin_arr,
        path_name='RAW',
        out_path=os.path.join(out_dir, 'confusion_matrix_raw.png')
    )
    save_confusion_matrix(
        y_true_arr, tars_bin_arr,
        path_name='TARS',
        out_path=os.path.join(out_dir, 'confusion_matrix_tars.png')
    )
    save_roc_comparison(
        y_true_arr, raw_scores_arr, tars_scores_arr,
        out_path=os.path.join(out_dir, 'roc_curve_comparison.png')
    )
    save_csv(
        records,
        out_path=os.path.join(out_dir, 'evaluation_results.csv')
    )

    print("\nEvaluation complete.")


if __name__ == '__main__':
    main()
