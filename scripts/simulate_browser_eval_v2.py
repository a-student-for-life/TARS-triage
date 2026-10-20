"""
simulate_browser_eval_v2.py
---------------------------
Identical to simulate_browser_eval.py with one addition:

After JPEG compression/decoding, a brightness normalisation step is applied
to the decompressed grayscale image BEFORE both RAW and TARS paths.

Normalisation:
  mean_val = img.mean()
  if mean_val >= 1:
      scale = 50.0 / mean_val
      img   = clip(img * scale, 0, 255).astype(uint8)

Purpose: reproduces the radiometric characteristics of real NASA ASF browse
products, which are systematically underexposed (mean intensity ~40–60).
This ensures the TARS brightness correction (+70) operates in the correct
radiometric regime during inference.

Output files (v2 suffix to avoid overwriting v1 results):
  browser_eval_v2_results.csv
  browser_confusion_raw_v2.png
  browser_confusion_tars_v2.png
  browser_roc_comparison_v2.png

NOTE: preprocess_style() in ml_logic.py writes debug_ai_input_v3.jpg as
a side-effect on every call. That file is overwritten silently 5,630
times during this run; it is harmless but expected.
"""

import os
# Force CPU to match the deployment environment
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import sys
# ml_logic.py uses emoji in print() — reconfigure stdout before importing it
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import csv
import time
import tempfile

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    accuracy_score,
)
from tensorflow.keras.applications.efficientnet import preprocess_input

# ---------------------------------------------------------------------------
# Reuse model loader, constants, and the TARS preprocessing function
# from the existing project code — no reimplementation.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_logic import load_ai_model, IMG_SIZE, OPTIMAL_THRESHOLD, preprocess_style

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# TODO: Update DATA_DIR to your local data directory before running.
DATA_DIR = (
    os.getenv("TARS_CSIRO_DIR", "path/to/csiro/data")
)

THRESHOLD     = OPTIMAL_THRESHOLD   # 0.75
BATCH_SIZE    = 32
JPEG_QUALITY  = 30                  # Simulates browser-preview compression
SEED          = 42
SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}

# Target mean intensity that approximates real NASA browse products (~40–60)
BRIGHTNESS_TARGET = 50.0

np.random.seed(SEED)

# Single temp file reused for all TARS calls (preprocess_style needs a path)
_TMP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '_browser_tmp_v2.jpg')

# ---------------------------------------------------------------------------
# Dataset discovery  (mirrors evaluate_model.py logic; not imported to
# avoid that module's stdout side-effects)
# ---------------------------------------------------------------------------

def _infer_label(folder_name: str) -> int | None:
    name = folder_name.lower().strip()
    if name == '0':
        return 0
    if name == '1':
        return 1
    oil_kw   = {'oil', 'spill', 'leak', 'slick', 'contamina'}
    clean_kw = {'clean', 'no_spill', 'nospill', 'sea', 'water',
                'clear', 'background', 'negative'}
    if any(k in name for k in oil_kw):
        return 1
    if any(k in name for k in clean_kw):
        return 0
    return None


def discover_images(data_dir: str) -> list[tuple[str, int]]:
    """
    Walk the two-level CSIRO structure:
        data_dir/
            *Class_0*/
                0/   <- images, label 0
            *Class_1*/
                1/   <- images, label 1
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            "Update DATA_DIR at the top of simulate_browser_eval_v2.py."
        )
    samples: list[tuple[str, int]] = []
    for top in sorted(os.scandir(data_dir), key=lambda e: e.name):
        if not top.is_dir():
            continue
        for inner in sorted(os.scandir(top.path), key=lambda e: e.name):
            if not inner.is_dir():
                continue
            label = _infer_label(inner.name)
            if label is None:
                continue
            for fname in sorted(os.listdir(inner.path)):
                if os.path.splitext(fname)[1].lower() in SUPPORTED_EXTS:
                    samples.append((os.path.join(inner.path, fname), label))
    return samples


# ---------------------------------------------------------------------------
# Step 1 – Browser simulation  (V2: adds brightness normalisation)
# ---------------------------------------------------------------------------

def simulate_browser(img_path: str) -> np.ndarray | None:
    """
    Load image → grayscale → JPEG compress at Q30 → decode →
    brightness normalise to mean ≈ BRIGHTNESS_TARGET (50).

    The normalisation step is the only difference from v1.  It reproduces
    the systematic underexposure of real NASA ASF browse products so that
    the TARS brightness-shift (+70) operates in its intended regime.

    Returns a single-channel uint8 array (H×W), or None on failure.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # JPEG compress / decompress (quality=30)
    ok, buf = cv2.imencode('.jpg', img,
                           [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        return None
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)

    # ── V2 ADDITION: brightness normalisation ────────────────────────────
    mean_val = img.mean()
    if mean_val >= 1.0:                        # skip if image is essentially black
        scale = BRIGHTNESS_TARGET / mean_val
        img   = np.clip(img.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    # ─────────────────────────────────────────────────────────────────────

    return img


# ---------------------------------------------------------------------------
# Step 2 – Preprocessing paths  (unchanged from v1)
# ---------------------------------------------------------------------------

def path_a_raw(compressed: np.ndarray) -> np.ndarray:
    """
    RAW baseline: resize compressed grayscale → 3-channel → preprocess_input.
    No domain adaptation whatsoever.
    """
    img = cv2.resize(compressed, (IMG_SIZE, IMG_SIZE))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return preprocess_input(img_bgr.astype(np.float32))


def path_b_tars(compressed: np.ndarray, seed_offset: int) -> np.ndarray | None:
    """
    TARS path: write compressed grayscale to temp file, then call
    preprocess_style() from ml_logic.py (exact project implementation).
    Deterministic per image via seed_offset.
    """
    # Save compressed (and brightness-normalised) grayscale to the shared temp file
    cv2.imwrite(_TMP_PATH, compressed)

    # Deterministic speckle per image across runs
    np.random.seed(SEED + seed_offset)

    # Reuse the exact TARS function from ml_logic.py
    processed = preprocess_style(_TMP_PATH, target_size=(IMG_SIZE, IMG_SIZE))
    if processed is None:
        return None

    # preprocess_style returns BGR uint8; apply EfficientNet normalization
    return preprocess_input(processed.astype(np.float32))


# ---------------------------------------------------------------------------
# Step 3 – Batched inference  (unchanged from v1)
# ---------------------------------------------------------------------------

def run_batched_inference(
    model,
    samples: list[tuple[str, int]],
) -> tuple[list[int], list[float], list[float]]:
    """
    Returns (y_true, scores_raw, scores_tars).
    Processes images in mini-batches of BATCH_SIZE.
    """
    y_true:      list[int]   = []
    scores_raw:  list[float] = []
    scores_tars: list[float] = []

    batch_labels:    list[int]        = []
    batch_raw_arrs:  list[np.ndarray] = []
    batch_tars_arrs: list[np.ndarray] = []

    def flush_batch():
        if not batch_raw_arrs:
            return
        raw_batch  = np.stack(batch_raw_arrs,  axis=0)
        tars_batch = np.stack(batch_tars_arrs, axis=0)
        raw_preds  = model.predict(raw_batch,  verbose=0).flatten().tolist()
        tars_preds = model.predict(tars_batch, verbose=0).flatten().tolist()
        y_true.extend(batch_labels)
        scores_raw.extend(raw_preds)
        scores_tars.extend(tars_preds)
        batch_labels.clear()
        batch_raw_arrs.clear()
        batch_tars_arrs.clear()

    t_start = time.time()
    n = len(samples)
    skipped = 0

    for idx, (img_path, label) in enumerate(samples, 1):
        # Step 1: browser simulation (with brightness normalisation in v2)
        compressed = simulate_browser(img_path)
        if compressed is None:
            skipped += 1
            continue

        # Step 2: both paths
        arr_raw  = path_a_raw(compressed)
        arr_tars = path_b_tars(compressed, seed_offset=idx)
        if arr_tars is None:
            skipped += 1
            continue

        batch_labels.append(label)
        batch_raw_arrs.append(arr_raw)
        batch_tars_arrs.append(arr_tars)

        # Flush when batch is full
        if len(batch_labels) == BATCH_SIZE:
            flush_batch()

        if idx % 500 == 0 or idx == n:
            elapsed = time.time() - t_start
            print(f"      {idx}/{n}  ({elapsed:.0f}s elapsed,  skipped={skipped})")

    flush_batch()  # remainder

    return y_true, scores_raw, scores_tars


# ---------------------------------------------------------------------------
# Step 4 / 6 – Metrics helpers  (unchanged from v1)
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict:
    y_bin = (scores > threshold).astype(int)
    fp    = int(np.sum((y_bin == 1) & (y_true == 0)))
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_bin, average='binary', zero_division=0
    )
    return {
        'auc'       : roc_auc_score(y_true, scores),
        'precision' : p,
        'recall'    : r,
        'f1'        : f1,
        'accuracy'  : accuracy_score(y_true, y_bin),
        'fp'        : fp,
        'y_bin'     : y_bin,
    }


def print_summary_table(metrics_raw: dict, metrics_tars: dict) -> None:
    header = f"\n{'Path':<8} {'AUC':>7} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Accuracy':>10} {'FP (Clean→Oil)':>16}"
    sep    = '-' * len(header)
    print(sep)
    print(header)
    print(sep)
    for name, m in [('RAW', metrics_raw), ('TARS', metrics_tars)]:
        print(
            f"{name:<8} "
            f"{m['auc']:>7.4f} "
            f"{m['precision']:>10.4f} "
            f"{m['recall']:>8.4f} "
            f"{m['f1']:>8.4f} "
            f"{m['accuracy']:>10.4f} "
            f"{m['fp']:>16}"
        )
    print(sep)
    print()

    # Delta row
    print("  TARS vs RAW deltas:")
    for key in ('auc', 'precision', 'recall', 'f1', 'accuracy'):
        delta = metrics_tars[key] - metrics_raw[key]
        sign  = '+' if delta >= 0 else ''
        print(f"    {key:<12}: {sign}{delta:.4f}")
    fp_delta = metrics_tars['fp'] - metrics_raw['fp']
    print(f"    {'fp':<12}: {fp_delta:+d}")
    print()


# ---------------------------------------------------------------------------
# Step 5 – Saving output artefacts  (v2 filenames)
# ---------------------------------------------------------------------------

def save_confusion_matrix(
    y_true: np.ndarray,
    y_bin:  np.ndarray,
    path_name: str,
    out_path:  str,
) -> None:
    cm = confusion_matrix(y_true, y_bin)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Clean Sea', 'Oil Spill']
    ).plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(
        f"Confusion Matrix – {path_name} path (browser-compressed + normalised)\n"
        f"JPEG Q{JPEG_QUALITY}  |  brightness target µ≈{BRIGHTNESS_TARGET:.0f}"
        f"  |  threshold = {THRESHOLD}  |  n = {len(y_true)}",
        fontsize=9
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def save_roc_comparison(
    y_true:      np.ndarray,
    scores_raw:  np.ndarray,
    scores_tars: np.ndarray,
    out_path:    str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for scores, label, colour in [
        (scores_raw,  'RAW (no adaptation)',       '#e74c3c'),
        (scores_tars, 'TARS (speckle pipeline)',   '#2ecc71'),
    ]:
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc_val = roc_auc_score(y_true, scores)
        ax.plot(fpr, tpr, color=colour, lw=2,
                label=f'{label}  (AUC = {auc_val:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.6, label='Random classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(
        f'ROC Curve – Browser-Compressed + Brightness-Normalised (JPEG Q{JPEG_QUALITY})\n'
        f'EfficientNet-B4  |  µ≈{BRIGHTNESS_TARGET:.0f}  |  n = {len(y_true)}'
        f'  |  threshold = {THRESHOLD}',
        fontsize=10
    )
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def save_csv(
    y_true:      list[int],
    scores_raw:  list[float],
    scores_tars: list[float],
    out_path:    str,
) -> None:
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['y_true', 'score_raw', 'score_tars'])
        for yt, sr, st in zip(y_true, scores_raw, scores_tars):
            writer.writerow([yt, f'{sr:.6f}', f'{st:.6f}'])
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    out_dir = os.path.dirname(os.path.abspath(__file__))

    print('\n' + '=' * 65)
    print('  OilSpillWatchdog – Browser-Compressed Evaluation  (V2)')
    print(f'  JPEG quality = {JPEG_QUALITY}  |  brightness target µ≈{BRIGHTNESS_TARGET:.0f}  |  threshold = {THRESHOLD}')
    print('=' * 65)

    # 1. Discover images ------------------------------------------------
    print(f'\n[1/5] Scanning data directory:\n      {DATA_DIR}')
    samples = discover_images(DATA_DIR)
    if not samples:
        print('ERROR: No images found. Check DATA_DIR.')
        return
    n0 = sum(1 for _, l in samples if l == 0)
    n1 = sum(1 for _, l in samples if l == 1)
    print(f'      Found {len(samples)} images  (Clean Sea: {n0}, Oil Spill: {n1})')

    # 2. Load model once ------------------------------------------------
    print('\n[2/5] Loading EfficientNetB4 …')
    model = load_ai_model()
    if model is None:
        print('ERROR: Model failed to load.')
        return
    print('      Model ready.')

    # 3. Run inference --------------------------------------------------
    print(f'\n[3/5] Browser simulation + brightness normalisation + inference'
          f' ({len(samples)} images × 2 paths) …')
    y_true, scores_raw, scores_tars = run_batched_inference(model, samples)
    print(f'      Done. Processed {len(y_true)} images successfully.')

    y_true_arr      = np.array(y_true)
    scores_raw_arr  = np.array(scores_raw)
    scores_tars_arr = np.array(scores_tars)

    # 4. Metrics --------------------------------------------------------
    print('\n[4/5] Computing metrics …')
    m_raw  = compute_metrics(y_true_arr, scores_raw_arr,  THRESHOLD)
    m_tars = compute_metrics(y_true_arr, scores_tars_arr, THRESHOLD)

    print('\n  RAW path — full report:')
    print(classification_report(
        y_true_arr, m_raw['y_bin'],
        target_names=['Clean Sea (0)', 'Oil Spill (1)'],
        zero_division=0
    ))
    print('  TARS path — full report:')
    print(classification_report(
        y_true_arr, m_tars['y_bin'],
        target_names=['Clean Sea (0)', 'Oil Spill (1)'],
        zero_division=0
    ))

    print_summary_table(m_raw, m_tars)

    # 5. Save artefacts -------------------------------------------------
    print('[5/5] Saving output files …')

    save_csv(
        y_true, scores_raw, scores_tars,
        out_path=os.path.join(out_dir, 'browser_eval_v2_results.csv')
    )
    save_confusion_matrix(
        y_true_arr, m_raw['y_bin'],
        path_name='RAW',
        out_path=os.path.join(out_dir, 'browser_confusion_raw_v2.png')
    )
    save_confusion_matrix(
        y_true_arr, m_tars['y_bin'],
        path_name='TARS',
        out_path=os.path.join(out_dir, 'browser_confusion_tars_v2.png')
    )
    save_roc_comparison(
        y_true_arr, scores_raw_arr, scores_tars_arr,
        out_path=os.path.join(out_dir, 'browser_roc_comparison_v2.png')
    )

    # Clean up temp file
    if os.path.exists(_TMP_PATH):
        os.remove(_TMP_PATH)

    print('\nBrowser evaluation V2 complete.')


if __name__ == '__main__':
    main()
