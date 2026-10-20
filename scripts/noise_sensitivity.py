"""
noise_sensitivity.py
--------------------
Evaluates EfficientNetB4 sensitivity to the TARS speckle noise parameter
(sigma_noise) across the full CSIRO SAR corpus using browser-preview
simulation identical to simulate_browser_eval_v2.py.

Browser simulation (per image, computed once and cached):
    grayscale → JPEG Q=30 encode/decode → brightness normalise (mean ≈ 50)

sigma values tested: [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    sigma = 0.0  → CLAHE + brightness +70 only (no speckle injection)
    sigma = 0.5  → production operating point

HOW sigma IS OVERRIDDEN without modifying ml_logic.py
    preprocess_style() calls  np.random.normal(0, noise_sigma, img.shape)
    where noise_sigma=0.5 is a local constant.  This script patches
    numpy.random.normal for the duration of each call via
    unittest.mock.patch, replacing the scale argument with our sigma.
    The patch is torn down immediately after each preprocess_style()
    returns, so no other code is affected.

Outputs:
    noise_sensitivity_results.csv
    noise_sensitivity_plot.png

Do NOT modify any existing files.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   # Force CPU (matches deployment)

import sys
# ml_logic.py uses emoji in print() — reconfigure stdout before import
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from unittest.mock import patch as _mock_patch
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from tensorflow.keras.applications.efficientnet import preprocess_input

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_logic import load_ai_model, IMG_SIZE, OPTIMAL_THRESHOLD, preprocess_style

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# TODO: Update DATA_DIR to your local data directory before running.
DATA_DIR = (
    os.getenv("TARS_CSIRO_DIR", "path/to/csiro/data")
)

THRESHOLD         = OPTIMAL_THRESHOLD       # 0.75
BATCH_SIZE        = 32
SEED              = 42
JPEG_QUALITY      = 30
BRIGHTNESS_TARGET = 50.0
OPERATING_SIGMA   = 0.5
SIGMA_VALUES      = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
SUPPORTED_EXTS    = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}

# Reused temp file for preprocess_style() (requires a file path, not an array)
_TMP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '_noise_sens_tmp.jpg')

# ---------------------------------------------------------------------------
# Dataset discovery  (mirrors simulate_browser_eval_v2.py exactly)
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
                0/   ← clean-sea images, label=0
            *Class_1*/
                1/   ← oil-spill images, label=1
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            "Update DATA_DIR at the top of noise_sensitivity.py."
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
# Browser simulation  (matches simulate_browser_eval_v2.py exactly)
# ---------------------------------------------------------------------------

def simulate_browser(img_path: str) -> np.ndarray | None:
    """
    Grayscale → JPEG Q=30 encode/decode → brightness normalise to mean ≈ 50.
    Returns a single-channel uint8 array (H×W), or None on failure.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    ok, buf = cv2.imencode('.jpg', img,
                           [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        return None
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)

    mean_val = img.mean()
    if mean_val >= 1.0:
        scale = BRIGHTNESS_TARGET / mean_val
        img   = np.clip(img.astype(np.float32) * scale, 0, 255).astype(np.uint8)

    return img


# ---------------------------------------------------------------------------
# TARS pipeline with sigma override
# ---------------------------------------------------------------------------

def _tars_with_sigma(compressed: np.ndarray, sigma: float, seed: int) -> np.ndarray | None:
    """
    Runs preprocess_style() (from ml_logic.py) with sigma_noise overridden.

    Strategy: preprocess_style has  `noise = np.random.normal(0, noise_sigma, img.shape)`
    where noise_sigma=0.5 is a local constant.  We patch numpy.random.normal for
    the duration of that call so our sigma is used instead.  Every other step of
    the pipeline (Gaussian blur, CLAHE clip=2.0, brightness +70) is untouched.

    sigma=0.0 → noise array is all zeros → multiplicative term vanishes →
                CLAHE + brightness correction only, no speckle injected.

    Returns BGR uint8 (380×380), or None on failure.
    """
    cv2.imwrite(_TMP_PATH, compressed)
    np.random.seed(seed)

    def _normal_override(loc=0.0, scale=1.0, size=None):
        # preprocess_style calls np.random.normal(0, 0.5, img.shape)
        # We substitute our sigma while keeping loc and size unchanged.
        if sigma == 0.0:
            # No speckle: return a zero array so the multiplicative term
            # img_float * noise  evaluates to zero everywhere.
            return np.zeros(size, dtype=np.float32) if size is not None else 0.0
        # Draw from N(0, sigma) using the already-seeded global RNG.
        # standard_normal is NOT patched, so the seeded state is consumed
        # correctly and results are reproducible across sigma sweeps.
        return np.random.standard_normal(size) * sigma + loc

    with _mock_patch('numpy.random.normal', side_effect=_normal_override):
        result = preprocess_style(_TMP_PATH, target_size=(IMG_SIZE, IMG_SIZE))

    return result   # BGR uint8, 380×380, or None


# ---------------------------------------------------------------------------
# Batched inference
# ---------------------------------------------------------------------------

def _predict_batch(bgr_batch: list[np.ndarray]) -> np.ndarray:
    """Stack a list of BGR uint8 arrays → preprocess_input → model.predict."""
    arr = np.stack([preprocess_input(x.astype(np.float32)) for x in bgr_batch])
    return load_ai_model().predict(arr, verbose=0).flatten()


# ---------------------------------------------------------------------------
# Per-sigma evaluation
# ---------------------------------------------------------------------------

def evaluate_sigma(
    sigma: float,
    samples: list[tuple[str, int]],
    compressed_cache: list[np.ndarray | None],
) -> dict:
    """Run full-corpus inference for one sigma value and return metrics."""
    y_true:   list[int]   = []
    y_scores: list[float] = []
    buf_imgs:   list[np.ndarray] = []
    buf_labels: list[int]        = []

    for idx, ((path, label), compressed) in enumerate(zip(samples, compressed_cache)):
        if compressed is None:
            continue

        processed = _tars_with_sigma(compressed, sigma, SEED + idx)
        if processed is None:
            continue

        buf_imgs.append(processed)
        buf_labels.append(label)

        if len(buf_imgs) == BATCH_SIZE:
            y_scores.extend(_predict_batch(buf_imgs).tolist())
            y_true.extend(buf_labels)
            buf_imgs, buf_labels = [], []

        if (idx + 1) % 1000 == 0:
            print(f'    [{idx + 1:,}/{len(samples):,}]', flush=True)

    # Flush remainder
    if buf_imgs:
        y_scores.extend(_predict_batch(buf_imgs).tolist())
        y_true.extend(buf_labels)

    y_true_arr   = np.array(y_true)
    y_scores_arr = np.array(y_scores)
    y_pred_arr   = (y_scores_arr >= THRESHOLD).astype(int)

    return {
        'sigma':           sigma,
        'n':               int(len(y_true_arr)),
        'auc':             round(float(roc_auc_score(y_true_arr, y_scores_arr)), 4),
        'precision':       round(float(precision_score(y_true_arr, y_pred_arr, zero_division=0)), 4),
        'recall':          round(float(recall_score(y_true_arr, y_pred_arr, zero_division=0)), 4),
        'f1':              round(float(f1_score(y_true_arr, y_pred_arr, zero_division=0)), 4),
        'false_positives': int(((y_pred_arr == 1) & (y_true_arr == 0)).sum()),
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def save_plot(results: list[dict]) -> None:
    sigmas = [r['sigma'] for r in results]
    aucs   = [r['auc']   for r in results]
    f1s    = [r['f1']    for r in results]

    fig, ax1 = plt.subplots(figsize=(9, 5))

    c_auc = '#1f77b4'
    c_f1  = '#d62728'

    ax1.set_xlabel('σ_noise  (speckle standard deviation)', fontsize=12)
    ax1.set_ylabel('AUC-ROC', color=c_auc, fontsize=12)
    ax1.plot(sigmas, aucs, 'o-', color=c_auc, linewidth=2.2,
             markersize=8, label='AUC-ROC')
    ax1.tick_params(axis='y', labelcolor=c_auc)
    ax1.set_ylim(0.4, 1.02)
    ax1.set_xticks(sigmas)

    ax2 = ax1.twinx()
    ax2.set_ylabel('F1-Score  (theta = 0.75)', color=c_f1, fontsize=12)
    ax2.plot(sigmas, f1s, 's--', color=c_f1, linewidth=2.2,
             markersize=8, label='F1-Score')
    ax2.tick_params(axis='y', labelcolor=c_f1)
    ax2.set_ylim(0.0, 1.02)

    ax1.axvline(x=OPERATING_SIGMA, color='#555555', linestyle='--',
                linewidth=1.5, label=f'Operating point  sigma = {OPERATING_SIGMA}')

    lines1, lbls1 = ax1.get_legend_handles_labels()
    lines2, lbls2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lbls1 + lbls2, loc='lower left', fontsize=10)

    plt.title(
        'TARS Speckle Noise Sensitivity — EfficientNet-B4 on CSIRO Benchmark\n'
        '(Browser-Compressed + Brightness-Normalised, n = 5,630)',
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig('noise_sensitivity_plot.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved: noise_sensitivity_plot.png')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print('=' * 68)
    print('  TARS Noise Sensitivity Analysis')
    print(f'  sigma values : {SIGMA_VALUES}')
    print(f'  Browser      : JPEG Q={JPEG_QUALITY}, brightness target = {BRIGHTNESS_TARGET}')
    print(f'  Threshold    : theta = {THRESHOLD}')
    print('=' * 68)

    # 1. Discover images
    print(f'\nDiscovering images in:\n  {DATA_DIR}')
    samples = discover_images(DATA_DIR)
    if not samples:
        sys.exit('ERROR: No images found — check DATA_DIR.')
    n0 = sum(1 for _, l in samples if l == 0)
    n1 = sum(1 for _, l in samples if l == 1)
    print(f'Found {len(samples):,} images  (Class 0: {n0:,}, Class 1: {n1:,})')

    # 2. Compute and cache browser-compressed images once
    #    (~900 MB RAM for 5,630 x 400x400 uint8 grayscale arrays)
    print(f'\nSimulating browser preview for {len(samples):,} images...')
    compressed_cache: list[np.ndarray | None] = []
    for i, (path, _) in enumerate(samples):
        compressed_cache.append(simulate_browser(path))
        if (i + 1) % 1000 == 0:
            print(f'  {i + 1:,}/{len(samples):,} compressed', flush=True)
    valid = sum(1 for c in compressed_cache if c is not None)
    print(f'  Done: {valid:,}/{len(samples):,} valid')

    # 3. Load model weights once
    print('\nLoading model weights...')
    load_ai_model()
    print('  Model ready')

    # 4. Sweep sigma values
    all_results: list[dict] = []
    for sigma in SIGMA_VALUES:
        tag = ''
        if sigma == 0.0:
            tag = ' (no speckle — CLAHE + brightness only)'
        elif sigma == OPERATING_SIGMA:
            tag = ' [production operating point]'
        print(f'\n--- sigma = {sigma}{tag} ---')
        result = evaluate_sigma(sigma, samples, compressed_cache)
        all_results.append(result)
        print(
            f'  AUC={result["auc"]:.4f}  '
            f'F1={result["f1"]:.4f}  '
            f'Prec={result["precision"]:.4f}  '
            f'Rec={result["recall"]:.4f}  '
            f'FP={result["false_positives"]}',
            flush=True,
        )

    # 5. Save CSV
    pd.DataFrame(all_results).to_csv('noise_sensitivity_results.csv', index=False)
    print('\nSaved: noise_sensitivity_results.csv')

    # 6. Save plot
    save_plot(all_results)

    # 7. Summary table
    print()
    print('=' * 68)
    header = f'{"sigma":>6}  {"AUC":>7}  {"Precision":>10}  {"Recall":>8}  {"F1":>7}  {"FP":>5}'
    print(header)
    print('-' * 68)
    for r in all_results:
        marker = '  <-- operating point' if r['sigma'] == OPERATING_SIGMA else ''
        print(
            f'{r["sigma"]:>6.1f}  '
            f'{r["auc"]:>7.4f}  '
            f'{r["precision"]:>10.4f}  '
            f'{r["recall"]:>8.4f}  '
            f'{r["f1"]:>7.4f}  '
            f'{r["false_positives"]:>5}'
            f'{marker}'
        )
    print('=' * 68)

    # Cleanup temp file
    if os.path.exists(_TMP_PATH):
        os.remove(_TMP_PATH)
        print(f'\nCleaned up: {_TMP_PATH}')


if __name__ == '__main__':
    main()
