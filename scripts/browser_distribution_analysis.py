"""
browser_distribution_analysis.py
---------------------------------
Analyses the distribution of RAW vs TARS inference scores across live
Sentinel-1 SAR acquisitions stored in temp_images.

Two source folders are configurable at the top of this file.  If a folder
does not exist it is silently skipped.  The source folder name is recorded
as a metadata column in the CSV so each image can be traced back to its
origin.

Browser simulation (identical to simulate_browser_eval_v2.py):
    grayscale → JPEG Q=30 encode/decode → brightness normalise (mean ≈ 50)

Outputs:
    browser_score_scatter.png    – RAW score (x) vs TARS score (y), quadrant-coloured
    browser_score_histogram.png  – overlapping score histograms
    browser_distribution_results.csv

No accuracy, precision, recall, or F1 is computed (no ground-truth labels).
Do NOT modify any existing files.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   # Force CPU (matches deployment)

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras.applications.efficientnet import preprocess_input

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_logic import load_ai_model, IMG_SIZE, OPTIMAL_THRESHOLD, preprocess_style

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# SOURCE_DIRS: list of (label, absolute_path) pairs.
# Each folder is scanned recursively for supported image extensions.
# If a folder does not exist it is skipped with a warning.
# Add or change directories here; source_folder in the CSV will show the label.
SOURCE_DIRS = [
    # The live acquisition pipeline saves images here:
    ("temp_images",
     os.path.join(_ROOT, "temp_images")),

    # Uncomment and update if a separate 63-image live-acquisition folder exists:
    # ("live_acquisitions",
    #  r"C:\path\to\live_acquisitions"),
]

THRESHOLD      = OPTIMAL_THRESHOLD   # 0.75
BATCH_SIZE     = 32
JPEG_QUALITY   = 30
BRIGHTNESS_TARGET = 50.0
SEED           = 42
SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}

_TMP_PATH = os.path.join(_ROOT, '_browser_dist_tmp.jpg')

# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------

def discover_images(source_dirs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """
    Walk each configured source directory and collect image paths.
    Returns a list of (file_path, source_label) tuples.
    Deduplicates by resolved absolute path so the same file is not counted
    twice if two labels point to overlapping directories.
    """
    seen: set[str] = set()
    samples: list[tuple[str, str]] = []

    for label, directory in source_dirs:
        if not os.path.isdir(directory):
            print(f'  WARNING: directory not found, skipping: {directory}')
            continue
        count = 0
        for root, _dirs, files in os.walk(directory):
            for fname in sorted(files):
                if os.path.splitext(fname)[1].lower() not in SUPPORTED_EXTS:
                    continue
                abs_path = os.path.realpath(os.path.join(root, fname))
                if abs_path in seen:
                    continue
                seen.add(abs_path)
                samples.append((abs_path, label))
                count += 1
        print(f'  {label}: {count} images found in {directory}')

    return samples


# ---------------------------------------------------------------------------
# Browser simulation  (identical to simulate_browser_eval_v2.py)
# ---------------------------------------------------------------------------

def simulate_browser(img_path: str) -> np.ndarray | None:
    """
    Grayscale → JPEG Q=30 encode/decode → brightness normalise to mean ≈ 50.
    Returns single-channel uint8 (H×W), or None on failure.
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
# Preprocessing paths  (identical to simulate_browser_eval_v2.py)
# ---------------------------------------------------------------------------

def path_raw(compressed: np.ndarray) -> np.ndarray:
    img = cv2.resize(compressed, (IMG_SIZE, IMG_SIZE))
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return preprocess_input(bgr.astype(np.float32))


def path_tars(compressed: np.ndarray, seed: int) -> np.ndarray | None:
    cv2.imwrite(_TMP_PATH, compressed)
    np.random.seed(seed)
    processed = preprocess_style(_TMP_PATH, target_size=(IMG_SIZE, IMG_SIZE))
    if processed is None:
        return None
    return preprocess_input(processed.astype(np.float32))


# ---------------------------------------------------------------------------
# Batched inference
# ---------------------------------------------------------------------------

def _flush_batch(
    model,
    buf_paths:  list[str],
    buf_labels: list[str],
    buf_raw:    list[np.ndarray],
    buf_tars:   list[np.ndarray],
    results:    list[dict],
) -> None:
    if not buf_raw:
        return
    raw_preds  = model.predict(np.stack(buf_raw),  verbose=0).flatten()
    tars_preds = model.predict(np.stack(buf_tars), verbose=0).flatten()
    for path, src, rs, ts in zip(buf_paths, buf_labels, raw_preds, tars_preds):
        raw_above  = bool(rs >= THRESHOLD)
        tars_above = bool(ts >= THRESHOLD)
        if raw_above and tars_above:
            quadrant = 'both_above'
        elif raw_above:
            quadrant = 'raw_only'
        elif tars_above:
            quadrant = 'tars_only'
        else:
            quadrant = 'both_below'
        results.append({
            'filename':            os.path.basename(path),
            'source_folder':       src,
            'raw_score':           round(float(rs),  6),
            'tars_score':          round(float(ts),  6),
            'delta':               round(float(ts - rs), 6),
            'raw_above_threshold': raw_above,
            'tars_above_threshold': tars_above,
            'quadrant':            quadrant,
        })
    buf_paths.clear()
    buf_labels.clear()
    buf_raw.clear()
    buf_tars.clear()


def run_inference(
    model,
    samples: list[tuple[str, str]],
) -> list[dict]:
    results:    list[dict]       = []
    buf_paths:  list[str]        = []
    buf_labels: list[str]        = []
    buf_raw:    list[np.ndarray] = []
    buf_tars:   list[np.ndarray] = []
    skipped = 0
    n = len(samples)

    for idx, (img_path, source_label) in enumerate(samples, 1):
        compressed = simulate_browser(img_path)
        if compressed is None:
            skipped += 1
            continue

        arr_raw = path_raw(compressed)
        arr_tars = path_tars(compressed, seed=SEED + idx)
        if arr_tars is None:
            skipped += 1
            continue

        buf_paths.append(img_path)
        buf_labels.append(source_label)
        buf_raw.append(arr_raw)
        buf_tars.append(arr_tars)

        if len(buf_raw) == BATCH_SIZE:
            _flush_batch(model, buf_paths, buf_labels, buf_raw, buf_tars, results)

        if idx % 50 == 0 or idx == n:
            print(f'  [{idx}/{n}]  skipped={skipped}', flush=True)

    _flush_batch(model, buf_paths, buf_labels, buf_raw, buf_tars, results)
    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

_QUADRANT_COLORS = {
    'both_below': '#888888',   # grey
    'raw_only':   '#ff8800',   # orange  (suppressed — RAW fires, TARS does not)
    'tars_only':  '#2255cc',   # blue    (raised    — TARS fires, RAW does not)
    'both_above': '#cc2222',   # red     (confirmed high)
}
_QUADRANT_LABELS = {
    'both_below': 'Both below θ',
    'raw_only':   'RAW only  (suppressed)',
    'tars_only':  'TARS only  (raised)',
    'both_above': 'Both above θ  (confirmed)',
}


def save_scatter(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))

    for q, color in _QUADRANT_COLORS.items():
        sub = df[df['quadrant'] == q]
        ax.scatter(
            sub['raw_score'], sub['tars_score'],
            c=color, alpha=0.7, s=40, label=f'{_QUADRANT_LABELS[q]} (n={len(sub)})',
            edgecolors='none',
        )

    ax.axhline(y=THRESHOLD, color='#444444', linestyle='--', linewidth=1.2,
               label=f'TARS threshold  θ = {THRESHOLD}')
    ax.axvline(x=THRESHOLD, color='#444444', linestyle='--', linewidth=1.2,
               label=f'RAW threshold  θ = {THRESHOLD}')

    # Quadrant count annotations in each corner
    n_both_below = len(df[df['quadrant'] == 'both_below'])
    n_raw_only   = len(df[df['quadrant'] == 'raw_only'])
    n_tars_only  = len(df[df['quadrant'] == 'tars_only'])
    n_both_above = len(df[df['quadrant'] == 'both_above'])

    ax.text(0.02, 0.02, f'n = {n_both_below}',
            transform=ax.transAxes, color=_QUADRANT_COLORS['both_below'],
            fontsize=11, va='bottom', ha='left')
    ax.text(0.98, 0.02, f'n = {n_raw_only}',
            transform=ax.transAxes, color=_QUADRANT_COLORS['raw_only'],
            fontsize=11, va='bottom', ha='right')
    ax.text(0.02, 0.98, f'n = {n_tars_only}',
            transform=ax.transAxes, color=_QUADRANT_COLORS['tars_only'],
            fontsize=11, va='top',    ha='left')
    ax.text(0.98, 0.98, f'n = {n_both_above}',
            transform=ax.transAxes, color=_QUADRANT_COLORS['both_above'],
            fontsize=11, va='top',    ha='right')

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('RAW Score  (resize → EfficientNet-B4)', fontsize=12)
    ax.set_ylabel('TARS Score  (browser-compressed + speckle + CLAHE + Δ70)', fontsize=12)
    ax.set_title(
        'RAW vs TARS Inference Score Distribution\n'
        f'Live SAR Acquisitions  (n = {len(df)},  θ = {THRESHOLD})',
        fontsize=13,
    )
    ax.legend(loc='center right', fontsize=9, framealpha=0.85)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig('browser_score_scatter.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved: browser_score_scatter.png')


def save_histogram(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    bins = np.linspace(0, 1, 41)   # 40 bins across [0, 1]
    ax.hist(df['raw_score'],  bins=bins, color='#cc2222', alpha=0.5,
            label=f'RAW score   (mean = {df["raw_score"].mean():.3f})')
    ax.hist(df['tars_score'], bins=bins, color='#2255cc', alpha=0.5,
            label=f'TARS score  (mean = {df["tars_score"].mean():.3f})')

    ax.axvline(x=THRESHOLD, color='#222222', linestyle='--', linewidth=1.5,
               label=f'θ = {THRESHOLD}')

    ax.set_xlabel('Sigmoid Score', fontsize=12)
    ax.set_ylabel('Image Count', fontsize=12)
    ax.set_title(
        'Score Distribution: RAW vs TARS Paths\n'
        f'Live SAR Acquisitions  (n = {len(df)})',
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig('browser_score_histogram.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved: browser_score_histogram.png')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print('=' * 65)
    print('  Browser Distribution Analysis')
    print(f'  Threshold : θ = {THRESHOLD}')
    print(f'  Browser   : JPEG Q={JPEG_QUALITY}, brightness target = {BRIGHTNESS_TARGET}')
    print('=' * 65)

    # 1. Discover images
    print('\nDiscovering images...')
    samples = discover_images(SOURCE_DIRS)
    if not samples:
        sys.exit(
            'ERROR: No images found.\n'
            'Check SOURCE_DIRS at the top of browser_distribution_analysis.py.'
        )
    print(f'Total: {len(samples)} images across {len(SOURCE_DIRS)} source(s)')

    # 2. Load model
    print('\nLoading model weights...')
    model = load_ai_model()
    if model is None:
        sys.exit('ERROR: Failed to load model.')
    print('  Model ready')

    # 3. Inference
    print(f'\nRunning RAW + TARS inference (batch={BATCH_SIZE})...')
    results = run_inference(model, samples)
    print(f'  Done: {len(results)} images processed')

    if not results:
        sys.exit('ERROR: No images were successfully processed.')

    df = pd.DataFrame(results)

    # 4. Save CSV
    df.to_csv('browser_distribution_results.csv', index=False)
    print('Saved: browser_distribution_results.csv')

    # 5. Save plots
    save_scatter(df)
    save_histogram(df)

    # 6. Cleanup temp file
    if os.path.exists(_TMP_PATH):
        os.remove(_TMP_PATH)

    # 7. Summary
    n_both_below = (df['quadrant'] == 'both_below').sum()
    n_raw_only   = (df['quadrant'] == 'raw_only').sum()
    n_tars_only  = (df['quadrant'] == 'tars_only').sum()
    n_both_above = (df['quadrant'] == 'both_above').sum()
    mean_raw     = df['raw_score'].mean()
    mean_tars    = df['tars_score'].mean()
    mean_delta   = df['delta'].mean()

    print()
    print('=' * 65)
    print('  SUMMARY')
    print('=' * 65)
    print(f'  Total images processed        : {len(df)}')
    print()
    print(f'  Both below θ  (clean, agreed) : {n_both_below}')
    print(f'  RAW only > θ  (suppressed)    : {n_raw_only}')
    print(f'    → TARS lowered score below threshold')
    print(f'  TARS only > θ (raised)        : {n_tars_only}')
    print(f'    → TARS elevated score above threshold')
    print(f'  Both above θ  (confirmed high): {n_both_above}')
    print()
    print(f'  Mean RAW score                : {mean_raw:.4f}')
    print(f'  Mean TARS score               : {mean_tars:.4f}')
    print(f'  Mean delta (TARS - RAW)       : {mean_delta:+.4f}')
    print('=' * 65)

    # Per-source breakdown if multiple sources
    if df['source_folder'].nunique() > 1:
        print()
        print('  Per-source breakdown:')
        for src, grp in df.groupby('source_folder'):
            print(f'    {src}: {len(grp)} images  '
                  f'(RAW mean={grp["raw_score"].mean():.3f}, '
                  f'TARS mean={grp["tars_score"].mean():.3f})')
        print()


if __name__ == '__main__':
    main()
