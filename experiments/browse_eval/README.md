# Real browse-image evaluation (Eastern Mediterranean, 2019)

Tests the deployed TARS classifier (`models/optisar_b4_retrained_weights.weights.h5`,
threshold 0.75) on real Sentinel-1 browse JPEGs, against independent expert labels.

## Labels

Yang, Y.-J. & Singha, S. (2025). *Oil slicks, look-alikes and other remarkable SAR
signatures in Sentinel-1 imagery in the Eastern Mediterranean Sea in 2019.* PANGAEA,
https://doi.org/10.1594/PANGAEA.980773 (CC-BY 4.0). Paper: Yang et al. (2025),
*Earth System Science Data* 17, 6807–6837, https://doi.org/10.5194/essd-17-6807-2025.

Oil objects were marked by two interpreters on full-resolution Sentinel-1 scenes
and kept only when both agreed. Look-alike (no-oil) patches are oceanic phenomena
that resemble oil. There is no in-situ ground truth (the dataset authors note this).

## Method

- 1,063 Sentinel-1 GRD products; footprints and browse JPEGs from NASA ASF.
- **Georeferencing.** Browse images are north-up rasters covering the footprint's
  lat/lon bounding box (~280 m/pixel). Verified by overlaying an offline land mask
  on ascending and descending scenes (`orientation_check_bbox.jpg`).
- **Units.**
  - *scene*: the whole browse image, which is what the live system classifies today.
    681 scenes: 558 containing labelled oil, 123 look-alike-only scenes inside the
    annotated area.
  - *crop*: a 25 km window centred on a labelled patch (simulates vessel-centred
    cropping). 2,100 crops: 1,220 oil patches, 880 look-alike patches, sampled from
    1,000 with a fixed seed. Look-alike crops that would also contain a labelled oil
    patch were dropped.
- **Paths.** `raw` = grayscale + resize; `rad` = STI with sigma 0 (blur, CLAHE, +70);
  `sti` = full STI, sigma 0.5, mean of 4 seeded draws.
- 95% intervals: bootstrap over whole scenes (patches from one scene are not independent).

## Results (threshold 0.75)

| Unit | Path | AUC | Recall | False-alarm rate |
|---|---|---|---|---|
| Whole scene | raw | 0.552 (CI 0.49–0.61) | 0.013 | 0.000 |
| Whole scene | sti | 0.588 | 0.002 | 0.008 |
| 25 km crop, all | raw | 0.715 | 0.365 | 0.091 |
| 25 km crop, all | rad | 0.731 | 0.301 | 0.055 |
| 25 km crop, all | sti | 0.738 | 0.129 | 0.016 |
| 25 km crop, open water | raw | 0.831 (CI 0.81–0.85) | 0.499 | 0.106 |
| 25 km crop, open water | sti | 0.827 | 0.174 | 0.015 |
| 25 km crop, coastal | raw | 0.492 | 0.012 | 0.014 |

Crops by size of the largest slick in the patch (raw path, AUC against all look-alike crops):

| Largest slick | n | AUC | Recall at 0.75 |
|---|---|---|---|
| < 1 km | 313 | 0.71 | 0.23 |
| 1–2 km | 335 | 0.66 | 0.30 |
| 2–5 km | 372 | 0.68 | 0.35 |
| 5–10 km | 124 | 0.85 | 0.69 |
| ≥ 10 km | 76 | 0.92 | 0.75 |

Open water with slicks ≥ 5 km: raw AUC 0.93 (CI 0.91–0.95).

## What this shows

1. **Whole-scene classification does not work** (AUC ≈ chance, recall ≈ 1%). At
   ~280 m/pixel, a whole scene squeezed to 380×380 leaves most slicks a few pixels wide.
2. **Vessel-centred cropping is essential.** The same model on 25 km crops reaches
   AUC 0.83 in open water and 0.92 for slicks of 10 km or more.
3. **Detectability depends on slick size.** Slicks under ~5 km are poorly detected
   from browse imagery; 5 km and above are detected reasonably well.
4. **STI does not improve ranking.** Open-water AUC difference STI − raw is −0.026 to
   +0.016 (95% CI). At the fixed 0.75 threshold STI mainly lowers scores: false alarms
   fall from 10.6% to 1.5% and recall from 50% to 17%. It behaves like a stricter
   threshold, not a domain fix.
5. **Coastal patches fail on every path** (AUC ≈ 0.5), which contradicts the earlier
   claim that STI helps most near coasts.

## Caveats

- One region (Eastern Mediterranean) and one year (2019).
- The training set (CSIRO) has 40 chips coded `ISR`; some may come from 2019 scenes in
  this area. Chip filenames carry no dates, so overlap cannot be ruled out, but it is
  at most 40 of 5,630 training chips.
- The 0.75 threshold was set on CSIRO chips, not browse images. Choosing a browse
  threshold needs a separate calibration split.
- Labels are expert interpretation, not in-situ confirmation.

## Reproduce

```bash
cd experiments/browse_eval
python fetch_data.py      # ~280 MB of browse images from NASA ASF
python run_eval.py        # resumable; ~1 h on 2 CPU cores, minutes on a GPU
python analyse.py
```

---

# Comparison: CSIRO-trained head vs browse-trained head

Both models use the **identical frozen ImageNet EfficientNet-B4 backbone** (the
original TARS backbone was frozen during training) and see the **identical 25 km
crops**. Only the head's training data differs. Splitting the deployed model into
backbone + head reproduces its scores exactly (max difference 0.0).

- **Browse-trained head:** same architecture as the deployed head; trained on browse
  crops; raw preprocessing (no STI).
- **Protocol:** 5-fold cross-validation grouped by **acquisition day**, so a test fold
  never shares a day, and hence an overlapping scene of the same slick, with training.
  Within each training fold, 20% of days are held out for early stopping and to set
  the threshold at 5% false alarms. Every crop gets one out-of-fold score. 95% CIs
  come from a bootstrap over days.

## v1: negatives = look-alikes only (`train_compare.py`)

AUC 0.928 vs 0.714. But the ESSD no-oil set contains only look-alikes, so a confound
check was run on 1,627 **ordinary-sea crops**: 25 km windows inside the annotated
area, more than 30 km from any labelled patch, less than 5% land
(`extract_background.py`). The v1 head flagged 24.5% of them as oil. It had partly
learned "look-alike vs ordinary sea". **Not used.**

## v2: negatives = look-alikes + ordinary sea (`train_compare_v2.py`)

| Task | Browse head AUC (95% CI) | Current head AUC (95% CI) |
|---|---|---|
| Oil vs look-alikes | 0.925 (0.912–0.937) | 0.714 (0.688–0.738) |
| Oil vs ordinary sea | 0.890 (0.873–0.906) | 0.549 (0.518–0.578) |
| **Oil vs all negatives** | **0.909 (0.896–0.921)** | **0.638 (0.610–0.664)** |
| Open water | 0.904 | 0.768 |
| Coastal | 0.896 | 0.501 |

At operating thresholds (browse: set on validation days; current: 0.75):

| | Browse head | Current head |
|---|---|---|
| Recall (1,220 oil crops) | **0.615** | 0.365 |
| False alarms, look-alikes (1,894) | **0.051** | 0.090 |
| False alarms, ordinary sea (1,627) | **0.038** | 0.157 |

By largest slick in the patch (vs all negatives), browse / current AUC:
<1 km 0.861 / 0.614 · 1–2 km 0.923 / 0.578 · 2–5 km 0.922 / 0.601 ·
5–10 km 0.949 / 0.810 · ≥10 km 0.918 / 0.896.

## Ship-echo check (`ship_confound.py`)

Oil crops contain more bright point targets (mean 8.6 vs ≈1 for ordinary sea), and
the browse score rises somewhat with them (Spearman 0.24 within every class). Within
strata of equal bright-point count, including **zero**, the browse head keeps AUC
0.86–0.90, while the current head falls to 0.54–0.78. So the head is not mainly
detecting ships. In deployment every crop is centred on a ship, so some extra
false alarms from ship echoes are still possible. A live run will measure this.

## Deployable head

`train_final_head.py` trains on all 4,741 crops (median CV epoch count) and saves
`models/browse_head_v1.weights.h5` with threshold 0.742 (median of fold thresholds)
in `models/browse_head_v1.json`. The pipeline uses it by default (`TARS_MODEL=browse_v1`);
`TARS_MODEL=csiro` restores the original head with STI.

## Remaining limitations

- One region (Eastern Mediterranean) and one year (2019). Performance elsewhere is
  unknown; a second labelled region is the most valuable next step.
- Ordinary-sea crops are *presumed* oil-free (no slick labelled within 30 km).
- Head-only training. Fine-tuning the backbone (needs a GPU) may do better.
- Patch-size AUC for slicks under 1 km is higher than resolution alone would
  suggest; patch context may contribute.

## Reproduce

```bash
python fetch_data.py
python extract_features.py      # crops -> frozen-backbone features (+ current head scores)
python extract_background.py    # ordinary-sea crops
python train_compare_v2.py      # cross-validated comparison
python ship_confound.py
python train_final_head.py      # deployable head
```
