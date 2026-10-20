# External test: Arabian/Persian Gulf (GulfOilSAR, 2021–2022)

## Data

Culot, A., Wang, Q. & Hanert, E. (2026). *GulfOilSAR: an annotated Sentinel-1 dataset
of oil spills and look-alikes in the Arabian/Persian Gulf (2021–2022).* Zenodo,
https://doi.org/10.5281/zenodo.23261453 (CC-BY 4.0). Published 9 October 2026; the
record does not yet describe how the polygons were labelled or verified.

- Oil polygons: 796 (2021) and 1,151 (2022), from 124 scenes.
- Look-alike polygons: 85,138, from 144 scenes.
- Each polygon carries its Sentinel-1 product ID.
- Browse images and footprints come from NASA ASF (156 scenes).

## Units

The 25 km browse crops are built exactly as in the Mediterranean study:

- **Oil:** centred on oil polygons (1,547).
- **Look-alike:** centred on look-alike polygons, at most 40 per scene (4,465).
- **Ordinary sea:** more than 30 km from any polygon, less than 5% land (374).

Both heads score the same frozen-backbone features. 95% intervals come from a
bootstrap over scenes.

## 1. Direct transfer: no Gulf data used in training (`gulf_eval.py`, `analyse_gulf.py`)

| Test | Browse head (Med-trained) | Current head (CSIRO) | Difference CI |
|---|---|---|---|
| Oil vs ordinary sea | 0.819 (0.773–0.863) | 0.745 (0.683–0.809) | +0.023 to +0.127 |
| Oil vs look-alikes | 0.859 (0.818–0.895) | 0.865 (0.826–0.903) | −0.036 to +0.026 |
| Oil vs all negatives | 0.856 (0.815–0.892) | 0.856 (0.816–0.895) | −0.031 to +0.033 |

At operating thresholds the browse head detects more slicks (77% vs 65%) but raises
far more false alarms (21% vs 9%). Its Mediterranean threshold does not transfer.

## 2. Held-out year: test on Gulf 2021 (`multi_region.py`)

Mean of 5 training seeds. Epochs and thresholds are chosen on training scenes only.

| Training data | AUC (all neg.) | vs ordinary sea | vs look-alikes | Recall | False alarms |
|---|---|---|---|---|---|
| Current (CSIRO, unchanged) | 0.844 | 0.698 | 0.854 | 0.610 | 0.086 |
| Mediterranean 2019 only | 0.827 ± 0.014 | 0.793 | 0.829 | 0.675 | 0.162 |
| **Gulf 2022 only** | **0.882 ± 0.007** | 0.773 | **0.890** | 0.580 | **0.048** |
| Mediterranean + Gulf 2022 | 0.860 ± 0.002 | 0.766 | 0.867 | 0.574 | 0.061 |

Test set: 705 oil, 1,931 look-alike and 127 ordinary-sea crops from 60 scenes.

## What this means

1. **The Mediterranean advantage is regional.** There, the browse head beat the
   current head by about 0.27 AUC. In the Gulf, the Med-trained head ties overall.
   It is better against ordinary sea and no better against Gulf look-alikes, which
   differ from Mediterranean ones.
2. **Training on browse crops from the same region helps.** It adds about +0.04 AUC
   and halves false alarms in the Gulf. Pooling a distant region dilutes this.
3. **The current CSIRO head is region-dependent too.** It scores 0.64 in the
   Mediterranean and 0.86 in the Gulf. The CSIRO set contains 74 chips coded `BAH`,
   which may be Bahrain in the Gulf, so it may have seen this region.
4. **Thresholds must be set per region.**

## Caveats

- Only two regions. The labelling method of GulfOilSAR is not yet documented.
- Ordinary-sea crops are presumed oil-free.
- Section 2 reports seed spread, not scene-bootstrap intervals; those should be added
  before publication.
- Several slicks can share one crop neighbourhood, so all intervals resample whole scenes.

## Reproduce

Put the four `.gpkg` files from Zenodo in `data/`. Build `data/footprints.json` and
`data/browse/` with the same ASF calls as `../browse_eval/fetch_data.py` (product IDs
are in each polygon's `source` field). Then run:

```bash
python gulf_eval.py
python analyse_gulf.py
python multi_region.py
```
