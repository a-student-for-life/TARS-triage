# Experiments for the Applied Geomatics revision

Each script answers one reviewer comment. Results files are committed; feature
files (`*.npz`) are not, because they are large and can be regenerated.

| Script | Answers | Output |
|---|---|---|
| `extract_csiro_features.py b4` / `b0` | 1, 2, 7 | `csiro_features_{b4,b0}.npz` (not committed) |
| `csiro_cv.py` | 1, 2, 7 | `csiro_cv.json`, `csiro_oof.csv` |
| `browse_b0_vs_b4.py` (after `TARS_BACKBONE=b0` runs of `../browse_eval/extract_features.py` and `extract_background.py`) | 2 | `browse_b0_vs_b4.json` |
| `inference_timing.py` | 2 | `inference_timing.json` |
| `layer2_sensitivity.py [oil_spill.db]` | 5 | `layer2_sensitivity.json` |
| `latency_availability.py` | 6 | `latency_availability.json`, `publication_lag.csv` |

`run_extract.sh` and `run_rest.sh` run everything in order on a CPU machine
(about 1.5 h on 2 cores). Set `SSL_CERT_FILE` if your network needs a custom CA.

## Main results

**CSIRO leakage (comment 7).** Chips are cut on a 200 px grid with 400 px windows, so
neighbouring chips from one source image share half their pixels. Under a chip-level
80/20 split, 27% of validation chips share a source image with a training chip.
B4, mean of 3 seeds:

| Split | AUC | F1 | False-negative rate |
|---|---|---|---|
| Chip-level (as original) | 0.993 | 0.948 | 0.055 |
| Grouped by source image | 0.992 | 0.947 | 0.056 |
| Region held out | 0.962 | 0.859 | 0.145 |

Overlap barely matters; geography does.

**B0 vs B4 (comment 2).** Same head, splits and seeds. Equal within a region (CSIRO and
browse crops). With a region held out, B4 has F1 0.859 vs 0.827 (fewer false positives),
B0 has lower false-negative rate (0.115 vs 0.145). B0: 29 ms and 4.7 M parameters per
image; B4: 131 ms and 18.6 M.

**Latency (comment 6).** Median wait for the next Sentinel-1 acquisition after a random
anomaly: 42.5 h (E. Med.), 31.2 h (Gulf). Median publication lag on ASF: 10.2 h. TARS
retrieval and processing: about 3 s.

**Layer 2 (comment 5).** Old cluster-size rule: 38% of 12,068 vessels flagged as possible
rendezvous. New rule: 0.09%; 0.04–0.6% across one-at-a-time threshold changes.

## Added after the first revision draft

**Off-centre crops** (`offcentre_extract.py`, `offcentre_score.py`, `offcentre_diff.py`;
manuscript Sect. 8.6). Every Mediterranean and Gulf 2021 oil and look-alike crop re-cut with
its centre moved 2, 5 and 10 km along one random bearing; scored with the same heads trained on
centred crops. Zero-shift crops reproduce the stored features exactly, and the re-trained
Mediterranean fold heads reproduce the stored out-of-fold scores (AUC 0.9091).

| Shift | Med AUC | Med recall | Gulf AUC | Gulf recall |
|---|---|---|---|---|
| 0 km | 0.911 | 0.620 | 0.881 | 0.584 |
| 2 km | 0.908 | 0.632 | 0.882 | 0.592 |
| 5 km | 0.900 | 0.598 | 0.878 | 0.582 |
| 10 km | 0.870 | 0.537 | 0.855 | 0.528 |

Paired sets: 2,913 Med and 2,462 Gulf crops valid at every shift. Med AUC change at 5 km
-0.011 (95% CI -0.018 to -0.005), at 10 km -0.041 (-0.050 to -0.031).
`offcentre_features.npz` (145 MB) is not committed; rerun `offcentre_extract.py`.

**Layer 3 on a national fleet** (`layer3_national.py`, `layer3_sensitivity.py`; Sect. 8.7,
Table 15). NOAA Marine Cadastre AIS for 14 June 2023 (8.84 M valid reports, 20,543 vessels),
unmodified `run_engine_cycle` at 03/09/15/21 UTC on the whole fleet. Each cycle escalates
8.4-10.1% of vessels, 74-82% of them on the Layer-3 criterion alone. Layer-3 flags outside dense
areas: 7.4% with automatic contamination (deployed), 0.42% with contamination 0.01.
Input: `https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2023/AIS_2023_06_14.zip`.

**Layer 3 alarm budget** (`layer3_national.py` with `TARS_ISO_CONTAMINATION=0.01`,
`TARS_L3_OUT=layer3_national_budget.json`; Sect. 8.7, Table 15). The released code uses a
contamination of 0.01 (`config.py`). On the same US day the whole cascade then escalates
2.2-2.6% of vessels per cycle (8.4-10.1% with "auto"); 60-65% from loitering, about one fifth
from rendezvous, 14-21% from Layer 3. A workload choice, not tuned for accuracy.

**Simple baseline** (`simple_baseline.py`; Sect. 8.6). Gradient boosting on 24 intensity and
texture statistics of the same crops, same folds and thresholds: AUC 0.840 (Med; CNN head 0.909,
paired difference 0.055-0.083) and 0.822 (Gulf 2021; CNN head 0.882). Logistic regression:
0.785 and 0.796.
