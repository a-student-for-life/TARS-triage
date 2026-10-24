# Changes since the version first submitted for review

This file lists how the system differs from the version described in the manuscript
as first submitted. Every change is covered by `tests/`.

## Data integrity

| Problem in the original | Change |
|---|---|
| Results were overwritten when a ship moved again, so past detections (e.g. the paper's 7 `OIL_DETECTED` cases) could not be recounted. | New append-only `alert_events` table. Every escalation, classification (all STI draw scores, threshold, sigma, weights path, git commit), expiry and operator action is kept permanently. |
| New ships got `anomaly_type = "UNKNOWN"`, which looked like a Layer-3 output in the statistics. | New ships start as `NONE`; only Layer 1 sets `SUDDEN_STOP` / `SUDDEN_ACCEL`. |
| AIS "not available" codes were used as real values: 129 ships at speed 102.3 kn (fake sudden accelerations), 17 ships at 0°N 0°E, test MMSI `999999999`. | `ais_validation.py` drops invalid positions, null island, placeholder and non-vessel MMSIs, and treats SOG ≥ 102.2 / COG 360 as unknown. Rejection counts at `GET /ingest/rejections`. |
| Server receive time was used as the report time. | The listener forwards the AIS message timestamp. |
| Images up to 48 h *before* the anomaly were classified (42 of 58 classified images in the May database predated their anomaly). | Only images acquired after the anomaly (1 h tolerance) and within 48 h are used. |

## Layer 1 (kinematic flag)

Sudden speed change > 10 kn is only counted when the two reports are at most
10 minutes apart (a change across a 3-hour gap is not "sudden"). As before, the
flag alone does not escalate a ship; it is recorded as a reason tag and lifts
the post-review cooldown.

## Layer 2 (spatial context)

The original used HDBSCAN cluster size alone. Measured on 12,068 real ships
from the May database, HDBSCAN with no distance scale grouped ships hundreds
of km apart, and a size-only rule called 38% of ships a possible "rendezvous".

Now:
- **dense area (SAFE veto):** HDBSCAN cluster of ≥ 6 ships whose median
  nearest-neighbour distance is ≤ 5 km, or ≥ 6 ships within 1 km.
- **rendezvous:** 2–5 ships within 1 km, all slower than 3 kn, and more than
  10 km from land (offline `global-land-mask`).
- **coastal:** not dense and within 10 km of land. Rendezvous and loitering
  rules don't apply here (harbours, marinas, anchorages).
- **isolated:** everything else.
- SAFE ships stay in the clustering input. Previously they were dropped, so
  ports shrank cycle by cycle until leftover ships looked like a rendezvous.

On the same 12,068 ships: rendezvous 0.09%, offshore loitering 1.6%.

## Layer 3 (behaviour)

The original Isolation Forest saw one snapshot per ship, `[lat, lon, speed]`,
with `contamination = 0.10` (10% of ships flagged every cycle by
construction). It could not detect zig-zags or drift, and "anomalous" partly
meant "in an unusual place".

Now each ship keeps its last 20 reports, and the forest uses five trajectory
features with no position: mean speed, speed std, max speed change between
reports, mean heading change, straightness. A ship must also be > 3.5 robust
z-scores from the fleet median in at least one feature (otherwise a uniform fleet
still produces "outliers").

The contamination is fixed at 0.01, an alarm budget of about 1% of the vessels
judged in each cycle. On one day of the US fleet the automatic setting flagged
7–9% of vessels per cycle and made Layer 3 the main source of escalations; with
0.01 the whole cascade escalates 2.2–2.6% (`experiments/revision`). This is a
workload choice, not a value tuned for accuracy. Override with
`TARS_ISO_CONTAMINATION`.

## Layer 4 (image + classifier)

- Watchdog: jobs without an image back off exponentially instead of being
  retried first on every loop (which blocked every other job). Misaligned
  jobs no longer get stuck. Jobs past the window become `EXPIRED`.
- STI (only used with `TARS_MODEL=csiro`) is random. Each image is scored as the mean of 8 draws with
  a seed derived from the file name, so the same image always gets the same
  score; the spread is logged.
- The backbone is still built with `weights="imagenet"`. This is required:
  Keras's EfficientNet only adds an input Rescaling layer in that case, and
  the trained model depends on it (building with `weights=None` changes every
  score; checked against `outputs/evaluation_results.csv`).
- No debug image is written on every inference (opt in with
  `TARS_SAVE_DEBUG_IMAGES=1`).

## Security / hygiene

- The AIS listener no longer disables TLS certificate checking.
- CORS limited to the local dashboard origins (override with `TARS_CORS_ORIGINS`).
- All thresholds and windows in `config.py`.

## Vessel-centred cropping and a browse-trained head

- The watchdog now classifies a 25 km crop around the vessel (`browse_geo.py`)
  instead of the whole ~250 km scene. Browse previews are north-up rasters over the
  footprint bounding box (verified against coastlines).
- **Region-aware heads (default `TARS_MODEL=regional`).** Browse-trained heads beat
  the original CSIRO head only inside their own region:
  - Eastern Mediterranean, 5-fold CV by day: AUC 0.909 vs 0.638.
  - Gulf, trained on 2022 and tested on 2021: AUC 0.882 vs 0.844.
  - A Mediterranean head applied in the Gulf only ties the CSIRO head.
  
  So the vessel's position selects the head (`models/regional_heads.json`), each with
  its own threshold. Outside every region, the CSIRO head scores the raw crop.
  `TARS_MODEL=browse_v1` forces the Mediterranean head; `TARS_MODEL=csiro` restores
  the original STI behaviour. See `experiments/browse_eval` and `experiments/gulf_eval`.

## Not changed yet

- **Training reproducibility (CSIRO model).** The notebook's validation split can't be
  reconstructed (files were renamed in filesystem order). Retraining should
  save the exact split.
