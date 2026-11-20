# TARS: Timely Anomaly Response System

TARS is an event-driven framework for marine oil spill triage. It watches live AIS
(Automatic Identification System) ship traffic. When a vessel behaves unusually, it
finds the next Sentinel-1 browse (quicklook) image of that location, cuts a 25 km
crop centred on the vessel, and scores the crop with a classifier head trained for
that region. The verdict is a triage decision for an analyst to review, not a
confirmed spill.

This repository holds the code, trained regional heads, result files and the scripts
behind every table and figure of the paper:

> J. Miranda, N. Nair, G. Shringare, N. Hussain, K. Bhadange, A. K. Tripathy (2026).
> *TARS: a framework for marine oil spill triage using AIS anomaly detection and SAR
> browse imagery.* **Applied Geomatics** 18(4):182 (Springer Nature).
> https://doi.org/10.1007/s12518-026-00837-5

The same code is archived with the article as Online Resource 1.

## How it works

```
AIS reports (aisstream.io WebSocket)
   │  node_ingestor/listener.js forwards every position report to POST /ingest
   ▼
Validation (ais_validation.py)
   drops "not available" positions, 0°N 0°E, placeholder and non-vessel MMSIs;
   treats SOG 102.2–102.3 kn and COG 360° as missing
   ▼
Anomaly engine (anomaly_engine.py), every 10 s on all vessels seen in the last 6 h
   Layer 1  speed change > 10 kn within 10 min (reason tag; lifts the review cooldown)
   Layer 2  HDBSCAN (haversine) + physical checks:
              dense  ≥ 6 vessels with median nearest-neighbour ≤ 5 km, or ≥ 6 within 1 km → SAFE
              coastal  within 10 km of land → no rendezvous/loitering rules
              rendezvous  2–5 vessels within 1 km, all < 3 kn, offshore → escalate
              loitering  isolated offshore and < 1 kn → escalate
   Layer 3  Isolation Forest on 5 trajectory features (last 20 reports),
              contamination 0.01 (alarm budget) + robust z-score > 3.5 → escalate
   24 h cooldown after a vessel has been reviewed
   ▼
Image worker (watchdog.py, satellite.py, browse_geo.py)
   first Sentinel-1 browse image acquired from 1 h before to 48 h after the anomaly;
   exponential back-off; EXPIRED after 60 h
   25 km crop centred on the vessel (geometry read from the preview: map-projected or radar)
   ▼
Classifier (ml_logic.py)
   frozen EfficientNet-B4 backbone + the head whose region contains the vessel
   (models/regional_heads.json): Eastern Mediterranean or Arabian/Persian Gulf;
   elsewhere the CSIRO-trained head at threshold 0.75
   ▼
Append-only event log (alert_events table) and React + Leaflet dashboard
```

Every threshold and window is in `config.py`, and each can be overridden with the
environment variable named there.

## Main results

| Test | Result |
|---|---|
| Whole browse scene, CSIRO-trained model (E. Mediterranean) | AUC 0.55 |
| 25 km vessel-centred crop, same model, open water | AUC 0.83 |
| Browse-trained head, E. Mediterranean (5-fold CV by day) | AUC 0.909 (CSIRO-trained 0.638, gradient boosting on 24 statistics 0.840) |
| Gulf-trained head, held-out year (train 2022, test 2021) | AUC 0.882 (CSIRO-trained 0.844) |
| Slick 5 km / 10 km from the crop centre | AUC change −0.011 / −0.041 |
| CSIRO benchmark, grouped by source image / region held out | F1 0.947 / 0.859 |
| Layer 2, 12,068 real vessels: possible rendezvous | 0.09% (cluster size alone: 38%) |
| Whole cascade, US fleet 14 June 2023 | 2.2–2.6% of vessels escalated per cycle |
| Wait for the next acquisition (median) / publication lag (median) | 31–43 h / 10.2 h |
| TARS retrieval and processing | about 3 s |

Details and scripts: `experiments/README` files listed below.

## Repository structure

```
run.py               starts the FastAPI backend and the anomaly engine
main.py              REST endpoints (/ingest, /alerts, /events, ...)
anomaly_engine.py    Layers 1-3 and the cascade
trajectory.py        trajectory features for Layer 3
coast.py             offline land mask checks
ais_validation.py    AIS report validation
watchdog.py          image worker: search, crop, classify, log
satellite.py         NASA ASF catalogue search and browse download
browse_geo.py        browse georeferencing and vessel-centred cropping
ml_logic.py          EfficientNet-B4 backbone, regional heads, (optional) STI
events.py            append-only event log
config.py            all settings
models/              regional heads (+ registry); CSIRO weights are a release download
node_ingestor/       AIS WebSocket listener
frontend/            React + Leaflet dashboard
tests/               pytest suite (no network, temporary database)
experiments/
  browse_eval/       E. Mediterranean browse evaluation and the browse-trained head
  gulf_eval/         Arabian/Persian Gulf evaluation and the Gulf head
  revision/          CSIRO re-validation, B0 vs B4, latency, AIS layers, off-centre,
                     simple baseline
  figures/           scripts that draw the paper's result figures
scripts/             earlier evaluation scripts for the CSIRO model and STI (kept for
                     reference; not used by the current system)
notebooks/TARS.ipynb training notebook for the CSIRO model
```

## Requirements and installation

Python 3.11 or newer, Node.js 18 or newer.

```bash
pip install -r requirements.txt
cd node_ingestor && npm install && cd ..
cd frontend && npm install && cd ..
```

**Model weights.** The two regional heads (`models/browse_head_v1.weights.h5`,
`models/gulf_head_v1.weights.h5`) are included. The EfficientNet-B4 backbone with the
CSIRO-trained head (`optisar_b4_retrained_weights.weights.h5`, 83 MB) is too large for
git. Download it from the repository's Releases page, or ask the authors, and place it in
`models/`. It is only needed to score crops outside the two trained regions or to retrain
heads from images; the paper's tables and figures run from the saved scores and features
in `experiments/`.

**AIS key.** Copy `node_ingestor/.env.example` to `node_ingestor/.env` and set
`AIS_API_KEY` (free from [aisstream.io](https://aisstream.io)).

## Running

Open four terminals in the project root:

```bash
python run.py                          # 1. backend + anomaly engine (http://127.0.0.1:8000, docs at /docs)
cd node_ingestor && node listener.js   # 2. AIS ingestor
python watchdog.py                     # 3. image worker
cd frontend && npm run dev             # 4. dashboard (http://localhost:5173)
```

On the dashboard, orange markers are vessels under investigation, red markers are
crops classified as oil (for analyst review) and green markers are vessels cleared by
imaging.

## Event log

The `oil_spill_alerts` table only holds each vessel's current state. Everything needed
to evaluate the system later is also written to the append-only `alert_events` table:
`KINEMATIC_FLAG`, `ESCALATED` (with reasons), `CLASSIFIED` (scores, thresholds, head,
crop geometry, image–anomaly delay, configuration and git commit), `CROP_FAILED`,
`EXPIRED`, `OPERATOR_ACTION`, and one `ENGINE_CYCLE` summary per engine pass.
Browse them at `GET /events?event_type=CLASSIFIED`.

## Tests

```bash
python -m pytest -q tests
```

The tests use a temporary database and never touch the network. They cover AIS
validation, trajectory features, each Layer 2/3 scenario (port veto, rendezvous,
harbour exclusion, zig-zag, loitering, cooldown), browse georeferencing and cropping,
the image worker's timing and back-off, STI reproducibility and the event log.

## Data

No raw data is included.

| Data | Source |
|---|---|
| CSIRO Sentinel-1 oil / no-oil chips (5,630, 26 region codes) | [Kaggle](https://www.kaggle.com/datasets/harikrishnacs/sentinel-1-sar-oil-spill-detection-dataset) |
| E. Mediterranean 2019 oil and look-alike labels (Yang et al. 2025) | PANGAEA, doi:10.1594/PANGAEA.980773 (CC-BY 4.0) |
| GulfOilSAR 2021–2022 labels (Culot et al. 2026) | Zenodo, doi:10.5281/zenodo.23261453 (CC-BY 4.0) |
| Sentinel-1 footprints and browse images | NASA Alaska Satellite Facility (`asf-search`) |
| US AIS, 14 June 2023 | NOAA Marine Cadastre |

The `experiments/*/README.md` files explain how to fetch each one and in which order
to run the scripts. `CHANGES.md` lists how the system differs from the version first
submitted for review.

## Statistical Texture Injection (optional)

An earlier design adapted the browse image towards the CSIRO training data with
Statistical Texture Injection (speckle-like noise, CLAHE and a brightness shift). On
labelled browse crops it did not improve discrimination, so the current system scores
the raw crop. It is still available for comparison with `TARS_MODEL=csiro`.

## Citation

If you use this code or the results, please cite the paper:

```bibtex
@article{miranda2026tars,
  title   = {{TARS}: a framework for marine oil spill triage using {AIS} anomaly detection and {SAR} browse imagery},
  author  = {Miranda, Janice and Nair, Navya and Shringare, Gargi and Hussain, Nazeefa and Bhadange, Kirti and Tripathy, Amiya Kumar},
  journal = {Applied Geomatics},
  year    = {2026},
  volume  = {18},
  number  = {4},
  pages   = {182},
  publisher = {Springer Nature},
  doi     = {10.1007/s12518-026-00837-5}
}
```

The labelled datasets belong to their authors: please also cite Yang et al. (2025) for
the Eastern Mediterranean labels, Culot et al. (2026) for GulfOilSAR, and the CSIRO
dataset if you use them.

## Licence

The code is released under the MIT licence (see `LICENSE`). The datasets used in the
paper keep their own licences.
