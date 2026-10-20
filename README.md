# TARS — Timely Anomaly Response System

## Overview

TARS is a real-time maritime oil spill detection system. It ingests global
AIS (Automatic Identification System) ship position data, identifies anomalous
vessel behaviour using unsupervised ML (HDBSCAN + Isolation Forest), then
automatically retrieves and classifies Sentinel-1 SAR browse imagery using a
EfficientNet-B4 classifier (ImageNet backbone frozen, classification head trained). Results are surfaced on an
interactive Leaflet map dashboard with per-ship status and satellite image popups.

## System Behaviour

| Visual signal | Meaning |
|---|---|
| **Red marker** | Vessel flagged as anomalous — under investigation |
| **Blinking pulsing dot** | Oil spill confirmed by AI (model score > 0.75) |
| **Green marker** | Vessel analysed and cleared — clean sea |

## Pipeline Architecture

```
Global AIS stream  (WebSocket — aisstream.io)
         │
         ▼
  node_ingestor/listener.js              ← Terminal 2
  Receives MMSI, lat, lon, speed
  POSTs every position report to /ingest
         │
         ▼
  FastAPI backend  (main.py)             ← Terminal 1  (python run.py)
  ┌──────────────────────────────────┐
  │  POST /ingest → SQLite DB        │
  │                                  │
  │  Anomaly Engine (background)     │
  │  ├─ HDBSCAN  (spatial density)   │  Port cluster vs open-ocean loner
  │  └─ Isolation Forest (kinematic) │  Speed-change anomaly detection
  │     ↓ status = PENDING_IMAGE     │
  └──────────────────────────────────┘
         │
         ▼
  watchdog.py                            ← Terminal 3  (python watchdog.py)
  Polls DB for PENDING_IMAGE jobs
  │
  ├── satellite.py
  │   └── NASA ASF API → Sentinel-1 GRD browse image (JPEG quicklook, ~500 KB)
  │
  └── ml_logic.py
      ├── TARS preprocessing pipeline:
      │     multiplicative speckle injection (sigma = 0.5)
      │   + Gaussian blur  (3×3)
      │   + CLAHE contrast enhancement  (clip = 2.0, tile = 8×8)
      │   + brightness shift  (+70)
      │   → adapts compressed JPEG previews to match SAR training distribution
      └── EfficientNet-B4 inference  (380×380 input)
          threshold = 0.75
          → OIL_DETECTED  or  SAFE
         │
         ▼
  frontend/  (React + Leaflet)           ← Terminal 4  (npm run dev)
  Polls GET /alerts every 3 s
  Renders markers + satellite image popups
  DISPATCH / IGNORE operator buttons
```

## Repository Structure

```
TARS/
├── run.py                 ← Terminal 1: starts FastAPI + anomaly engine
├── watchdog.py            ← Terminal 3: SAR fetch + ML inference worker
├── main.py                ← FastAPI app, REST endpoints, anomaly engine
├── ml_logic.py            ← EfficientNetB4 architecture + TARS preprocessing
├── satellite.py           ← NASA ASF API: Sentinel-1 browse image search/download
├── database.py            ← SQLAlchemy SQLite engine (WAL mode)
├── models.py              ← ORM: OilSpillAlert table + JobStatus enum
├── schemas.py             ← Pydantic request/response schemas
│
├── node_ingestor/
│   └── listener.js        ← Terminal 2: AIS WebSocket → HTTP bridge
│
├── frontend/              ← Terminal 4: React + Leaflet dashboard
│   └── src/
│       ├── App.jsx        ← Dashboard: polling, stats overlay
│       └── Map.jsx        ← Leaflet map with markers and popups
│
├── scripts/               ← Standalone evaluation and research scripts
│   ├── evaluate_model.py          ← Full CSIRO corpus: RAW vs TARS metrics
│   ├── simulate_browser_eval.py   ← Browser-preview simulation eval v1
│   ├── simulate_browser_eval_v2.py ← Browser eval v2 (brightness normalisation)
│   ├── browser_distribution_analysis.py ← Score distribution on live acquisitions
│   ├── run_ablation.py            ← A/B ablation on temp_images/
│   ├── noise_sensitivity.py       ← Speckle sigma sweep analysis
│   ├── check_image.py             ← Image texture-strength QC utility
│   ├── compare_histograms.py      ← Pixel histogram comparison utility
│   └── domain_adapt.py            ← Legacy domain adaptation visualisation
│
├── notebooks/
│   └── TARS.ipynb         ← Research notebook (EDA, model development)
│
├── models/                ← Model weights (gitignored — see Data section)
│   └── optisar_b4_retrained_weights.weights.h5   ← ACTIVE weights
│
├── data/
│   └── sample/            ← Reference SAR chips for manual testing
│
├── outputs/               ← Generated figures and CSVs (gitignored)
├── temp_images/           ← Runtime SAR downloads (gitignored)
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Requirements

- **Python** >= 3.10
- **Node.js** >= 18
- **npm** >= 9

```bash
# Python backend
pip install -r requirements.txt

# Node AIS ingestor
cd node_ingestor
npm install

# React frontend
cd frontend
npm install
```

## How to Run

Open **four terminals** in the project root:

**Terminal 1 — Python backend (FastAPI + anomaly engine)**
```bash
python run.py
```
Backend runs at `http://127.0.0.1:8000`.
Interactive API docs: `http://127.0.0.1:8000/docs`

**Terminal 2 — Node AIS ingestor**
```bash
cd node_ingestor
node listener.js
```

**Terminal 3 — Watchdog (SAR download + ML inference)**
```bash
python watchdog.py
```

**Terminal 4 — Frontend dashboard**
```bash
cd frontend
npm run dev
```
Dashboard available at `http://localhost:5173` (or the port Vite prints).

## Evaluation Scripts

All scripts are in `scripts/` and must be run from the project root:

```bash
python scripts/evaluate_model.py
```

> **Important:** Each script has a `VAL_DIR` / `DATA_DIR` / `SEARCH_DIR` constant
> near the top that must be updated to your local data path before running.

| Script | Purpose |
|---|---|
| `evaluate_model.py` | Full CSIRO corpus evaluation: confusion matrices, ROC curves, AUC/F1 |
| `simulate_browser_eval.py` | Browser-compression simulation (JPEG Q=30) eval v1 |
| `simulate_browser_eval_v2.py` | Browser eval v2 — adds brightness normalisation (mean ≈ 50) |
| `browser_distribution_analysis.py` | RAW vs TARS score distribution on live SAR acquisitions |
| `run_ablation.py` | A/B ablation: RAW vs TARS per-image on `temp_images/` |
| `noise_sensitivity.py` | Speckle sigma sweep: sigma ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9} |

## Data

Raw data is **not included** in this repository.

| Dataset | Description | Source |
|---|---|---|
| **CSIRO S1SAR benchmark** | 5,630 Sentinel-1 chips (3,725 clean sea / 1,905 oil spill, 400×400 px) | [Kaggle](https://www.kaggle.com/datasets/harikrishnacs/sentinel-1-sar-oil-spill-detection-dataset) — chips from 23 regions (region code in each filename) |
| **Sentinel-1 browse images** | Quicklook JPEG previews (~500 KB) fetched automatically at runtime | NASA ASF via `asf-search` |

**AIS stream:** Live ship positions from [AISstream.io](https://aisstream.io)
via WebSocket. An API key is required: copy `node_ingestor/.env.example` to `node_ingestor/.env` and set `AIS_API_KEY`.

**Model weights:** The two regional browse heads (`models/browse_head_v1.weights.h5`,
`models/gulf_head_v1.weights.h5`, 11 MB each) are included. The EfficientNet-B4 backbone with
the CSIRO-trained head (`optisar_b4_retrained_weights.weights.h5`, 83 MB) is too large for git:
download it from the repository's **Releases** page and place it in `models/`.

## Important System Constraints

**Satellite revisit interval:** Sentinel-1 revisits a location every few days.
The watchdog only accepts images acquired *after* the anomaly (1 h clock
tolerance) and up to 48 h later. Jobs with no image yet are retried with
exponential backoff; after the window (+12 h publication slack) they become
`EXPIRED`. All windows are in `config.py`.

**Browse images only:** TARS uses Sentinel-1 **browse (quicklook) products** via
NASA ASF, not full-resolution Level-1 GRD products. The TARS preprocessing
pipeline (speckle injection + CLAHE) is specifically designed to compensate for
JPEG compression artefacts in these ~500 KB previews. Do not replace browse
images with GRD data without retraining.



## Tests

```bash
pip install -r requirements.txt
python -m pytest -q tests
```

The tests use a temporary in-memory database and never touch `oil_spill.db`,
`temp_images/` or the network. They cover AIS validation, trajectory features,
each Layer 2/3 scenario (port veto, rendezvous, harbour exclusion, zig-zag,
loitering, cooldown), watchdog timing and backoff, STI reproducibility, and the
event log.

## Configuration and the event log

Every tunable that affects results is in `config.py` (each can be overridden
with the environment variable named there).

The `oil_spill_alerts` table holds only each ship's *current* state and is
overwritten as ships move. Everything worth counting later is also written to
the append-only `alert_events` table: `KINEMATIC_FLAG`, `ESCALATED`,
`CLASSIFIED` (with every STI draw's score, the settings used and the git commit),
`EXPIRED`, `OPERATOR_ACTION`, and one `ENGINE_CYCLE` summary per engine pass.
Browse recent events at `GET /events?event_type=CLASSIFIED`.

See `CHANGES.md` for what changed relative to the version used in the
original paper and what that means for reported numbers.
