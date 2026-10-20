"""Sensitivity of the Layer-2 spatial-context rules to their thresholds.

Snapshot: every vessel whose last report is within 1 h of the newest report in
the earlier deployment's database (25 May 2026), after AIS validation
(12,068 vessels). Each threshold is varied one at a time around the deployed
setting; the outcome is how many vessels each context receives.

  python layer2_sensitivity.py path/to/oil_spill.db  -> layer2_sensitivity.json
"""
import json, os, sys, time
from types import SimpleNamespace

import pandas as pd
import sqlite3

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
import config  # noqa: E402
import anomaly_engine  # noqa: E402
from ais_validation import clean_report  # noqa: E402

db = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "oil_spill.db")
d = pd.read_sql("select ship_id, lat, lon, speed, updated_at from oil_spill_alerts", sqlite3.connect(db))
d["u"] = pd.to_datetime(d.updated_at, format="mixed", utc=True)
d = d[d.u >= d.u.max() - pd.Timedelta(hours=1)]
ships = []
for r in d.itertuples():
    c, _ = clean_report(int(r.ship_id), r.lat, r.lon, r.speed)
    if c is not None:
        ships.append(SimpleNamespace(lat=c.lat, lon=c.lon, speed=c.speed))
print(len(ships), "vessels", flush=True)

DEFAULT = dict(RENDEZVOUS_MAX_KM=1.0, RENDEZVOUS_MAX_SPEED_KNOTS=3.0, OFFSHORE_KM=10.0,
               DENSE_CLUSTER_MIN_SIZE=6, DENSE_MAX_NN_KM=5.0)
SWEEP = dict(RENDEZVOUS_MAX_KM=[0.5, 1.0, 2.0, 5.0],
             RENDEZVOUS_MAX_SPEED_KNOTS=[1.0, 3.0, 5.0, 1e9],
             OFFSHORE_KM=[5.0, 10.0, 20.0],
             DENSE_CLUSTER_MIN_SIZE=[4, 6, 8],
             DENSE_MAX_NN_KM=[2.0, 5.0, 10.0])


def run(settings):
    for k, v in settings.items():
        setattr(config, k, v)
    t = time.time()
    ctx = anomaly_engine._cluster(ships)
    counts = pd.Series([c[0] for c in ctx]).value_counts().to_dict()
    return {k: int(counts.get(k, 0)) for k in ["dense", "coastal", "isolated", "rendezvous"]}, time.time() - t


results = {"n": len(ships), "default": DEFAULT, "runs": []}
base, secs = run(DEFAULT)
results["baseline"] = base
results["seconds_per_run"] = round(secs, 1)
print("baseline", base, flush=True)
for key, values in SWEEP.items():
    for v in values:
        s = dict(DEFAULT, **{key: v})
        counts, _ = run(s)
        results["runs"].append({"param": key, "value": v, **counts})
        print(key, v, counts, flush=True)
json.dump(results, open(os.path.join(HERE, "layer2_sensitivity.json"), "w"), indent=1)
print("DONE")
