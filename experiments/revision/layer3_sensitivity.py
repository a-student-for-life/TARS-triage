"""Layer-3 sensitivity on the national fleet (14 June 2023, native sampling, 4 snapshots).

Same inputs as layer3_national.py (latest position per vessel fresh in 6 h, last 20
valid reports), but the Layer-3 decision is recomputed for other settings of its two
thresholds: the Isolation Forest contamination ('auto' as deployed, or a fixed share)
and the robust z-score a vessel must exceed in at least one feature (3.5 as deployed).
Reported: vessels flagged by Layer 3 outside dense areas (the ones that are escalated).
The deployed cell reproduces the engine's count. No labels exist, so this shows the
alarm volume each setting produces, not its accuracy.
Output: layer3_sensitivity.json
"""
import json, os, sys, zipfile
from types import SimpleNamespace
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO); os.chdir(REPO)
import config, anomaly_engine  # noqa: E402
from trajectory import trajectory_features  # noqa: E402
from ais_validation import is_vessel_mmsi, SOG_NOT_AVAILABLE_MIN, COG_NOT_AVAILABLE  # noqa: E402

parts = []
with zipfile.ZipFile(sys.argv[1]) as z:
    for ch in pd.read_csv(z.open(z.namelist()[0]), usecols=["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG"],
                          chunksize=1_000_000):
        for c in ("MMSI", "LAT", "LON", "SOG", "COG"):
            ch[c] = pd.to_numeric(ch[c], errors="coerce")
        ch = ch.dropna(subset=["MMSI", "LAT", "LON"])
        ch["t"] = pd.to_datetime(ch.BaseDateTime, utc=True)
        parts.append(ch.drop(columns="BaseDateTime").astype(dict(MMSI="int64", SOG="float32", COG="float32")))
a = pd.concat(parts, ignore_index=True); del parts
ok = {m for m in a.MMSI.unique() if is_vessel_mmsi(int(m))}
a = a[a.MMSI.isin(ok) & a.LAT.between(-90, 90) & a.LON.between(-180, 180) & ~((a.LAT == 0) & (a.LON == 0))]
a.loc[~a.SOG.between(0, SOG_NOT_AVAILABLE_MIN, inclusive="left"), "SOG"] = np.nan
a.loc[~a.COG.between(0, COG_NOT_AVAILABLE, inclusive="left"), "COG"] = np.nan
a = a.sort_values(["MMSI", "t"]).reset_index(drop=True)

CONTAM = ["auto", 0.05, 0.02, 0.01]
ZS = [3.5, 5.0, 7.0, 10.0]
day = a.t.dt.floor("D").iloc[0]
res = {f"{c}|{z}": [] for c in CONTAM for z in ZS}
per = []
for h in (3, 9, 15, 21):
    t = day + pd.Timedelta(hours=h)
    upto = a[a.t <= t]
    last = upto[upto.t > t - pd.Timedelta(hours=config.STALE_POSITION_HOURS)].groupby("MMSI").tail(1)
    ships = [SimpleNamespace(ship_id=str(r.MMSI), lat=r.LAT, lon=r.LON, speed=None if np.isnan(r.SOG) else float(r.SOG))
             for r in last.itertuples()]
    dense = {s.ship_id for s, c in zip(ships, anomaly_engine._cluster(ships)) if c[0] == "dense"}
    hist = upto[upto.MMSI.isin(last.MMSI)].groupby("MMSI").tail(config.HISTORY_LEN)
    feats, owners = [], []
    for m, g in hist.groupby("MMSI"):
        if len(g) < config.MIN_HISTORY_FOR_BEHAVIOUR:
            continue
        reps = [SimpleNamespace(lat=r.LAT, lon=r.LON, speed=None if np.isnan(r.SOG) else float(r.SOG),
                                course=None if np.isnan(r.COG) else float(r.COG)) for r in g.itertuples()]
        f = trajectory_features(reps)
        if f is not None:
            feats.append(f); owners.append(str(m))
    X = np.vstack(feats); owners = np.array(owners)
    eligible = ~np.isin(owners, list(dense))
    med = np.median(X, axis=0); mad = np.median(np.abs(X - med), axis=0) * 1.4826
    rz = (np.abs(X - med) / (mad + anomaly_engine.FEATURE_FLOOR)).max(axis=1)
    per.append(dict(time=str(t), vessels=len(ships), judged=len(owners), non_dense_judged=int(eligible.sum())))
    for c in CONTAM:
        pred = IsolationForest(contamination=c, random_state=config.ISO_RANDOM_STATE).fit_predict(X) == -1
        for z in ZS:
            n = int((pred & (rz > z) & eligible).sum())
            res[f"{c}|{z}"].append(dict(flagged=n, pct_of_vessels=round(100 * n / len(ships), 2)))
    print(t, per[-1], {k: v[-1]["flagged"] for k, v in res.items()}, flush=True)

table = {k: dict(mean_flagged=float(np.mean([x["flagged"] for x in v])),
                 mean_pct=float(np.mean([x["pct_of_vessels"] for x in v])),
                 range_pct=[min(x["pct_of_vessels"] for x in v), max(x["pct_of_vessels"] for x in v)]) for k, v in res.items()}
json.dump(dict(snapshots=per, settings=table), open(os.path.join(HERE, "layer3_sensitivity.json"), "w"), indent=1)
for k, v in table.items():
    print(k, round(v["mean_flagged"]), v["mean_pct"], v["range_pct"])
