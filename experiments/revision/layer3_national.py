"""Layer 3 (and the full engine cycle) on a real national fleet.

The 12,068-vessel snapshot of the deployment database has no position histories, so
Layer 3 could not be evaluated on it. Here one full day of US AIS (NOAA Marine
Cadastre, 14 June 2023, all US coastal receivers) is used. For each snapshot time t:
  - a fresh TARS database is created;
  - every vessel with a valid report in (t - 6 h, t] is inserted with its latest
    position and speed (what the live table holds), and its last 20 valid reports up
    to t are inserted as its position history (what _append_history keeps);
  - the unmodified anomaly_engine.run_engine_cycle runs once at t, on the whole fleet.
Reports pass the backend's own validation (ais_validation.clean_report rules).
Marine Cadastre provides at most about one report per vessel per minute; the deployed
system received every message, so its 20-report histories were shorter in time.
A second run keeps at most one report per vessel per 5 min (as in the replays).
Layer 1 is not replayed (it only lifts the review cooldown, which is empty here).
Output: layer3_national.json
"""
import json, os, sys, zipfile
from datetime import timezone
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
ZIP = sys.argv[1]
db_path = "/dev/shm/tars_l3nat.db"
os.environ["TARS_DB_URL"] = f"sqlite:///{db_path}"
sys.path.insert(0, REPO); os.chdir(REPO)
import config, database, models, anomaly_engine  # noqa: E402
from ais_validation import is_vessel_mmsi, SOG_NOT_AVAILABLE_MIN, COG_NOT_AVAILABLE  # noqa: E402
from sqlalchemy import insert  # noqa: E402

parts, n_raw = [], 0
with zipfile.ZipFile(ZIP) as z:
    for ch in pd.read_csv(z.open(z.namelist()[0]), usecols=["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG"],
                          chunksize=1_000_000):
        n_raw += len(ch)
        for c in ("MMSI", "LAT", "LON", "SOG", "COG"):
            ch[c] = pd.to_numeric(ch[c], errors="coerce")
        ch = ch.dropna(subset=["MMSI", "LAT", "LON"])
        ch["t"] = pd.to_datetime(ch.BaseDateTime, utc=True)
        parts.append(ch.drop(columns="BaseDateTime").astype(dict(MMSI="int64", LAT="float64", LON="float64",
                                                                  SOG="float32", COG="float32")))
a = pd.concat(parts, ignore_index=True); del parts
ok_mmsi = {m for m in a.MMSI.unique() if is_vessel_mmsi(int(m))}
a = a[a.MMSI.isin(ok_mmsi) & a.LAT.between(-90, 90) & a.LON.between(-180, 180) & ~((a.LAT == 0) & (a.LON == 0))]
a.loc[~a.SOG.between(0, SOG_NOT_AVAILABLE_MIN, inclusive="left"), "SOG"] = np.nan
a.loc[~a.COG.between(0, COG_NOT_AVAILABLE, inclusive="left"), "COG"] = np.nan
a = a.sort_values(["MMSI", "t"]).reset_index(drop=True)
print("raw", n_raw, "valid", len(a), "vessels", a.MMSI.nunique(), flush=True)


def thin(df, gap_s):
    if gap_s <= 0:
        return df
    keep, last = np.zeros(len(df), bool), {}
    for i, (m, t) in enumerate(zip(df.MMSI.values, df.t.values.astype("datetime64[ns]").astype("int64"))):
        p = last.get(m)
        if p is None or t - p >= gap_s * 1e9:
            keep[i] = True; last[m] = t
    return df[keep]


def snapshot(df, t):
    for s in ("", "-wal", "-shm"):
        if os.path.exists(db_path + s):
            os.remove(db_path + s)
    database.engine.dispose()
    database.Base.metadata.create_all(bind=database.engine)
    database.ensure_alert_schema()
    upto = df[df.t <= t]
    fresh = upto[upto.t > t - pd.Timedelta(hours=config.STALE_POSITION_HOURS)]
    last = fresh.groupby("MMSI").tail(1)
    hist = upto[upto.MMSI.isin(last.MMSI)].groupby("MMSI").tail(config.HISTORY_LEN)
    py = lambda x: x.to_pydatetime()
    with database.engine.begin() as con:
        con.execute(insert(models.OilSpillAlert.__table__), [
            dict(ship_id=str(r.MMSI), lat=r.LAT, lon=r.LON, speed=None if np.isnan(r.SOG) else float(r.SOG),
                 anomaly_type="NONE", status=models.JobStatus.UNPROCESSED, anomaly_reason="UNKNOWN",
                 last_report_at=py(r.t), created_at=py(r.t), updated_at=py(r.t)) for r in last.itertuples()])
        con.execute(insert(models.PositionReport.__table__), [
            dict(ship_id=str(r.MMSI), reported_at=py(r.t), lat=r.LAT, lon=r.LON,
                 speed=None if np.isnan(r.SOG) else float(r.SOG), course=None if np.isnan(r.COG) else float(r.COG))
            for r in hist.itertuples()])
    db = database.SessionLocal()
    ships = db.query(models.OilSpillAlert).all()
    ctx = pd.Series([c[0] for c in anomaly_engine._cluster(ships)]).value_counts().to_dict()
    summary = anomaly_engine.run_engine_cycle(db, now=py(t))
    ev = pd.read_sql("select ship_id, reason from alert_events where event_type='ESCALATED'", database.engine)
    db.close()
    n = len(last)
    reasons = ev.reason.str.replace(r"_CLUSTER_\d+", "", regex=True).value_counts().to_dict()
    only_l3 = int((ev.reason == "BEHAVIOUR_OUTLIER_ISO_FOREST").sum())
    return dict(time=str(t), vessels=n, contexts={k: int(v) for k, v in ctx.items()},
                judged_by_layer3=int(summary.get("judged_by_behaviour", 0)),
                layer3_outliers=int(summary.get("behaviour_outliers", 0)),
                escalated=int(summary.get("escalated", 0)), escalated_pct=round(100 * summary.get("escalated", 0) / n, 2),
                escalated_layer3_only=only_l3, reasons={k: int(v) for k, v in reasons.items()},
                median_history_span_min=float(hist.groupby("MMSI").t.agg(lambda s: (s.max() - s.min()).total_seconds() / 60).median()))


out = dict(source=os.path.basename(ZIP), raw_reports=n_raw, valid_reports=len(a), vessels_in_day=int(a.MMSI.nunique()))
day = a.t.dt.floor("D").iloc[0]
times = [day + pd.Timedelta(hours=h) for h in (3, 9, 15, 21)]
for name, gap in (("native_1min", 0), ("thinned_5min", 300)):
    df = thin(a, gap)
    out[name] = [snapshot(df, t) for t in times]
    for s in out[name]:
        print(name, s["time"], s["vessels"], s["contexts"], "L3", s["layer3_outliers"], "/", s["judged_by_layer3"],
              "esc", s["escalated"], f"({s['escalated_pct']}%)", s["reasons"], flush=True)
json.dump(out, open(os.path.join(HERE, os.getenv("TARS_L3_OUT", "layer3_national.json")), "w"), indent=1)
