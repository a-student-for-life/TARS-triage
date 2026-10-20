"""End-to-end latency components that do not depend on TARS itself.

1. Acquisition availability: for random open-sea points and random anomaly times
   in a 30-day window, the wait until the next Sentinel-1 IW GRD acquisition whose
   footprint covers the point (ASF search API footprints).
2. Publication lag: acquisition stop time to the Last-Modified time of the browse
   JPEG on the ASF data pool (when it became downloadable), for a sample of products.

Output: latency_availability.json (+ publication_lag.csv)
"""
import csv, json, os, random, sys, time, urllib.request, ssl
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import numpy as np
from shapely.geometry import Point, shape
from global_land_mask import globe

HERE = os.path.dirname(os.path.abspath(__file__))
CTX = ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE"))
REGIONS = {"eastern_med": (27.0, 30.0, 37.0, 37.0), "arabian_gulf": (47.5, 22.5, 60.0, 30.5)}  # lon0, lat0, lon1, lat1
START, END = "2026-09-01T00:00:00Z", "2026-10-15T00:00:00Z"
random.seed(0)


def get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    return urllib.request.urlopen(req, context=CTX, timeout=60)


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


out, lag_rows = {}, []
for name, (x0, y0, x1, y1) in REGIONS.items():
    url = ("https://api.daac.asf.alaska.edu/services/search/param?platform=SENTINEL-1&beamMode=IW"
           f"&processingLevel=GRD_HD&bbox={x0},{y0},{x1},{y1}&start={START}&end={END}&output=geojson&maxResults=5000")
    feats = json.load(get(url))["features"]
    acq = sorted(((ts(f["properties"]["startTime"]), shape(f["geometry"]), f["properties"]) for f in feats),
                 key=lambda a: a[0])
    t0, t1 = ts(START), ts(END)
    # 1. waits: anomaly times uniform in the first 30 days so the window after each is observed
    waits = []
    while len(waits) < 2000:
        lon, lat = random.uniform(x0, x1), random.uniform(y0, y1)
        if globe.is_land(lat, lon):
            continue
        t = t0 + (t1 - t0) * random.uniform(0, 30 / 44)
        p = Point(lon, lat)
        nxt = next((a for a, g, _ in acq if a >= t and g.contains(p)), None)
        waits.append((nxt - t).total_seconds() / 3600 if nxt else float("inf"))
    w = np.array(waits)
    # 2. publication lag on a sample of products
    sample = random.sample(acq, min(60, len(acq)))
    lags = []
    for a, _, pr in sample:
        try:
            r = get(pr["browse"][0], {"Range": "bytes=0-0"})
            lm = parsedate_to_datetime(r.headers["Last-Modified"])
            lag = (lm - ts(pr["stopTime"])).total_seconds() / 3600
            lags.append(lag)
            lag_rows.append([name, pr["sceneName"], pr["stopTime"], lm.isoformat(), f"{lag:.3f}"])
        except Exception as e:  # noqa: BLE001
            print("skip", pr["sceneName"], e, flush=True)
        time.sleep(0.2)
    lg = np.array(lags)
    out[name] = {
        "n_products": len(acq), "n_points": len(w), "never_covered_within_14d": int(np.isinf(w).sum()),
        "wait_h_median": float(np.median(w)), "wait_h_p90": float(np.percentile(w, 90)),
        "wait_h_mean_finite": float(w[np.isfinite(w)].mean()),
        "share_wait_le_24h": float(np.mean(w <= 24)), "share_wait_le_48h": float(np.mean(w <= 48)),
        "n_lag": len(lg), "pub_lag_h_median": float(np.median(lg)), "pub_lag_h_p10": float(np.percentile(lg, 10)),
        "pub_lag_h_p90": float(np.percentile(lg, 90)), "pub_lag_h_min": float(lg.min()), "pub_lag_h_max": float(lg.max()),
    }
    print(name, out[name], flush=True)

out["window"] = [START, END]
json.dump(out, open(os.path.join(HERE, "latency_availability.json"), "w"), indent=1)
with open(os.path.join(HERE, "publication_lag.csv"), "w", newline="") as fh:
    csv.writer(fh).writerows([["region", "scene", "stop_time", "browse_last_modified", "lag_h"], *lag_rows])
print("DONE")
