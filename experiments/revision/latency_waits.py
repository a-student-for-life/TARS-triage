"""Re-run the acquisition-wait part of latency_availability.py (same seed, same catalogue window)
and save every wait, so the distribution can be plotted. Publication lag is not re-measured;
publication_lag.csv from the original run is used. Output: latency_waits.csv"""
import os, json, random, ssl, urllib.request, os
from datetime import datetime
import numpy as np, pandas as pd
from shapely.geometry import Point, shape
from global_land_mask import globe
CTX = ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE"))
REGIONS = {"eastern_med": (27.0, 30.0, 37.0, 37.0), "arabian_gulf": (47.5, 22.5, 60.0, 30.5)}
START, END = "2026-09-01T00:00:00Z", "2026-10-15T00:00:00Z"
random.seed(0)
ts = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))
rows = []
for name, (x0, y0, x1, y1) in REGIONS.items():
    url = ("https://api.daac.asf.alaska.edu/services/search/param?platform=SENTINEL-1&beamMode=IW"
           f"&processingLevel=GRD_HD&bbox={x0},{y0},{x1},{y1}&start={START}&end={END}&output=geojson&maxResults=5000")
    feats = json.load(urllib.request.urlopen(url, context=CTX, timeout=120))["features"]
    acq = sorted(((ts(f["properties"]["startTime"]), shape(f["geometry"]), f["properties"]) for f in feats), key=lambda a: a[0])
    t0, t1 = ts(START), ts(END); waits = []
    while len(waits) < 2000:
        lon, lat = random.uniform(x0, x1), random.uniform(y0, y1)
        if globe.is_land(lat, lon):
            continue
        t = t0 + (t1 - t0) * random.uniform(0, 30 / 44); p = Point(lon, lat)
        nxt = next((a for a, g, _ in acq if a >= t and g.contains(p)), None)
        waits.append((nxt - t).total_seconds() / 3600 if nxt else float("inf"))
    random.sample(acq, min(60, len(acq)))          # keep the random stream identical to the original run
    w = np.array(waits); print(name, len(acq), np.median(w), np.mean(w <= 24), np.mean(w <= 48))
    rows += [dict(region=name, wait_h=x) for x in waits]
pd.DataFrame(rows).to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "latency_waits.csv"), index=False)
