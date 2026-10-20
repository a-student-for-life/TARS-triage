"""Download everything run_eval.py needs (about 280 MB of browse images).

1. ESSD / PANGAEA table of labelled oil slicks and look-alikes (CC-BY 4.0):
   Yang & Singha (2025), https://doi.org/10.1594/PANGAEA.980773
2. Footprints and browse URLs for every Sentinel-1 product, from NASA ASF.
3. The browse JPEGs themselves, into ./browse/.
"""
import io, json, os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
COLS = ['set', 'jpg', 'xml', 'tag', 'patch', 'start', 'end', 'sid', 'pw', 'ph', 'pul_lon', 'pul_lat', 'pur_lon',
        'pur_lat', 'pbr_lon', 'pbr_lat', 'pbl_lon', 'pbl_lat', 'oul_lon', 'oul_lat', 'our_lon', 'our_lat', 'obr_lon',
        'obr_lat', 'obl_lon', 'obl_lat', 'xmin', 'ymin', 'xmax', 'ymax', 'size']

table = os.path.join(HERE, 'essd_table.csv')
if not os.path.exists(table):
    txt = requests.get('https://doi.pangaea.de/10.1594/PANGAEA.980773?format=textfile', timeout=120).text
    d = pd.read_csv(io.StringIO(txt.split('*/\n', 1)[1]), sep='\t'); d.columns = COLS
    d.to_csv(table, index=False)
d = pd.read_csv(table)

fp_path = os.path.join(HERE, 'footprints.json')
if not os.path.exists(fp_path):
    ids = sorted({p.replace('.SAFE', '') for s in d.sid for p in s.split(';')})
    fp = {}
    for i in range(0, len(ids), 40):
        r = requests.post('https://api.daac.asf.alaska.edu/services/search/param', timeout=120,
                          data={'granule_list': ','.join(ids[i:i + 40]), 'output': 'geojson', 'processingLevel': 'GRD_HD'})
        r.raise_for_status()
        for f in r.json()['features']:
            p = f['properties']
            fp[p['sceneName']] = {'browse': p.get('browse'), 'flight': p.get('flightDirection'),
                                  'geometry': f['geometry'], 'start': p.get('startTime')}
    json.dump(fp, open(fp_path, 'w'))
fp = json.load(open(fp_path))

os.makedirs(os.path.join(HERE, 'browse'), exist_ok=True)
def get(item):
    name, info = item
    path = os.path.join(HERE, 'browse', name + '.jpg')
    if os.path.exists(path):
        return
    for _ in range(3):
        try:
            r = requests.get(info['browse'][0], timeout=120)
            if r.ok:
                open(path, 'wb').write(r.content); return
        except requests.RequestException:
            pass
with ThreadPoolExecutor(8) as ex:
    list(ex.map(get, fp.items()))
print(len(os.listdir(os.path.join(HERE, 'browse'))), 'browse images ready')
