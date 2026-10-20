"""Map lat/lon to browse-image pixels via the 4 footprint corners (bilinear)."""
import numpy as np
from global_land_mask import globe

# 8 ways to assign footprint corners c0..c3 to image corners (TL, TR, BR, BL)
ORIENTS = []
for start in range(4):
    for direction in (1, -1):
        ORIENTS.append([(start + direction * k) % 4 for k in range(4)])


def corners(geom):
    pts = geom["coordinates"][0][:4]
    return np.array(pts, dtype=float)  # (4, [lon, lat])


def pixel_grid_latlon(c, orient, h, w):
    """lat/lon of every pixel centre for corner assignment `orient`."""
    tl, tr, br, bl = (c[i] for i in orient)
    u = (np.arange(w) + 0.5) / w
    v = (np.arange(h) + 0.5) / h
    U, V = np.meshgrid(u, v)
    pts = (tl[None, None] * ((1 - U) * (1 - V))[..., None] + tr[None, None] * (U * (1 - V))[..., None]
           + br[None, None] * (U * V)[..., None] + bl[None, None] * ((1 - U) * V)[..., None])
    return pts[..., 1], pts[..., 0]


def latlon_to_pixel(c, orient, lat, lon, h, w, iters=20):
    """Invert the bilinear map with Newton iterations. Returns (x, y) pixels."""
    tl, tr, br, bl = (c[i] for i in orient)
    target = np.array([lon, lat])
    u, v = 0.5, 0.5
    for _ in range(iters):
        p = tl * (1 - u) * (1 - v) + tr * u * (1 - v) + br * u * v + bl * (1 - u) * v
        du = -tl * (1 - v) + tr * (1 - v) + br * v - bl * v
        dv = -tl * (1 - u) - tr * u + br * u + bl * (1 - u)
        J = np.stack([du, dv], axis=1)
        step = np.linalg.solve(J, target - p)
        u, v = u + step[0], v + step[1]
    return u * w, v * h


def land_mask(c, orient, h, w, step=4):
    lat, lon = pixel_grid_latlon(c, orient, h // step, w // step)
    return globe.is_land(np.clip(lat, -89.9, 89.9), ((lon + 180) % 360) - 180)


def bbox(geom):
    c = corners(geom)
    return c[:, 0].min(), c[:, 0].max(), c[:, 1].min(), c[:, 1].max()  # lon0, lon1, lat0, lat1


def bbox_latlon_to_pixel(geom, lat, lon, h, w):
    """Browse images are north-up rasters covering the footprint's lat/lon bounding box."""
    lon0, lon1, lat0, lat1 = bbox(geom)
    return (lon - lon0) / (lon1 - lon0) * w, (lat1 - lat) / (lat1 - lat0) * h


def bbox_land_mask(geom, h, w, step=4):
    lon0, lon1, lat0, lat1 = bbox(geom)
    xs = lon0 + (np.arange(w // step) + 0.5) / (w // step) * (lon1 - lon0)
    ys = lat1 - (np.arange(h // step) + 0.5) / (h // step) * (lat1 - lat0)
    LON, LAT = np.meshgrid(xs, ys)
    return globe.is_land(LAT, LON)
