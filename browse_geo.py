"""Vessel-centred cropping of Sentinel-1 browse previews.

ASF browse JPEGs come in two geometries, and the image itself tells which:
  * map-projected: a north-up raster of the footprint's lat/lon bounding box. The
    footprint is tilted inside the frame, so all four image corners are no-data
    (black). A lat/lon maps to a pixel linearly.
  * radar geometry: the image is the footprint itself (no black corners). A lat/lon
    is mapped through the four footprint corners (bilinear); the corner assignment
    was checked against coastlines in the Mediterranean, the Gulf and California.
Treating a radar-geometry image as map-projected puts the crop tens of km away from
the vessel, so crop_around() always picks the mapping from the image.

Why it matters: on 681 expert-labelled scenes, classifying the whole browse image
gave AUC 0.55 (chance); 25 km crops gave AUC 0.83 in open water
(experiments/browse_eval/README.md).
"""
import math
import os

import cv2
import numpy as np

import config

EARTH_RADIUS_KM = 6371.0088
NO_DATA_LEVEL = 3  # browse pixels outside the footprint are (near) black


def footprint_bbox(geometry):
    """(lon_min, lon_max, lat_min, lat_max) of a GeoJSON polygon."""
    pts = np.array(geometry["coordinates"][0], dtype=float)
    return pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()


# footprint corner k (ASF polygon order) that sits at image TL, TR, BR, BL in radar geometry
RADAR_CORNERS = (1, 2, 3, 0)


def is_map_projected(image, frac=0.03, share=0.9):
    """True when all four image corners are no-data, i.e. a north-up bounding-box raster."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    k = max(4, int(frac * min(h, w)))
    corners = [gray[:k, :k], gray[:k, -k:], gray[-k:, :k], gray[-k:, -k:]]
    return all(np.mean(c < NO_DATA_LEVEL) > share for c in corners)


def _radar_corners(geometry):
    pts = np.array(geometry["coordinates"][0][:4], dtype=float)  # (lon, lat)
    return [pts[i] for i in RADAR_CORNERS]


def _km(lon1, lat1, lon2, lat2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def latlon_to_pixel(geometry, lat, lon, height, width, projected=True):
    if projected:
        lon0, lon1, lat0, lat1 = footprint_bbox(geometry)
        return (lon - lon0) / (lon1 - lon0) * width, (lat1 - lat) / (lat1 - lat0) * height
    tl, tr, br, bl = _radar_corners(geometry)
    target, u, v = np.array([lon, lat], dtype=float), 0.5, 0.5
    for _ in range(25):  # invert the bilinear map (Newton)
        p = tl * (1 - u) * (1 - v) + tr * u * (1 - v) + br * u * v + bl * (1 - u) * v
        du = -tl * (1 - v) + tr * (1 - v) + br * v - bl * v
        dv = -tl * (1 - u) - tr * u + br * u + bl * (1 - u)
        step = np.linalg.solve(np.stack([du, dv], axis=1), target - p)
        u, v = u + step[0], v + step[1]
    return float(u * width), float(v * height)


def pixel_size_km(geometry, width, projected=True):
    if projected:
        lon0, lon1, lat0, lat1 = footprint_bbox(geometry)
        mid = math.radians((lat0 + lat1) / 2)
        return (lon1 - lon0) * math.cos(mid) * math.pi / 180 * EARTH_RADIUS_KM / width
    tl, tr, br, bl = _radar_corners(geometry)
    return (_km(*tl, *tr) + _km(*bl, *br)) / 2 / width


def crop_around(image, geometry, lat, lon, crop_km=None):
    """Cut a square window of crop_km centred on (lat, lon).

    Returns (crop, info). crop is None when the window doesn't fit inside the
    image or is mostly outside the footprint; info always says why."""
    crop_km = crop_km or config.CROP_KM
    h, w = image.shape[:2]
    projected = is_map_projected(image)
    x, y = latlon_to_pixel(geometry, lat, lon, h, w, projected)
    px_km = pixel_size_km(geometry, w, projected)
    half = int(round(crop_km / px_km / 2))
    x0, y0 = int(round(x)) - half, int(round(y)) - half
    info = {"centre_px": [round(x, 1), round(y, 1)], "half_px": half, "crop_km": crop_km,
            "pixel_km": round(px_km, 4), "geometry": "map-projected" if projected else "radar"}
    if half < 8:
        return None, {**info, "reason": "crop_too_small"}
    if x0 < 0 or y0 < 0 or x0 + 2 * half > w or y0 + 2 * half > h:
        return None, {**info, "reason": "vessel_too_close_to_image_edge"}
    crop = image[y0:y0 + 2 * half, x0:x0 + 2 * half]
    gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    no_data = float(np.mean(gray < NO_DATA_LEVEL))
    info.update(box_px=[x0, y0, x0 + 2 * half, y0 + 2 * half], no_data_frac=round(no_data, 3))
    if no_data > config.CROP_MAX_NO_DATA:
        return None, {**info, "reason": "vessel_outside_imaged_area"}
    return crop, {**info, "reason": None}


def save_crop(preview_path, geometry, lat, lon, tag):
    """Crop a downloaded preview around (lat, lon) and save it losslessly (PNG,
    so no second round of JPEG artefacts). Returns (path or None, info)."""
    image = cv2.imread(preview_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, {"reason": "unreadable_preview"}
    crop, info = crop_around(image, geometry, lat, lon)
    if crop is None:
        return None, info
    stem = os.path.splitext(preview_path)[0]
    path = f"{stem}_crop_{tag}.png"
    cv2.imwrite(path, crop)
    return path, info
