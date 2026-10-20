"""Vessel-centred cropping of Sentinel-1 browse previews.

ASF browse JPEGs are north-up rasters covering the lat/lon bounding box of the
product footprint (verified against coastlines on ascending and descending
scenes; see experiments/browse_eval/orientation_check_bbox.jpg). So a lat/lon
maps to a pixel linearly, and a window of CROP_KM around the flagged vessel can
be cut out exactly.

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


def latlon_to_pixel(geometry, lat, lon, height, width):
    lon0, lon1, lat0, lat1 = footprint_bbox(geometry)
    return (lon - lon0) / (lon1 - lon0) * width, (lat1 - lat) / (lat1 - lat0) * height


def pixel_size_km(geometry, width):
    lon0, lon1, lat0, lat1 = footprint_bbox(geometry)
    mid = math.radians((lat0 + lat1) / 2)
    return (lon1 - lon0) * math.cos(mid) * math.pi / 180 * EARTH_RADIUS_KM / width


def crop_around(image, geometry, lat, lon, crop_km=None):
    """Cut a square window of crop_km centred on (lat, lon).

    Returns (crop, info). crop is None when the window doesn't fit inside the
    image or is mostly outside the footprint; info always says why."""
    crop_km = crop_km or config.CROP_KM
    h, w = image.shape[:2]
    x, y = latlon_to_pixel(geometry, lat, lon, h, w)
    half = int(round(crop_km / pixel_size_km(geometry, w) / 2))
    x0, y0 = int(round(x)) - half, int(round(y)) - half
    info = {"centre_px": [round(x, 1), round(y, 1)], "half_px": half, "crop_km": crop_km,
            "pixel_km": round(pixel_size_km(geometry, w), 4)}
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
