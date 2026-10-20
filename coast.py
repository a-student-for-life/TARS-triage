"""Offshore test using an offline global land mask (~1 km resolution).

Stopped ships near the coast are usually in harbours, marinas or anchorages.
The same behaviour far from land is what the rendezvous and loitering rules
are meant to catch, so both rules only apply offshore.
"""
import numpy as np
from global_land_mask import globe

from trajectory import EARTH_RADIUS_KM

N_BEARINGS = 16


def _ring(lat, lon, radius_km):
    """Points on a circle of radius_km around each (lat, lon). Shapes: (n, N_BEARINGS)."""
    lat = np.radians(np.asarray(lat, dtype=float))[:, None]
    lon = np.radians(np.asarray(lon, dtype=float))[:, None]
    brg = np.linspace(0, 2 * np.pi, N_BEARINGS, endpoint=False)[None, :]
    d = radius_km / EARTH_RADIUS_KM
    lat2 = np.arcsin(np.sin(lat) * np.cos(d) + np.cos(lat) * np.sin(d) * np.cos(brg))
    lon2 = lon + np.arctan2(np.sin(brg) * np.sin(d) * np.cos(lat), np.cos(d) - np.sin(lat) * np.sin(lat2))
    lat2 = np.clip(np.degrees(lat2), -89.999, 89.999)
    lon2 = (np.degrees(lon2) + 180.0) % 360.0 - 180.0
    return lat2, lon2


def near_land(lats, lons, within_km):
    """Boolean array: is there land at the point or within within_km of it?"""
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    hit = globe.is_land(lats, lons)
    for frac in (0.25, 0.5, 1.0):
        rlat, rlon = _ring(lats, lons, within_km * frac)
        hit |= globe.is_land(rlat, rlon).any(axis=1)
    return hit
