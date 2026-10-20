"""Trajectory features for the Layer-3 behavioural outlier detector.

The original engine gave the Isolation Forest one snapshot per ship
([lat, lon, speed]), so it could not see zig-zags, drift or speed changes,
and "anomalous" partly meant "in an unusual place". These features describe
*how* a ship has been moving over its recent reports and contain no position.
"""
import math

import numpy as np

EARTH_RADIUS_KM = 6371.0088
MIN_MOVE_KM = 0.05  # below this, a step is treated as GPS jitter for heading purposes

FEATURE_NAMES = [
    "mean_speed",       # knots
    "speed_std",        # knots
    "max_speed_change", # knots between consecutive reports
    "mean_turn_deg",    # mean absolute heading change per step
    "straightness",     # net displacement / path length (1 = straight line)
]


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def bearing_deg(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlmb = math.radians(lon2 - lon1)
    x = math.sin(dlmb) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlmb)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def angle_diff(a, b):
    """Smallest absolute difference between two headings, 0-180 degrees."""
    d = abs(a - b) % 360.0
    return 360.0 - d if d > 180.0 else d


def trajectory_features(reports):
    """reports: time-ordered sequence of objects with lat, lon, speed, course.
    Returns a numpy array matching FEATURE_NAMES, or None if too short."""
    if len(reports) < 2:
        return None

    speeds = [r.speed for r in reports if r.speed is not None]
    mean_speed = float(np.mean(speeds)) if speeds else 0.0
    speed_std = float(np.std(speeds)) if speeds else 0.0

    speed_changes = [
        abs(b.speed - a.speed)
        for a, b in zip(reports, reports[1:])
        if a.speed is not None and b.speed is not None
    ]
    max_speed_change = max(speed_changes) if speed_changes else 0.0

    headings = []
    path_km = 0.0
    for a, b in zip(reports, reports[1:]):
        step = haversine_km(a.lat, a.lon, b.lat, b.lon)
        path_km += step
        if step >= MIN_MOVE_KM:
            headings.append(bearing_deg(a.lat, a.lon, b.lat, b.lon))
    turns = [angle_diff(a, b) for a, b in zip(headings, headings[1:])]
    mean_turn = float(np.mean(turns)) if turns else 0.0

    net_km = haversine_km(reports[0].lat, reports[0].lon, reports[-1].lat, reports[-1].lon)
    straightness = net_km / path_km if path_km >= MIN_MOVE_KM else 1.0

    return np.array([mean_speed, speed_std, max_speed_change, mean_turn, straightness], dtype=float)
