"""Clean raw AIS position reports before they reach the pipeline.

AIS uses reserved values to mean "not available" (ITU-R M.1371):
  speed over ground 102.3 kn, latitude 91, longitude 181, course 360.
Speeds of 102.2 kn mean "102.2 or more" and are not usable for kinematics.
Treating these as real values creates fake sudden accelerations and
points off the map.
"""
from dataclasses import dataclass
from typing import Optional

SOG_NOT_AVAILABLE_MIN = 102.2   # 102.2 = ">=102.2", 102.3 = not available
COG_NOT_AVAILABLE = 360.0

# MMSIs that are placeholders or test values rather than real vessels.
PLACEHOLDER_MMSIS = {0, 111111111, 123456789, 999999999}


@dataclass
class CleanReport:
    mmsi: int
    lat: float
    lon: float
    speed: Optional[float]   # None when not available
    course: Optional[float]  # None when not available


def is_vessel_mmsi(mmsi: int) -> bool:
    """Ship stations have 9-digit MMSIs whose first digit is 2-7 (the MID).
    Base stations (00...), SAR aircraft (111...), aids to navigation (99...)
    and other special ranges are excluded."""
    return 200_000_000 <= mmsi <= 799_999_999 and mmsi not in PLACEHOLDER_MMSIS


def clean_report(mmsi, lat, lon, speed, course=None):
    """Return (CleanReport, None) or (None, rejection_reason)."""
    try:
        mmsi = int(mmsi)
        lat = float(lat)
        lon = float(lon)
    except (TypeError, ValueError):
        return None, "unparseable"

    if not is_vessel_mmsi(mmsi):
        return None, "non_vessel_or_placeholder_mmsi"
    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
        return None, "position_not_available"
    if lat == 0.0 and lon == 0.0:
        return None, "null_island"

    clean_speed = None
    if speed is not None:
        speed = float(speed)
        if 0.0 <= speed < SOG_NOT_AVAILABLE_MIN:
            clean_speed = speed

    clean_course = None
    if course is not None:
        course = float(course)
        if 0.0 <= course < COG_NOT_AVAILABLE:
            clean_course = course

    return CleanReport(mmsi, lat, lon, clean_speed, clean_course), None
