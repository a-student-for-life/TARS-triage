"""Central configuration for TARS.

Every tunable that affects results lives here, so a run can be described
exactly (and reproduced) by the values in this file plus any environment
overrides. Each value can be overridden with the environment variable shown.
"""
import os


def _float(name, default):
    return float(os.getenv(name, default))


def _int(name, default):
    return int(os.getenv(name, default))


# --- Storage -----------------------------------------------------------------
DB_URL = os.getenv("TARS_DB_URL", "sqlite:///./oil_spill.db")
TEMP_IMAGE_DIR = os.getenv("TARS_TEMP_IMAGE_DIR", "temp_images")

# --- Layer 1: kinematic flag -------------------------------------------------
# A speed change larger than this between two reports counts as sudden...
SUDDEN_SPEED_DELTA_KNOTS = _float("TARS_SUDDEN_SPEED_DELTA_KNOTS", 10.0)
# ...but only if the two reports are close enough in time to be "sudden".
SUDDEN_MAX_GAP_SECONDS = _float("TARS_SUDDEN_MAX_GAP_SECONDS", 600)

# --- Position history (feeds Layer 3) -----------------------------------------
HISTORY_LEN = _int("TARS_HISTORY_LEN", 20)            # reports kept per ship
MIN_HISTORY_FOR_BEHAVIOUR = _int("TARS_MIN_HISTORY", 5)  # needed for trajectory features
STALE_POSITION_HOURS = _float("TARS_STALE_POSITION_HOURS", 6)  # older ships ignored by the engine

# --- Layer 2: spatial clustering ----------------------------------------------
HDBSCAN_MIN_CLUSTER_SIZE = _int("TARS_HDBSCAN_MIN_CLUSTER_SIZE", 2)
DENSE_CLUSTER_MIN_SIZE = _int("TARS_DENSE_CLUSTER_MIN_SIZE", 6)   # >5 ships = dense area (veto)
# A big cluster only counts as a dense area (port, anchorage, busy lane) if its
# ships are actually close together: median nearest-neighbour distance below this.
DENSE_MAX_NN_KM = _float("TARS_DENSE_MAX_NN_KM", 5.0)
# A possible ship-to-ship transfer: 2-5 ships this close together, all nearly stopped.
# (Ships passing each other in a lane are close but fast.)
RENDEZVOUS_MAX_KM = _float("TARS_RENDEZVOUS_MAX_KM", 1.0)
RENDEZVOUS_MAX_SPEED_KNOTS = _float("TARS_RENDEZVOUS_MAX_SPEED_KNOTS", 3.0)
LOITER_SPEED_KNOTS = _float("TARS_LOITER_SPEED_KNOTS", 1.0)
# Rendezvous and loitering rules only apply this far from land (harbours,
# marinas and anchorages are excluded). Uses an offline ~1 km land mask.
OFFSHORE_KM = _float("TARS_OFFSHORE_KM", 10.0)

# --- Layer 3: behavioural outliers --------------------------------------------
# Alarm budget: the share of judged ships the Isolation Forest may label as
# outliers in one cycle. On a full day of US traffic, "auto" (the score offset of
# the original Isolation Forest paper) flagged about 7% of all ships per cycle, far
# more than an analyst can review; 0.01 flags about 0.4% (experiments/revision,
# layer3_national.py and layer3_sensitivity.py). This is a workload choice, not a
# value tuned for accuracy. Set "auto" or another fraction via TARS_ISO_CONTAMINATION.
ISO_CONTAMINATION = os.getenv("TARS_ISO_CONTAMINATION", "0.01")
ISO_RANDOM_STATE = 42

# After a ship has been imaged and cleared, don't re-escalate it for this long
# unless it triggers a new kinematic flag.
REVIEW_COOLDOWN_HOURS = _float("TARS_REVIEW_COOLDOWN_HOURS", 24)

# --- Layer 4: image acquisition -----------------------------------------------
# Only images acquired AFTER the anomaly can show its consequences. A small
# tolerance before the anomaly covers AIS/clock jitter.
IMAGE_MAX_BEFORE_HOURS = _float("TARS_IMAGE_MAX_BEFORE_HOURS", 1.0)
IMAGE_MAX_AFTER_HOURS = _float("TARS_IMAGE_MAX_AFTER_HOURS", 48.0)
RETRY_BASE_SECONDS = _float("TARS_RETRY_BASE_SECONDS", 300)
RETRY_MAX_SECONDS = _float("TARS_RETRY_MAX_SECONDS", 6 * 3600)

# --- Layer 4: vessel-centred crop ---------------------------------------------
# Classify a window around the flagged vessel, not the whole ~250 km scene
# (whole scenes: AUC 0.55 on expert labels; 25 km crops: 0.83 in open water).
CROP_ENABLED = os.getenv("TARS_CROP_ENABLED", "1") == "1"
CROP_KM = _float("TARS_CROP_KM", 25.0)
CROP_MAX_NO_DATA = _float("TARS_CROP_MAX_NO_DATA", 0.2)  # max fraction of the crop outside the footprint

# --- Layer 4: classifier ------------------------------------------------------
# MODEL_KIND:
#   "regional"  - (default) score the vessel crop with the head trained for the
#                 vessel's region (models/regional_heads.json); outside every
#                 region, use the original CSIRO head on the raw crop. Browse-trained
#                 heads only beat the CSIRO head inside their own region
#                 (experiments/browse_eval, experiments/gulf_eval).
#   "browse_v1" - always use the Eastern Mediterranean head.
#   "csiro"     - original behaviour: CSIRO head with STI.
MODEL_KIND = os.getenv("TARS_MODEL", "regional")
REGIONAL_HEADS = os.getenv("TARS_REGIONAL_HEADS", "models/regional_heads.json")
WEIGHTS_PATH = os.getenv("TARS_WEIGHTS_PATH", "models/optisar_b4_retrained_weights.weights.h5")
IMG_SIZE = 380
# CSIRO head threshold, chosen on its validation split. Regional heads read their
# own thresholds from their info files (chosen for ~5% false alarms).
OPTIMAL_THRESHOLD = _float("TARS_THRESHOLD", 0.75)
STI_SIGMA = _float("TARS_STI_SIGMA", 0.5)            # 0.0 = CLAHE + brightness only
STI_DRAWS = _int("TARS_STI_DRAWS", 8)                # speckle is random: average several draws
STI_BRIGHTNESS_SHIFT = _int("TARS_STI_BRIGHTNESS_SHIFT", 70)
SAVE_DEBUG_IMAGES = os.getenv("TARS_SAVE_DEBUG_IMAGES", "0") == "1"


def snapshot():
    """All settings as a dict, stored with every classification event."""
    return {k: v for k, v in globals().items() if k.isupper()}
