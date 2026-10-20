"""Layers 2 and 3 of the TARS cascade, as one testable function.

Each cycle:
  1. Takes every ship with a recent AIS report (any status), so dense areas
     stay dense even after ships in them are marked SAFE.
  2. Clusters positions with HDBSCAN (Layer 2), with physical distance checks
     (see _cluster): dense area -> SAFE veto; tight 2-5 ship group -> possible
     rendezvous; anything else -> isolated.
  3. Scores each ship's recent trajectory with an Isolation Forest (Layer 3)
     fitted on the whole current fleet.
  4. Escalates eligible ships to PENDING_IMAGE and logs every escalation, plus
     one summary event per cycle, to the append-only event log.

Layer 1 (sudden speed change) is computed at ingest time. It does not by
itself escalate a ship; it is recorded as a reason tag and it lifts the
review cooldown, so a ship cleared yesterday is re-examined if it suddenly
stops today.
"""
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

import hdbscan
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import BallTree

import config
import models
from coast import near_land
from events import log_event, set_status
from trajectory import EARTH_RADIUS_KM, haversine_km, trajectory_features

ELIGIBLE = (models.JobStatus.UNPROCESSED, models.JobStatus.SAFE)
MIN_FLEET_FOR_ISO = 20  # too few ships -> no meaningful "normal" to compare against
ROBUST_Z_MIN = 3.5      # how far from the fleet median a feature must be
# Minimum spread per feature (speed kn, speed std kn, speed change kn, turn deg,
# straightness) so tiny differences in a uniform fleet don't count as "far".
FEATURE_FLOOR = np.array([1.0, 0.5, 1.0, 5.0, 0.05])


def utc_now():
    return datetime.now(timezone.utc)


def _aware(dt):
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _cluster(ships):
    """Classify each ship's spatial context with HDBSCAN (Layer 2).

    Returns a list of (category, group_size, group_max_km) per ship, where
    category is "dense", "rendezvous", "coastal" or "isolated".

    HDBSCAN has no physical distance scale: with allow_single_cluster it will
    happily put ships hundreds of km apart into one cluster, which both hides
    real meetings and fakes "ports". So:
      dense      - (vetoed as SAFE) either an HDBSCAN cluster of >= DENSE_CLUSTER_MIN_SIZE
                   ships with median nearest-neighbour distance <= DENSE_MAX_NN_KM,
                   or >= DENSE_CLUSTER_MIN_SIZE ships within RENDEZVOUS_MAX_KM
      coastal    - not dense and within OFFSHORE_KM of land (offline land mask)
      rendezvous - offshore, 1-4 other ships within RENDEZVOUS_MAX_KM, all of them
                   (this ship included) slower than RENDEZVOUS_MAX_SPEED_KNOTS.
                   A radius rule: HDBSCAN cluster membership is not used here.
    Everything else counts as isolated (offshore, alone).
    """
    n = len(ships)
    if n < 2:
        return [("isolated", 0, None)] * n

    coords = np.radians(np.array([[s.lat, s.lon] for s in ships]))
    labels = hdbscan.HDBSCAN(
        min_cluster_size=config.HDBSCAN_MIN_CLUSTER_SIZE,
        metric="haversine",
        allow_single_cluster=True,
    ).fit_predict(coords)

    tree = BallTree(coords, metric="haversine")
    nn_km = tree.query(coords, k=2)[0][:, 1] * EARTH_RADIUS_KM
    # Ships within RENDEZVOUS_MAX_KM of each ship (including itself).
    close = tree.query_radius(coords, r=config.RENDEZVOUS_MAX_KM / EARTH_RADIUS_KM)

    groups = defaultdict(list)
    for idx, label in enumerate(labels):
        if label != -1:
            groups[label].append(idx)

    dense_labels = {
        label for label, idxs in groups.items()
        if len(idxs) >= config.DENSE_CLUSTER_MIN_SIZE
        and float(np.median(nn_km[idxs])) <= config.DENSE_MAX_NN_KM
    }

    offshore = ~near_land([s.lat for s in ships], [s.lon for s in ships], config.OFFSHORE_KM)

    contexts = []
    for idx, label in enumerate(labels):
        if label in dense_labels:
            contexts.append(("dense", len(groups[label]), None))
            continue
        group = close[idx]
        if not offshore[idx] and len(group) < config.DENSE_CLUSTER_MIN_SIZE:
            # Near land: small groups of stopped ships are harbours/anchorages,
            # so neither the rendezvous nor the loitering rule applies.
            contexts.append(("coastal", len(group), None))
        elif len(group) >= config.DENSE_CLUSTER_MIN_SIZE:
            # Crowded locally even if HDBSCAN didn't label it: treat as dense.
            contexts.append(("dense", len(group), None))
        elif len(group) >= 2 and all(
            ships[j].speed is not None and ships[j].speed < config.RENDEZVOUS_MAX_SPEED_KNOTS
            for j in group
        ):
            max_km = max(
                haversine_km(ships[a].lat, ships[a].lon, ships[b].lat, ships[b].lon)
                for i, a in enumerate(group) for b in group[i + 1:]
            )
            contexts.append(("rendezvous", len(group), max_km))
        else:
            contexts.append(("isolated", 0, None))
    return contexts


def _histories(db, ship_ids):
    rows = (
        db.query(models.PositionReport)
        .filter(models.PositionReport.ship_id.in_(ship_ids))
        .order_by(models.PositionReport.ship_id, models.PositionReport.reported_at)
        .all()
    )
    by_ship = defaultdict(list)
    for r in rows:
        by_ship[r.ship_id].append(r)
    return by_ship


def _behaviour_outliers(db, ships):
    """Return {ship_id: True/False} for ships with enough history to judge."""
    ids = [s.ship_id for s in ships]
    histories = {}
    for chunk_start in range(0, len(ids), 900):  # stay under SQLite's variable limit
        histories.update(_histories(db, ids[chunk_start:chunk_start + 900]))

    feats, owners = [], []
    for sid, hist in histories.items():
        if len(hist) >= config.MIN_HISTORY_FOR_BEHAVIOUR:
            f = trajectory_features(hist)
            if f is not None:
                feats.append(f)
                owners.append(sid)
    if len(feats) < MIN_FLEET_FOR_ISO:
        return {}

    contamination = config.ISO_CONTAMINATION
    if contamination != "auto":
        contamination = float(contamination)
    forest = IsolationForest(contamination=contamination, random_state=config.ISO_RANDOM_STATE)
    X = np.vstack(feats)
    preds = forest.fit_predict(X)

    # Isolation Forest always ranks *someone* as most isolated, even in a fleet
    # where everyone behaves the same. Require the ship to also be far from the
    # fleet median (robust z-score) in at least one feature.
    median = np.median(X, axis=0)
    mad = np.median(np.abs(X - median), axis=0) * 1.4826
    robust_z = np.abs(X - median) / (mad + FEATURE_FLOOR)
    far = (robust_z > ROBUST_Z_MIN).any(axis=1)
    return {sid: bool(pred == -1 and f) for sid, pred, f in zip(owners, preds, far)}


def _mark_new_anomaly_event(alert, now):
    alert.anomaly_timestamp = now
    alert.granule_id = None
    alert.image_url = None
    alert.image_timestamp = None
    alert.image_alignment_status = None
    alert.prediction_confidence = None
    alert.raw_score = None
    alert.final_result = None
    alert.dispatched_at = None
    alert.resolved_at = None
    alert.image_status = models.ImageStatus.PENDING
    alert.image_attempts = 0
    alert.next_attempt_at = None


def run_engine_cycle(db, now=None):
    """Run one engine pass and commit. Returns a summary dict."""
    now = now or utc_now()
    fresh_after = now - timedelta(hours=config.STALE_POSITION_HOURS)

    ships = [
        s for s in db.query(models.OilSpillAlert).all()
        if _aware(s.last_report_at or s.updated_at or s.created_at) is not None
        and _aware(s.last_report_at or s.updated_at or s.created_at) >= fresh_after
    ]
    summary = Counter(fresh_ships=len(ships))
    if not ships:
        return dict(summary)

    contexts = _cluster(ships)
    outliers = _behaviour_outliers(db, ships)
    summary["judged_by_behaviour"] = len(outliers)
    summary["behaviour_outliers"] = sum(outliers.values())

    cooldown = timedelta(hours=config.REVIEW_COOLDOWN_HOURS)
    flag_window = timedelta(hours=config.STALE_POSITION_HOURS)

    for ship, (category, size, max_km) in zip(ships, contexts):
        if ship.status not in ELIGIBLE:
            continue

        reviewed = _aware(ship.reviewed_at)
        flagged = _aware(ship.kinematic_flag_at)
        recent_flag = flagged is not None and flagged >= now - flag_window
        if reviewed is not None and now - reviewed < cooldown and not (flagged and flagged > reviewed):
            summary["in_cooldown"] += 1
            continue

        if category == "dense":
            summary["dense_area"] += 1
            if ship.status != models.JobStatus.SAFE:
                ship.status = models.JobStatus.SAFE
                ship.anomaly_reason = "DENSE_AREA_HDBSCAN"
            continue

        reasons = []
        if category == "rendezvous":
            reasons.append(f"RENDEZVOUS_CLUSTER_{size}")
        if outliers.get(ship.ship_id):
            reasons.append("BEHAVIOUR_OUTLIER_ISO_FOREST")
        if category == "isolated" and ship.speed is not None and ship.speed < config.LOITER_SPEED_KNOTS:
            reasons.append("LOITERING_ALONE")

        if not reasons:
            summary["normal"] += 1
            if ship.status != models.JobStatus.UNPROCESSED or ship.anomaly_reason != "NO_ANOMALY":
                ship.status = models.JobStatus.UNPROCESSED
                ship.anomaly_reason = "NO_ANOMALY"
            continue

        if recent_flag:
            reasons.append("KINEMATIC_FLAG")
        reason = " + ".join(reasons)
        _mark_new_anomaly_event(ship, now)
        set_status(db, ship, models.JobStatus.PENDING_IMAGE, reason=reason, event_type="ESCALATED",
                   details={"context": category, "cluster_size": size, "cluster_max_km": max_km})
        summary["escalated"] += 1
        for r in reasons:
            summary[f"reason:{r.split('_CLUSTER_')[0]}"] += 1

    log_event(db, "ENGINE_CYCLE", details=dict(summary))
    db.commit()
    return dict(summary)
