"""Layer 4 worker: fetch a browse image for each escalated ship and classify it.

Changes from the original version:
  * Jobs without an image are retried with exponential backoff (next_attempt_at)
    instead of being picked first on every loop, which starved all other jobs.
  * A job that gets no image within IMAGE_MAX_AFTER_HOURS (+ ASF publication
    slack) becomes EXPIRED instead of waiting forever.
  * Only images acquired after the anomaly are used (see satellite.choose_candidate).
  * Every classification is written to the append-only event log with its
    scores, settings and code version, so results survive later status changes.
"""
import sys
import time
from datetime import datetime, timedelta, timezone

import browse_geo
import config
import ml_logic
import models
import satellite
from database import SessionLocal, engine, ensure_alert_schema
from events import log_event, set_status
from models import ImageStatus, JobStatus, OilSpillAlert
from sqlalchemy import or_

# Products can appear in ASF's catalogue hours after acquisition.
PUBLICATION_SLACK_HOURS = 12

RECORDED_SETTINGS = ("MODEL_KIND", "OPTIMAL_THRESHOLD", "STI_SIGMA", "STI_DRAWS", "STI_BRIGHTNESS_SHIFT",
                     "IMAGE_MAX_BEFORE_HOURS", "IMAGE_MAX_AFTER_HOURS", "WEIGHTS_PATH",
                     "CROP_ENABLED", "CROP_KM")


def utc_now():
    return datetime.now(timezone.utc)


def _aware(dt):
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _schedule_retry(db, job, now, image_status, note):
    job.image_attempts = (job.image_attempts or 0) + 1
    delay = min(config.RETRY_BASE_SECONDS * 2 ** (job.image_attempts - 1), config.RETRY_MAX_SECONDS)
    job.next_attempt_at = now + timedelta(seconds=delay)
    job.image_status = image_status
    job.status = JobStatus.PENDING_IMAGE
    print(f"   {note} Retry #{job.image_attempts} in {delay / 60:.0f} min.", flush=True)
    db.commit()


def next_job(db, now):
    return (
        db.query(OilSpillAlert)
        .filter(
            OilSpillAlert.status == JobStatus.PENDING_IMAGE,
            or_(OilSpillAlert.next_attempt_at.is_(None), OilSpillAlert.next_attempt_at <= now),
        )
        .order_by(OilSpillAlert.next_attempt_at, OilSpillAlert.anomaly_timestamp, OilSpillAlert.id)
        .first()
    )


def process_next_job(db, now=None, find_image=None, download=None, analyze=None):
    """Handle at most one due job. Returns the job (or None if nothing was due).
    The three callables can be replaced in tests."""
    now = _aware(now) or utc_now()
    find_image = find_image or satellite.find_relevant_image
    download = download or satellite.download_preview
    analyze = analyze or ml_logic.analyze_image

    job = next_job(db, now)
    if job is None:
        return None

    if job.anomaly_timestamp is None:
        job.anomaly_timestamp = job.updated_at or job.created_at
    anomaly_at = _aware(job.anomaly_timestamp)

    deadline = anomaly_at + timedelta(hours=config.IMAGE_MAX_AFTER_HOURS + PUBLICATION_SLACK_HOURS)
    if now > deadline:
        job.image_status = ImageStatus.UNAVAILABLE
        set_status(db, job, JobStatus.EXPIRED, event_type="EXPIRED",
                   details={"attempts": job.image_attempts or 0})
        db.commit()
        print(f"[EXPIRED] Ship {job.ship_id}: no image within the acquisition window.", flush=True)
        return job

    print(f"\n[ALERT] Ship {job.ship_id} at {job.lat}, {job.lon} ({job.anomaly_reason})", flush=True)
    job.status = JobStatus.SEARCHING_IMG
    job.image_status = ImageStatus.PENDING
    db.commit()

    # Images that already failed cropping for this anomaly are skipped.
    tried = [
        e.granule_id for e in db.query(models.AlertEvent)
        .filter(models.AlertEvent.alert_id == job.id, models.AlertEvent.event_type == "CROP_FAILED")
        if _aware(e.created_at) >= anomaly_at
    ]
    try:
        result = find_image(job.lat, job.lon, anomaly_at, exclude=tuple(tried))
    except Exception as exc:
        _schedule_retry(db, job, now, ImageStatus.FAILED, f"ASF search failed ({exc}).")
        return job
    if result is None:
        _schedule_retry(db, job, now, ImageStatus.UNAVAILABLE, "No post-anomaly image yet.")
        return job

    job.granule_id = result["granule_id"]
    job.image_timestamp = result["image_timestamp"]
    job.image_alignment_status = result["image_alignment_status"]
    if not result["browse_url"]:
        _schedule_retry(db, job, now, ImageStatus.UNAVAILABLE, "Image found but no browse preview.")
        return job

    local_path = download(result["browse_url"])
    if not local_path:
        job.image_url = None
        _schedule_retry(db, job, now, ImageStatus.FAILED, "Preview download failed.")
        return job

    job.image_url = satellite.build_public_image_url(local_path)
    job.image_status = ImageStatus.AVAILABLE

    target, crop_info = local_path, None
    if config.CROP_ENABLED:
        footprint = result.get("footprint")
        if not footprint:
            _schedule_retry(db, job, now, ImageStatus.FAILED, "No footprint to crop with.")
            return job
        target, crop_info = browse_geo.save_crop(local_path, footprint, job.lat, job.lon, tag=str(job.id))
        if target is None:
            # The vessel isn't usefully inside this image; another pass may cover it.
            log_event(db, "CROP_FAILED", job, reason=crop_info.get("reason"), details={"crop": crop_info})
            _schedule_retry(db, job, now, ImageStatus.UNAVAILABLE,
                            f"Crop not possible ({crop_info.get('reason')}).")
            return job

    job.status = JobStatus.ANALYZING
    db.commit()

    analysis = analyze(target, lat=job.lat, lon=job.lon)
    if not analysis:
        set_status(db, job, JobStatus.ERROR, event_type="ANALYSIS_FAILED")
        db.commit()
        return job

    job.prediction_confidence = analysis["confidence"]
    job.raw_score = analysis["raw_score"]
    job.reviewed_at = now
    details = {
        "is_oil": analysis["is_oil"],
        "score_std": analysis.get("score_std"),
        "scores": analysis.get("scores"),
        "n_draws": analysis.get("n_draws"),
        "image_delta_hours": (_aware(job.image_timestamp) - anomaly_at).total_seconds() / 3600
        if job.image_timestamp else None,
        "alignment": getattr(job.image_alignment_status, "value", job.image_alignment_status),
        "settings": {k: getattr(config, k) for k in RECORDED_SETTINGS},
        "threshold_used": analysis.get("threshold_used"),
        "crop": crop_info,
    }
    if analysis["is_oil"]:
        job.final_result = "OIL_DETECTED"
        new_status = JobStatus.OIL_DETECTED
        if "AI_CONFIRMED" not in (job.anomaly_reason or ""):
            job.anomaly_reason = f"{job.anomaly_reason} + AI_CONFIRMED" if job.anomaly_reason else "AI_CONFIRMED"
    else:
        job.final_result = "CLEAN_SEA"
        new_status = JobStatus.SAFE
    set_status(db, job, new_status, event_type="CLASSIFIED", raw_score=analysis["raw_score"], details=details)
    db.commit()
    print(f"   [RESULT] {job.final_result} (score {analysis['raw_score']:.3f})", flush=True)
    return job


def start_watchdog():
    models.Base.metadata.create_all(bind=engine)
    ensure_alert_schema()
    print("Watchdog (Layer 4) is awake. Monitoring for suspicious ships...")
    ml_logic.load_ai_model()

    while True:
        db = SessionLocal()
        try:
            job = process_next_job(db)
            if job is None:
                sys.stdout.write(".")
                sys.stdout.flush()
                time.sleep(5)
        except Exception as exc:
            print(f"\nWatchdog Error: {exc}", flush=True)
            db.rollback()
            time.sleep(5)
        finally:
            db.close()


if __name__ == "__main__":
    start_watchdog()
