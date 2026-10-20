"""Layer 4 scheduling, image timing and logging, with the network and model faked."""
import json
import os
import tempfile
from datetime import timedelta

import cv2
import numpy as np

import models
import satellite
import watchdog
from conftest import NOW, add_ship

S = models.JobStatus

# A fake north-up browse preview covering lon 59-63, lat 8-11; test ships sit at (10, 60).
FOOTPRINT = {"type": "Polygon", "coordinates": [[[59, 8], [63, 8], [63, 11], [59, 11], [59, 8]]]}
PREVIEW = os.path.join(tempfile.mkdtemp(), "S1A_TEST_preview.jpg")
cv2.imwrite(PREVIEW, np.full((800, 1000), 60, np.uint8))


def fetch(_url):
    return PREVIEW


def pending(db, ship_id, anomaly_at=NOW - timedelta(hours=1)):
    return add_ship(db, ship_id, 10.0, 60.0, status=S.PENDING_IMAGE,
                    anomaly_timestamp=anomaly_at, anomaly_reason="LOITERING_ALONE")


def image(hours_after, browse=True, anomaly_at=NOW - timedelta(hours=1)):
    t = anomaly_at + timedelta(hours=hours_after)
    return {"granule_id": f"S1A_TEST_{hours_after}", "browse_url": "http://x/a.jpg" if browse else None,
            "image_timestamp": t, "image_alignment_status": models.ImageAlignmentStatus.ALIGNED,
            "delta_hours": hours_after, "image_status": models.ImageStatus.AVAILABLE,
            "footprint": FOOTPRINT}


def clean(_path, **kw):
    return {"is_oil": False, "confidence": 90.0, "raw_score": 0.1, "score_std": 0.01,
            "scores": [0.1], "threshold_used": 0.75, "sigma": 0.5, "n_draws": 1}


def oil(_path, **kw):
    return {**clean(_path), "is_oil": True, "confidence": 91.0, "raw_score": 0.91}


# --- image selection -------------------------------------------------------

def test_images_taken_before_the_anomaly_are_never_chosen():
    anomaly = NOW
    cands = [image(-30, anomaly_at=anomaly), image(-5, anomaly_at=anomaly)]
    assert satellite.choose_candidate(cands, anomaly) is None


def test_earliest_image_after_the_anomaly_is_chosen():
    anomaly = NOW
    cands = [image(-30, anomaly_at=anomaly), image(20, anomaly_at=anomaly), image(3, anomaly_at=anomaly)]
    assert satellite.choose_candidate(cands, anomaly)["granule_id"] == "S1A_TEST_3"


def test_small_clock_tolerance_before_anomaly_is_allowed():
    anomaly = NOW
    assert satellite.choose_candidate([image(-0.5, anomaly_at=anomaly)], anomaly) is not None


def test_images_beyond_the_window_are_ignored():
    anomaly = NOW
    assert satellite.choose_candidate([image(60, anomaly_at=anomaly)], anomaly) is None


def test_alignment_is_signed():
    status, delta = satellite.classify_alignment(NOW, NOW - timedelta(hours=20))
    assert status == models.ImageAlignmentStatus.MISALIGNED and delta < 0


# --- scheduling ------------------------------------------------------------

def test_job_without_image_backs_off_and_does_not_block_others(db):
    """Regression: the oldest image-less job used to be retried every loop forever."""
    pending(db, 300000001, anomaly_at=NOW - timedelta(hours=3))  # older, no image
    pending(db, 300000002, anomaly_at=NOW - timedelta(hours=1))

    def find(lat, lon, anomaly, exclude=()):
        return None if anomaly < NOW - timedelta(hours=2) else image(0.5, anomaly_at=anomaly)

    first = watchdog.process_next_job(db, now=NOW, find_image=find, download=fetch, analyze=clean)
    assert first.ship_id == "300000001" and first.next_attempt_at is not None
    second = watchdog.process_next_job(db, now=NOW, find_image=find, download=fetch, analyze=clean)
    assert second.ship_id == "300000002" and second.final_result == "CLEAN_SEA"


def test_backoff_grows(db):
    job = pending(db, 300000003)
    delays = []
    t = NOW
    for _ in range(3):
        watchdog.process_next_job(db, now=t, find_image=lambda *a, **k: None)
        db.refresh(job)
        nxt = job.next_attempt_at if job.next_attempt_at.tzinfo else job.next_attempt_at.replace(tzinfo=NOW.tzinfo)
        delays.append((nxt - t).total_seconds())
        t = nxt
    assert delays[0] < delays[1] < delays[2]


def test_job_expires_after_window(db):
    pending(db, 300000004, anomaly_at=NOW - timedelta(hours=100))
    job = watchdog.process_next_job(db, now=NOW, find_image=lambda *a, **k: None)
    assert job.status == S.EXPIRED
    assert db.query(models.AlertEvent).filter_by(event_type="EXPIRED").count() == 1


def test_search_errors_are_retried_not_fatal(db):
    pending(db, 300000005)

    def boom(*a, **k):
        raise RuntimeError("ASF down")

    job = watchdog.process_next_job(db, now=NOW, find_image=boom)
    assert job.status == S.PENDING_IMAGE and job.image_status == models.ImageStatus.FAILED


# --- results are logged permanently -------------------------------------------

def test_classification_is_logged_and_survives_status_changes(db):
    pending(db, 300000006)
    job = watchdog.process_next_job(db, now=NOW, find_image=lambda *a, **k: image(2),
                                    download=fetch, analyze=oil)
    assert job.status == S.OIL_DETECTED and job.raw_score == 0.91

    job.status = S.UNPROCESSED   # e.g. the ship moves on and is re-evaluated
    job.raw_score = None
    db.commit()

    event = db.query(models.AlertEvent).filter_by(event_type="CLASSIFIED").one()
    details = json.loads(event.details)
    assert event.raw_score == 0.91 and details["is_oil"] is True
    assert details["settings"]["OPTIMAL_THRESHOLD"] == 0.75
    assert details["image_delta_hours"] == 2.0
    assert "code_version" in details


# --- vessel-centred crop ---------------------------------------------------------

def test_classifier_sees_a_crop_around_the_vessel_not_the_whole_scene(db):
    pending(db, 300000007)
    seen = {}

    def analyze(path, **kw):
        seen["shape"] = cv2.imread(path, cv2.IMREAD_GRAYSCALE).shape
        return clean(path)

    job = watchdog.process_next_job(db, now=NOW, find_image=lambda *a, **k: image(2),
                                    download=fetch, analyze=analyze)
    assert job.final_result == "CLEAN_SEA"
    assert seen["shape"][0] == seen["shape"][1] < 100   # ~25 km square, not 800x1000
    details = json.loads(db.query(models.AlertEvent).filter_by(event_type="CLASSIFIED").one().details)
    assert details["crop"]["crop_km"] == 25.0 and details["crop"]["reason"] is None


def test_failed_crop_retries_with_a_different_image(db):
    job = add_ship(db, 300000008, 10.0, 62.99, status=S.PENDING_IMAGE,   # at the image edge
                   anomaly_timestamp=NOW - timedelta(hours=1), anomaly_reason="LOITERING_ALONE")
    calls = []

    def find(lat, lon, anomaly, exclude=()):
        calls.append(tuple(exclude))
        return image(2)

    watchdog.process_next_job(db, now=NOW, find_image=find, download=fetch, analyze=clean)
    db.refresh(job)
    assert job.status == S.PENDING_IMAGE and job.next_attempt_at is not None
    assert db.query(models.AlertEvent).filter_by(event_type="CROP_FAILED").count() == 1

    watchdog.process_next_job(db, now=job.next_attempt_at, find_image=find, download=fetch, analyze=clean)
    assert calls[-1] == ("S1A_TEST_2",)   # the failed image is excluded next time
