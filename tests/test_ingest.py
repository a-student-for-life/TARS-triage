"""Ingest endpoint: validation, Layer-1 flag, history cap, event logging."""
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

import config
import main
import models
from database import get_db

T0 = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def client(db):
    main.app.dependency_overrides[get_db] = lambda: db
    yield TestClient(main.app)  # no context manager: the background engine is not started
    main.app.dependency_overrides.clear()


def post(client, mmsi=366123456, lat=40.0, lon=-70.0, speed=12.0, course=90.0, t=T0):
    return client.post("/ingest", json={"mmsi": mmsi, "lat": lat, "lon": lon, "speed": speed,
                                        "course": course, "timestamp": t.isoformat()}).json()


def test_invalid_reports_are_rejected_and_counted(client, db):
    assert post(client, mmsi=999999999)["status"] == "rejected"
    assert post(client, lat=0.0, lon=0.0)["reason"] == "null_island"
    assert db.query(models.OilSpillAlert).count() == 0
    assert client.get("/ingest/rejections").json()["null_island"] >= 1


def test_speed_not_available_does_not_create_sudden_acceleration(client, db):
    post(client, speed=10.0, t=T0)
    post(client, speed=102.3, t=T0 + timedelta(seconds=30))
    ship = db.query(models.OilSpillAlert).one()
    assert ship.kinematic_flag_at is None and ship.speed is None


def test_sudden_stop_sets_flag_and_logs_event(client, db):
    post(client, speed=14.0, t=T0)
    post(client, speed=0.5, t=T0 + timedelta(seconds=60))
    ship = db.query(models.OilSpillAlert).one()
    assert ship.anomaly_type == "SUDDEN_STOP" and ship.kinematic_flag_at is not None
    assert db.query(models.AlertEvent).filter_by(event_type="KINEMATIC_FLAG").count() == 1


def test_speed_change_across_a_long_gap_is_not_sudden(client, db):
    post(client, speed=14.0, t=T0)
    post(client, speed=0.5, t=T0 + timedelta(hours=3))
    assert db.query(models.OilSpillAlert).one().kinematic_flag_at is None


def test_new_ships_start_with_neutral_type_not_unknown(client, db):
    post(client)
    assert db.query(models.OilSpillAlert).one().anomaly_type == "NONE"


def test_history_is_capped(client, db):
    for i in range(config.HISTORY_LEN + 7):
        post(client, lon=-70.0 + 0.01 * i, t=T0 + timedelta(minutes=i))
    assert db.query(models.PositionReport).count() == config.HISTORY_LEN


def test_operator_actions_are_logged(client, db):
    post(client)
    ship = db.query(models.OilSpillAlert).one()
    ship.status = models.JobStatus.OIL_DETECTED
    db.commit()
    r = client.put(f"/alert/{ship.id}/status", json={"status": "IGNORED"})
    assert r.status_code == 200
    ev = db.query(models.AlertEvent).filter_by(event_type="OPERATOR_ACTION").one()
    assert (ev.from_status, ev.to_status) == ("OIL_DETECTED", "IGNORED")
