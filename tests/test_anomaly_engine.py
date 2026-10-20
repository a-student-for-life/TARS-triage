"""Scenario tests for Layers 2-3. Each test builds a small synthetic fleet."""
import json
from datetime import timedelta

import models
from anomaly_engine import run_engine_cycle
from conftest import NOW, add_ship, straight_track

S = models.JobStatus


def open_ocean_fleet(db, n=30, start_id=300000000):
    """Normal traffic: ships ~200 km apart, each on a straight track at 11-14 kn."""
    for i in range(n):
        lat, lon = -30.0 + 2.0 * (i % 15), -150.0 + 3.0 * (i // 15)
        add_ship(db, start_id + i, lat, lon + 0.07, speed=11 + i % 4,
                 history=straight_track(lat, lon, speed=11 + i % 4))


def port(db, n=10, start_id=400000000, lat=1.26, lon=103.82):
    """Ships packed within ~1 km, moored."""
    for i in range(n):
        add_ship(db, start_id + i, lat + 0.001 * (i % 5), lon + 0.001 * (i // 5), speed=0.0,
                 history=[(lat, lon, 0.0)] * 6)


def status(db, ship_id):
    return db.query(models.OilSpillAlert).filter_by(ship_id=str(ship_id)).one().status


def test_port_ships_are_vetoed_as_dense_area(db):
    open_ocean_fleet(db)
    port(db)
    run_engine_cycle(db, now=NOW)
    assert all(status(db, 400000000 + i) == S.SAFE for i in range(10))


def test_port_stays_dense_on_later_cycles(db):
    """Regression: SAFE ships used to be dropped from clustering, shrinking the
    port cluster until remaining ships looked like a 2-5 ship rendezvous."""
    open_ocean_fleet(db)
    port(db)
    for k in range(3):
        run_engine_cycle(db, now=NOW + timedelta(seconds=10 * k))
    assert all(status(db, 400000000 + i) == S.SAFE for i in range(10))


def test_normal_open_ocean_traffic_is_not_escalated(db):
    open_ocean_fleet(db)
    summary = run_engine_cycle(db, now=NOW)
    assert summary.get("escalated", 0) <= 2  # Isolation Forest may flag the odd edge case
    assert summary["judged_by_behaviour"] == 30


def test_two_ships_meeting_in_open_water_escalate_as_rendezvous(db):
    open_ocean_fleet(db)
    add_ship(db, 500000001, -45.0, -120.0, speed=0.3)
    add_ship(db, 500000002, -45.003, -120.002, speed=0.2)  # ~400 m away
    run_engine_cycle(db, now=NOW)
    ship = db.query(models.OilSpillAlert).filter_by(ship_id="500000001").one()
    assert ship.status == S.PENDING_IMAGE
    assert "RENDEZVOUS_CLUSTER_2" in ship.anomaly_reason
    assert ship.anomaly_timestamp is not None


def test_far_apart_ships_are_not_a_rendezvous_even_if_hdbscan_pairs_them(db):
    add_ship(db, 500000001, -45.0, -120.0, speed=12.0)
    add_ship(db, 500000002, -45.0, -119.0, speed=12.0)  # ~80 km away
    run_engine_cycle(db, now=NOW)
    assert status(db, 500000001) != S.PENDING_IMAGE


def test_sparse_fleet_is_not_mistaken_for_a_port(db):
    """HDBSCAN can put widely spaced ships into one big cluster; they must not be vetoed."""
    open_ocean_fleet(db)
    run_engine_cycle(db, now=NOW)
    reasons = {s.anomaly_reason for s in db.query(models.OilSpillAlert)}
    assert "DENSE_AREA_HDBSCAN" not in reasons


def test_zigzag_at_constant_speed_is_flagged_by_behaviour(db):
    open_ocean_fleet(db)
    zig = [(-40.0 + (0.02 if i % 2 else 0.0), -100.0 + 0.01 * i, 12.0) for i in range(8)]
    add_ship(db, 600000001, -40.0, -99.93, speed=12.0, history=zig)
    run_engine_cycle(db, now=NOW)
    ship = db.query(models.OilSpillAlert).filter_by(ship_id="600000001").one()
    assert ship.status == S.PENDING_IMAGE
    assert "BEHAVIOUR_OUTLIER_ISO_FOREST" in ship.anomaly_reason


def test_isolated_stationary_ship_is_flagged_as_loitering(db):
    open_ocean_fleet(db)
    add_ship(db, 600000002, -50.0, -80.0, speed=0.2)
    run_engine_cycle(db, now=NOW)
    assert "LOITERING_ALONE" in db.query(models.OilSpillAlert).filter_by(ship_id="600000002").one().anomaly_reason


def test_recently_cleared_ship_is_not_re_escalated_until_new_flag(db):
    open_ocean_fleet(db)
    ship = add_ship(db, 600000003, -50.0, -80.0, speed=0.2, status=S.SAFE,
                    reviewed_at=NOW - timedelta(hours=2))
    run_engine_cycle(db, now=NOW)
    assert status(db, 600000003) == S.SAFE

    ship.kinematic_flag_at = NOW - timedelta(minutes=5)  # sudden stop after the review
    db.commit()
    run_engine_cycle(db, now=NOW + timedelta(seconds=10))
    ship = db.query(models.OilSpillAlert).filter_by(ship_id="600000003").one()
    assert ship.status == S.PENDING_IMAGE
    assert "KINEMATIC_FLAG" in ship.anomaly_reason


def test_stale_ships_are_ignored(db):
    add_ship(db, 600000004, -50.0, -80.0, speed=0.2, now=NOW - timedelta(days=2))
    summary = run_engine_cycle(db, now=NOW)
    assert summary["fresh_ships"] == 0


def test_escalations_and_cycles_are_logged(db):
    open_ocean_fleet(db)
    add_ship(db, 600000002, -50.0, -80.0, speed=0.2)
    summary = run_engine_cycle(db, now=NOW)
    events = db.query(models.AlertEvent).all()
    escalations = [e for e in events if e.event_type == "ESCALATED"]
    cycles = [e for e in events if e.event_type == "ENGINE_CYCLE"]
    assert len(escalations) == summary["escalated"] >= 1
    assert len(cycles) == 1 and json.loads(cycles[0].details)["fresh_ships"] == 31


def test_two_stopped_boats_in_a_harbour_are_not_a_rendezvous(db):
    open_ocean_fleet(db)
    add_ship(db, 500000011, 43.296, 5.364, speed=0.0)   # Marseille old port
    add_ship(db, 500000012, 43.2965, 5.3645, speed=0.0)
    run_engine_cycle(db, now=NOW)
    for sid in (500000011, 500000012):
        ship = db.query(models.OilSpillAlert).filter_by(ship_id=str(sid)).one()
        assert "RENDEZVOUS" not in (ship.anomaly_reason or "")
        assert "LOITERING" not in (ship.anomaly_reason or "")


def test_fast_ships_passing_close_are_not_a_rendezvous(db):
    open_ocean_fleet(db)
    add_ship(db, 500000021, -45.0, -120.0, speed=14.0)
    add_ship(db, 500000022, -45.003, -120.002, speed=13.0)
    run_engine_cycle(db, now=NOW)
    assert "RENDEZVOUS" not in (db.query(models.OilSpillAlert).filter_by(ship_id="500000021").one().anomaly_reason or "")
