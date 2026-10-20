"""FastAPI backend: AIS ship ingest, HDBSCAN + Isolation Forest anomaly engine, alert API."""
from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import os
import threading
import time

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import config
import models
import schemas
from ais_validation import clean_report
from anomaly_engine import run_engine_cycle
from database import SessionLocal, engine, ensure_alert_schema, get_db
from events import log_event
from notifications import send_dispatch_notification
from sqlalchemy.orm import Session

models.Base.metadata.create_all(bind=engine)
ensure_alert_schema()
os.makedirs(config.TEMP_IMAGE_DIR, exist_ok=True)

ACTIVE_ALERT_EXCLUSIONS = [
    models.JobStatus.SAFE,
    models.JobStatus.COMPLETED,
    models.JobStatus.RESOLVED,
    models.JobStatus.IGNORED,
    models.JobStatus.EXPIRED,
]
ENGINE_INTERVAL_SECONDS = 10


def utc_now():
    return datetime.now(timezone.utc)


def background_anomaly_engine():
    """Runs Layers 2-3 (see anomaly_engine.py) every ENGINE_INTERVAL_SECONDS."""
    while True:
        db = SessionLocal()
        try:
            summary = run_engine_cycle(db)
            if summary.get("escalated"):
                print(f"Engine: {summary}", flush=True)
        except Exception as exc:
            db.rollback()
            print(f"Engine Error: {exc}", flush=True)
        finally:
            db.close()
        time.sleep(ENGINE_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("System Startup: Launching Anomaly Engine...")
    thread = threading.Thread(target=background_anomaly_engine, daemon=True)
    thread.start()

    yield

    print("System Shutdown: Engine stopping...")


app = FastAPI(lifespan=lifespan, title="SAR Oil Spill Watchdog")
app.mount("/images", StaticFiles(directory=config.TEMP_IMAGE_DIR), name="images")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("TARS_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(","),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/alert/{id}", response_model=schemas.ShipAlertResponse)
def get_alert_status(id: int, db: Session = Depends(get_db)):
    report = db.query(models.OilSpillAlert).filter(models.OilSpillAlert.id == id).first()

    if not report:
        raise HTTPException(status_code=404, detail="Alert not found")
    return report


@app.get("/alerts", response_model=list[schemas.ShipAlertResponse])
def get_all_alerts(db: Session = Depends(get_db)):
    return db.query(models.OilSpillAlert).filter(
        models.OilSpillAlert.status.notin_(ACTIVE_ALERT_EXCLUSIONS)
    ).all()


@app.get("/alerts/active", response_model=list[schemas.ShipAlertResponse])
def get_active_alerts(db: Session = Depends(get_db)):
    return db.query(models.OilSpillAlert).filter(
        models.OilSpillAlert.status.notin_(ACTIVE_ALERT_EXCLUSIONS)
    ).all()


@app.get("/alerts/history", response_model=list[schemas.ShipAlertResponse])
def get_history_alerts(db: Session = Depends(get_db)):
    return db.query(models.OilSpillAlert).filter(
        models.OilSpillAlert.status.in_(
            [
                models.JobStatus.SAFE,
                models.JobStatus.COMPLETED,
                models.JobStatus.RESOLVED,
                models.JobStatus.IGNORED,
            ]
        )
    ).all()


REJECTED_COUNTS = Counter()


def _report_time(raw_ts):
    """AIS message time if the ingestor sent one, otherwise now."""
    if raw_ts is None:
        return utc_now()
    return raw_ts if raw_ts.tzinfo else raw_ts.replace(tzinfo=timezone.utc)


def _append_history(db, ship_id, reported_at, report):
    db.add(models.PositionReport(
        ship_id=ship_id, reported_at=reported_at,
        lat=report.lat, lon=report.lon, speed=report.speed, course=report.course,
    ))
    db.flush()
    stale = (
        db.query(models.PositionReport.id)
        .filter(models.PositionReport.ship_id == ship_id)
        .order_by(models.PositionReport.reported_at.desc())
        .offset(config.HISTORY_LEN)
        .all()
    )
    if stale:
        db.query(models.PositionReport).filter(
            models.PositionReport.id.in_([row.id for row in stale])
        ).delete(synchronize_session=False)


@app.post("/ingest")
async def ingest_ship_data(ship: schemas.RawShipData, db: Session = Depends(get_db)):
    report, rejection = clean_report(ship.mmsi, ship.lat, ship.lon, ship.speed, ship.course)
    if rejection:
        REJECTED_COUNTS[rejection] += 1
        return {"status": "rejected", "reason": rejection}

    reported_at = _report_time(ship.timestamp)
    ship_id = str(report.mmsi)
    existing = db.query(models.OilSpillAlert).filter(models.OilSpillAlert.ship_id == ship_id).first()

    if existing is None:
        existing = models.OilSpillAlert(
            ship_id=ship_id, lat=report.lat, lon=report.lon, speed=report.speed,
            anomaly_type="NONE", status=models.JobStatus.UNPROCESSED,
            last_report_at=reported_at,
        )
        db.add(existing)
    else:
        # Layer 1: sudden speed change between two close-in-time valid reports.
        previous_at = existing.last_report_at
        if previous_at is not None and previous_at.tzinfo is None:
            previous_at = previous_at.replace(tzinfo=timezone.utc)
        gap = (reported_at - previous_at).total_seconds() if previous_at else None
        if (
            existing.speed is not None and report.speed is not None
            and gap is not None and 0 < gap <= config.SUDDEN_MAX_GAP_SECONDS
        ):
            delta = report.speed - existing.speed
            if abs(delta) > config.SUDDEN_SPEED_DELTA_KNOTS:
                existing.anomaly_type = "SUDDEN_STOP" if delta < 0 else "SUDDEN_ACCEL"
                existing.kinematic_flag_at = reported_at
                log_event(db, "KINEMATIC_FLAG", existing, reason=existing.anomaly_type,
                          details={"speed_from": existing.speed, "speed_to": report.speed, "gap_s": gap})

        existing.lat = report.lat
        existing.lon = report.lon
        existing.speed = report.speed
        existing.last_report_at = reported_at

    _append_history(db, ship_id, reported_at, report)
    db.commit()
    return {"status": "ok"}


@app.get("/ingest/rejections")
def ingest_rejections():
    """Counts of AIS reports dropped by validation since the server started."""
    return dict(REJECTED_COUNTS)


@app.get("/events")
def list_events(event_type: str | None = None, limit: int = 200, db: Session = Depends(get_db)):
    """Recent rows from the append-only event log."""
    query = db.query(models.AlertEvent)
    if event_type:
        query = query.filter(models.AlertEvent.event_type == event_type)
    rows = query.order_by(models.AlertEvent.id.desc()).limit(min(limit, 5000)).all()
    return [
        {c.name: getattr(r, c.name) for c in models.AlertEvent.__table__.columns}
        for r in rows
    ]


@app.put("/alert/{id}/status", response_model=schemas.ShipAlertResponse)
def update_alert_status(id: int, update: schemas.StatusUpdate, db: Session = Depends(get_db)):
    alert = db.query(models.OilSpillAlert).filter(models.OilSpillAlert.id == id).first()

    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    requested_status = update.status
    previous_status = alert.status
    if requested_status == models.JobStatus.RESPONSE_INITIATED:
        requested_status = models.JobStatus.DISPATCHED

    should_notify = requested_status == models.JobStatus.DISPATCHED and alert.dispatched_at is None
    now = utc_now()

    alert.status = requested_status
    if requested_status == models.JobStatus.DISPATCHED and alert.dispatched_at is None:
        alert.dispatched_at = now
    if requested_status == models.JobStatus.DISPATCHED:
        alert.resolved_at = None
    if requested_status in [models.JobStatus.IGNORED, models.JobStatus.RESOLVED]:
        alert.resolved_at = now

    log_event(db, "OPERATOR_ACTION", alert, from_status=previous_status, to_status=requested_status,
              raw_score=alert.raw_score)
    db.commit()
    db.refresh(alert)

    if should_notify:
        delivered, _ = send_dispatch_notification(alert)
        print(
            f"[DISPATCH] Alert {alert.id} for MMSI {alert.ship_id} "
            f"{'emailed' if delivered else 'simulated'} at {alert.dispatched_at.isoformat()}",
            flush=True,
        )
    elif requested_status in [models.JobStatus.IGNORED, models.JobStatus.RESOLVED]:
        print(
            f"[ARCHIVE] Alert {alert.id} for MMSI {alert.ship_id} archived at {alert.resolved_at.isoformat()}",
            flush=True,
        )

    return alert
