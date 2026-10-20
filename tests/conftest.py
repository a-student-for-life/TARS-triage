"""Shared test setup. Tests never touch oil_spill.db or temp_images/."""
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

_TMP = tempfile.mkdtemp(prefix="tars_tests_")
os.environ["TARS_DB_URL"] = f"sqlite:///{os.path.join(_TMP, 'import_time.db')}"
os.environ["TARS_TEMP_IMAGE_DIR"] = os.path.join(_TMP, "images")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import models  # noqa: E402

NOW = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def db():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    models.Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False)()
    yield session
    session.close()


def add_ship(db, ship_id, lat, lon, speed=10.0, status=models.JobStatus.UNPROCESSED,
             history=None, now=NOW, **kw):
    """Create a ship row plus optional (lat, lon, speed) history, 1 minute apart."""
    ship = models.OilSpillAlert(ship_id=str(ship_id), lat=lat, lon=lon, speed=speed,
                                status=status, anomaly_type="NONE", last_report_at=now, **kw)
    db.add(ship)
    for i, (hlat, hlon, hspeed) in enumerate(history or []):
        db.add(models.PositionReport(
            ship_id=str(ship_id), reported_at=now - timedelta(minutes=len(history) - i),
            lat=hlat, lon=hlon, speed=hspeed, course=None,
        ))
    db.commit()
    return ship


def straight_track(lat, lon, n=8, step=0.01, speed=12.0):
    return [(lat, lon + i * step, speed) for i in range(n)]
