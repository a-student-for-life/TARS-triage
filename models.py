from datetime import datetime, timezone
import enum

from sqlalchemy import Column, DateTime, Enum, Float, Index, Integer, String, Text
from sqlalchemy.sql import func

from database import Base


def utc_now():
    return datetime.now(timezone.utc)


class JobStatus(str, enum.Enum):
    # 1. New Data (The Listener just dropped it here)
    UNPROCESSED = "UNPROCESSED"

    # 2. The Filter (Analysis decided these are safe/boring)
    SAFE = "SAFE"

    # 3. The Suspects (DBSCAN/Physics found an issue, needs an image)
    PENDING_IMAGE = "PENDING_IMAGE"

    # 4. The Workflow (Watchdog is downloading/analyzing)
    SEARCHING_IMG = "SEARCHING_IMG"
    PROCESSING_IMG = "PROCESSING_IMG"
    ANALYZING = "ANALYZING"

    # 5. The Verdict (Final Outcomes)
    OIL_DETECTED = "OIL_DETECTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

    # 6. Human Actions
    RESPONSE_INITIATED = "RESPONSE_INITIATED"
    DISPATCHED = "DISPATCHED"
    IGNORED = "IGNORED"
    RESOLVED = "RESOLVED"

    # 7. Processing failure
    ERROR = "ERROR"

    # 8. No usable image arrived within the acquisition window
    EXPIRED = "EXPIRED"


class ImageStatus(str, enum.Enum):
    PENDING = "PENDING"
    AVAILABLE = "AVAILABLE"
    FAILED = "FAILED"
    UNAVAILABLE = "UNAVAILABLE"


class ImageAlignmentStatus(str, enum.Enum):
    ALIGNED = "ALIGNED"
    WEAK_ALIGNMENT = "WEAK_ALIGNMENT"
    MISALIGNED = "MISALIGNED"


class OilSpillAlert(Base):
    __tablename__ = "oil_spill_alerts"
    __table_args__ = (Index("ix_alert_status", "status"),)

    # -- Section A: The Case Info --
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), default=utc_now, server_default=func.now(), onupdate=utc_now)

    # Default status is now UNPROCESSED so the Anomaly Engine picks it up
    status = Column(Enum(JobStatus), default=JobStatus.UNPROCESSED)

    # -- Section B: The Trigger (From Ship Anomaly) --
    ship_id = Column(String)
    lat = Column(Float)
    lon = Column(Float)
    speed = Column(Float, nullable=True)
    anomaly_type = Column(String)
    anomaly_reason = Column(String, default="UNKNOWN")
    anomaly_timestamp = Column(DateTime(timezone=True), nullable=True)

    # -- Section C: The Investigation (From Satellite/SAR/ML) --
    granule_id = Column(String, nullable=True)
    image_url = Column(String, nullable=True)
    image_timestamp = Column(DateTime(timezone=True), nullable=True)
    image_status = Column(Enum(ImageStatus), nullable=True)
    image_alignment_status = Column(Enum(ImageAlignmentStatus), nullable=True)
    prediction_confidence = Column(Float, nullable=True)
    final_result = Column(String, nullable=True)

    # -- Section D: Human Workflow --
    dispatched_at = Column(DateTime(timezone=True), nullable=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)

    # -- Section E: Pipeline bookkeeping --
    last_report_at = Column(DateTime(timezone=True), nullable=True)   # time of latest AIS report
    kinematic_flag_at = Column(DateTime(timezone=True), nullable=True)  # last Layer-1 flag
    reviewed_at = Column(DateTime(timezone=True), nullable=True)       # last image classification
    raw_score = Column(Float, nullable=True)                           # mean model score (0-1)
    image_attempts = Column(Integer, nullable=True)
    next_attempt_at = Column(DateTime(timezone=True), nullable=True)


class PositionReport(Base):
    """Recent AIS reports per ship (capped at config.HISTORY_LEN) for trajectory features."""
    __tablename__ = "position_reports"
    __table_args__ = (Index("ix_position_ship_time", "ship_id", "reported_at"),)

    id = Column(Integer, primary_key=True)
    ship_id = Column(String, nullable=False)
    reported_at = Column(DateTime(timezone=True), nullable=False)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    speed = Column(Float, nullable=True)   # None = not available
    course = Column(Float, nullable=True)  # course over ground, degrees; None = not available


class AlertEvent(Base):
    """Append-only log. Rows are never updated or deleted, so every escalation,
    classification and operator action can be counted and audited later."""
    __tablename__ = "alert_events"
    __table_args__ = (Index("ix_event_type_time", "event_type", "created_at"),)

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    alert_id = Column(Integer, nullable=True)
    ship_id = Column(String, nullable=True)
    event_type = Column(String, nullable=False)
    from_status = Column(String, nullable=True)
    to_status = Column(String, nullable=True)
    reason = Column(String, nullable=True)
    lat = Column(Float, nullable=True)
    lon = Column(Float, nullable=True)
    speed = Column(Float, nullable=True)
    granule_id = Column(String, nullable=True)
    image_timestamp = Column(DateTime(timezone=True), nullable=True)
    raw_score = Column(Float, nullable=True)
    details = Column(Text, nullable=True)  # JSON: scores per draw, settings, code version
