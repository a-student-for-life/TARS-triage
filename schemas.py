from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict

from models import ImageAlignmentStatus, ImageStatus, JobStatus


class ShipAlertCreate(BaseModel):
    ship_id: str
    lat: float
    lon: float
    anomaly_type: str


class RawShipData(BaseModel):
    mmsi: int
    lat: float
    lon: float
    speed: Optional[float] = None
    course: Optional[float] = None        # course over ground, degrees
    timestamp: Optional[datetime] = None  # AIS message time (UTC); server time if absent


class ShipAlertResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    ship_id: str
    lat: float
    lon: float
    anomaly_type: Optional[str] = None
    anomaly_reason: Optional[str] = None
    status: JobStatus
    created_at: datetime
    updated_at: Optional[datetime] = None
    anomaly_timestamp: Optional[datetime] = None

    granule_id: Optional[str] = None
    image_url: Optional[str] = None
    image_timestamp: Optional[datetime] = None
    image_status: Optional[ImageStatus] = None
    image_alignment_status: Optional[ImageAlignmentStatus] = None
    prediction_confidence: Optional[float] = None
    raw_score: Optional[float] = None
    final_result: Optional[str] = None
    reviewed_at: Optional[datetime] = None

    dispatched_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None


class StatusUpdate(BaseModel):
    status: JobStatus
