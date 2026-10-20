"""Append-only event log helpers.

The alerts table holds only the *current* state of each ship and is
overwritten as ships move. Anything that needs to be counted later (for a
paper, an audit, or a reviewer) is written here as a new row instead.
"""
import json
import subprocess
from functools import lru_cache

import models


@lru_cache(maxsize=1)
def code_version():
    """Short git commit hash of the running code, or 'unknown'."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, timeout=5,
        )
        version = out.stdout.strip() or "unknown"
        return f"{version}-dirty" if dirty.stdout.strip() else version
    except Exception:
        return "unknown"


def _status_value(status):
    return status.value if hasattr(status, "value") else status


def log_event(db, event_type, alert=None, *, from_status=None, to_status=None,
              reason=None, raw_score=None, details=None):
    """Add an event row to the session. The caller commits."""
    payload = dict(details or {})
    payload.setdefault("code_version", code_version())
    event = models.AlertEvent(
        alert_id=getattr(alert, "id", None),
        ship_id=getattr(alert, "ship_id", None),
        event_type=event_type,
        from_status=_status_value(from_status),
        to_status=_status_value(to_status),
        reason=reason,
        lat=getattr(alert, "lat", None),
        lon=getattr(alert, "lon", None),
        speed=getattr(alert, "speed", None),
        granule_id=getattr(alert, "granule_id", None),
        image_timestamp=getattr(alert, "image_timestamp", None),
        raw_score=raw_score,
        details=json.dumps(payload, default=str),
    )
    db.add(event)
    return event


def set_status(db, alert, new_status, reason=None, event_type="STATUS_CHANGE", **kw):
    """Change an alert's status and log the transition. No-op if unchanged."""
    old = alert.status
    if old == new_status and (reason is None or reason == alert.anomaly_reason):
        return False
    alert.status = new_status
    if reason is not None:
        alert.anomaly_reason = reason
    log_event(db, event_type, alert, from_status=old, to_status=new_status, reason=reason, **kw)
    return True
