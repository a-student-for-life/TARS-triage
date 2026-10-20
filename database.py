from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

import config

SQLite_URL = config.DB_URL
engine = create_engine(
    SQLite_URL, connect_args={"check_same_thread": False}
)


# 2. OPTIONAL: Enable Write-Ahead Logging (WAL) for speed
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

ALERT_SCHEMA_UPDATES = {
    "updated_at": "ALTER TABLE oil_spill_alerts ADD COLUMN updated_at DATETIME",
    "image_timestamp": "ALTER TABLE oil_spill_alerts ADD COLUMN image_timestamp DATETIME",
    "anomaly_timestamp": "ALTER TABLE oil_spill_alerts ADD COLUMN anomaly_timestamp DATETIME",
    "image_status": "ALTER TABLE oil_spill_alerts ADD COLUMN image_status VARCHAR(16)",
    "image_alignment_status": "ALTER TABLE oil_spill_alerts ADD COLUMN image_alignment_status VARCHAR(16)",
    "dispatched_at": "ALTER TABLE oil_spill_alerts ADD COLUMN dispatched_at DATETIME",
    "resolved_at": "ALTER TABLE oil_spill_alerts ADD COLUMN resolved_at DATETIME",
    "last_report_at": "ALTER TABLE oil_spill_alerts ADD COLUMN last_report_at DATETIME",
    "kinematic_flag_at": "ALTER TABLE oil_spill_alerts ADD COLUMN kinematic_flag_at DATETIME",
    "reviewed_at": "ALTER TABLE oil_spill_alerts ADD COLUMN reviewed_at DATETIME",
    "raw_score": "ALTER TABLE oil_spill_alerts ADD COLUMN raw_score FLOAT",
    "image_attempts": "ALTER TABLE oil_spill_alerts ADD COLUMN image_attempts INTEGER",
    "next_attempt_at": "ALTER TABLE oil_spill_alerts ADD COLUMN next_attempt_at DATETIME",
}


def ensure_alert_schema():
    with engine.begin() as connection:
        tables = {
            row[0]
            for row in connection.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if "oil_spill_alerts" not in tables:
            return

        existing_columns = {
            row[1]
            for row in connection.exec_driver_sql(
                "PRAGMA table_info('oil_spill_alerts')"
            ).fetchall()
        }

        for column_name, ddl in ALERT_SCHEMA_UPDATES.items():
            if column_name not in existing_columns:
                connection.exec_driver_sql(ddl)

        connection.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS ix_alert_status ON oil_spill_alerts(status)"
        )

        # Safe backfills for previously created rows.
        connection.exec_driver_sql(
            "UPDATE oil_spill_alerts SET updated_at = COALESCE(updated_at, created_at) "
            "WHERE updated_at IS NULL"
        )
        connection.exec_driver_sql(
            "UPDATE oil_spill_alerts SET anomaly_timestamp = COALESCE(anomaly_timestamp, created_at) "
            "WHERE anomaly_timestamp IS NULL"
        )
        connection.exec_driver_sql(
            "UPDATE oil_spill_alerts SET image_status = 'AVAILABLE' "
            "WHERE image_status IS NULL AND image_url IS NOT NULL"
        )
        connection.exec_driver_sql(
            "UPDATE oil_spill_alerts SET image_status = 'PENDING' "
            "WHERE image_status IS NULL AND status IN "
            "('PENDING_IMAGE', 'SEARCHING_IMG', 'PROCESSING_IMG', 'ANALYZING')"
        )
        connection.exec_driver_sql(
            "UPDATE oil_spill_alerts SET resolved_at = COALESCE(resolved_at, updated_at, created_at) "
            "WHERE resolved_at IS NULL AND status IN ('SAFE', 'COMPLETED', 'RESOLVED', 'IGNORED')"
        )
        connection.exec_driver_sql(
            "UPDATE oil_spill_alerts SET dispatched_at = COALESCE(dispatched_at, updated_at, created_at) "
            "WHERE dispatched_at IS NULL AND status IN ('DISPATCHED', 'RESPONSE_INITIATED')"
        )
        connection.exec_driver_sql(
            "UPDATE oil_spill_alerts SET "
            "granule_id = NULL, "
            "image_timestamp = NULL, "
            "image_status = NULL, "
            "image_alignment_status = NULL "
            "WHERE status IN ('SAFE', 'COMPLETED', 'RESOLVED', 'IGNORED') "
            "AND image_url IS NULL "
            "AND image_status IN ('PENDING', 'FAILED', 'UNAVAILABLE')"
        )


# Dependency creation
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
