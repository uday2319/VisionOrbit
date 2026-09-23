"""Database session management, connection pooling, and repository helper functions."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Generator, Sequence

from sqlalchemy import create_engine, func, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from ..config import get_settings
from ..core.logging import get_logger
from .models import AnalysisRecord, Base, UploadedImageRecord

logger = get_logger(__name__)
settings = get_settings()

connect_args = {"check_same_thread": False} if settings.is_sqlite else {}
engine = create_engine(
    settings.database_url,
    connect_args=connect_args,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Initialize database tables, then add any columns a pre-existing file is missing. Idempotent."""
    settings.ensure_dirs()
    Base.metadata.create_all(bind=engine)
    _add_missing_columns()


# Columns added to `analysis_records` after the first release. `create_all` only creates missing
# *tables*, so a database file written by an earlier version keeps its old shape and every query
# naming a new column fails. This is a deliberately minimal forward-only migration: SQLite's
# ADD COLUMN is safe, non-destructive and idempotent here because we check first. A prototype at
# this scale does not warrant Alembic (§50), but silently breaking an existing local database
# would be worse than either.
_ANALYSIS_COLUMN_MIGRATIONS: tuple[tuple[str, str], ...] = (
    ("tool", "VARCHAR"),
    ("tier", "VARCHAR"),
    ("status", "VARCHAR NOT NULL DEFAULT 'success'"),
    ("input_count", "INTEGER NOT NULL DEFAULT 0"),
    ("modalities_json", "TEXT NOT NULL DEFAULT '[]'"),
    ("duration_ms", "INTEGER"),
)


def _add_missing_columns() -> None:
    """Add post-release columns to an existing `analysis_records` table, if absent."""
    inspector = inspect(engine)
    if "analysis_records" not in inspector.get_table_names():
        return
    existing = {col["name"] for col in inspector.get_columns("analysis_records")}
    missing = [(name, ddl) for name, ddl in _ANALYSIS_COLUMN_MIGRATIONS if name not in existing]
    if not missing:
        return
    with engine.begin() as conn:
        for name, ddl in missing:
            conn.execute(text(f"ALTER TABLE analysis_records ADD COLUMN {name} {ddl}"))
    logger.info(
        "added missing analysis_records columns",
        extra={"columns": [name for name, _ in missing]},
    )


def get_db() -> Generator[Session, None, None]:
    """Dependency injection yield for DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def save_image_record(
    db: Session,
    image_id: str,
    filename: str,
    file_path: str,
    width: int,
    height: int,
    bands: int,
    dtype: str,
    modality: str,
    crs_epsg: int | None = None,
    crs_wkt: str | None = None,
    georeferenced: bool = False,
    preview_path: str | None = None,
) -> UploadedImageRecord:
    rec = UploadedImageRecord(
        id=image_id,
        filename=filename,
        file_path=file_path,
        width=width,
        height=height,
        bands=bands,
        dtype=dtype,
        modality=modality,
        crs_epsg=crs_epsg,
        crs_wkt=crs_wkt,
        georeferenced=1 if georeferenced else 0,
        preview_path=preview_path,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return rec


def get_image_record(db: Session, image_id: str) -> UploadedImageRecord | None:
    return db.query(UploadedImageRecord).filter(UploadedImageRecord.id == image_id).first()


def save_analysis_record(
    db: Session,
    analysis_id: str,
    query: str,
    mode: str,
    task: str,
    answer: str,
    confidence_level: str,
    confidence_score: float,
    image_ids: list[str],
    data: dict[str, Any],
    trace: dict[str, Any],
    *,
    status: str,
    input_count: int,
    modalities: list[str],
    tool: str | None = None,
    tier: str | None = None,
    duration_ms: int | None = None,
) -> AnalysisRecord:
    """Persist one completed analysis with everything §7 requires to audit it later.

    ``mode``, ``status``, ``input_count`` and ``modalities`` are required keyword facts rather than
    defaulted ones: the caller always knows them, and a default here would write a plausible lie
    into the permanent record.
    """
    rec = AnalysisRecord(
        id=analysis_id,
        query=query,
        mode=mode,
        task=task,
        tool=tool,
        tier=tier,
        status=status,
        answer=answer,
        confidence_level=confidence_level,
        confidence_score=confidence_score,
        input_count=input_count,
        modalities_json=json.dumps(modalities),
        image_ids_json=json.dumps(image_ids),
        data_json=json.dumps(data),
        trace_json=json.dumps(trace),
        duration_ms=duration_ms,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return rec


def get_analysis_record(db: Session, analysis_id: str) -> AnalysisRecord | None:
    return db.query(AnalysisRecord).filter(AnalysisRecord.id == analysis_id).first()


def list_analysis_records(db: Session, limit: int = 50) -> Sequence[AnalysisRecord]:
    return (
        db.query(AnalysisRecord)
        .order_by(AnalysisRecord.created_at.desc())
        .limit(limit)
        .all()
    )


def get_next_analysis_id(db: Session) -> str:
    """Allocate the next display analysis ID (``SAT-<year>-<sequence>``) for the current year.

    The sequence comes from the highest existing id **for this year**, not from the total row
    count: a count restarts the numbering whenever the year rolls over, which would collide with
    the previous year's rows only by luck of the id including the year, and would renumber
    unpredictably after any deletion. Scanning for the max is cheap at this scale and correct.

    The final probe loop closes the remaining gap: two concurrent requests can compute the same
    candidate, and the id is the primary key, so we advance past anything already present. A true
    race between the probe and the insert would still raise ``IntegrityError`` rather than silently
    overwrite — the failure mode is a visible error, never a lost analysis.
    """
    from ..core.ids import current_year, format_analysis_id

    year = current_year()
    prefix = f"SAT-{year:04d}-"
    highest: str | None = (
        db.query(func.max(AnalysisRecord.id))
        .filter(AnalysisRecord.id.like(f"{prefix}%"))
        .scalar()
    )
    seq = 1
    if highest:
        tail = highest.rsplit("-", 1)[-1]
        if tail.isdigit():
            seq = int(tail) + 1
    candidate = format_analysis_id(year, seq)
    while db.query(AnalysisRecord.id).filter(AnalysisRecord.id == candidate).first() is not None:
        seq += 1
        candidate = format_analysis_id(year, seq)
    return candidate
