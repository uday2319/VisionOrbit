"""Database models for persisting analysis records and uploaded image metadata.

Written against the SQLAlchemy 2.0 declarative API (:class:`DeclarativeBase` +
:class:`Mapped` / :func:`mapped_column`) rather than the legacy ``declarative_base()`` factory:
the legacy form produced a dynamically-created base that a type checker cannot see through, which
is the whole of the module's former type-check failure.

Two honesty points live in this file:

* ``mode`` has **no default**. A row is written by the analysis pipeline, which always knows the
  canonical mode it routed against; a column default of ``"single_optical"`` meant any row written
  without one silently claimed to be a single optical analysis.
* :meth:`AnalysisRecord.to_summary` reports the analysis's *stored* status. It used to emit a
  constant ``"success"``, so a failed run appeared successful in the history list.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Base(DeclarativeBase):
    """Declarative base for every table in this application."""


class UploadedImageRecord(Base):
    """DB table storing uploaded image metadata."""

    __tablename__ = "uploaded_images"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    file_path: Mapped[str] = mapped_column(String, nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    bands: Mapped[int] = mapped_column(Integer, nullable=False)
    dtype: Mapped[str] = mapped_column(String, nullable=False)
    modality: Mapped[str] = mapped_column(String, nullable=False)
    crs_epsg: Mapped[int | None] = mapped_column(Integer, nullable=True)
    crs_wkt: Mapped[str | None] = mapped_column(Text, nullable=True)
    georeferenced: Mapped[int] = mapped_column(Integer, default=0)
    preview_path: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[str] = mapped_column(String, default=_utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.id,
            "filename": self.filename,
            "file_path": self.file_path,
            "width": self.width,
            "height": self.height,
            "bands": self.bands,
            "dtype": self.dtype,
            "modality": self.modality,
            "crs_epsg": self.crs_epsg,
            "georeferenced": bool(self.georeferenced),
            "created_at": self.created_at,
        }


class AnalysisRecord(Base):
    """DB table storing analysis results and execution traces (§7).

    §7 requires that every run persists what it was and what ran it: the id, the canonical mode,
    the task, the tool, the input count and modalities, and the timestamp. Those are columns rather
    than facts buried inside the trace JSON, so history and audit queries do not have to parse a
    blob to answer "which tool produced this?".
    """

    __tablename__ = "analysis_records"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    #: Canonical InputMode value. No default: the writer always knows it (§5).
    mode: Mapped[str] = mapped_column(String, nullable=False)
    task: Mapped[str] = mapped_column(String, nullable=False)
    #: The tool that actually produced the answer, or NULL when none ran.
    tool: Mapped[str | None] = mapped_column(String, nullable=True)
    #: That tool's fallback-chain rung, so "classical" is recorded rather than implied.
    tier: Mapped[str | None] = mapped_column(String, nullable=True)
    #: Terminal AnalysisStatus value — success / partial / failed.
    status: Mapped[str] = mapped_column(String, nullable=False, default="success")
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    confidence_level: Mapped[str] = mapped_column(String, nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    input_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    modalities_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    image_ids_json: Mapped[str] = mapped_column(Text, nullable=False)
    data_json: Mapped[str] = mapped_column(Text, nullable=False)
    trace_json: Mapped[str] = mapped_column(Text, nullable=False)
    #: Measured end-to-end wall clock for the run (§51). NULL when a run was not timed.
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[str] = mapped_column(String, default=_utc_now_iso)

    @staticmethod
    def _loads(blob: str | None, fallback: Any) -> Any:
        """Parse a stored JSON column, tolerating a legacy NULL rather than raising."""
        if not blob:
            return fallback
        return json.loads(blob)

    @property
    def image_ids(self) -> list[str]:
        return list(self._loads(self.image_ids_json, []))

    @property
    def modalities(self) -> list[str]:
        return list(self._loads(self.modalities_json, []))

    @property
    def data(self) -> dict[str, Any]:
        return dict(self._loads(self.data_json, {}))

    @property
    def trace(self) -> dict[str, Any]:
        return dict(self._loads(self.trace_json, {}))

    def to_summary(self) -> dict[str, Any]:
        return {
            "analysis_id": self.id,
            "id": self.id,
            "query": self.query,
            "title": self.query,
            "mode": self.mode,
            "task": self.task,
            "tool": self.tool,
            "confidence_level": self.confidence_level,
            "confidence": self.confidence_level,
            "confidence_score": self.confidence_score,
            "answer": self.answer,
            "image_count": self.input_count or len(self.image_ids),
            "status": self.status,
            "created_at": self.created_at,
        }
