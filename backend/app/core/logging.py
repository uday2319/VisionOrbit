"""Structured logging and correlation IDs.

Every log line carries the ``request_id`` (and ``analysis_id`` when known) so a single
request can be reconstructed from logs (brief §30). The IDs live in ``contextvars`` so they
propagate through async call stacks without being threaded through every signature.
"""
from __future__ import annotations

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_analysis_id: ContextVar[str | None] = ContextVar("analysis_id", default=None)

# Attributes present on every LogRecord; anything else was passed via `extra=` and is
# therefore structured context worth emitting.
_STD_ATTRS = frozenset(
    {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "taskName", "message", "asctime",
    }
)


def new_request_id() -> str:
    """Generate a request correlation ID."""
    return f"req_{uuid.uuid4().hex[:16]}"


def set_request_id(value: str) -> None:
    _request_id.set(value)


def get_request_id() -> str | None:
    return _request_id.get()


def set_analysis_id(value: str | None) -> None:
    _analysis_id.set(value)


def get_analysis_id() -> str | None:
    return _analysis_id.get()


class JsonFormatter(logging.Formatter):
    """Render records as single-line JSON, enriched with correlation IDs."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if rid := _request_id.get():
            payload["request_id"] = rid
        if aid := _analysis_id.get():
            payload["analysis_id"] = aid
        for key, value in record.__dict__.items():
            if key not in _STD_ATTRS and not key.startswith("_"):
                payload[key] = _coerce(value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    """Human-friendly formatter for local development."""

    def format(self, record: logging.LogRecord) -> str:
        base = f"{self.formatTime(record, '%H:%M:%S')} {record.levelname:<7} {record.name}: {record.getMessage()}"
        if rid := _request_id.get():
            base += f"  [{rid}]"
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def _coerce(value: Any) -> Any:
    """Make arbitrary extra-values JSON friendly."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return [_coerce(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _coerce(v) for k, v in value.items()}
    return str(value)


def configure_logging(level: str = "INFO", as_json: bool = True) -> None:
    """Install the root handler. Safe to call repeatedly (replaces existing handlers)."""
    root = logging.getLogger()
    root.setLevel(level.upper())
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if as_json else TextFormatter())
    root.addHandler(handler)
    # Uvicorn's access log duplicates our structured request log.
    logging.getLogger("uvicorn.access").disabled = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger."""
    return logging.getLogger(name)
