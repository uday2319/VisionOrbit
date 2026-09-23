"""Identifier generation.

Analysis IDs are human-quotable (``SAT-2026-000123``, brief §13) because judges read them off
a report; image and artifact IDs are opaque.
"""
from __future__ import annotations

import re
import threading
import uuid
from datetime import UTC, datetime

# `\A`/`\Z` rather than `^`/`$`: in Python `$` also matches *before* a trailing newline, so
# `"SAT-2026-000123\n"` would validate and then flow into a filename or a log line. The
# sequence is `{6,}` so that anything :func:`format_analysis_id` produces validates — a
# generator that can emit IDs its own validator rejects is a latent 500.
_ANALYSIS_ID_RE = re.compile(r"\ASAT-\d{4}-\d{6,}\Z")
_lock = threading.Lock()
_counter = 0


def format_analysis_id(year: int, sequence: int) -> str:
    """Format a display analysis ID, e.g. ``SAT-2026-000123``."""
    return f"SAT-{year:04d}-{sequence:06d}"


def is_analysis_id(value: str) -> bool:
    """Whether ``value`` is a well-formed analysis ID."""
    return bool(_ANALYSIS_ID_RE.match(value))


def next_analysis_sequence() -> int:
    """Process-local monotonic counter.

    Used only as a fallback; the authoritative sequence comes from the database so IDs stay
    unique across workers and restarts.
    """
    global _counter
    with _lock:
        _counter += 1
        return _counter


def current_year() -> int:
    return datetime.now(tz=UTC).year


def new_image_id() -> str:
    return f"img_{uuid.uuid4().hex[:16]}"


def new_artifact_id() -> str:
    return f"art_{uuid.uuid4().hex[:16]}"


def utc_now() -> datetime:
    """Timezone-aware current UTC time (never a naive datetime)."""
    return datetime.now(tz=UTC)
