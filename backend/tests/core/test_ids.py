"""Tests for identifier generation.

The analysis-ID format is a user-visible contract (brief §13) — judges read it off a report —
so its shape is pinned exactly rather than checked loosely.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from app.core import ids


class TestAnalysisIdFormat:
    def test_matches_the_documented_example(self):
        assert ids.format_analysis_id(2026, 123) == "SAT-2026-000123"

    def test_sequence_is_zero_padded_to_six_digits(self):
        assert ids.format_analysis_id(2026, 1) == "SAT-2026-000001"

    def test_large_sequence_is_not_truncated(self):
        """Overflowing the padding must widen the field, never drop digits."""
        assert ids.format_analysis_id(2026, 1_234_567) == "SAT-2026-1234567"

    def test_year_is_zero_padded_to_four_digits(self):
        assert ids.format_analysis_id(7, 1) == "SAT-0007-000001"

    def test_generated_ids_validate(self):
        assert ids.is_analysis_id(ids.format_analysis_id(2026, 42))

    def test_generated_ids_validate_past_the_padding_width(self):
        """A generator whose own output its validator rejects is a latent failure."""
        assert ids.is_analysis_id(ids.format_analysis_id(2026, 1_234_567))

    def test_recognises_only_the_exact_shape(self):
        for bad in (
            "SAT-2026-123",        # too few digits
            "SAT-26-000123",       # short year
            "sat-2026-000123",     # wrong case
            "SAT-2026-000123 ",    # trailing space
            "xSAT-2026-000123",    # prefixed
            "SAT-2026-00012a",     # non-digit
            "",
        ):
            assert not ids.is_analysis_id(bad), bad

    def test_rejects_injection_attempts(self):
        """Regression: `$` also matches before a trailing newline, so a newline-suffixed ID
        validated and could reach a filename or log line. The pattern is anchored with
        `\\A`/`\\Z` instead."""
        for bad in ("SAT-2026-000123/../etc", "SAT-2026-000123\n", "../SAT-2026-000123",
                    "SAT-2026-000123\r\n", "SAT-2026-000123\x00"):
            assert not ids.is_analysis_id(bad), bad


class TestSequenceCounter:
    def test_is_monotonic(self):
        first = ids.next_analysis_sequence()
        assert ids.next_analysis_sequence() == first + 1

    def test_is_thread_safe(self):
        """A duplicate here would mean two analyses sharing an ID."""
        with ThreadPoolExecutor(max_workers=8) as pool:
            got = list(pool.map(lambda _: ids.next_analysis_sequence(), range(400)))
        assert len(set(got)) == 400


class TestOpaqueIds:
    def test_image_ids_are_prefixed_and_unique(self):
        got = {ids.new_image_id() for _ in range(500)}
        assert len(got) == 500
        assert all(re.fullmatch(r"img_[0-9a-f]{16}", v) for v in got)

    def test_artifact_ids_are_prefixed_and_unique(self):
        got = {ids.new_artifact_id() for _ in range(500)}
        assert len(got) == 500
        assert all(re.fullmatch(r"art_[0-9a-f]{16}", v) for v in got)

    def test_image_and_artifact_ids_never_collide(self):
        assert not ({ids.new_image_id() for _ in range(50)} & {ids.new_artifact_id() for _ in range(50)})


class TestClock:
    def test_utc_now_is_timezone_aware(self):
        """A naive timestamp would compare wrongly against database values."""
        now = ids.utc_now()
        assert now.tzinfo is not None
        assert now.utcoffset() == datetime.now(tz=UTC).utcoffset()

    def test_current_year_is_plausible(self):
        assert ids.current_year() == datetime.now(tz=UTC).year

    def test_current_year_matches_utc_now(self):
        assert ids.current_year() == ids.utc_now().year
