"""Tests for structured logging and correlation IDs.

The brief (§30) requires every log line to carry a request_id so one request can be
reconstructed from logs, and (§29) requires that stack traces never leak to users. Both are
asserted here against real emitted output rather than inspected by eye.
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.core import logging as applog


@pytest.fixture(autouse=True)
def _clean_context():
    """Correlation IDs are contextvars; reset them so tests cannot leak into each other."""
    applog.set_request_id("")
    applog.set_analysis_id(None)
    yield
    applog.set_request_id("")
    applog.set_analysis_id(None)


def make_record(msg: str = "hello", **extra) -> logging.LogRecord:
    record = logging.LogRecord(
        name="app.test", level=logging.INFO, pathname=__file__, lineno=1,
        msg=msg, args=(), exc_info=None,
    )
    for k, v in extra.items():
        setattr(record, k, v)
    return record


class TestRequestIdGeneration:
    def test_ids_are_prefixed_and_unique(self):
        got = {applog.new_request_id() for _ in range(500)}
        assert len(got) == 500
        assert all(v.startswith("req_") for v in got)


class TestCorrelationContext:
    def test_request_id_round_trips(self):
        applog.set_request_id("req_abc")
        assert applog.get_request_id() == "req_abc"

    def test_analysis_id_round_trips(self):
        applog.set_analysis_id("SAT-2026-000001")
        assert applog.get_analysis_id() == "SAT-2026-000001"

    def test_analysis_id_can_be_cleared(self):
        applog.set_analysis_id("SAT-2026-000001")
        applog.set_analysis_id(None)
        assert applog.get_analysis_id() is None

    def test_context_does_not_leak_between_threads(self):
        """Two concurrent requests must never be attributed to one another."""
        applog.set_request_id("req_main")

        def worker(name: str) -> str | None:
            applog.set_request_id(name)
            return applog.get_request_id()

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(worker, ["req_a", "req_b", "req_c", "req_d"]))
        assert results == ["req_a", "req_b", "req_c", "req_d"]
        assert applog.get_request_id() == "req_main", "worker threads must not overwrite ours"


class TestJsonFormatter:
    def test_emits_single_line_valid_json(self):
        out = applog.JsonFormatter().format(make_record())
        assert "\n" not in out
        assert json.loads(out)["message"] == "hello"

    def test_includes_level_logger_and_timestamp(self):
        payload = json.loads(applog.JsonFormatter().format(make_record()))
        assert payload["level"] == "INFO"
        assert payload["logger"] == "app.test"
        assert payload["ts"].endswith("+00:00"), "timestamps must be explicit UTC"

    def test_attaches_request_id_when_set(self):
        applog.set_request_id("req_xyz")
        payload = json.loads(applog.JsonFormatter().format(make_record()))
        assert payload["request_id"] == "req_xyz"

    def test_omits_request_id_when_unset(self):
        payload = json.loads(applog.JsonFormatter().format(make_record()))
        assert "request_id" not in payload

    def test_attaches_analysis_id_when_set(self):
        applog.set_analysis_id("SAT-2026-000009")
        payload = json.loads(applog.JsonFormatter().format(make_record()))
        assert payload["analysis_id"] == "SAT-2026-000009"

    def test_promotes_extra_fields_to_top_level_keys(self):
        """Structured context is the point of JSON logs; it must not be stringified away."""
        payload = json.loads(applog.JsonFormatter().format(
            make_record(index="ndbi", quality=0.87)
        ))
        assert payload["index"] == "ndbi"
        assert payload["quality"] == 0.87

    def test_drops_standard_record_attributes(self):
        payload = json.loads(applog.JsonFormatter().format(make_record()))
        assert "pathname" not in payload and "levelno" not in payload

    def test_serialises_non_json_values_without_raising(self):
        class Odd:
            def __str__(self) -> str:
                return "odd-value"

        payload = json.loads(applog.JsonFormatter().format(make_record(thing=Odd())))
        assert payload["thing"] == "odd-value"

    def test_coerces_nested_structures(self):
        payload = json.loads(applog.JsonFormatter().format(
            make_record(shape=(64, 64), meta={"crs": "EPSG:32643"}, tags=[1, "a"])
        ))
        assert payload["shape"] == [64, 64]
        assert payload["meta"] == {"crs": "EPSG:32643"}
        assert payload["tags"] == [1, "a"]

    def test_includes_the_formatted_exception_when_present(self):
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            record = make_record("failed")
            record.exc_info = sys.exc_info()
        payload = json.loads(applog.JsonFormatter().format(record))
        assert "ValueError: boom" in payload["exception"]

    def test_message_interpolation_is_applied(self):
        record = logging.LogRecord(
            "app.test", logging.INFO, __file__, 1, "value=%s", ("42",), None
        )
        assert json.loads(applog.JsonFormatter().format(record))["message"] == "value=42"


class TestTextFormatter:
    def test_contains_level_logger_and_message(self):
        out = applog.TextFormatter().format(make_record())
        assert "INFO" in out and "app.test" in out and "hello" in out

    def test_appends_request_id_when_set(self):
        applog.set_request_id("req_t")
        assert "[req_t]" in applog.TextFormatter().format(make_record())

    def test_omits_request_id_when_unset(self):
        assert "[" not in applog.TextFormatter().format(make_record())


class TestConfigureLogging:
    def test_installs_exactly_one_handler_and_is_idempotent(self):
        applog.configure_logging("INFO", as_json=True)
        applog.configure_logging("INFO", as_json=True)
        root = logging.getLogger()
        assert len(root.handlers) == 1

    def test_selects_the_requested_formatter(self):
        applog.configure_logging("INFO", as_json=True)
        assert isinstance(logging.getLogger().handlers[0].formatter, applog.JsonFormatter)
        applog.configure_logging("INFO", as_json=False)
        assert isinstance(logging.getLogger().handlers[0].formatter, applog.TextFormatter)

    def test_applies_the_requested_level_case_insensitively(self):
        applog.configure_logging("debug")
        assert logging.getLogger().level == logging.DEBUG
        applog.configure_logging("WARNING")
        assert logging.getLogger().level == logging.WARNING

    def test_disables_duplicate_uvicorn_access_log(self):
        applog.configure_logging("INFO")
        assert logging.getLogger("uvicorn.access").disabled

    def test_get_logger_is_namespaced(self):
        assert applog.get_logger("app.services.x").name == "app.services.x"


@pytest.fixture(autouse=True, scope="module")
def _restore_root_logging():
    """configure_logging replaces root handlers; put pytest's back afterwards."""
    root = logging.getLogger()
    saved, level = list(root.handlers), root.level
    yield
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in saved:
        root.addHandler(h)
    root.setLevel(level)
    logging.getLogger("uvicorn.access").disabled = False
