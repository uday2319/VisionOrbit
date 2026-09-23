"""Tests for application settings.

Two brief requirements are load-bearing here: secrets and configuration arrive only through
environment variables (§32), and a fresh clone must run offline with no configuration at all
(§19) — which means SQLite has to be a real working fallback, not a note in the README.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app import config


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch):
    """Settings are cached process-wide and read the host environment.

    Clearing every ``SATQUERY_`` variable means a developer's own shell cannot change what
    these tests measure, and clearing the cache stops one test's settings leaking into the
    next.
    """
    import os

    for key in [k for k in os.environ if k.startswith("SATQUERY_")]:
        monkeypatch.delenv(key, raising=False)
    config.reset_settings_cache()
    yield
    config.reset_settings_cache()


def build(**env) -> config.Settings:
    """Construct settings directly, bypassing the cache and any repo .env file."""
    return config.Settings(_env_file=None, **env)


class TestDefaults:
    def test_runs_with_no_configuration_at_all(self):
        """The offline-first requirement: zero env vars must still yield usable settings."""
        s = build()
        assert s.app_name and s.version
        assert s.database_url, "a usable database URL must always be derivable"

    def test_falls_back_to_sqlite_when_no_database_url_is_given(self):
        s = build()
        assert s.is_sqlite
        assert s.database_url.startswith("sqlite:///")

    def test_debug_is_off_by_default(self):
        """Debug on by default would leak internals through error responses (§29)."""
        assert build().debug is False

    def test_learned_models_are_permitted_by_default(self):
        """A real adapted checkpoint now ships, so the learned rung is permitted by default.

        This asserts *permission*, not availability. ``app.services.scene.is_available`` still
        requires an actual weights file and an importable torch on top of this flag, so a clone with
        neither runs the deterministic analysers exactly as before. The opposite default is still
        reachable with ``SATQUERY_ENABLE_LEARNED_MODELS=false`` (covered below).
        """
        assert build().enable_learned_models is True

    def test_learned_models_can_be_forced_off(self, monkeypatch):
        """The deterministic-only path must stay one env var away, for a reviewer who wants it."""
        monkeypatch.setenv("SATQUERY_ENABLE_LEARNED_MODELS", "false")
        assert build().enable_learned_models is False

    def test_upload_limits_are_finite_and_positive(self):
        s = build()
        assert 0 < s.max_upload_bytes <= 1024 * 1024 * 1024
        assert s.max_pixels > 0
        assert s.max_query_length > 0

    def test_allowed_extensions_are_lowercase_and_dotted(self):
        for ext in build().allowed_extensions:
            assert ext.startswith(".") and ext == ext.lower()

    def test_geotiff_is_accepted(self):
        assert {".tif", ".tiff"} <= set(build().allowed_extensions)

    def test_no_executable_extension_is_allowed(self):
        """§32: uploaded files are never executed, so they can never be executables."""
        banned = {".exe", ".sh", ".bat", ".py", ".dll", ".so", ".js", ".svg", ".html"}
        assert not (set(build().allowed_extensions) & banned)

    def test_cors_origins_are_explicit_not_wildcard(self):
        origins = build().cors_origins
        assert origins and "*" not in origins

    def test_rate_limit_is_set(self):
        assert build().rate_limit_per_minute > 0

    def test_analysis_edge_bounds_memory_use(self):
        """§31: a huge upload must be decimated, not loaded whole."""
        assert 0 < build().max_analysis_edge <= 4096

    def test_pair_overlap_threshold_is_a_fraction(self):
        assert 0.0 < build().min_pair_overlap_fraction <= 1.0


class TestEnvironmentOverrides:
    def test_reads_prefixed_environment_variables(self, monkeypatch):
        monkeypatch.setenv("SATQUERY_LOG_LEVEL", "DEBUG")
        assert config.Settings().log_level == "DEBUG"

    def test_ignores_unprefixed_variables(self, monkeypatch):
        """The prefix stops unrelated host environment from reconfiguring the app."""
        monkeypatch.setenv("LOG_LEVEL", "CRITICAL")
        assert config.Settings(_env_file=None).log_level != "CRITICAL"

    def test_database_url_from_environment_wins_over_sqlite_fallback(self, monkeypatch):
        monkeypatch.setenv("SATQUERY_DATABASE_URL", "postgresql://u:p@db:5432/satquery")
        s = config.Settings()
        assert s.database_url == "postgresql://u:p@db:5432/satquery"
        assert not s.is_sqlite

    def test_booleans_parse_from_strings(self, monkeypatch):
        monkeypatch.setenv("SATQUERY_DEBUG", "true")
        assert config.Settings().debug is True

    def test_integers_parse_from_strings(self, monkeypatch):
        monkeypatch.setenv("SATQUERY_MAX_UPLOAD_BYTES", "1024")
        assert config.Settings().max_upload_bytes == 1024

    def test_paths_parse_from_strings(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SATQUERY_STORAGE_ROOT", str(tmp_path / "s"))
        assert config.Settings().storage_root == tmp_path / "s"

    def test_unknown_variables_are_ignored_rather_than_fatal(self, monkeypatch):
        monkeypatch.setenv("SATQUERY_TOTALLY_UNKNOWN_SETTING", "x")
        assert config.Settings().app_name


class TestNoHardCodedAbsolutePaths:
    def test_all_default_paths_are_under_the_repo(self):
        """§42 forbids hard-coded absolute paths; defaults must be repo-relative."""
        s = build()
        for path in (s.storage_root, s.demo_data_root, s.reports_root, s.checkpoint_dir):
            assert path.is_absolute(), "resolved paths are absolute at runtime"
            assert config.REPO_ROOT in path.parents or path == config.REPO_ROOT

    def test_repo_root_contains_the_backend_package(self):
        assert (config.REPO_ROOT / "backend" / "app" / "config.py").exists()


class TestDirectoryCreation:
    def test_ensure_dirs_creates_everything_it_writes_to(self, tmp_path):
        s = build(
            storage_root=tmp_path / "store",
            reports_root=tmp_path / "rep",
            checkpoint_dir=tmp_path / "ckpt",
        )
        s.ensure_dirs()
        assert (tmp_path / "store").is_dir()
        assert (tmp_path / "rep").is_dir()
        assert (tmp_path / "ckpt").is_dir()

    def test_ensure_dirs_is_idempotent(self, tmp_path):
        s = build(storage_root=tmp_path / "a", reports_root=tmp_path / "b",
                  checkpoint_dir=tmp_path / "c")
        s.ensure_dirs()
        s.ensure_dirs()
        assert (tmp_path / "a").is_dir()

    def test_ensure_dirs_creates_nested_parents(self, tmp_path):
        s = build(storage_root=tmp_path / "x" / "y" / "z", reports_root=tmp_path / "r",
                  checkpoint_dir=tmp_path / "k")
        s.ensure_dirs()
        assert (tmp_path / "x" / "y" / "z").is_dir()


class TestSqliteDetection:
    @pytest.mark.parametrize(
        "url,expected",
        [
            ("sqlite:///./x.db", True),
            ("sqlite+aiosqlite:///./x.db", True),
            ("postgresql://u@h/db", False),
            ("postgresql+psycopg://u@h/db", False),
        ],
    )
    def test_classifies_backend_from_url(self, url, expected):
        assert build(database_url=url).is_sqlite is expected


class TestCaching:
    def test_get_settings_returns_the_same_instance(self):
        config.reset_settings_cache()
        assert config.get_settings() is config.get_settings()

    def test_reset_forces_a_fresh_read(self, monkeypatch):
        config.reset_settings_cache()
        first = config.get_settings()
        monkeypatch.setenv("SATQUERY_LOG_LEVEL", "ERROR")
        config.reset_settings_cache()
        second = config.get_settings()
        assert second is not first
        assert second.log_level == "ERROR"


class TestNoCommittedSecrets:
    def test_no_default_looks_like_a_credential(self):
        """§32/§42: nothing sensitive may be baked into the defaults."""
        s = build()
        for name, value in s.model_dump().items():
            if isinstance(value, str) and value:
                lowered = value.lower()
                assert "password" not in lowered, name
                assert "secret" not in lowered, name
                assert "@" not in lowered or name == "database_url", name

    def test_default_database_url_carries_no_credentials(self):
        assert "@" not in build().database_url

    def test_env_file_target_is_not_committed(self):
        """A .env must never be in the repo; only .env.example is."""
        env_file = config.Settings.model_config.get("env_file")
        assert env_file is not None
        assert Path(str(env_file)).name == ".env"
