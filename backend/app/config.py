"""Application settings, loaded from environment with safe local defaults.

Secrets only ever arrive via environment variables (brief §32); nothing sensitive is
committed. Every default here is chosen so a fresh clone runs offline with no configuration.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root = backend/app/config.py -> app -> backend -> <root>
REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration."""

    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="SATQUERY_",
    )

    # --- application ---
    app_name: str = "SatQuery AI"
    version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"
    log_json: bool = True

    # --- database -------------------------------------------------------------
    # PostgreSQL is the target architecture; SQLite is the supported offline
    # fallback so the prototype runs with zero infrastructure (brief §15).
    database_url: str = Field(default="")

    # --- storage --------------------------------------------------------------
    storage_root: Path = Field(default=REPO_ROOT / "storage")
    demo_data_root: Path = Field(default=REPO_ROOT / "data" / "demo")
    reports_root: Path = Field(default=REPO_ROOT / "reports")

    # --- upload limits (security, brief §32) ---------------------------------
    max_upload_bytes: int = 200 * 1024 * 1024  # 200 MB
    max_pixels: int = 200_000_000  # guards decompression bombs
    allowed_extensions: tuple[str, ...] = (
        ".tif",
        ".tiff",
        ".png",
        ".jpg",
        ".jpeg",
    )

    # --- processing -----------------------------------------------------------
    # Analysis is downsampled to this longest edge. Full-resolution scenes are
    # never loaded into the browser, and huge rasters are decimated on read so a
    # 10000x10000 upload cannot exhaust server memory (brief §31, §23).
    max_analysis_edge: int = 1024
    preview_edge: int = 1024
    min_pair_overlap_fraction: float = 0.30
    request_timeout_seconds: int = 120

    # --- query limits ---------------------------------------------------------
    max_query_length: int = 1000

    # --- optional learned models ---------------------------------------------
    # On by default, because a real adapted checkpoint now exists: ml/checkpoints/
    # satquery-rs-visual-v1, produced by `python -m ml.adaptation.train` on EuroSAT (RGB). The flag
    # only *permits* a checkpoint to be used — it never fabricates one. Availability is gated on all
    # three of this flag, an actual weights file on disk and an importable torch, so a clone without
    # the checkpoint (or without torch) still runs every deterministic analyser unchanged and simply
    # reports the learned rung as UNAVAILABLE. Set SATQUERY_ENABLE_LEARNED_MODELS=false to force the
    # deterministic path even when the checkpoint is present.
    enable_learned_models: bool = True
    checkpoint_dir: Path = Field(default=REPO_ROOT / "ml" / "checkpoints")

    # --- api ------------------------------------------------------------------
    cors_origins: tuple[str, ...] = (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    )
    rate_limit_per_minute: int = 30

    @field_validator("database_url", mode="after")
    @classmethod
    def _default_sqlite(cls, v: str) -> str:
        """Fall back to a local SQLite file when no database URL is supplied."""
        if v:
            return v
        db_path = REPO_ROOT / "storage" / "satquery.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{db_path.as_posix()}"

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    def ensure_dirs(self) -> None:
        """Create the directories the app writes to. Idempotent."""
        for p in (self.storage_root, self.reports_root, self.checkpoint_dir):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()


def reset_settings_cache() -> None:
    """Clear the settings cache. Used by tests that patch environment variables."""
    get_settings.cache_clear()


__all__ = ["REPO_ROOT", "Settings", "get_settings", "reset_settings_cache"]
