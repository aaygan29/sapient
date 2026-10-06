"""Runtime configuration, sourced entirely from environment variables.

No secrets are hardcoded. On Modal, secrets (HF token, etc.) arrive via the
secret store and show up as env vars; locally they come from a gitignored .env.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    engine_mode: str            # "mock" | "real"
    model_id: str               # HF checkpoint id (real mode)
    model_version: str          # human-readable version string surfaced in responses
    keys_file: str              # path to the hashed API-key store
    rate_limit_per_min: int     # per-client requests/minute
    daily_quota: int            # per-client requests/day (anti-extraction + cost cap)
    result_ttl_seconds: int     # how long a result is retrievable before purge
    max_upload_mb: int          # reject uploads larger than this
    include_full_parcels_default: bool  # never True in prod; per-client opt-in instead
    docs_enabled: bool          # expose Swagger UI

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


def load_settings() -> Settings:
    return Settings(
        engine_mode=os.getenv("SAPIENT_ENGINE", "mock").strip().lower(),
        model_id=os.getenv("SAPIENT_MODEL_ID", "The-Sapient-Company/sapient-1-llama"),
        model_version=os.getenv("SAPIENT_MODEL_VERSION", "sapient-1-llama@mock"),
        keys_file=os.getenv("SAPIENT_KEYS_FILE", "keys.json"),
        rate_limit_per_min=_int(os.getenv("SAPIENT_RATE_LIMIT_PER_MIN"), 12),
        daily_quota=_int(os.getenv("SAPIENT_DAILY_QUOTA"), 200),
        result_ttl_seconds=_int(os.getenv("SAPIENT_RESULT_TTL"), 900),
        max_upload_mb=_int(os.getenv("SAPIENT_MAX_UPLOAD_MB"), 200),
        include_full_parcels_default=_bool(os.getenv("SAPIENT_INCLUDE_FULL_PARCELS"), False),
        docs_enabled=_bool(os.getenv("SAPIENT_DOCS"), False),
    )


settings = load_settings()
