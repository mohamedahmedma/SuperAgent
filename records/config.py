"""Environment configuration, read in exactly one place.

The rule, the same one `sis/config.py` and `identity/config.py` state: **nothing under
`domain/`, `ports/` or `application/` may import this file.** A use case that reads
`os.getenv` cannot be unit-tested without arranging the environment, and its behaviour
then depends on which test ran first. What a use case needs — a grading policy, a
timeout, a pool size — is handed to it by `api/deps.py` or by the composition root, which
are the only layers allowed to know an environment exists.

## Read lazily, cached, and re-readable

`settings()` resolves on first use and caches. `reset_settings()` drops the cache, which
is what lets a test repoint the service after import — the failure mode
`records/identity.py` had to work around by rebuilding its config object per call.

Two settings are deliberately **not** cached and are read per request instead:
`RECORDS_API_KEY`, so rotating it is a restart rather than a rebuild, and
`IDENTITY_PUBLIC_KEY_PEM`, which a test may set after import. Both are noted where they
are read.

## Junk falls back to the documented default

A typo'd `SIS_TIMEOUT_SECONDS=ten` must not take the facade down for a school; it should
run with the documented default and say so in the log.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

_LOADED = False


def load_env() -> None:
    """Load the project's `.env`, once, before anything reads the environment.

    `records/` is deployed as its own process, so nothing else has loaded the project's
    settings for it. Without this every `os.getenv` in the service sees only what the
    shell happened to export — which is how a service comes up configured for a deployment
    nobody described, reports itself healthy, and answers wrongly.

    `override=False`, the default, so a variable already exported wins over the file. That
    is what keeps a shell or the Windows launcher, a container injecting real secrets,
    and a test that sets its own value all behaving exactly as before.
    """
    global _LOADED
    if _LOADED:
        return
    load_dotenv(PROJECT_ROOT / ".env")
    _LOADED = True


def env_value(name: str) -> str:
    """The variable, stripped, or `""` when absent or blank.

    One rule, matching `backend/env.py`: a variable set to an empty string counts as
    unset. `.env` files routinely carry `FOO=` for something someone meant to disable.
    """
    return (os.getenv(name) or "").strip()


def int_env(name: str, default: int) -> int:
    raw = env_value(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s is not a number (%r); using %d.", name, raw, default)
        return default
    if value <= 0:
        logger.warning("%s must be positive (got %d); using %d.", name, value, default)
        return default
    return value


def float_env(name: str, default: float) -> float:
    raw = env_value(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("%s is not a number (%r); using %s.", name, raw, default)
        return default
    return value if value > 0 else default


#: How many connections each client to the SIS may hold open.
#:
#: One number for all three — the marks adapter, the guardian directory and the calendar —
#: because they are three clients to **one** service, called from **one** worker pool, and
#: sizing them separately means two of them are wrong.
#:
#: It must not be smaller than that worker pool. FastAPI serves sync endpoints from an
#: anyio threadpool of 40 by default, so 40 requests can be in flight while only
#: `max_connections` may hold a connection; the rest block inside `httpx`, already counted
#: as in-flight, queueing on this side of the wire where the SIS cannot see it.
#:
#: Measured against a stub answering in 20ms, with 40 calling threads:
#:
#:     max_connections=10   231 req/s   p50  86ms   p95 453ms
#:     max_connections=40   715 req/s   p50  47ms   p95  84ms
#:
#: Raise it alongside the worker count if this service is given more threads; lower it
#: only to protect a SIS that genuinely cannot take the concurrency, knowing the queue
#: moves here rather than disappearing.
_DEFAULT_POOL_SIZE: Final[int] = 40

#: A hung call must not hang a chat turn. Failing at 10s with "I can't reach the records"
#: beats succeeding at 90s to a parent watching a streamed answer.
_DEFAULT_SIS_TIMEOUT: Final[float] = 10.0

#: The directory and the calendar answer smaller questions than the marks call, so they
#: get a shorter budget: a slow one of these should fail fast rather than spend the whole
#: request's patience before the marks call has even started.
_DEFAULT_LOOKUP_TIMEOUT: Final[float] = 5.0

#: A term's dates change when a registrar edits the school year. Ten minutes of staleness
#: costs nothing; asking on every parent question costs two SIS calls each.
_DEFAULT_CALENDAR_CACHE_SECONDS: Final[float] = 600.0

_VALID_PRIMARY: Final[tuple[str, ...]] = ("academic", "official")


@dataclass(frozen=True, slots=True)
class Settings:
    """Resolved configuration. Frozen so nothing mutates it mid-request."""

    lms_backend: str
    sis_base_url: str
    sis_api_key: str
    sis_timeout_seconds: float
    lookup_timeout_seconds: float
    pool_size: int
    calendar_cache_seconds: float
    identity_issuer: str
    identity_audience: str
    identity_jwks_url: str
    identity_jwks_ttl_seconds: int
    primary_figure: str

    @property
    def uses_sis(self) -> bool:
        return self.lms_backend == "sis"


def primary_figure() -> str:
    """Which of the two percentages leads, read loudly rather than silently.

    A school that grades attendance produces two legitimate answers to "how is she doing
    in maths" — the official course total, and the assessments alone. This says which one
    a parent is shown first.

    A typo would otherwise change that number quietly, which is the worst possible way for
    a configuration mistake to present, so an unrecognised value warns and falls back.

    The variable is `RECORDS_PRIMARY_GRADE` and stays that: it is set in deployments, and
    renaming it here would silently move every school back to the default.
    """
    raw = (env_value("RECORDS_PRIMARY_GRADE") or _VALID_PRIMARY[0]).strip().lower()
    if raw in _VALID_PRIMARY:
        return raw
    logger.warning(
        "RECORDS_PRIMARY_GRADE=%r is not one of %s — falling back to %r",
        raw, _VALID_PRIMARY, _VALID_PRIMARY[0],
    )
    return _VALID_PRIMARY[0]


@lru_cache(maxsize=1)
def settings() -> Settings:
    """The process's configuration, resolved on first use and cached."""
    load_env()
    return Settings(
        lms_backend=(env_value("RECORDS_LMS") or "fake").lower(),
        sis_base_url=env_value("SIS_BASE_URL"),
        sis_api_key=env_value("SIS_API_KEY"),
        sis_timeout_seconds=float_env("SIS_TIMEOUT_SECONDS", _DEFAULT_SIS_TIMEOUT),
        lookup_timeout_seconds=float_env(
            "SIS_LOOKUP_TIMEOUT_SECONDS", _DEFAULT_LOOKUP_TIMEOUT
        ),
        pool_size=int_env("RECORDS_POOL_SIZE", _DEFAULT_POOL_SIZE),
        calendar_cache_seconds=float_env(
            "RECORDS_CALENDAR_CACHE_SECONDS", _DEFAULT_CALENDAR_CACHE_SECONDS
        ),
        identity_issuer=env_value("IDENTITY_ISSUER") or "school-identity",
        identity_audience=env_value("IDENTITY_AUDIENCE") or "school-services",
        identity_jwks_url=env_value("IDENTITY_JWKS_URL"),
        identity_jwks_ttl_seconds=int_env("IDENTITY_JWKS_TTL_SECONDS", 600),
        primary_figure=primary_figure(),
    )


def api_key() -> str:
    """The secret this service admits, or `""` when none is configured.

    Read per request rather than captured, so rotating it is a restart of this process and
    not a rebuild — and so a test can set it without reloading the module. Deliberately
    outside `Settings` for that reason.
    """
    return env_value("RECORDS_API_KEY")


def outbound_pool_size() -> int:
    """Kept as a function so the adapters can ask without importing `Settings`."""
    return settings().pool_size


def reset_settings() -> None:
    """Drop the cached settings so a test can repoint the service."""
    settings.cache_clear()


__all__ = [
    "PROJECT_ROOT",
    "Settings",
    "api_key",
    "env_value",
    "float_env",
    "int_env",
    "load_env",
    "outbound_pool_size",
    "primary_figure",
    "reset_settings",
    "settings",
]
