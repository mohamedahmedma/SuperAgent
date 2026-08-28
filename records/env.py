"""Load the project's `.env`, once, before anything reads the environment.

`records/` is deployed as its own process, so nothing else has loaded the project's
settings for it. Without this every `os.getenv` in the service sees only what the shell
happened to export — which is how a service comes up configured for a deployment nobody
described, reports itself healthy, and answers wrongly.

`override=False`, the default, so a variable already exported wins over the file. That is
what keeps `run_all.bat`, a container injecting real secrets, and a test that sets its own
value all behaving exactly as before.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOADED = False


def load_env() -> None:
    """Idempotent. Safe to call from every entry point, and cheap after the first."""
    global _LOADED
    if _LOADED:
        return
    load_dotenv(PROJECT_ROOT / ".env")
    _LOADED = True


def int_env(name: str, default: int) -> int:
    """The variable as a positive int, or the documented default on anything unusable.

    A typo'd `SIS_POOL_SIZE=forty` must not take the facade down; it should run with the
    documented default. The rule `sis.config._int_env` follows, for the same reason.
    """
    raw = env_value(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def outbound_pool_size() -> int:
    """How many connections each client to the SIS may hold open.

    One number for all three — the marks adapter, the guardian directory and the calendar
    — because they are three clients to **one** service, called from **one** worker pool,
    and sizing them separately means two of them are wrong.

    It must not be smaller than that worker pool. FastAPI serves sync endpoints from an
    anyio threadpool of 40 by default, so 40 requests can be in flight while only
    `max_connections` may hold a connection; the rest block inside `httpx`, already
    counted as in-flight, queueing on this side of the wire where the SIS cannot see it.

    Measured against a stub answering in 20ms, with 40 calling threads:

        max_connections=10   231 req/s   p50  86ms   p95 453ms
        max_connections=40   715 req/s   p50  47ms   p95  84ms

    Raise `RECORDS_POOL_SIZE` alongside the worker count if this service is given more
    threads; lower it only to protect a SIS that genuinely cannot take the concurrency,
    knowing the queue moves here rather than disappearing.
    """
    return int_env("RECORDS_POOL_SIZE", 40)


def env_value(name: str) -> str:
    """The variable, stripped, or `""` when absent or blank.

    One rule, matching `backend/env.py`: a variable set to an empty string counts as
    unset. `.env` files routinely carry `FOO=` for something someone meant to disable.
    """
    return (os.getenv(name) or "").strip()


__all__ = ["env_value", "int_env", "load_env", "outbound_pool_size", "PROJECT_ROOT"]
