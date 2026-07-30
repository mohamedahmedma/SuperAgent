"""Load .env from the project root; called once by the app entry point or a standalone
script before importing other backend modules.

Also home to the typed environment readers. They exist so that one rule holds
everywhere: **a variable set to an empty string counts as unset.** `.env` files
routinely contain `FOO=` for a value someone meant to disable, and the bare
`os.getenv(name, default)` form returns `""` in that case rather than the default —
which then either crashes (`int("")`) or, worse, silently evaluates to the wrong
branch. These readers also keep the modules consistent with
backend/profiles/registry.py, which applies the same blank-is-unset rule when
overlaying env onto a profile.
"""
import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOADED = False

_TRUE_VALUES = ("true", "1", "yes", "on")
_FALSE_VALUES = ("false", "0", "no", "off")


def load_env() -> None:
    global _LOADED
    if _LOADED:
        return
    load_dotenv(PROJECT_ROOT / ".env")
    _LOADED = True


def env_value(name: str) -> Optional[str]:
    """The variable's value, or None when absent or blank."""
    raw = (os.getenv(name) or "").strip()
    return raw or None


def env_int(name: str, default: int, minimum: Optional[int] = None) -> int:
    raw = env_value(name)
    if raw is None:
        return default if minimum is None else max(default, minimum)
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid integer %s=%r — using %r", name, raw, default)
        return default if minimum is None else max(default, minimum)
    return value if minimum is None else max(value, minimum)


def env_float(name: str, default: float, minimum: Optional[float] = None) -> float:
    raw = env_value(name)
    if raw is None:
        return default if minimum is None else max(default, minimum)
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Invalid float %s=%r — using %r", name, raw, default)
        return default if minimum is None else max(default, minimum)
    return value if minimum is None else max(value, minimum)


def env_bool(name: str, default: bool) -> bool:
    raw = env_value(name)
    if raw is None:
        return default
    lowered = raw.lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    logger.warning("Invalid boolean %s=%r — using %r", name, raw, default)
    return default
