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
    """Load `.env`, then collapse the selected LLM provider block onto the generic names.

    The provider step has to happen here and not at any call site: the modules that read
    `MODEL` / `BASE_URL` / `ARK_API_KEY` capture them at import, several at module scope,
    so the only moment the selection can still take effect is between `load_dotenv` and
    the first of those imports. An unset `LLM_PROVIDER` makes it a no-op — see
    backend/llm_provider.py for what it resolves and in what order.
    """
    global _LOADED
    if _LOADED:
        return
    # `.env.local` first, and it WINS, because `load_dotenv` does not override a name
    # that is already set.
    #
    # The split exists because one file could not be both things. `.env` is the
    # PRODUCTION file: it is uploaded to the server as it stands, so it has to carry the
    # deployment's addresses and the runtime secrets compose declares required. A
    # developer then edited that same file to point at localhost, and the two uses fought
    # — a local run reached the deployed estate, and an upload of the local copy
    # overwrote production's secrets with values that were never in it.
    #
    # So `.env` is now written for the server and left alone, and everything a laptop
    # needs differently goes in `.env.local`, which is gitignored and never uploaded.
    # Same precedence Vite already applies to the same two filenames, so the Python
    # services and the Vue app agree about which file wins.
    load_dotenv(PROJECT_ROOT / ".env.local")
    load_dotenv(PROJECT_ROOT / ".env")

    # Imported here rather than at module scope so that `backend.env` keeps importing
    # nothing from the rest of the backend, which is what lets every other module import
    # it without thinking about cycles.
    from backend.llm_provider import apply_provider_env

    apply_provider_env()
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


#: Where the records facade lives when nothing says otherwise. The port `records/`
#: documents for itself.
RECORDS_BASE_URL_DEFAULT = "http://localhost:8100"


def records_base_url() -> str:
    """The records facade's origin, with no trailing slash.

    Read here rather than in the two modules that need it. `backend/tools/records.py` and
    `backend/chat/child_roster.py` each used to call `os.getenv` with their own copy of the
    default, and the second one carried a comment explaining why: `tools.records` imports
    from `backend.chat`, so importing back the other way is a cycle. The explanation was
    correct and the conclusion was not — `backend.env` imports nothing from either, so it
    can hold the value both need.

    Two copies of a default is a slow failure. Change one and the marks arrive from the
    configured facade while the child roster is fetched from wherever the other copy
    points, and a parent is offered a list of children that does not match the records
    behind it.

    Blank counts as unset, per this module's rule. That is a change of one kind: the old
    `os.getenv(name, default)` returned `""` for `RECORDS_BASE_URL=`, which made every
    request go to a bare path and fail. Falling back to the default cannot be worse.
    """
    return (env_value("RECORDS_BASE_URL") or RECORDS_BASE_URL_DEFAULT).rstrip("/")


def records_api_key() -> str:
    """The secret the records facade admits, or `""` when none is configured.

    Empty is meaningful and is preserved: `records/` FAILS CLOSED on it, refusing every
    request with a 503 rather than admitting everyone. Substituting anything here would
    turn a loud misconfiguration into a quiet one.
    """
    return env_value("RECORDS_API_KEY") or ""
