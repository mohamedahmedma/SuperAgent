"""Load the project's `.env`, once, before anything reads the environment.

`sis/` is deployed as its own process, so nothing else has loaded the project's
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


def env_value(name: str) -> str:
    """The variable, stripped, or `""` when absent or blank.

    One rule, matching `backend/env.py`: a variable set to an empty string counts as
    unset. `.env` files routinely carry `FOO=` for something someone meant to disable.
    """
    return (os.getenv(name) or "").strip()


__all__ = ["env_value", "load_env", "PROJECT_ROOT"]
