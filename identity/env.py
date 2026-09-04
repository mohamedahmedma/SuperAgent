"""Load `.env` from the project root, once, before anything reads the environment.

`identity/` is deployed as its own process — `uvicorn identity.app:app` — and until now
nothing in it loaded the project's `.env`. Every `os.getenv` in the service therefore saw
only what the shell happened to export, which in practice meant nothing.

That is not a quiet degradation. With `IDENTITY_WHATSAPP_NUMBER` unset the sign-in link
becomes `https://wa.me/?text=...` — no number in the path — and WhatsApp responds by
opening the **contact picker**. The parent is asked to choose who to send the school's
verification code to, and in production they have no idea which contact that is. The link
works, the app opens, the message is prefilled, and the flow is dead.

`override=False`, the default, so a variable already exported wins over the file. That is
what keeps a test that sets its own value, and a container that injects real secrets,
working exactly as before.
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
    # `.env.local` first, and it WINS: `load_dotenv` does not override a name already
    # set. `.env` is the PRODUCTION file, uploaded to the server as it stands; anything a
    # laptop needs differently goes in `.env.local`, which is gitignored and never
    # uploaded. See backend/env.py for the incident that split them.
    load_dotenv(PROJECT_ROOT / ".env.local")
    load_dotenv(PROJECT_ROOT / ".env")
    _LOADED = True


def env_value(name: str) -> str:
    """The variable, stripped, or `""` when absent or blank.

    One rule, matching `backend/env.py`: a variable set to an empty string counts as
    unset. `.env` files routinely carry `FOO=` for something someone meant to disable,
    and the bare `os.getenv` form returns `""` there rather than falling back.
    """
    return (os.getenv(name) or "").strip()


__all__ = ["env_value", "load_env", "PROJECT_ROOT"]
