"""What LangSmith is about to do, in one line at boot.

The sibling of `log_provider_status()`, `log_database_status()` and
`log_vision_status()`, and it exists for the reason they do: tracing is configured
entirely through the environment — nothing in this codebase turns it on — so a
deployment that believes it is tracing and is not has no symptom at all. The runs
simply never arrive, and from inside the application a switch that is off, a project
nobody is looking at, and a key the endpoint rejects are indistinguishable.

Measured twice on this estate. Tracing came back only when the services were recreated
against a `.env` carrying the intended key, and went away again on the next release,
which recreated them against a different one. The two files' `LANGSMITH_API_KEY` and
`LANGSMITH_PROJECT` differed and nothing anywhere said so. See DEVOPS.md.

The key is never printed. What is printed is the first eight hex characters of its
SHA-256 — exactly the shorthand DEVOPS.md compares `.env` files with — so this line can
be matched against the file a deployment believes it is using without either being read
aloud.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Sequence, Tuple
from urllib.parse import urlsplit

from backend.env import env_value

logger = logging.getLogger(__name__)

#: Each setting under both spellings the SDK accepts, `LANGSMITH_` first because that is
#: the order the SDK reads them in. Naming the variable that actually supplied the value
#: is the point: a deployment carrying both spellings with different values looks fine in
#: `.env` and behaves like only one of them.
_SWITCH = ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2")
_API_KEY = ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY")
_PROJECT = ("LANGSMITH_PROJECT", "LANGCHAIN_PROJECT")
_ENDPOINT = ("LANGSMITH_ENDPOINT", "LANGCHAIN_ENDPOINT")

#: Where the SDK sends runs when no endpoint is configured. Stated here so the line can
#: say where traces are going even then — "the default" is an answer an operator looking
#: for missing runs in a self-hosted or EU workspace needs to be given, not left to infer.
DEFAULT_ENDPOINT = "https://api.smith.langchain.com"

_ENABLED = {"true", "1", "yes", "on", "t", "y"}


def _first_set(names: Sequence[str]) -> Tuple[str, str]:
    """`(variable, value)` for the first of `names` that carries one, else `("", "")`."""
    for name in names:
        value = env_value(name)
        if value:
            return name, value
    return "", ""


def key_fingerprint(secret: str) -> str:
    """The first eight hex of the SHA-256, matching how DEVOPS.md compares `.env` files.

    Deliberately not the last four characters of the key: a fingerprint that reveals part
    of the secret cannot be pasted into a chat transcript, and being pasteable is the
    whole reason this is here.
    """
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()[:8]


def describe_tracing() -> str:
    """One sentence: whether runs are being sent, where, and under which key."""
    name, switch = _first_set(_SWITCH)
    if switch.lower() not in _ENABLED:
        return f"tracing off ({name or _SWITCH[0]}={switch or 'unset'})"

    project = _first_set(_PROJECT)[1] or "default"
    endpoint = _first_set(_ENDPOINT)[1] or DEFAULT_ENDPOINT
    host = urlsplit(endpoint).netloc or endpoint
    key_name, key = _first_set(_API_KEY)
    if not key:
        return (
            f"tracing on for project {project!r} at {host}, but {_API_KEY[0]} is unset "
            "— no run will be ingested"
        )
    return (
        f"tracing on -> project {project!r} at {host} "
        f"({key_name} fp {key_fingerprint(key)})"
    )


def log_tracing_status() -> None:
    """Say it at boot, whichever way it comes out.

    Logged when tracing is OFF as well, because "why are there no traces" is asked far
    more often than "why are there traces", and an absent line answers neither.
    """
    logger.info("LangSmith: %s", describe_tracing())
