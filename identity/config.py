"""Environment configuration, read in exactly one place.

The rule this module exists to protect, stated the same way `sis/config.py` states it:
**nothing under `domain/` or `application/` may import this file.** A use case that reads
`os.getenv` cannot be unit-tested without arranging the environment, and its behaviour
then depends on which test ran first. Values a service needs — the lockout threshold, the
challenge TTL, the token lifetime — are passed into its constructor by `api/deps.py`,
which is the only layer allowed to know that an environment exists.

## Read lazily, never at import

This service already learned this once. `infrastructure/db/session.py` carries a long note
about an engine built at module scope from a URL read at module scope, and the cross-suite
failure that produced. The same bug was still live in three other places when this file
was written::

    auth.py     MAX_FAILED_ATTEMPTS = int(os.getenv(...))   # at import
    tokens.py   ACCESS_TTL_MINUTES  = int(os.getenv(...))   # at import
    keys.py     _DEV_KEY_PATH       = Path(os.getenv(...))  # at import

Each captured whatever the environment held at the moment something first imported the
module. `uvicorn --reload`, a pytest fixture and `load_env()` all set variables *after*
that moment, so the setting that was read was rarely the setting that was configured —
and the symptom was never "wrong config", it was a lockout that triggered at the wrong
count, or a suite writing into another suite's database.

`settings()` is cached rather than re-read per access, so the hot paths pay one dict
lookup. `reset_settings()` drops the cache for a test that repoints the service.

## Junk falls back to the documented default

A typo'd `IDENTITY_ACCESS_TTL_MINUTES=thirty` must not take authentication down for a
school; it should run with the documented default and say so in the log. The failure this
avoids is a whole school losing sign-in because somebody edited an env file at 7am.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Final

from identity.env import env_value, load_env

from schoolauth import DEFAULT_AUDIENCE, DEFAULT_ISSUER

logger = logging.getLogger(__name__)

#: PBKDF2 iterations for new hashes. A password *is* a low-entropy secret, so stretching
#: is exactly right here — unlike the refresh tokens and poll secrets elsewhere in this
#: service, which are high-entropy machine-generated values where it would only add
#: latency to a request a parent is waiting on.
_DEFAULT_PBKDF2_ROUNDS: Final[int] = 310_000

#: Then locked for `lockout_minutes`. Per account rather than per IP: the threat is
#: credential stuffing against a known parent, and an attacker has more IPs than the
#: school has parents.
_DEFAULT_MAX_FAILED_ATTEMPTS: Final[int] = 8
_DEFAULT_LOCKOUT_MINUTES: Final[int] = 15

#: Bounds the revocation window. Access tokens cannot be revoked — verification elsewhere
#: is offline — so keeping them short is the only control over a binding that changed.
_DEFAULT_ACCESS_TTL_MINUTES: Final[int] = 30
_DEFAULT_REFRESH_TTL_DAYS: Final[int] = 30

#: Long enough for a parent to find the WhatsApp message, short enough that a
#: screenshotted link shared later is worthless.
_DEFAULT_VERIFICATION_TTL_MINUTES: Final[int] = 10

#: How long the guardian directory may take before a lookup is abandoned. Two budgets,
#: because the two calls sit in different places.
#:
#: `resolve` is the sign-in itself — the parent cannot proceed without it, so it gets the
#: full budget. `children_of` only decorates a token with a convenience claim that the
#: chat backend looks up for itself anyway, and it runs *inside* the latency of a parent's
#: sign-in, so it gets a tight one. Before this split, a slow SIS added five seconds to
#: every parent's login in exchange for saving the backend one call it makes regardless.
_DEFAULT_DIRECTORY_TIMEOUT_SECONDS: Final[float] = 5.0
_DEFAULT_CHILDREN_TIMEOUT_SECONDS: Final[float] = 1.5

#: See `Settings.whatsapp_graph_version`.
_DEFAULT_GRAPH_VERSION: Final[str] = "v22.0"

_TRUTHY: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})


@dataclass(frozen=True, slots=True)
class Settings:
    """Resolved configuration. Frozen so nothing mutates it mid-request."""

    # -- storage ------------------------------------------------------------
    database_url: str
    db_pool_size: int
    db_max_overflow: int
    db_pool_recycle_seconds: int
    db_pool_timeout_seconds: int

    # -- signing ------------------------------------------------------------
    private_key_pem: str
    private_key_file: str
    dev_key_file: str
    issuer: str
    audience: str
    access_ttl_minutes: int
    refresh_ttl_days: int

    # -- passwords and lockout ----------------------------------------------
    pbkdf2_rounds: int
    max_failed_attempts: int
    lockout_minutes: int
    admin_key: str
    admin_invite_code: str

    # -- WhatsApp -----------------------------------------------------------
    whatsapp_number: str
    whatsapp_phone_number_id: str
    whatsapp_token: str
    whatsapp_verify_token: str
    whatsapp_app_secret: str
    whatsapp_log_codes: bool
    #: Meta's Graph API version. Pinned, never "latest" — Meta ships breaking changes
    #: between versions and an unpinned client starts failing on a date nobody chose.
    #: Overridable because the version a school's app was created against is a fact
    #: about that app: the console shows the one it generated its examples for, and
    #: following it removes one variable from every "why did this stop working"
    #: conversation.
    whatsapp_graph_version: str
    verification_ttl_minutes: int

    # -- the school's system of record --------------------------------------
    sis_base_url: str
    sis_api_key: str
    directory_timeout_seconds: float
    children_timeout_seconds: float

    # -- multi-school -------------------------------------------------------
    school_codes: tuple[str, ...]

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    @property
    def is_multi_school(self) -> bool:
        return bool(self.school_codes)


def _int_env(name: str, default: int) -> int:
    """The variable as an int, or the documented default on anything unusable."""
    raw = env_value(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s is not a number (%r); using the default %d.", name, raw, default)
        return default
    if value <= 0:
        logger.warning("%s must be positive (got %d); using the default %d.", name, value, default)
        return default
    return value


def _float_env(name: str, default: float) -> float:
    raw = env_value(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("%s is not a number (%r); using the default %s.", name, raw, default)
        return default
    return value if value > 0 else default


def _bool_env(name: str) -> bool:
    return env_value(name).lower() in _TRUTHY


def _first_env(*names: str) -> str:
    """The first of `names` that is set, or `""`.

    Exists for one shape of setting: the SIS's address and key, which this service and
    `records/` both need and which used to be spelled twice — `IDENTITY_SIS_BASE_URL` here,
    `SIS_BASE_URL` there. Two names for one service is not redundancy, it is a
    disagreement waiting to happen, and a silent one: fill in the records spelling only and
    identity keeps an empty in-memory directory, so the chat backend answers a parent's
    questions about marks perfectly while sign-in tells her that her number is not
    registered.

    The specific name still wins where it is set, because a deployment may legitimately
    run identity against a different SIS from the records facade — which is the only
    reason the second name was defensible in the first place.
    """
    for name in names:
        value = env_value(name)
        if value:
            return value
    return ""


def _codes_env(name: str) -> tuple[str, ...]:
    """Comma-separated school codes, upper-cased and de-duplicated, in order.

    Normalisation matches `sis.domain.value_objects._Code`, so `ncs`, ` NCS ` and `NCS`
    name one school in both services. A code legal here and refused there would be a
    school that can sign a parent in and then fail every lookup made on their behalf.
    """
    raw = env_value(name)
    if not raw:
        return ()
    seen: list[str] = []
    for part in raw.split(","):
        code = part.strip().upper()
        if code and code not in seen:
            seen.append(code)
    return tuple(seen)


@lru_cache(maxsize=1)
def settings() -> Settings:
    """The process's configuration, resolved on first use and cached.

    `load_env()` first, so a deployment started as `uvicorn identity.app:app` — with
    nothing else in the estate having loaded the project's `.env` for it — reads the same
    values a developer sees. It is idempotent and cheap after the first call.
    """
    load_env()
    return Settings(
        database_url=env_value("IDENTITY_DATABASE_URL") or "sqlite:///./identity.db",
        db_pool_size=_int_env("IDENTITY_DB_POOL_SIZE", 10),
        db_max_overflow=_int_env("IDENTITY_DB_MAX_OVERFLOW", 10),
        db_pool_recycle_seconds=_int_env("IDENTITY_DB_POOL_RECYCLE_SECONDS", 1800),
        db_pool_timeout_seconds=_int_env("IDENTITY_DB_POOL_TIMEOUT_SECONDS", 30),
        private_key_pem=env_value("IDENTITY_PRIVATE_KEY_PEM"),
        private_key_file=env_value("IDENTITY_PRIVATE_KEY_FILE"),
        dev_key_file=env_value("IDENTITY_DEV_KEY_FILE") or "./identity-dev-key.pem",
        # Defaults from `schoolauth`, the package every verifier in the estate compiles
        # in. This service MINTS with them, so a literal here drifting from the
        # verifiers' is the one version of this bug that breaks everything at once.
        issuer=env_value("IDENTITY_ISSUER") or DEFAULT_ISSUER,
        audience=env_value("IDENTITY_AUDIENCE") or DEFAULT_AUDIENCE,
        access_ttl_minutes=_int_env("IDENTITY_ACCESS_TTL_MINUTES", _DEFAULT_ACCESS_TTL_MINUTES),
        refresh_ttl_days=_int_env("IDENTITY_REFRESH_TTL_DAYS", _DEFAULT_REFRESH_TTL_DAYS),
        pbkdf2_rounds=_int_env("IDENTITY_PBKDF2_ROUNDS", _DEFAULT_PBKDF2_ROUNDS),
        max_failed_attempts=_int_env(
            "IDENTITY_MAX_FAILED_ATTEMPTS", _DEFAULT_MAX_FAILED_ATTEMPTS
        ),
        lockout_minutes=_int_env("IDENTITY_LOCKOUT_MINUTES", _DEFAULT_LOCKOUT_MINUTES),
        admin_key=env_value("IDENTITY_ADMIN_KEY"),
        admin_invite_code=env_value("IDENTITY_ADMIN_INVITE_CODE"),
        whatsapp_number=env_value("IDENTITY_WHATSAPP_NUMBER"),
        whatsapp_phone_number_id=env_value("IDENTITY_WHATSAPP_PHONE_NUMBER_ID"),
        whatsapp_token=env_value("IDENTITY_WHATSAPP_TOKEN"),
        whatsapp_verify_token=env_value("IDENTITY_WHATSAPP_VERIFY_TOKEN"),
        whatsapp_app_secret=env_value("IDENTITY_WHATSAPP_APP_SECRET"),
        whatsapp_log_codes=_bool_env("IDENTITY_WHATSAPP_LOG_CODES"),
        whatsapp_graph_version=(
            env_value("IDENTITY_WHATSAPP_GRAPH_VERSION") or _DEFAULT_GRAPH_VERSION
        ),
        verification_ttl_minutes=_int_env(
            "IDENTITY_VERIFICATION_TTL_MINUTES", _DEFAULT_VERIFICATION_TTL_MINUTES
        ),
        sis_base_url=_first_env("IDENTITY_SIS_BASE_URL", "SIS_BASE_URL"),
        sis_api_key=_first_env("IDENTITY_SIS_API_KEY", "SIS_API_KEY"),
        directory_timeout_seconds=_float_env(
            "IDENTITY_SIS_TIMEOUT_SECONDS", _DEFAULT_DIRECTORY_TIMEOUT_SECONDS
        ),
        children_timeout_seconds=_float_env(
            "IDENTITY_SIS_CHILDREN_TIMEOUT_SECONDS", _DEFAULT_CHILDREN_TIMEOUT_SECONDS
        ),
        school_codes=_codes_env("IDENTITY_SCHOOLS"),
    )


def env_suffix(code: str) -> str:
    """The environment-variable suffix a school code folds to.

    `.` and `-` are legal in a school code and illegal in a variable name, so they fold to
    `_` — matching `sis.tenancy._suffix`. Two codes can therefore collide, which
    `infrastructure/whatsapp/registry.py` refuses at startup rather than resolving.
    """
    return code.replace(".", "_").replace("-", "_")


def school_env(prefix: str, code: str) -> str:
    """One school's value for a per-school setting, e.g. `IDENTITY_WHATSAPP_TOKEN_NCS`."""
    return env_value(f"{prefix}_{env_suffix(code)}")


def reset_settings() -> None:
    """Drop the cached settings so a test can repoint the service."""
    settings.cache_clear()


__all__ = ["Settings", "env_suffix", "reset_settings", "school_env", "settings"]
