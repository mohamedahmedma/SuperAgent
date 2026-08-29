"""Environment configuration, read in exactly one place.

The rule this module exists to protect: **no service in `application/services/` may
import this file.** A use case that reads `os.getenv` cannot be unit-tested without
arranging the environment, and its behaviour changes depending on which test ran
first. Values that a service needs — the preview TTL, the upload ceiling — are passed
into its constructor by the API layer, which is the only layer allowed to know that an
environment exists.

Settings are read lazily and cached rather than captured at import time. Alembic's
`env.py`, pytest fixtures and `uvicorn --reload` all set variables *after* the package
is first imported, and an import-time snapshot silently ignores them.
"""
import os
from dataclasses import dataclass
from functools import lru_cache

# A roster for a whole school is tens of kilobytes. The ceiling exists so a mistaken
# upload of a 400 MB export is refused at the door rather than parsed into memory.
_DEFAULT_MAX_UPLOAD_BYTES = 5 * 1024 * 1024

# Long enough for a registrar to read every rejected row and decide, short enough that
# a preview cannot be committed against a roster that has since been superseded.
_DEFAULT_PREVIEW_TTL_MINUTES = 30

# Egypt. A default rather than a required setting because a school deploying this has one
# country, and making every install state it would only mean most of them state it wrong
# once. A number typed with its own `+` prefix is unaffected by this at any value.
_DEFAULT_COUNTRY_CODE = "+20"


@dataclass(frozen=True, slots=True)
class Settings:
    """Resolved configuration. Frozen so nothing mutates it mid-request."""

    database_url: str
    max_upload_bytes: int
    import_preview_ttl_minutes: int
    #: Prepended to guardian phone numbers a registrar types in national form. Needed
    #: because a spreadsheet says `01001234567` and never says which country that is —
    #: and Excel, which formats a phone column as a number, has usually eaten the leading
    #: zero before the file arrives.
    default_country_code: str
    db_pool_size: int
    db_max_overflow: int
    db_pool_recycle_seconds: int
    db_pool_timeout_seconds: int

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")


def _int_env(name: str, default: int) -> int:
    """Fall back to the default on junk rather than crashing at startup.

    A typo'd `SIS_MAX_UPLOAD_BYTES=5mb` should not take the service down; it should
    run with the documented default. The failure this avoids is a whole school losing
    its SIS because someone edited an env file at 7am.
    """
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _country_code_env(name: str, default: str) -> str:
    """Read a dialling code and normalise it to `+NN`.

    A bare `20` in an env file means the same thing a human means by `+20`, and refusing
    it — or worse, using it unprefixed — would turn every guardian phone in the school
    into a number that reaches nobody. Junk falls back to the default for the reason
    `_int_env` does: a typo at 7am must not take the service down.
    """
    raw = (os.getenv(name) or "").strip()
    digits = "".join(character for character in raw if character.isdigit())
    return f"+{digits}" if digits else default


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """The process's settings. Cached; call `reset_settings_cache()` in tests."""
    return Settings(
        # SQLite by default so the service runs with no infrastructure at all. Point
        # it at Postgres for anything holding real student records.
        database_url=os.getenv("SIS_DATABASE_URL") or "sqlite:///./sis.db",
        max_upload_bytes=_int_env("SIS_MAX_UPLOAD_BYTES", _DEFAULT_MAX_UPLOAD_BYTES),
        import_preview_ttl_minutes=_int_env(
            "SIS_IMPORT_PREVIEW_TTL_MINUTES", _DEFAULT_PREVIEW_TTL_MINUTES
        ),
        default_country_code=_country_code_env(
            "SIS_DEFAULT_COUNTRY_CODE", _DEFAULT_COUNTRY_CODE
        ),
        db_pool_size=_int_env("SIS_DB_POOL_SIZE", 10),
        db_max_overflow=_int_env("SIS_DB_MAX_OVERFLOW", 10),
        db_pool_recycle_seconds=_int_env("SIS_DB_POOL_RECYCLE_SECONDS", 1800),
        db_pool_timeout_seconds=_int_env("SIS_DB_POOL_TIMEOUT_SECONDS", 30),
    )


def reset_settings_cache() -> None:
    """Drop the cached settings so a test can point the service at a temp database."""
    get_settings.cache_clear()
