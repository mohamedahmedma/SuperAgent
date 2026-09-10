import logging
import os
import time
from typing import Optional

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import declarative_base, sessionmaker

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@localhost:5432/langchain_app",
)

# Sized to the server's worker threads, not left at SQLAlchemy's default.
#
# The default is pool_size=5 with max_overflow=10, so fifteen connections. Starlette
# runs synchronous endpoints on a forty-thread pool, and every turn touches Postgres to
# load and save the conversation — so past fifteen concurrent turns, threads block in
# the pool waiting for a connection and then raise TimeoutError after thirty seconds.
# That ceiling is invisible until it is hit, and then it looks like the database being
# slow rather than the client refusing to open connections.
#
# Overshooting is not free either: every connection is a backend process on the server,
# and `max_connections` is shared by every replica. Total across the fleet is what has
# to fit — replicas x (POOL_SIZE + MAX_OVERFLOW) under Postgres's limit.
_POOL_OPTIONS = {
    "pool_size": int(os.getenv("DB_POOL_SIZE") or 20),
    "max_overflow": int(os.getenv("DB_MAX_OVERFLOW") or 20),
    # Recycle below the typical idle timeout of a proxy or managed Postgres, so a
    # connection is retired by us rather than discovered dead by a user's request.
    "pool_recycle": int(os.getenv("DB_POOL_RECYCLE_SECONDS") or 1800),
    "pool_timeout": int(os.getenv("DB_POOL_TIMEOUT_SECONDS") or 30),
}

# SQLite is a normal thing to point DATABASE_URL at while developing, and its pools
# (SingletonThreadPool, NullPool) reject these arguments outright rather than ignoring
# them — so sizing a pool it does not have would turn a convenience into an import
# error. The sizing is for the server dialects that actually pool connections.
if DATABASE_URL.startswith("sqlite"):
    _POOL_OPTIONS = {}

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    **_POOL_OPTIONS,
)


import re
import unicodedata

# Regex matching non-printable C0/C1 control characters (regular whitespace \t, \n, \r are kept)
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Regex matching zero-width characters and invisible formatting control characters
_INVISIBLE_CHAR_RE = re.compile(r"[\u200b-\u200d\ufeff\u200f\u202a-\u202e]")


def _clean_nul_chars(val):
    """
    Recursively strip any non-standard characters from a Python data structure that could
    corrupt the underlying relational driver, cause JSON parsing errors, or produce garbled text.
    1. Normalization: automatically normalize to the standard Unicode NFC form.
    2. Strip invisible and non-printable characters: remove zero-width spaces, forced
       direction control characters, non-printable control characters, and PUA private-use blocks.
    3. Collapse and strip surrogates: safely strip any broken, isolated UTF-16 surrogates
       using utf-8 ignore.
    """
    if isinstance(val, str):
        # 1. Normalize to NFC
        val = unicodedata.normalize("NFC", val)
        # 2. Strip zero-width, invisible control, and BOM characters
        val = _INVISIBLE_CHAR_RE.sub("", val)
        # 3. Strip non-printable C0/C1 characters and PUA private-use characters
        val = _CONTROL_CHAR_RE.sub("", val)
        val = re.sub(r"[\ue000-\uf8ff]", "", val)
        # 4. Convert and ensure 100% compliant UTF-8 (equivalent to PG's utf8mb4 standard)
        try:
            return val.encode("utf-8", "ignore").decode("utf-8", "ignore")
        except Exception:
            chars = []
            for char in val:
                if 0xD800 <= ord(char) <= 0xDFFF:
                    continue
                chars.append(char)
            return "".join(chars)
    elif isinstance(val, dict):
        return {k: _clean_nul_chars(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [_clean_nul_chars(v) for v in val]
    elif isinstance(val, tuple):
        return tuple(_clean_nul_chars(v) for v in val)
    return val


@event.listens_for(engine, "before_cursor_execute", retval=True)
def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    """
    Global SQLAlchemy engine listener: intercepts all underlying SQL execution and the
    bound parameters passed in.
    Automatically filters and strips \x00 characters from all parameters before the
    underlying driver executes, cleanly and thoroughly avoiding the error where PostgreSQL
    rejects NUL (0x00) bytes written into VARCHAR/TEXT columns, so the business layer
    doesn't need to hand-write replace() calls everywhere.
    """
    if parameters is not None:
        if isinstance(parameters, dict):
            for k, v in list(parameters.items()):
                parameters[k] = _clean_nul_chars(v)
        elif isinstance(parameters, list):
            for i, v in enumerate(parameters):
                parameters[i] = _clean_nul_chars(v)
        elif isinstance(parameters, tuple):
            parameters = tuple(_clean_nul_chars(v) for v in parameters)
    return statement, parameters


SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
Base = declarative_base()


def init_db() -> None:
    # Delayed import to avoid circular dependency.
    import backend.db.models  # noqa: F401

    Base.metadata.create_all(bind=engine)


# Postgres SQLSTATEs for a credential the server rejected outright: 28P01 is
# invalid_password, 28000 the wider invalid_authorization_specification (which also
# covers a role that does not exist). Neither is transient, so neither is retried —
# backing off from a wrong password only delays the diagnosis by the length of the wait.
_AUTH_SQLSTATES = frozenset({"28P01", "28000"})


def _is_auth_failure(exc: Exception) -> bool:
    code = getattr(getattr(exc, "orig", None), "pgcode", None)
    if code:
        return code in _AUTH_SQLSTATES
    # psycopg2 leaves pgcode unset when the failure happened during the connection
    # handshake rather than in reply to a statement — which is exactly this case — so
    # the message is the only signal left.
    return "authentication failed" in str(exc).lower()


def describe_database() -> str:
    """The connection target with the password removed, for logs and error messages."""
    try:
        url = make_url(DATABASE_URL)
    except Exception:
        return "unparseable DATABASE_URL"
    if url.get_backend_name() == "sqlite":
        return f"sqlite file={url.database or ':memory:'}"
    return (
        f"{url.drivername} host={url.host or '-'}:{url.port or 5432} "
        f"db={url.database or '-'} user={url.username or '-'} "
        f"password={'set' if url.password else 'ABSENT'}"
    )


def log_database_status() -> None:
    """One line at boot saying which database this process is about to use.

    The sibling of `log_provider_status()`, for the same reason: the failure worth
    catching at boot is a deployment that believes it is pointed somewhere it is not.
    Without it, "which database is this, with whose credentials" is answered by reading
    `.env` and reasoning about compose interpolation — which is the step that gets got
    wrong in the first place.
    """
    source = "DATABASE_URL" if os.getenv("DATABASE_URL") else "the built-in default"
    logger.info("Database: %s (from %s)", describe_database(), source)


def verify_connectivity(attempts: int = 5, delay_seconds: float = 2.0) -> None:
    """Open one connection before the app reports itself started, and fail legibly.

    `init_db()` opens one immediately afterwards, so this costs nothing and buys a
    diagnosis. A rejected password used to arrive as a hundred and fifty lines of
    SQLAlchemy pool internals whose single informative line sat below the default
    `--tail`, and it arrived in the same shape whether the cause was a credential, an
    unresolvable host or a stopped container. Separating the permanent failure from the
    transient one is the whole point: a wrong password is reported at once and named,
    and a postgres still coming up is waited for.
    """
    last: Optional[Exception] = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return
        except OperationalError as exc:
            last = exc
            if _is_auth_failure(exc):
                logger.error(
                    "DATABASE CREDENTIALS REJECTED — %s. POSTGRES_PASSWORD applies only "
                    "while postgres initialises an EMPTY data directory, so on an estate "
                    "whose volume already exists a rotated .env changes what this service "
                    "SENDS and never what the role ACCEPTS — while pg_isready reports the "
                    "container healthy throughout. Reconcile the role with .env: "
                    "bash deploy/scripts/apply-env.sh",
                    describe_database(),
                )
                # `from None` deliberately: the psycopg2/SQLAlchemy chain is what buried
                # the diagnosis, and the line above has already said everything it said.
                raise RuntimeError("database credentials rejected") from None
            if attempt < attempts:
                logger.warning(
                    "Database not reachable yet (attempt %d/%d): %s",
                    attempt,
                    attempts,
                    getattr(exc, "orig", None) or exc,
                )
                time.sleep(delay_seconds)

    logger.error(
        "DATABASE UNREACHABLE after %d attempts — %s. Last error: %s",
        attempts,
        describe_database(),
        getattr(last, "orig", None) or last,
    )
    raise RuntimeError("database unreachable") from None
