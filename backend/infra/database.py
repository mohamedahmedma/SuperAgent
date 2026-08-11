import os
from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker

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
