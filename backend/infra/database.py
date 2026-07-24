import os
from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@localhost:5432/langchain_app",
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
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
