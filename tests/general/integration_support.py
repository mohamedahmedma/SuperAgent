"""Shared plumbing for tests that run against the real thing.

Everything in tests/test_integration_*.py and tests/test_e2e_*.py talks to a live
Postgres, Redis, Milvus or LLM rather than a mock. That buys the one property unit
tests cannot have — it proves the wiring, the drivers, the SQL types and the network
paths actually work — at the cost of two obligations this module exists to discharge:

**Skip, never fail, when a service is absent.** A developer without Docker running
should still be able to run the suite. An unavailable dependency is a skipped test with
a reason, not a red build.

**Never touch real data.** These tests share a database and a vector store with the
running application. Everything written here is namespaced under a per-run marker and
removed afterwards, and nothing deletes by any broader predicate. Read-only assertions
against the real corpus are fine and are marked as such.
"""
from __future__ import annotations

import os
import socket
import time
import unittest
import uuid
from contextlib import contextmanager
from functools import lru_cache
from typing import Optional

from backend.env import load_env

load_env()

# Everything this suite creates carries this, so cleanup can be exact and a stray row
# is always identifiable as test residue.
RUN_ID = uuid.uuid4().hex[:8]
TEST_PREFIX = f"itest-{RUN_ID}"


def _reachable(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except Exception:
        return False


def _split_hostport(url: str, default_port: int) -> tuple:
    """Host and port out of a URL, without pulling in a URL parser for four cases."""
    body = url.split("://", 1)[-1]
    if "@" in body:
        body = body.rsplit("@", 1)[-1]
    body = body.split("/", 1)[0]
    if ":" in body:
        host, _, port = body.rpartition(":")
        try:
            return host, int(port)
        except ValueError:
            return body, default_port
    return body, default_port


@lru_cache(maxsize=1)
def postgres_available() -> bool:
    host, port = _split_hostport(
        os.getenv("DATABASE_URL", "postgresql://postgres@127.0.0.1:5432/x"), 5432
    )
    if not _reachable(host, port):
        return False
    try:
        from sqlalchemy import text

        from backend.infra.database import engine

        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def redis_available() -> bool:
    host, port = _split_hostport(os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0"), 6379)
    if not _reachable(host, port):
        return False
    try:
        from backend.infra.cache import cache

        cache.set_json(f"{TEST_PREFIX}:probe", {"ok": True}, ttl=5)
        return cache.get_json(f"{TEST_PREFIX}:probe") == {"ok": True}
    except Exception:
        return False


@lru_cache(maxsize=1)
def milvus_available() -> bool:
    host = os.getenv("MILVUS_HOST", "127.0.0.1")
    port = int(os.getenv("MILVUS_PORT", "19530"))
    if not _reachable(host, port):
        return False
    try:
        from backend.indexing.milvus_client import MilvusStore

        return MilvusStore().has_collection()
    except Exception:
        return False


@lru_cache(maxsize=1)
def corpus_indexed() -> int:
    """How many chunks are actually in the collection. 0 disables corpus assertions."""
    if not milvus_available():
        return 0
    try:
        from backend.indexing.milvus_client import MilvusStore

        return len(MilvusStore().query_all(output_fields=["chunk_id"]))
    except Exception:
        return 0


@lru_cache(maxsize=1)
def embedder_available() -> bool:
    try:
        from backend.indexing.embedding import embedding_service

        vector = embedding_service.get_embeddings(["readiness probe"])
        return bool(vector) and len(vector[0]) > 0
    except Exception:
        return False


@lru_cache(maxsize=1)
def llm_available() -> bool:
    """A real call, because a configured key is not the same as a working one.

    Opt-out via RUN_LLM_TESTS=false: these cost tokens and depend on someone else's
    uptime, which is a reasonable thing to decline in CI.
    """
    if (os.getenv("RUN_LLM_TESTS") or "").strip().lower() in ("false", "0", "no", "off"):
        return False
    if not (os.getenv("BASE_URL") and os.getenv("ARK_API_KEY")):
        return False
    try:
        from langchain.chat_models import init_chat_model

        model = init_chat_model(
            model=os.getenv("FAST_MODEL") or os.getenv("MODEL"),
            model_provider="openai",
            api_key=os.getenv("ARK_API_KEY"),
            base_url=os.getenv("BASE_URL"),
            temperature=0.0,
            max_tokens=16,
        )
        model.invoke([{"role": "user", "content": "ping"}])
        return True
    except Exception:
        return False


requires_postgres = unittest.skipUnless(postgres_available(), "no reachable Postgres")
requires_redis = unittest.skipUnless(redis_available(), "no reachable Redis")
requires_milvus = unittest.skipUnless(milvus_available(), "no reachable Milvus collection")
requires_embedder = unittest.skipUnless(embedder_available(), "embedder unavailable")


def requires_corpus(minimum: int = 1):
    return unittest.skipUnless(
        corpus_indexed() >= minimum, f"corpus has fewer than {minimum} indexed chunks"
    )


def requires_llm(func):
    """Deferred: probing the LLM at import time would call it even for a run that
    selects no LLM tests."""
    return unittest.skipUnless(llm_available(), "LLM endpoint unavailable")(func)


@contextmanager
def temporary_profile(name: Optional[str] = None):
    """A profile name nothing else uses, with every row under it removed afterwards.

    Section summaries and corpus digests are keyed by profile, so a synthetic profile
    is a clean namespace inside the real database — no fixture database, no migrations,
    and no possibility of deleting a row this test did not create.
    """
    profile = name or f"{TEST_PREFIX}-{uuid.uuid4().hex[:6]}"
    try:
        yield profile
    finally:
        try:
            from sqlalchemy import text

            from backend.infra.database import engine

            with engine.begin() as connection:
                connection.execute(
                    text("DELETE FROM section_summaries WHERE profile = :p"), {"p": profile}
                )
                connection.execute(
                    text("DELETE FROM corpus_digests WHERE profile = :p"), {"p": profile}
                )
        except Exception:
            pass


@contextmanager
def temporary_user(username: Optional[str] = None, password: str = "Test-passw0rd!"):
    """A real user row, removed afterwards. Yields (username, password)."""
    name = username or f"{TEST_PREFIX}-{uuid.uuid4().hex[:6]}"
    try:
        yield name, password
    finally:
        try:
            from sqlalchemy import text

            from backend.infra.database import engine

            with engine.begin() as connection:
                connection.execute(
                    text("DELETE FROM chat_messages WHERE session_id LIKE :p"),
                    {"p": f"{TEST_PREFIX}%"},
                )
                connection.execute(
                    text("DELETE FROM chat_sessions WHERE user_id IN "
                         "(SELECT id FROM users WHERE username = :u)"),
                    {"u": name},
                )
                connection.execute(text("DELETE FROM users WHERE username = :u"), {"u": name})
        except Exception:
            pass


def wait_until(predicate, timeout: float = 10.0, interval: float = 0.05) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if predicate():
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False


def infrastructure_report() -> str:
    return (
        f"postgres={postgres_available()} redis={redis_available()} "
        f"milvus={milvus_available()} chunks={corpus_indexed()} "
        f"embedder={embedder_available()}"
    )
