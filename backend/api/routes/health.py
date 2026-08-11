"""Liveness and readiness, which are different questions.

Running N interchangeable API containers behind a load balancer needs both, and
conflating them is what makes a rolling deploy drop requests.

  /health   Is this process alive? Cheap, no dependencies. A failing answer means
            "restart me". It must NOT check the database or the embedder: a shared
            Postgres blip would otherwise fail every container's liveness probe at
            once and the orchestrator would restart the entire fleet.

  /ready    Should this process receive traffic? A failing answer means "route
            elsewhere", not "restart me". This one does check dependencies, because
            being unable to serve is exactly what it is for.

The distinction matters most at startup here. Loading bge-m3 takes roughly 110 seconds,
during which the process accepts connections and answers /health perfectly well while
being unable to embed anything. Without a readiness gate a load balancer sends real
users into that window and they wait it out — or time out. `/ready` stays false until
the embedder is actually built.
"""
import logging
import os
import time

from fastapi import APIRouter, Response, status

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

_STARTED_AT = time.time()


@router.get("/health")
def health() -> dict:
    """Liveness. Deliberately dependency-free."""
    return {"status": "ok", "uptime_seconds": round(time.time() - _STARTED_AT, 1)}


@router.get("/ready")
def ready(response: Response) -> dict:
    """Readiness: can this process actually serve a turn right now?

    Reports every dependency rather than short-circuiting on the first failure, so a
    probe that fails says which one — the alternative is an operator reading "not
    ready" with no way to tell a cold embedder from a dead database.
    """
    checks = {
        "embedder": _check_embedder(),
        "database": _check_database(),
    }
    ok = all(item["ok"] for item in checks.values())
    if not ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "ready" if ok else "not_ready",
        "uptime_seconds": round(time.time() - _STARTED_AT, 1),
        "checks": checks,
    }


def _check_embedder() -> dict:
    """Built, not merely importable.

    Deliberately does not embed anything. A probe that ran a forward pass would consume
    the very CPU the request path is contending for, on every probe, from every replica.
    """
    try:
        from backend.indexing.embedding import embedding_service

        if embedding_service.is_ready:
            return {"ok": True, "detail": "loaded"}
        return {"ok": False, "detail": "loading"}
    except Exception as exc:  # pragma: no cover - import failure is not a normal state
        return {"ok": False, "detail": f"unavailable: {exc}"}


def _check_database() -> dict:
    try:
        from sqlalchemy import text

        from backend.infra.database import engine

        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return {"ok": True, "detail": "connected"}
    except Exception as exc:
        logger.warning("readiness: database check failed", exc_info=True)
        return {"ok": False, "detail": str(exc)[:200]}
