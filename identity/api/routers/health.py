"""Liveness."""
from fastapi import APIRouter

router = APIRouter(tags=["ops"])


@router.get("/health")
def health() -> dict:
    """Deliberately shallow.

    It answers "is this process serving" and nothing else — it opens no database
    connection and calls no dependency. A health check that reached SIS would take this
    service out of a load balancer because a *different* service was slow, and a health
    check that touched the database would be one more connection per poll, forever.
    """
    return {"status": "ok", "service": "identity"}


__all__ = ["router"]
