"""Liveness. One route, no authentication, no database.

The omission is the design. A health check that opens a connection reports the database's
mood as the *process's* health, and an orchestrator reading it restarts a perfectly good
container every time the pool is briefly saturated — which removes capacity at exactly
the moment there is none to spare, and does nothing at all for the database. This answers
one question, "is this process serving HTTP", and answers it in microseconds.

Unauthenticated for the same reason: a probe that needs a key is a probe that reports
`false` when the key expires, and the operator is paged for an outage that is not one.
Nothing here reveals anything a port scan does not already know.
"""
from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(tags=["health"])

_SERVICE_NAME = "sis"


class HealthOut(BaseModel):
    """What a probe reads. Deliberately says nothing about stored data."""

    status: Literal["ok"] = "ok"
    service: str = Field(examples=[_SERVICE_NAME])
    checked_at: datetime = Field(
        description="When this process answered, in UTC. A stale value in a proxied "
        "response is how a cached health page is spotted."
    )


@router.get(
    "/health",
    response_model=HealthOut,
    summary="Is this process serving HTTP",
    description="Liveness only. It does not touch the database — see the module note on "
    "why a database-backed probe makes an overloaded system restart itself.",
)
def read_health() -> HealthOut:
    return HealthOut(service=_SERVICE_NAME, checked_at=datetime.now(UTC))
