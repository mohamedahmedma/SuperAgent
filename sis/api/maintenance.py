"""HTTP maintenance gate for the SIS.

The gate sits at the edge so it also covers legacy integration routes.  Authentication
still runs normally for bearer sessions; a global System Administrator grant is the only
bypass.  Liveness and sign-in remain reachable so an administrator can diagnose and
recover a paused service.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from sis.application.services.access import read_system_state, resolve
from sis.infrastructure.db.unit_of_work import SqlAlchemyUnitOfWork
from sis.tenancy import get_registry


_ALWAYS_AVAILABLE = frozenset({"/health", "/v1/auth/login", "/v1/auth/logout"})
_READ_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


class SystemMaintenanceMiddleware(BaseHTTPMiddleware):
    """Apply the persisted system status before a protected request reaches a route."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.url.path in _ALWAYS_AVAILABLE or not request.url.path.startswith("/v1/"):
            return await call_next(request)

        school_code = _school_code(request)
        # Let the normal tenant dependency render missing/unknown school headers.  The
        # maintenance gate must not replace a precise 404/422 with an unrelated 500.
        if get_registry().is_multi_school and school_code is None:
            return await call_next(request)

        with SqlAlchemyUnitOfWork(school_code=school_code) as uow:
            state = read_system_state(uow._session)
            if state.status.value == "active":
                return await call_next(request)

            profile = _bearer_profile(request, uow._session)
            if profile is not None and profile.is_system_admin:
                return await call_next(request)

        if state.status.allows_reads and request.method.upper() in _READ_METHODS:
            return await call_next(request)

        message = state.note.strip() or "The SIS is temporarily unavailable for maintenance."
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "detail": {
                    "code": "system_maintenance",
                    "message": message,
                    "field": None,
                    "system_status": state.status.value,
                }
            },
            headers={"Retry-After": "300", "Cache-Control": "no-store"},
        )


def _school_code(request: Request) -> str | None:
    registry = get_registry()
    if not registry.is_multi_school:
        return None
    presented = request.headers.get("X-School-Code", "").strip()
    if not presented or presented not in registry.codes:
        return None
    return registry.get(presented).code


def _bearer_profile(request: Request, session):  # noqa: ANN001, ANN202
    scheme, _, token = request.headers.get("Authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return resolve(session, token=token.strip())


__all__ = ["SystemMaintenanceMiddleware"]
