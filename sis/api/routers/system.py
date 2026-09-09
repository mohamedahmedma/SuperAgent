"""System Administrator controls: estate status and maintenance windows."""
from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from sis.api.deps import UowFactoryDep, require_user_permission
from sis.api.routers import error_responses
from sis.application.services.access import read_system_state, write_system_state
from sis.domain.rbac import AccessProfile, Permission, SystemStatus

router = APIRouter(prefix="/v1/admin/system", tags=["system administration"])

SystemAdministrator = Annotated[
    AccessProfile, Depends(require_user_permission(Permission.SYSTEM_MANAGE))
]
StatusAdministrator = Annotated[
    AccessProfile, Depends(require_user_permission(Permission.SYSTEM_STATUS_WRITE))
]


class SystemStatusIn(BaseModel):
    status: SystemStatus
    note: str = Field(default="", max_length=2000)


class SystemStatusOut(BaseModel):
    status: SystemStatus
    note: str = ""
    updated_by: str = ""
    updated_at: datetime | None = None


def _out(state) -> SystemStatusOut:  # noqa: ANN001
    return SystemStatusOut(
        status=state.status,
        note=state.note,
        updated_by=state.updated_by,
        updated_at=state.updated_at,
    )


@router.get(
    "/status",
    response_model=SystemStatusOut,
    responses=error_responses(401, 403),
)
def get_system_status(
    administrator: SystemAdministrator, uow_factory: UowFactoryDep
) -> SystemStatusOut:
    with uow_factory() as uow:
        return _out(read_system_state(uow._session))


@router.put(
    "/status",
    response_model=SystemStatusOut,
    responses=error_responses(401, 403, 422),
)
def set_system_status(
    body: SystemStatusIn,
    administrator: StatusAdministrator,
    uow_factory: UowFactoryDep,
) -> SystemStatusOut:
    with uow_factory() as uow:
        state = write_system_state(
            uow._session,
            status=body.status,
            note=body.note.strip(),
            actor=administrator.username,
        )
        uow.commit()
        return _out(state)


__all__ = ["router"]
