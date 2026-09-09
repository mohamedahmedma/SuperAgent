"""Append-only ORM mutation capture for the SIS record of truth."""
from __future__ import annotations

from contextvars import ContextVar
from datetime import date, datetime, time

from sqlalchemy import event, inspect
from sqlalchemy.orm import Session

from sis.infrastructure.db import models as m

actor_context: ContextVar[tuple[int | None, str]] = ContextVar(
    "sis_audit_actor", default=(None, "integration")
)

_SKIP = {m.AuditLog, m.AccessAudit, m.ApiKey, m.UserSession}
_SECRET = {"password", "password_hash", "token", "token_hash", "key_hash"}


def _value(value):  # noqa: ANN001
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    return value


def _snapshot(obj, *, old: bool = False) -> dict[str, object]:  # noqa: ANN001
    state = inspect(obj)
    result: dict[str, object] = {}
    for column in state.mapper.columns:
        name = column.key
        if name.lower() in _SECRET:
            continue
        history = state.attrs[name].history
        value = history.deleted[0] if old and history.deleted else getattr(obj, name)
        result[name] = _value(value)
    return result


@event.listens_for(Session, "before_flush")
def record_mutations(session: Session, _flush_context, _instances) -> None:  # noqa: ANN001
    # SQLAlchemy event registrations are process-wide.  The identity service uses its
    # own declarative base but the same ``Session`` class, so its account/bootstrap
    # writes also arrive here when both services are imported by the integration
    # suite.  Auditing those objects with the SIS model would attempt to insert into
    # ``audit_log`` in the identity database (where that table intentionally does
    # not exist).  Only SIS-mapped entities belong in the SIS audit trail.
    def is_sis_entity(obj: object) -> bool:
        return isinstance(obj, m.Base)

    if session.info.get("audit_flushing"):
        return
    actor_user_id, actor = actor_context.get()
    entries = []
    for obj in tuple(session.new):
        if not is_sis_entity(obj) or type(obj) in _SKIP:
            continue
        action = {
            m.Student: "student_created",
            m.ClassEnrolment: "enrollment_created",
            m.StudentDocument: "document_uploaded",
        }.get(type(obj), "create")
        entries.append((action, obj, None, _snapshot(obj)))
    for obj in tuple(session.dirty):
        if (
            not is_sis_entity(obj)
            or type(obj) in _SKIP
            or not session.is_modified(obj, include_collections=False)
        ):
            continue
        before, after = _snapshot(obj, old=True), _snapshot(obj)
        action = "restore" if before.get("is_active") is False and after.get("is_active") is True else (
            "soft_delete" if before.get("is_active") is True and after.get("is_active") is False else "update"
        )
        if type(obj) is m.StudentDocument and before.get("deleted_at") is None and after.get("deleted_at") is not None:
            action = "soft_delete"
        entries.append((action, obj, before, after))
    for obj in tuple(session.deleted):
        if is_sis_entity(obj) and type(obj) not in _SKIP:
            entries.append(("delete", obj, _snapshot(obj), None))
    if not entries:
        return
    session.info["audit_flushing"] = True
    try:
        for action, obj, old, new in entries:
            identity = inspect(obj).identity
            entity_id = ":".join(str(v) for v in identity) if identity else "pending"
            session.add(m.AuditLog(
                actor_user_id=actor_user_id, actor=actor[:64], action=action,
                entity_type=type(obj).__name__, entity_id=entity_id,
                old_values=old, new_values=new,
            ))
    finally:
        session.info["audit_flushing"] = False
