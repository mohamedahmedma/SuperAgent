"""Signing a person in, and working out what they may do.

This is the second door into the service. The first — `X-API-Key`, in `sis/api/deps.py` —
answers "which *system* is calling" and is what `records/` and the import jobs use. This
one answers "which *person* is calling", and the two are kept apart deliberately: folding
them together would mean either giving `records/` a user account or giving a teacher an
API key, and each is worse than having two doors.

Three operations:

    sign_in     username + password -> a session token and a profile
    resolve     a session token -> a profile, or nothing
    sign_out    revoke one session

**The lockout is counted on the account, and it fails closed.** Five wrong passwords earn
fifteen minutes. That is slow enough to make an online guess pointless and short enough
that it is not a way to lock a colleague out of their afternoon.

**Every refusal says the same thing.** No such user, wrong password, disabled account and
locked account are one message and one status. A caller who can tell them apart can
enumerate a school's staff usernames, and those are people's names.

This module holds a `Session` because it is the only way to read the RBAC tables, and the
tables it reads have no repository — they are read once per request by primary key and a
repository per table would be five classes of pass-through. It takes the session as a
parameter rather than making one, so a test drives it against its own database.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from sis.domain.rbac import (
    AccessProfile,
    BUILT_IN_ROLES,
    Permission,
    RoleAssignment,
    Scope,
    ScopeType,
    SystemStatus,
    build_profile,
)
from sis.domain.staff import LOCKOUT, MAX_FAILED_ATTEMPTS, SESSION_LIFETIME
from sis.infrastructure.crypto import (
    dummy_hash,
    generate_session_token,
    hash_password,
    hash_session_token,
    needs_rehash,
    verify_password,
)
from sis.infrastructure.db import models as m

log = logging.getLogger("sis.access")

# The key `system_settings` keeps the estate's status under. Duplicated from the seeder
# as a literal rather than imported, because `sis/demo/` is tooling and nothing the
# service runs may depend on it. Kept in step by name: both spell it `system.status`.
SYSTEM_STATUS_KEY = "system.status"


def _utc(moment: datetime) -> datetime:
    """SQLite drops timezone metadata; restore UTC before security comparisons."""
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)


class AuthenticationFailed(Exception):
    """One exception for every way a sign-in can fail. See the module docstring."""


@dataclass(frozen=True, slots=True)
class SignedIn:
    """What a successful sign-in produces: the raw token, and who the caller now is."""

    token: str
    expires_at: datetime
    profile: AccessProfile
    # Denormalised for the response, so the console does not need a second call to render
    # a header with the person's name in their own language.
    full_name_en: str
    full_name_ar: str
    preferred_language: str
    school_code: str | None


def sign_in(
    session: Session,
    *,
    username: str,
    password: str,
    now: datetime | None = None,
    client_ip: str = "",
) -> SignedIn:
    """Verify a password, open a session, and return the caller's whole profile.

    The KDF runs even when the username matches nothing, against `dummy_hash()`. Skipping
    it would make "no such user" measurably faster than "wrong password", which turns a
    stopwatch into a list of who works at the school.
    """
    at = now or datetime.now(UTC)
    # Cheap on all but the first sign-in after an upgrade — see `ensure_catalogue`. It runs
    # here rather than at startup so a school that has never been seeded still gets a
    # working role table the first time somebody signs in.
    ensure_catalogue(session)
    handle = str(username or "").strip()

    user = session.scalars(
        select(m.User).where(m.User.username == handle)
    ).one_or_none()

    presented_ok = verify_password(password or "", user.password_hash if user else dummy_hash())

    if user is None or not presented_ok:
        if user is not None:
            _record_failure(session, user, at)
        log.info("sign-in refused for %r", handle)
        raise AuthenticationFailed("Incorrect username or password.")

    if not user.is_active or (user.locked_until is not None and user.locked_until > at):
        # Deliberately the same message and the same exception as a wrong password. A
        # locked account that says so is a confirmation that the username is real.
        log.info("sign-in refused for %r: account not usable", handle)
        raise AuthenticationFailed("Incorrect username or password.")

    # Success: clear the counter, record the login, and upgrade the hash if the cost
    # floor has risen since it was written.
    user.failed_attempts = 0
    user.locked_until = None
    user.last_login_at = at
    if needs_rehash(user.password_hash):
        user.password_hash = hash_password(password)

    # Expired rows are pruned here rather than by a job: it is one indexed delete on the
    # path that creates a row, and a cron nobody notices has stopped is how a session
    # table becomes a million rows.
    session.execute(delete(m.UserSession).where(m.UserSession.expires_at <= at))

    raw, token_hash = generate_session_token()
    expires_at = at + SESSION_LIFETIME
    session.add(
        m.UserSession(
            user_id=user.id,
            token_hash=token_hash,
            expires_at=expires_at,
            client_ip=client_ip[:64],
            created_at=at,
        )
    )
    session.flush()

    return SignedIn(
        token=raw,
        expires_at=expires_at,
        profile=profile_for(session, user),
        full_name_en=user.full_name_en,
        full_name_ar=user.full_name_ar,
        preferred_language=user.preferred_language,
        school_code=_school_code(session, user.school_id),
    )


def _record_failure(session: Session, user: m.User, at: datetime) -> None:
    """Count a wrong password, and lock the account once there have been enough."""
    user.failed_attempts = int(user.failed_attempts or 0) + 1
    if user.failed_attempts >= MAX_FAILED_ATTEMPTS:
        user.locked_until = at + LOCKOUT
        log.warning(
            "account %r locked until %s after %d failed attempts",
            user.username,
            user.locked_until.isoformat(),
            user.failed_attempts,
        )


def resolve(
    session: Session, *, token: str, now: datetime | None = None
) -> AccessProfile | None:
    """A bearer token to a profile. `None` for anything that is not a live session.

    Looked up by the hash, which is indexed and unique, so this is one row by key —
    which matters because it runs on every authenticated request.
    """
    at = now or datetime.now(UTC)
    raw = str(token or "").strip()
    if not raw:
        return None

    row = session.scalars(
        select(m.UserSession).where(m.UserSession.token_hash == hash_session_token(raw))
    ).one_or_none()
    if row is None or row.revoked_at is not None or _utc(row.expires_at) <= _utc(at):
        return None

    user = session.get(m.User, row.user_id)
    if user is None or not user.is_active:
        return None
    return profile_for(session, user)


def sign_out(session: Session, *, token: str, now: datetime | None = None) -> bool:
    """Revoke one session. Returns whether there was a live one to revoke."""
    at = now or datetime.now(UTC)
    row = session.scalars(
        select(m.UserSession).where(m.UserSession.token_hash == hash_session_token(token))
    ).one_or_none()
    if row is None or row.revoked_at is not None:
        return False
    row.revoked_at = at
    return True


def profile_for(session: Session, user: m.User) -> AccessProfile:
    """Flatten a user's grants into the object every authorisation check reads.

    Two queries regardless of how many roles the user holds: their assignments joined to
    the role codes, and the whole role→permission map. The second is small — seven roles
    and a couple of hundred rows — and reading all of it beats a per-role query in a loop.
    """
    assignments_rows = session.execute(
        select(m.UserRole, m.Role.code)
        .join(m.Role, m.UserRole.role_id == m.Role.id)
        .where(m.UserRole.user_id == user.id)
    ).all()

    assignments = []
    for grant, role_code in assignments_rows:
        scope_type = ScopeType(grant.scope_type)
        # A global grant carries no id; anything else without one covers nothing, and is
        # dropped rather than raising — a malformed row must not stop a login.
        if scope_type is not ScopeType.GLOBAL and grant.scope_id is None:
            log.warning(
                "user %r holds %s at %s with no scope id; ignoring the grant",
                user.username,
                role_code,
                scope_type.value,
            )
            continue
        scope = Scope(
            type=scope_type,
            id=None if scope_type is ScopeType.GLOBAL else int(grant.scope_id),
        )
        assignments.append(
            RoleAssignment(role_code=role_code, scope=scope, granted_by=grant.granted_by)
        )

    return build_profile(
        user_id=user.id,
        username=user.username,
        assignments=assignments,
        permissions_by_role=permissions_by_role(session),
        school_id=user.school_id,
    )


#: Which table each scope type's `scope_id` points into, and the column holding its code.
#: `GLOBAL` is absent because it names nothing.
#:
#: One table, read by everything that has to turn a scope id into something a person can
#: read. It was written out twice — once here and once in the access router — and two
#: copies of "which table does `year_level` mean" is the kind of pair that stays correct
#: until somebody adds a seventh scope to one of them.
SCOPE_TABLES: dict[ScopeType, tuple[object, object]] = {
    ScopeType.SCHOOL: (m.School, m.School.code),
    ScopeType.TRACK: (m.EducationalSystem, m.EducationalSystem.code),
    ScopeType.YEAR_LEVEL: (m.YearLevel, m.YearLevel.code),
    ScopeType.CLASS_SECTION: (m.ClassSection, m.ClassSection.code),
    ScopeType.SUBJECT: (m.Subject, m.Subject.code),
}


def scope_codes(
    session: Session, profile: AccessProfile
) -> dict[tuple[str, int], str]:
    """The human code behind each scope id this profile is bounded by.

    Exists for the console. A grant is stored against a surrogate id, and a screen holds
    codes — it navigated to `P1A`, not to class 47 — so without this the browser cannot
    tell whether a class-scoped grant covers the class it is drawing, and every scope check
    in the UI degrades to "does this person hold the permission anywhere". Which is the
    check that shows a teacher of one room an edit button on all of them.

    One query per scope *type* the profile actually uses, not one per grant: a teacher of
    eight classes costs one statement. Types the profile does not use cost nothing.
    """
    wanted: dict[ScopeType, set[int]] = {}
    for assignment in profile.assignments:
        if assignment.scope.type is ScopeType.GLOBAL or assignment.scope.id is None:
            continue
        wanted.setdefault(assignment.scope.type, set()).add(assignment.scope.id)

    found: dict[tuple[str, int], str] = {}
    for scope_type, ids in wanted.items():
        table, code_column = SCOPE_TABLES[scope_type]
        for identifier, code in session.execute(
            select(table.id, code_column).where(table.id.in_(sorted(ids)))
        ).all():
            found[(scope_type.value, int(identifier))] = code
    return found


def permissions_by_role(session: Session) -> dict[str, tuple[Permission, ...]]:
    """The whole role→permission map, as domain enums.

    A permission code in the table that this build's enum does not have is skipped rather
    than raising: it is a row from a newer version of the service, and an older process
    must degrade to "cannot do that" rather than to "cannot start".
    """
    rows = session.execute(
        select(m.Role.code, m.PermissionRow.code)
        .join(m.RolePermission, m.RolePermission.role_id == m.Role.id)
        .join(m.PermissionRow, m.RolePermission.permission_id == m.PermissionRow.id)
    ).all()

    mapping: dict[str, list[Permission]] = {}
    for role_code, permission_code in rows:
        try:
            permission = Permission(permission_code)
        except ValueError:
            continue
        mapping.setdefault(role_code, []).append(permission)
    return {role: tuple(permissions) for role, permissions in mapping.items()}


#: The `system_settings` key holding the fingerprint of the catalogue currently in the
#: database. See `ensure_catalogue` for what it buys.
CATALOGUE_KEY = "rbac.catalogue"


def permission_label(permission: Permission) -> str:
    """A readable name derived from the code rather than written twice.

    `noun.verb` reads as "Verb noun", which is a better label than most hand-written
    ones and — more to the point — cannot fall out of step with the code it labels.
    """
    noun, _, verb = permission.value.rpartition(".")
    return f"{verb.replace('_', ' ').title()} {noun.replace('.', ' ')}"


def catalogue_fingerprint() -> str:
    """A short digest of the role and permission catalogue this build ships.

    Every permission, and every role with the bundle it carries, in a stable order. Any
    edit to `sis/domain/rbac.py` changes it; nothing else does.
    """
    material = "\n".join(
        [*(p.value for p in Permission)]
        + [
            f"{d.code.value}:{d.default_scope.value}:" + ",".join(p.value for p in d.permissions)
            for d in BUILT_IN_ROLES
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


def ensure_catalogue(session: Session) -> bool:
    """Reconcile the catalogue, but only when this build ships a different one.

    `sync_builtin_rbac` below is a dozen statements and it used to run on **every single
    sign-in**. Two things were wrong with that. The obvious one is cost on the hottest
    authenticated path in the service. The one that would have hurt: it writes, so two
    people signing in at the same moment on Postgres race to insert the same permission
    row, and the loser's login fails with an integrity error on a table they were only
    reading from.

    So the reconcile is guarded by a fingerprint of the catalogue in the code, stored in
    `system_settings`. A matching fingerprint means the tables already say what this build
    says, and the check is one indexed read. A mismatch — a fresh database, or a release
    that added a permission — runs the reconcile once and records the new value.

    Returns whether it actually reconciled, which is what the tooling reports.
    """
    wanted = catalogue_fingerprint()
    row = session.scalars(
        select(m.SystemSetting).where(m.SystemSetting.key == CATALOGUE_KEY)
    ).one_or_none()
    if row is not None and row.value == wanted:
        return False

    sync_builtin_rbac(session)
    if row is None:
        row = m.SystemSetting(key=CATALOGUE_KEY)
        session.add(row)
    row.value = wanted
    row.note = "Fingerprint of the built-in role catalogue; managed by the service."
    row.updated_at = datetime.now(UTC)
    session.flush()
    log.info("rbac catalogue reconciled to %s", wanted)
    return True


def sync_builtin_rbac(session: Session) -> None:
    """Write the built-in role and permission catalogue into the tables. Idempotent.

    **Grants are never touched.** This function only ever writes `permissions`, `roles`
    and `role_permissions` — the catalogue. `user_roles`, the table that says who is what,
    is not in any statement below, so an upgrade cannot revoke somebody's access. Roles
    are updated in place rather than replaced for the same reason: a role a school has
    granted to forty people keeps its id.

    **A built-in role's bundle is owned by the code**, so a `role_permissions` row that
    this build does not declare is removed. That is what makes the definitions in
    `sis/domain/rbac.py` the single statement of what a Principal may do, rather than a
    starting point that drifts. A school that wants a different bundle adds a role rather
    than editing a built-in one — nothing here deletes or reconciles a role it did not
    declare.

    Prefer `ensure_catalogue` as the entry point; this one always writes.
    """
    permission_rows = {row.code: row for row in session.scalars(select(m.PermissionRow)).all()}
    for permission in Permission:
        row = permission_rows.get(permission.value)
        label = permission_label(permission)
        if row is None:
            row = m.PermissionRow(code=permission.value, name_en=label, name_ar=label)
            session.add(row)
            permission_rows[permission.value] = row
        else:
            # Only fill a blank. A school that has translated a label keeps it.
            row.name_en = row.name_en or label
            row.name_ar = row.name_ar or label
    session.flush()

    roles = {row.code: row for row in session.scalars(select(m.Role)).all()}
    for definition in BUILT_IN_ROLES:
        row = roles.get(definition.code.value)
        if row is None:
            row = m.Role(code=definition.code.value, created_at=datetime.now(UTC))
            session.add(row)
            roles[definition.code.value] = row
        row.name_en, row.name_ar = definition.name_en, definition.name_ar
        row.description, row.default_scope = definition.description_en, definition.default_scope.value
        row.is_builtin = True
    session.flush()

    for definition in BUILT_IN_ROLES:
        role = roles[definition.code.value]
        wanted = {permission_rows[p.value].id for p in definition.permissions}
        session.execute(
            delete(m.RolePermission).where(
                m.RolePermission.role_id == role.id,
                m.RolePermission.permission_id.not_in(wanted),
            )
        )
        existing = set(
            session.scalars(
                select(m.RolePermission.permission_id).where(
                    m.RolePermission.role_id == role.id
                )
            ).all()
        )
        session.add_all(
            m.RolePermission(role_id=role.id, permission_id=pid)
            for pid in sorted(wanted - existing)
        )
    session.flush()


def _school_code(session: Session, school_id: int | None) -> str | None:
    if school_id is None:
        return None
    return session.scalar(select(m.School.code).where(m.School.id == school_id))


# ---------------------------------------------------------------------------
# The estate's status
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SystemState:
    """Whether the service is answering, why, and who last said so."""

    status: SystemStatus
    note: str = ""
    updated_by: str = ""
    updated_at: datetime | None = None


def read_system_state(session: Session) -> SystemState:
    """The current status. An unset or unrecognised value reads as `ACTIVE`.

    Failing open is the right default here and it is worth saying why: this switch exists
    so an administrator can *deliberately* close the service. A missing row, a typo, or a
    value written by a newer version must not close a school's SIS by accident — the
    failure that matters is being down when nobody asked for it.
    """
    row = session.scalars(
        select(m.SystemSetting).where(m.SystemSetting.key == SYSTEM_STATUS_KEY)
    ).one_or_none()
    if row is None:
        return SystemState(status=SystemStatus.ACTIVE)
    try:
        status = SystemStatus(row.value)
    except ValueError:
        log.warning("system status %r is not a known value; treating as active", row.value)
        status = SystemStatus.ACTIVE
    return SystemState(
        status=status, note=row.note, updated_by=row.updated_by, updated_at=row.updated_at
    )


def write_system_state(
    session: Session,
    *,
    status: SystemStatus,
    note: str = "",
    actor: str = "",
    now: datetime | None = None,
) -> SystemState:
    """Set the status. Upserts the single row, and always records who and when."""
    at = now or datetime.now(UTC)
    row = session.scalars(
        select(m.SystemSetting).where(m.SystemSetting.key == SYSTEM_STATUS_KEY)
    ).one_or_none()
    if row is None:
        row = m.SystemSetting(key=SYSTEM_STATUS_KEY)
        session.add(row)
    row.value = status.value
    row.note = note or ""
    row.updated_by = actor or ""
    row.updated_at = at
    session.flush()
    log.warning("system status set to %s by %s", status.value, actor or "<unknown>")
    return SystemState(status=status, note=row.note, updated_by=row.updated_by, updated_at=at)


__all__ = [
    "CATALOGUE_KEY",
    "SYSTEM_STATUS_KEY",
    "AuthenticationFailed",
    "SignedIn",
    "SystemState",
    "catalogue_fingerprint",
    "ensure_catalogue",
    "permission_label",
    "permissions_by_role",
    "SCOPE_TABLES",
    "scope_codes",
    "profile_for",
    "read_system_state",
    "resolve",
    "sign_in",
    "sign_out",
    "sync_builtin_rbac",
    "write_system_state",
]
