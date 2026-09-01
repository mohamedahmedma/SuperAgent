"""Signing in, and saying who may do what — the RBAC surface over HTTP.

Two groups of routes, and they answer different questions.

`/v1/auth/*` is the session: sign in, ask who you are, sign out. It is the door that
already existed and none of it changed for Stage 9 beyond what the profile now carries.

`/v1/rbac/*` is the model itself. The catalogue routes (`roles`, `permissions`, `scopes`)
describe what the service *can* express, and they exist so a console does not have to hard
code any of it — the six roles, the twenty-odd permissions and the six scope types are
served as data, and a screen built against them keeps working when a seventh role is
added. The grant routes (`users/{id}/roles`) are how a person comes to hold one.

**Grants are additive here as they are everywhere else.** `POST` adds a role at a scope
and takes nothing away, so making a teacher an attendance supervisor leaves them a
teacher. There is no "set the user's role" route on purpose: an endpoint shaped that way
is one an administrator eventually calls with a shorter list than they meant, and the
person quietly loses the classes they had.

**Nobody can grant what they do not hold.** See `_authorised_to_grant`.
"""
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import delete, select, update

from sis.api.errors import error_detail
from sis.api.deps import SessionProfile, UowFactoryDep, require_user_permission
from sis.application.services.access import (
    SCOPE_TABLES,
    AuthenticationFailed,
    ensure_catalogue,
    permission_label,
    permissions_by_role,
    scope_codes,
    sign_in,
    sign_out,
)
from sis.application.services.scopes import ScopeResolver
from sis.domain.rbac import (
    ROLE_CODE_ALIASES,
    SCOPE_CATALOGUE,
    AccessProfile,
    Permission,
    RoleCode,
    ScopeType,
    Target,
)
from sis.infrastructure.db import models as m
from sis.infrastructure.crypto import hash_password
from sis.domain.staff import PASSWORD_MIN_LENGTH, USERNAME_MAX_LENGTH

router = APIRouter(prefix="/v1", tags=["access"])


def _refuse(status_code: int, code: str, message: str) -> HTTPException:
    """Every refusal in this router, in the envelope the rest of the service uses."""
    return HTTPException(status_code=status_code, detail=error_detail(code, message))


# ---------------------------------------------------------------------------
# What a session says about the person holding it
# ---------------------------------------------------------------------------


class RoleGrantOut(BaseModel):
    """One role, held at one scope. The unit an administrator adds and removes."""

    role_code: str
    scope_type: ScopeType
    scope_id: int | None
    scope_code: str | None = Field(
        default=None,
        description="The code of the thing `scope_id` points at — `P1A`, `AR-P1`. Null "
        "for a system scope, which names nothing, and for a row whose target has since "
        "been deleted.",
    )
    granted_by: str


class GrantOut(BaseModel):
    """One *permission* at one scope — a role assignment already flattened.

    Sent alongside the roles because the console needs it to decide what to draw. Knowing
    that somebody is a Teacher tells a screen nothing without also shipping the role table;
    knowing they hold `grades.write` on class `P1A` tells it exactly which register to
    offer an edit button on. The flattening happens on the server so one implementation of
    "what does this role mean" exists rather than two that disagree.
    """

    permission: str
    scope_type: ScopeType
    scope_id: int | None
    scope_code: str | None = Field(
        default=None,
        description="As on a role grant. A browser holds codes and never surrogate ids, "
        "so this is the field a client-side scope check can actually compare against.",
    )


class ProfileOut(BaseModel):
    user_id: int
    username: str
    #: The school this account is bound to. `null` for a system administrator, who is
    #: bound to none — which is a fact the console renders differently, not an error.
    school_id: int | None = None
    is_system_admin: bool = False
    roles: list[RoleGrantOut]
    permissions: list[str] = Field(
        description="Every permission held **somewhere**, de-duplicated. Good enough to "
        "decide whether a screen exists at all; not good enough to decide whether a "
        "particular class may be edited — use `grants` for that."
    )
    grants: list[GrantOut] = Field(
        default_factory=list,
        description="The same permissions, each with the scope it is held at.",
    )

    @classmethod
    def of(
        cls, profile: AccessProfile, codes: dict[tuple[str, int], str] | None = None
    ) -> "ProfileOut":
        """Render a profile, optionally resolving each scope id to its code.

        `codes` is optional so this stays callable without a session — but every route
        that answers a browser passes it, because a scope the client cannot name is a
        scope the client cannot check against.
        """
        found = codes or {}

        def code_for(scope) -> str | None:
            if scope.id is None:
                return None
            return found.get((scope.type.value, scope.id))

        return cls(
            user_id=profile.user_id,
            username=profile.username,
            school_id=profile.school_id,
            is_system_admin=profile.is_system_admin,
            roles=[
                RoleGrantOut(
                    role_code=a.role_code,
                    scope_type=a.scope.type,
                    scope_id=a.scope.id,
                    scope_code=code_for(a.scope),
                    granted_by=a.granted_by,
                )
                for a in profile.assignments
            ],
            permissions=list(profile.permissions()),
            grants=[
                GrantOut(
                    permission=grant.permission.value,
                    scope_type=grant.scope.type,
                    scope_id=grant.scope.id,
                    scope_code=code_for(grant.scope),
                )
                for grant in profile.grants
            ],
        )


class LoginIn(BaseModel):
    username: str
    password: str


class AccountOut(BaseModel):
    """Who is signed in: their names, their language, and everything they may do."""

    full_name_en: str
    full_name_ar: str
    preferred_language: str
    school_code: str | None
    profile: ProfileOut


class LoginOut(AccountOut):
    token: str = Field(description="Bearer session token; keep it in session storage only.")
    expires_at: datetime


@router.post("/auth/login", response_model=LoginOut)
def login(body: LoginIn, request: Request, uow_factory: UowFactoryDep) -> LoginOut:
    with uow_factory() as uow:
        try:
            result = sign_in(
                uow._session,
                username=body.username,
                password=body.password,
                client_ip=request.client.host if request.client else "",
            )
        except AuthenticationFailed as exc:
            # Committed, not rolled back: the failed-attempt counter and the lockout it
            # leads to are security state. Discarding them with the rest of the
            # transaction would make the lockout unreachable by repeated guessing, which
            # is the one case it exists for.
            uow.commit()
            raise _refuse(401, "not_authorized", str(exc)) from None
        codes = scope_codes(uow._session, result.profile)
        uow.commit()
    return LoginOut(
        token=result.token,
        expires_at=result.expires_at,
        full_name_en=result.full_name_en,
        full_name_ar=result.full_name_ar,
        preferred_language=result.preferred_language,
        school_code=result.school_code,
        profile=ProfileOut.of(result.profile, codes),
    )


@router.get("/auth/me", response_model=AccountOut)
def me(profile: SessionProfile, uow_factory: UowFactoryDep) -> AccountOut:
    """The whole account behind this token, in the shape `login` returned it.

    Same shape minus the token on purpose: a console that reloads calls this instead of
    asking the person to sign in again, and it must not have to assemble its header from
    two differently-shaped payloads depending on how it got here.
    """
    with uow_factory() as uow:
        user = uow._session.get(m.User, profile.user_id)
        if user is None:
            raise _refuse(401, "not_authorized", "The user session is invalid or expired.")
        school_code = (
            uow._session.scalar(select(m.School.code).where(m.School.id == user.school_id))
            if user.school_id is not None
            else None
        )
        account = AccountOut(
            full_name_en=user.full_name_en,
            full_name_ar=user.full_name_ar,
            preferred_language=user.preferred_language,
            school_code=school_code,
            profile=ProfileOut.of(profile, scope_codes(uow._session, profile)),
        )
    return account


@router.post("/auth/logout", status_code=204)
def logout(
    authorization: Annotated[str | None, Header(alias="Authorization")],
    uow_factory: UowFactoryDep,
) -> None:
    """Revoke this session. Never says whether there was one to revoke.

    204 either way: a caller that could tell a live token from a dead one has an oracle
    for testing stolen tokens, and there is nothing a legitimate client does differently
    on the two answers.
    """
    _, _, token = (authorization or "").partition(" ")
    with uow_factory() as uow:
        sign_out(uow._session, token=token)
        uow.commit()


# ---------------------------------------------------------------------------
# The catalogue: what this service can express
# ---------------------------------------------------------------------------


class ScopeOut(BaseModel):
    """One kind of boundary a grant can be drawn at."""

    type: ScopeType
    name_en: str
    name_ar: str
    names_a: str = Field(
        description="Which thing `scope_id` identifies. Empty for `global`, which "
        "identifies nothing and must be sent with no id."
    )
    depth: int = Field(
        description="Widest first. A grant at a lower depth answers for everything "
        "beneath it, so `school` covers every grade and class in that school."
    )


class RoleOut(BaseModel):
    code: str
    name_en: str
    name_ar: str
    description: str
    default_scope: ScopeType = Field(
        description="Where this role is normally granted — what an assignment screen "
        "should offer first. Advisory: a school-wide attendance supervisor is legitimate "
        "and nothing refuses one."
    )
    is_builtin: bool
    permissions: list[str]
    aliases: list[str] = Field(
        default_factory=list,
        description="Other spellings this role may be granted under. `grade_supervisor` "
        "and `year_supervisor` are one role, not two.",
    )


class PermissionOut(BaseModel):
    code: str
    name_en: str
    name_ar: str
    group: str = Field(description="The noun half of the code — how a listing is grouped.")


@router.get("/rbac/scopes", response_model=list[ScopeOut])
def list_scopes(profile: SessionProfile) -> list[ScopeOut]:
    """The scope ladder, widest first.

    Served rather than hard-coded in each client so "System, School, Track, Grade, Class,
    Subject" is stated once. A console builds its scope picker from this and needs no
    release when a seventh kind of boundary is added.
    """
    return [
        ScopeOut(
            type=descriptor.type,
            name_en=descriptor.name_en,
            name_ar=descriptor.name_ar,
            names_a=descriptor.names_a,
            depth=descriptor.depth,
        )
        for descriptor in SCOPE_CATALOGUE
    ]


class YearLevelScopeOut(BaseModel):
    id: int
    code: str
    name_en: str
    name_ar: str


@router.get("/rbac/year-level-scopes", response_model=list[YearLevelScopeOut])
def list_year_level_scopes(
    profile: Annotated[
        AccessProfile, Depends(require_user_permission(Permission.ROLES_ASSIGN))
    ],
    uow_factory: UowFactoryDep,
    school: str,
) -> list[YearLevelScopeOut]:
    """Grades available when a manager bounds an attendance account."""
    with uow_factory() as uow:
        school_row = uow._session.scalar(select(m.School).where(m.School.code == school))
        if school_row is None:
            raise _refuse(404, "unknown_reference", "No such school.")
        if profile.school_id is not None and profile.school_id != school_row.id:
            raise _refuse(403, "not_authorized", "That school is outside your authority.")
        levels = uow._session.scalars(select(m.YearLevel).where(
            m.YearLevel.school_id == school_row.id
        ).order_by(m.YearLevel.display_order, m.YearLevel.code)).all()
        return [YearLevelScopeOut(id=row.id, code=row.code, name_en=row.name_en, name_ar=row.name_ar) for row in levels]


@router.get("/rbac/roles", response_model=list[RoleOut])
def list_role_catalogue(profile: SessionProfile, uow_factory: UowFactoryDep) -> list[RoleOut]:
    """Every role this deployment has, with the permissions each one carries.

    Read from the tables rather than from `BUILT_IN_ROLES`, so a role a school added
    appears beside the ones the service ships. The catalogue is reconciled first, which
    matters on a database that has been migrated but never seeded: the alternative is an
    empty list and a console that offers no roles at all.
    """
    with uow_factory() as uow:
        session = uow._session
        ensure_catalogue(session)
        uow.commit()

    with uow_factory() as uow:
        session = uow._session
        by_role = permissions_by_role(session)
        rows = session.scalars(select(m.Role).order_by(m.Role.id)).all()
        aliases: dict[str, list[str]] = {}
        for spelling, code in ROLE_CODE_ALIASES.items():
            aliases.setdefault(code, []).append(spelling)
        catalogue = [
            RoleOut(
                code=row.code,
                name_en=row.name_en,
                name_ar=row.name_ar,
                description=row.description,
                default_scope=ScopeType(row.default_scope),
                is_builtin=bool(row.is_builtin),
                permissions=sorted(p.value for p in by_role.get(row.code, ())),
                aliases=sorted(aliases.get(row.code, ())),
            )
            for row in rows
            if row.code != RoleCode.SUBJECT_COORDINATOR.value
        ]
    return catalogue


@router.get("/rbac/permissions", response_model=list[PermissionOut])
def list_permission_catalogue(profile: SessionProfile) -> list[PermissionOut]:
    """Every verb the service can authorise, grouped by the noun it acts on.

    Straight from the enum, not the table: this is the contract of *this build*, and a row
    left behind by an older one would list a permission nothing checks.
    """
    return [
        PermissionOut(
            code=permission.value,
            name_en=permission_label(permission),
            name_ar=permission_label(permission),
            group=permission.value.rpartition(".")[0],
        )
        for permission in Permission
    ]


# ---------------------------------------------------------------------------
# Granting and revoking
# ---------------------------------------------------------------------------


class RoleGrantIn(BaseModel):
    role_code: RoleCode = Field(
        description="A role code, or one of the alternative spellings the catalogue "
        "lists under `aliases`."
    )
    scope_type: ScopeType
    scope_id: int | None = None


RoleManager = Annotated[
    AccessProfile, Depends(require_user_permission(Permission.ROLES_ASSIGN))
]
"""Somebody who may change what other people are. A person, never an integration."""

UserReader = Annotated[
    AccessProfile, Depends(require_user_permission(Permission.USERS_READ))
]
UserManager = Annotated[
    AccessProfile, Depends(require_user_permission(Permission.USERS_WRITE))
]


class UserOut(BaseModel):
    """A person who can sign in, and everything they currently hold."""

    id: int
    username: str
    full_name_en: str
    full_name_ar: str
    school_id: int | None
    is_active: bool
    roles: list[RoleGrantOut]


class UserCreateIn(BaseModel):
    username: str = Field(min_length=1, max_length=USERNAME_MAX_LENGTH)
    password: str = Field(min_length=PASSWORD_MIN_LENGTH, max_length=1024)
    email: str = Field(default="", max_length=255)
    full_name_en: str = Field(default="", max_length=255)
    full_name_ar: str = Field(default="", max_length=255)
    preferred_language: str = Field(default="en", pattern=r"^(en|ar)$")
    school_id: int | None = None
    is_active: bool = True


class UserUpdateIn(BaseModel):
    password: str | None = Field(default=None, min_length=PASSWORD_MIN_LENGTH, max_length=1024)
    email: str | None = Field(default=None, max_length=255)
    full_name_en: str | None = Field(default=None, max_length=255)
    full_name_ar: str | None = Field(default=None, max_length=255)
    preferred_language: str | None = Field(default=None, pattern=r"^(en|ar)$")
    is_active: bool | None = None


def _user_out(session, user: m.User) -> UserOut:  # noqa: ANN001
    return UserOut(
        id=user.id,
        username=user.username,
        full_name_en=user.full_name_en,
        full_name_ar=user.full_name_ar,
        school_id=user.school_id,
        is_active=bool(user.is_active),
        roles=_grants_of(session, user.id),
    )


@router.post("/rbac/users", response_model=UserOut, status_code=201)
def create_user(
    body: UserCreateIn, manager: UserManager, uow_factory: UowFactoryDep
) -> UserOut:
    """Create an account. Roles are granted separately and remain additive."""
    with uow_factory() as uow:
        session = uow._session
        username = body.username.strip()
        if session.scalar(select(m.User.id).where(m.User.username == username)) is not None:
            raise _refuse(409, "duplicate_code", "That username already exists.")
        school_id = body.school_id
        if manager.school_id is not None:
            if school_id is not None and school_id != manager.school_id:
                raise _refuse(403, "not_authorized", "That school is outside your authority.")
            school_id = manager.school_id
        elif school_id is not None and session.get(m.School, school_id) is None:
            raise _refuse(404, "unknown_reference", "No such school.")
        user = m.User(
            username=username,
            password_hash=hash_password(body.password),
            email=body.email.strip(),
            full_name_en=body.full_name_en.strip(),
            full_name_ar=body.full_name_ar.strip(),
            preferred_language=body.preferred_language,
            school_id=school_id,
            is_active=body.is_active,
        )
        session.add(user)
        session.flush()
        uow.commit()
        return _user_out(session, user)


@router.patch("/rbac/users/{user_id}", response_model=UserOut)
def update_user(
    user_id: int,
    body: UserUpdateIn,
    manager: UserManager,
    uow_factory: UowFactoryDep,
) -> UserOut:
    """Update or disable an account without deleting its history or role audit."""
    with uow_factory() as uow:
        session = uow._session
        user = _subject_user(session, manager, user_id)
        changes = body.model_dump(exclude_unset=True)
        password = changes.pop("password", None)
        for field, value in changes.items():
            if isinstance(value, str):
                value = value.strip()
            setattr(user, field, value)
        if password is not None:
            user.password_hash = hash_password(password)
        # Password changes and deactivation revoke existing bearer sessions. Keeping an
        # already-issued token alive would make both controls cosmetic until expiry.
        if password is not None or body.is_active is False:
            session.execute(
                update(m.UserSession)
                .where(m.UserSession.user_id == user.id, m.UserSession.revoked_at.is_(None))
                .values(revoked_at=datetime.now().astimezone())
            )
        session.flush()
        uow.commit()
        return _user_out(session, user)


@router.get("/rbac/users", response_model=list[UserOut])
def list_users(reader: UserReader, uow_factory: UowFactoryDep) -> list[UserOut]:
    """The accounts this caller may see, with their grants.

    Narrowed to the caller's own school unless they are bound to none. A principal
    listing the staff of another branch is not a feature, and the narrowing is here rather
    than in the query builder so it is visible in the route that returns the names.
    """
    with uow_factory() as uow:
        session = uow._session
        statement = select(m.User).order_by(m.User.username)
        if reader.school_id is not None:
            statement = statement.where(m.User.school_id == reader.school_id)
        users = session.scalars(statement).all()
        held: dict[int, list[RoleGrantOut]] = {}
        for grant, code in session.execute(
            select(m.UserRole, m.Role.code).join(m.Role, m.UserRole.role_id == m.Role.id)
        ).all():
            scope_type = ScopeType(grant.scope_type)
            held.setdefault(grant.user_id, []).append(
                RoleGrantOut(
                    role_code=code,
                    scope_type=scope_type,
                    scope_id=grant.scope_id,
                    scope_code=_scope_code(session, scope_type, grant.scope_id),
                    granted_by=grant.granted_by,
                )
            )
        listing = [
            UserOut(
                id=user.id,
                username=user.username,
                full_name_en=user.full_name_en,
                full_name_ar=user.full_name_ar,
                school_id=user.school_id,
                is_active=bool(user.is_active),
                roles=held.get(user.id, []),
            )
            for user in users
        ]
    return listing


def _authorised_to_grant(session, manager: AccessProfile, role_code: str) -> None:
    """Refuse to hand out a permission the granter does not have.

    Without this, `roles.assign` is the only permission anybody needs. A principal holds
    it at their school; nothing else in this router stops them granting `system_admin` at
    that same school, and `system_admin` carries every permission there is — including
    `roles.assign` at a wider scope. One route call and a school administrator owns the
    estate.

    So a grant is refused unless the granter already holds every permission the role
    carries. A system administrator passes trivially, holding all of them. A principal can
    make teachers, supervisors and owners, and cannot make another administrator.

    Checked against `holds` — the permission anywhere — rather than at the target scope.
    Scope is the *other* half of the decision and `_validate_scope` makes it; folding them
    together here would stop a principal granting a class-scoped role they hold at school
    scope, which is the ordinary case.
    """
    if manager.is_system_admin:
        return
    if manager.has_role(RoleCode.PRINCIPAL.value):
        if role_code in {
            RoleCode.TEACHER.value,
            RoleCode.YEAR_SUPERVISOR.value,
            RoleCode.ATTENDANCE_SUPERVISOR.value,
        }:
            return
        raise _refuse(
            403,
            "not_authorized",
            "A School Manager may assign only Teacher, Grade Supervisor, and "
            "Attendance Supervisor roles.",
        )
    carried = permissions_by_role(session).get(role_code, ())
    missing = sorted({p.value for p in carried if not manager.holds(p)})
    if missing:
        raise _refuse(
            403,
            "not_authorized",
            f"You cannot grant {role_code}: it carries {', '.join(missing)}, "
            "which this account does not hold.",
        )


def _validate_scope(
    session, profile: AccessProfile, kind: ScopeType, identifier: int | None
) -> None:
    """Check that the scope named is real, and inside the granter's own school."""
    if kind is ScopeType.GLOBAL:
        if identifier is not None:
            raise _refuse(422, "invalid_value", "A system scope has no scope_id.")
        if not profile.is_system_admin:
            raise _refuse(
                403, "not_authorized", "Only a system administrator may grant system scope."
            )
        return
    if identifier is None:
        raise _refuse(422, "invalid_value", f"A {kind.value} scope requires scope_id.")

    model, _ = SCOPE_TABLES[kind]
    row = session.get(model, identifier)
    if row is None:
        raise _refuse(404, "unknown_reference", f"No {kind.value} {identifier}.")

    # A school-bound manager cannot manufacture a grant into another school.
    if profile.school_id is not None:
        # A school *is* its own school. Reading `school_id` off it finds nothing — the
        # column does not exist — and the comparison below then refused every school-scoped
        # grant a principal tried to make, which is most of them.
        school_id = row.id if kind is ScopeType.SCHOOL else getattr(row, "school_id", None)
        if school_id is None and kind is ScopeType.CLASS_SECTION:
            school_id = session.scalar(
                select(m.AcademicYear.school_id)
                .join(m.ClassSection, m.ClassSection.academic_year_id == m.AcademicYear.id)
                .where(m.ClassSection.id == identifier)
            )
        if school_id is None and kind is ScopeType.SUBJECT:
            school_id = session.scalar(
                select(m.AcademicYear.school_id)
                .join(m.Subject, m.Subject.academic_year_id == m.AcademicYear.id)
                .where(m.Subject.id == identifier)
            )
        if school_id != profile.school_id:
            raise _refuse(403, "not_authorized", "That scope belongs to another school.")

    # And the granter's *own* `roles.assign` has to reach the place being granted.
    #
    # The school check above is not this check. It stops a grant crossing into another
    # school; this one stops a grant reaching past the granter's own boundary *inside*
    # their school. Nothing in the built-in catalogue holds `roles.assign` at a narrow
    # scope today — the principal holds it school-wide — but the whole point of the model
    # is that a school can delegate it, and the day somebody makes a grade supervisor a
    # role-granter for their own rung, "for their own rung" has to mean something.
    #
    # Skipped for a system administrator, whose global grant covers every target anyway,
    # and for a target the resolver cannot place, which the school check has already
    # bounded.
    if not profile.is_system_admin:
        target = _target_of(session, kind, identifier, row)
        if target is not None and not profile.allows(Permission.ROLES_ASSIGN, target):
            raise _refuse(
                403,
                "not_authorized",
                f"Your authority to assign roles does not reach that {kind.value}.",
            )


def _target_of(session, kind: ScopeType, identifier: int, row) -> Target | None:
    """Where a scope being granted sits, named fully enough for `allows` to judge it.

    Fully enough is the operative part: `Scope.covers` matches on ids the target actually
    carries, so a target naming only the class would be invisible to a grade-scoped
    granter's own authority and would refuse them their own rung.
    """
    if kind is ScopeType.SCHOOL:
        return Target(school_id=row.id)
    if kind is ScopeType.TRACK:
        return Target(school_id=row.school_id, track_id=row.id)
    if kind is ScopeType.YEAR_LEVEL:
        return Target(
            school_id=row.school_id,
            track_id=row.educational_system_id,
            year_level_id=row.id,
        )
    if kind is ScopeType.CLASS_SECTION:
        return ScopeResolver(session).for_class_id(identifier)
    if kind is ScopeType.SUBJECT:
        school_id = session.scalar(
            select(m.AcademicYear.school_id).where(
                m.AcademicYear.id == row.academic_year_id
            )
        )
        return Target(school_id=school_id, subject_id=row.id)
    return None


def _subject_user(session, manager: AccessProfile, user_id: int) -> m.User:
    """The account being changed, once it is established the manager may change it."""
    user = session.get(m.User, user_id)
    if user is None:
        raise _refuse(404, "unknown_reference", "No such user.")
    if manager.school_id is not None and user.school_id != manager.school_id:
        raise _refuse(403, "not_authorized", "That user belongs to another school.")
    return user


def _scope_code(session, scope_type: ScopeType, scope_id: int | None) -> str | None:
    """The code behind one scope id, for a listing an administrator reads.

    One statement per row, unlike `scope_codes` which batches — this runs over the grants
    of a single person, which is a handful, and a listing of the whole staff room resolves
    at most a few dozen. Batching that would be machinery for no measurable gain.

    The table it reads comes from `SCOPE_TABLES` rather than from a second map written out
    here: which table `year_level` means is one fact, and two statements of it stay in step
    only until somebody adds a scope to one.
    """
    if scope_id is None or scope_type is ScopeType.GLOBAL:
        return None
    table, column = SCOPE_TABLES[scope_type]
    return session.scalar(select(column).where(table.id == scope_id))


def _grants_of(session, user_id: int) -> list[RoleGrantOut]:
    rows = session.execute(
        select(m.UserRole, m.Role.code)
        .join(m.Role, m.UserRole.role_id == m.Role.id)
        .where(m.UserRole.user_id == user_id)
        .order_by(m.UserRole.id)
    ).all()
    return [
        RoleGrantOut(
            role_code=code,
            scope_type=ScopeType(row.scope_type),
            scope_id=row.scope_id,
            scope_code=_scope_code(session, ScopeType(row.scope_type), row.scope_id),
            granted_by=row.granted_by,
        )
        for row, code in rows
    ]


@router.get("/rbac/users/{user_id}/roles", response_model=list[RoleGrantOut])
def list_roles(
    user_id: int, manager: RoleManager, uow_factory: UowFactoryDep
) -> list[RoleGrantOut]:
    with uow_factory() as uow:
        _subject_user(uow._session, manager, user_id)
        return _grants_of(uow._session, user_id)


@router.post("/rbac/users/{user_id}/roles", response_model=list[RoleGrantOut])
def add_role(
    user_id: int, body: RoleGrantIn, manager: RoleManager, uow_factory: UowFactoryDep
) -> list[RoleGrantOut]:
    """Give a person a role at a scope, and return everything they now hold.

    Adding a role they already hold at that scope is a success that changes nothing —
    the unique key on `user_roles` is the whole grant, so this is idempotent and an
    administrator double-clicking does not create a duplicate to puzzle over later.

    Returns the full set rather than the one row added, because the point of this model is
    that a person holds several at once and an interface showing only the newest is the
    interface that made somebody think a role had replaced the others.
    """
    with uow_factory() as uow:
        session = uow._session
        subject_user = _subject_user(session, manager, user_id)
        if body.role_code is RoleCode.SUBJECT_COORDINATOR:
            raise _refuse(422, "invalid_value", "Subject Coordinator is no longer an assignable role.")
        if body.role_code is RoleCode.ATTENDANCE_SUPERVISOR:
            is_teacher = session.scalar(
                select(m.Teacher.id).where(m.Teacher.user_id == subject_user.id).limit(1)
            )
            has_teacher_role = session.scalar(
                select(m.UserRole.id)
                .join(m.Role, m.UserRole.role_id == m.Role.id)
                .where(
                    m.UserRole.user_id == subject_user.id,
                    m.Role.code == RoleCode.TEACHER.value,
                )
                .limit(1)
            )
            if is_teacher is not None or has_teacher_role is not None:
                raise _refuse(
                    422, "invalid_value",
                    "An attendance supervisor must have a non-teacher account.",
                )
        _authorised_to_grant(session, manager, body.role_code.value)
        _validate_scope(session, manager, body.scope_type, body.scope_id)

        role = session.scalar(select(m.Role).where(m.Role.code == body.role_code.value))
        if role is None:
            raise _refuse(404, "unknown_reference", "That role is not configured.")

        exists = session.scalar(
            select(m.UserRole).where(
                m.UserRole.user_id == user_id,
                m.UserRole.role_id == role.id,
                m.UserRole.scope_type == body.scope_type.value,
                m.UserRole.scope_id.is_(None)
                if body.scope_id is None
                else m.UserRole.scope_id == body.scope_id,
            )
        )
        if exists is None:
            session.add(
                m.UserRole(
                    user_id=user_id,
                    role_id=role.id,
                    scope_type=body.scope_type.value,
                    scope_id=body.scope_id,
                    granted_by=manager.username,
                )
            )
        uow.commit()
        return _grants_of(session, user_id)


@router.delete("/rbac/users/{user_id}/roles", response_model=list[RoleGrantOut])
def remove_role(
    user_id: int, body: RoleGrantIn, manager: RoleManager, uow_factory: UowFactoryDep
) -> list[RoleGrantOut]:
    """Take one grant away, and return what is left.

    One grant, not the role: a teacher of 3A and 3B who is removed from 3A is still the
    teacher of 3B, and the scope in the body is what says which. Removing a grant that was
    never there succeeds — the caller asked for a state, and the state holds.
    """
    with uow_factory() as uow:
        session = uow._session
        _subject_user(session, manager, user_id)
        role_id = session.scalar(select(m.Role.id).where(m.Role.code == body.role_code.value))
        if role_id is not None:
            session.execute(
                delete(m.UserRole).where(
                    m.UserRole.user_id == user_id,
                    m.UserRole.role_id == role_id,
                    m.UserRole.scope_type == body.scope_type.value,
                    m.UserRole.scope_id.is_(None)
                    if body.scope_id is None
                    else m.UserRole.scope_id == body.scope_id,
                )
            )
        uow.commit()
        return _grants_of(session, user_id)
