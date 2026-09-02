"""Request-scoped wiring: who is calling, which transaction, and which use case.

Composition happens here. A router declares `caller: RegistrarCaller` and
`service: GradeImportServiceDep` and receives objects that are already authenticated and
already bound to a transaction; it never constructs a unit of work, never reads
`sis.config`, and never learns which repository implementation it is writing through.
That is what makes every service in this package testable with fakes: the only place
that knows an environment and a database exist is this file, and a test replaces it
wholesale through `app.dependency_overrides`.

**There are two doors, and they answer different questions.**

*The integration door* is `_require_scopes`, and **it authenticates nobody**. `X-API-Key` is
not read, the `api_keys` table is not consulted on a request, and no caller is refused for
want of a credential. That was a deliberate removal and it stands: `records/` and
`identity/` call this service with no credential, and closing this door would stop them.
Anyone who can reach this process can read and write everything in it through it, so keep
it on a network the school controls, or behind a proxy that authenticates in front of it.
`_require_scopes` remains the single place that has to change to bring key checking back,
and `tests/sis/test_authentication.py` fails the day it does — on purpose, so that day is a
deliberate edit rather than a surprise.

*The person door* is `Authorization: Bearer <session token>`, minted by
`POST /v1/auth/login`. A request carrying one is judged by that person's roles: the union
of every role they hold, each bounded to the school, track, grade, class or subject it was
granted over. `Principal` is the object both doors produce, so a handler is written once
and works for either.

**The role layer sits over the old arrangement, not in place of it.** A request with no
session behaves exactly as it did before roles existed. That is what keeps a nightly import
working at three in the morning after a permissions change, and it is asserted rather than
hoped for — see `test_an_integration_is_unaffected_by_any_of_this`.

**The route scopes survive both.** Each route still names `registrar`, `reader` or both,
because that is a statement about what the route is for rather than about who is calling.

**Which school a request is about is still decided here**, by `X-School-Code`, and is
unaffected by any of the above: it chooses the database a request is answered from, not
who may ask.
"""
from __future__ import annotations

import hashlib
import logging
import secrets
from collections.abc import Callable, Collection, Iterator, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from typing import Annotated, Final

from fastapi import Depends, Header, HTTPException, status

from sis.application.dto import Page, PageRequest
from sis.application.ports.unit_of_work import UnitOfWork
from sis.application.services.attendance import AttendanceService
from sis.application.services.grade_import import GradeImportService
from sis.application.services.guardian_import import GuardianImportService
from sis.application.services.queries import QueryService
from sis.application.services.roster_import import RosterImportService
from sis.application.services.structure import StructureGenerationService
from sis.application.services.timetable import TimetableService
from sis.application.services.teachers import TeacherManagementService
from sis.application.services.marks import MarkSheetService
from sis.application.services.teaching import TeachingService
from sis.application.services.access import resolve
from sis.application.services.scopes import ScopeResolver
from sis.api.errors import error_detail
from sis.domain.rbac import ANYWHERE, AccessProfile, Permission, ScopeType, Target
from sis.config import get_settings
from sis.domain.auth import PREFIX_LENGTH, ApiKey, Scope
from sis.domain.errors import ImportBatchNotFound, UnknownReference, ValidationError
from sis.domain.imports import ImportBatch, ImportRow, RowOutcome
from sis.domain.people import ClassEnrolment, Student
from sis.domain.structure import (
    SCHOOL_CONFIGURATION,
    AcademicTrack,
    AcademicYear,
    ClassSection,
    School,
    Stage,
    Subject,
    Term,
    YearLevel,
)
from sis.domain.value_objects import (
    AcademicYearCode,
    ClassCode,
    SchoolCode,
    StudentNumber,
    SubjectCode,
    YearCode,
)
from sis.application.ports.repositories import GradeSubjects
from sis.application.dto import TERM_LABELS, TermPlan, term_code_for
from sis.infrastructure.db.unit_of_work import SqlAlchemyUnitOfWork
from sis.infrastructure.parsers import (
    SpreadsheetGradeParser,
    SpreadsheetGuardianParser,
    SpreadsheetFamilyRosterParser,
)
from sis.tenancy import get_registry

logger = logging.getLogger(__name__)

_KEY_BYTES: Final[int] = 32

API_KEY_HEADER: Final[str] = "X-API-Key"

REQUEST_ID_HEADER: Final[str] = "X-Request-Id"
"""Correlates an access decision back to whatever caused it, one service further out."""

#: Longest correlation id this service will store. A caller controls this header entirely,
#: and the audit column is 64 characters — truncating here rather than at the database
#: keeps an over-long value from failing an authorised read.
_MAX_REQUEST_ID = 64


def get_request_id(
    x_request_id: Annotated[str | None, Header(alias=REQUEST_ID_HEADER)] = None,
) -> str:
    """The caller's correlation id, or `""`.

    Optional on purpose: a registrar reading a report card through the console has no chat
    turn behind it, and demanding one would refuse a legitimate request over a field that
    only helps somebody reading an audit later.

    Never trusted for anything but correlation. It is written to the audit and read by a
    human; nothing branches on it, so a caller inventing one gains nothing.
    """
    return (x_request_id or "").strip()[:_MAX_REQUEST_ID]


RequestId = Annotated[str, Depends(get_request_id)]
"""This request's correlation id, already trimmed and length-capped."""


SCHOOL_HEADER: Final[str] = "X-School-Code"
"""Names the school a request is about, and therefore the database that answers it."""


class MissingSchoolHeader(ValidationError):
    """A multi-school deployment was asked a question that names no school.

    Refused rather than defaulted. The tempting fallbacks — the first configured school,
    or "the only one" while there happens to be one — are both a request meant for one
    branch answered out of another branch's database the day a second school is added,
    and nothing in the response would say so.
    """

    def __init__(self, known: Collection[str]) -> None:
        super().__init__(
            f"this service holds several schools, so {SCHOOL_HEADER} is required. "
            f"Configured schools: {', '.join(sorted(known)) or 'none'}.",
            field="school_code",
        )


# ---------------------------------------------------------------------------
# Key material
# ---------------------------------------------------------------------------


def hash_api_key(raw: str) -> str:
    """The stored verifier for a presented key.

    SHA-256 rather than bcrypt or PBKDF2, which is the right call *here* and nowhere
    near a password: the input is 32 bytes of CSPRNG output, so there is no dictionary
    to attack and stretching would only add latency to every single request. Key
    stretching protects low-entropy secrets; a random 256-bit token is not one.

    This is the only implementation in the service. A second one that differed by so
    much as an encoding would make every previously stored key unverifiable, with no
    symptom other than "all our integrations stopped working at once".
    """
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def key_prefix(raw: str) -> str:
    """The public handle of a key: enough to name it in an audit, not enough to use it."""
    return raw[:PREFIX_LENGTH]


def generate_api_key() -> tuple[str, str, str]:
    """Return `(full_key, prefix, key_hash)`. The full key is shown once, then lost."""
    raw = secrets.token_urlsafe(_KEY_BYTES)
    return raw, key_prefix(raw), hash_api_key(raw)


# Compared against when no key row matched, so a wrong prefix and a wrong secret do the
# same work. Skipping the comparison on a miss turns prefix enumeration into a timing
# measurement, which is how an attacker learns which handles are real before guessing.
_ABSENT_HASH: Final[str] = hash_api_key("")


@dataclass(frozen=True, slots=True)
class Caller:
    """The authenticated *system* behind a request. Never a person.

    An API key proves which integration is calling — the registrar UI, a reporting job,
    the `records/` adapter. It does not identify a human, and nothing downstream should
    read it as though it did.
    """

    prefix: str
    scope: Scope
    is_bootstrap: bool = False

    def __str__(self) -> str:
        return self.prefix


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


ANONYMOUS_CALLER: Final[str] = "anonymous"
"""The `actor` written to audit rows now that no request names a credential.

Sixteen characters is the audit column's width; this is nine, so nothing truncates.
It reads as what it is in an import history: nobody in particular.
"""


def _require_scopes(*allowed: Scope) -> Callable[..., Caller]:
    """Admit every caller. **This door authenticates nobody, and that is deliberate.**

    API-key authentication was removed on purpose and has not come back: no header is read,
    no key table is consulted, and no request is ever refused here. `records/` and
    `identity/` reach this service through this door with no credential at all.

    Sign-in with a username and password now exists — see `require_permission` — but it was
    added *beside* this, not in front of it. A request carrying a session is judged by that
    person's roles; a request carrying nothing behaves exactly as it did before roles
    existed. Putting the two in series instead would have been the same thing as deleting
    the integration door, and every job that calls this service would have stopped.

    The scopes a route lists are still spelled out at its call site, because they say what
    a route is *for*. The first one listed is simply granted, so `require_registrar` and
    `require_read_access` both hand back a caller their routes accept.

    **What this costs, still.** Every write in this service — rewriting a term's marks,
    importing a roster, provisioning a school — is available to anyone who can open a
    socket to it *without* a session. The roles added in Stage 9 bound what a signed-in
    person may do; they do not narrow this door, and reading them as though they did would
    be the mistake this paragraph exists to prevent. Put the service behind a network that
    only the school reaches, or behind a reverse proxy that authenticates in front of it,
    for as long as this stands.

    To close it, this function is the only place that has to change: give it back a body
    that identifies the caller and refuses when it cannot.
    """

    granted: Scope = allowed[0]

    def dependency() -> Caller:
        return Caller(prefix=ANONYMOUS_CALLER, scope=granted)

    return dependency


require_registrar = _require_scopes(Scope.REGISTRAR)
"""Writes: structure generation, imports, key management."""

require_reader = _require_scopes(Scope.READER)
"""Read-only integrations: `records/` and `identity/` ask their questions through here."""

require_read_access = _require_scopes(Scope.REGISTRAR, Scope.READER)
"""Reads that both the console and a parent-facing service legitimately perform.

All three of these admit everyone while authentication is switched off. They stay
distinct because they record what each route is for, and because that is what a
username-and-password sign-in will have to tell apart when it arrives.
"""


# ---------------------------------------------------------------------------
# Who is calling: a signed-in person, or an integration
# ---------------------------------------------------------------------------
#
# Two doors, and the code below keeps them apart rather than merging them.
#
# The **integration door** is `_require_scopes` above, unchanged and still open: `records/`
# and `identity/` call this service with no credential and must keep working. Nothing in
# this section closes it, and that is deliberate — role-based access is being added *over*
# the existing arrangement, not in place of it.
#
# The **person door** is `Authorization: Bearer <session token>`, minted by
# `POST /v1/auth/login`. When a request carries one, the person's grants are authoritative
# and a missing permission is a refusal. Silently falling back to the open door for a
# signed-in user would make every permission in the service decorative.


def get_access_profile(
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
    school_code: SchoolCodeDep = None,
) -> AccessProfile:
    """The person behind a bearer token, with every role they hold, flattened.

    One database read per request. The alternative — putting the grants in the token —
    means a revoked role stays live until the token expires, and "we removed her access an
    hour ago and she can still change marks" is not a sentence a school should ever have
    to hear.
    """
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=error_detail("not_authorized", "A user session is required."),
        )
    with SqlAlchemyUnitOfWork(school_code=school_code) as uow:
        profile = resolve(uow._session, token=token.strip())
    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=error_detail("not_authorized", "The user session is invalid or expired."),
        )
    return profile


SessionProfile = Annotated[AccessProfile, Depends(get_access_profile)]
"""A signed-in person. Routes that only a person can call declare this."""


def _forbidden(permission: Permission) -> HTTPException:
    """One refusal, naming the permission that was missing.

    The permission code is in the message rather than only in a log, because the person
    reading it is usually an administrator working out which role to grant, and "403" on
    its own sends them to whoever holds the server.
    """
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=error_detail(
            "not_authorized", f"This account does not hold {permission.value}."
        ),
    )


def _is_read(permission: Permission) -> bool:
    """Whether a permission only looks. Decides which API-key scopes the old door accepts."""
    return permission.value.endswith(".read")


@dataclass(frozen=True, slots=True)
class Principal:
    """Whoever is behind this request, and what they may do — either door.

    Routers take one of these instead of a `Caller`, so a handler is written once and
    works for both an integration and a signed-in teacher. Everything a router used of a
    `Caller` still works: `.prefix` is the audit actor and `str()` is the name in a log.

    **An integration is allowed everything.** `allows` returns `True` when there is no
    profile, which is exactly the behaviour this service had before roles existed. That is
    the promise being kept: adding RBAC did not quietly refuse a job that ran last night.

    **A person is allowed what their grants say**, unioned over every role they hold. Two
    checks make that practical, and the split matters:

      `ensure`  the wide one. "Do you hold this permission anywhere?" Answered from memory
                before the handler does any work, and it is what the route dependency runs.

      `narrow`  the exact one. "Do you hold it *here* — on this class, this rung?" Needs
                the school's structure, so it costs a query, and it is only run by routes
                that name something narrow enough for the answer to differ.

    Running only the wide check would let a teacher of 3A take 3B's register. Running only
    the narrow one would put a lookup in front of every request including the ones where a
    school-wide grant already settles it. So both, in that order.
    """

    caller: Caller | None = None
    profile: AccessProfile | None = None
    #: Which school's database this request is answered from; the resolver needs it.
    school_code: str | None = None

    @property
    def is_person(self) -> bool:
        return self.profile is not None

    @property
    def prefix(self) -> str:
        """The actor written to the audit trail. A username for a person, a key handle
        otherwise. Capped at the audit column's width so an authorised write is never
        refused by the database over the length of a name."""
        if self.profile is not None:
            return self.profile.username[:16]
        return self.caller.prefix if self.caller is not None else ANONYMOUS_CALLER

    @property
    def username(self) -> str:
        return self.profile.username if self.profile is not None else self.prefix

    @property
    def school_id(self) -> int | None:
        return self.profile.school_id if self.profile is not None else None

    def __str__(self) -> str:
        return self.prefix

    def allows(self, permission: Permission, target: Target = ANYWHERE) -> bool:
        if self.profile is None:
            return True
        return self.profile.allows(permission, target)

    def ensure(self, permission: Permission, target: Target = ANYWHERE) -> None:
        if not self.allows(permission, target):
            raise _forbidden(permission)

    def narrow(
        self,
        permission: Permission,
        locate: Callable[[ScopeResolver], Target | Sequence[Target]],
    ) -> None:
        """Refuse unless the caller holds `permission` over the thing `locate` finds.

        `locate` is a callback rather than a `Target` so the lookup it needs is not paid
        for when it cannot change the answer — a system administrator and a principal are
        settled from memory, and only a genuinely narrow grant reaches the database.

        It may answer with several places rather than one, and any of them passing is
        enough. That is not a loosening: a child enrolled in two rooms this year really was
        in both, so a grant on either is a grant over part of what is being asked about.

        A route calls this *after* its dependency has already run the wide check, so the
        only thing left to establish is where the caller's grant reaches.
        """
        if self._settled_without_lookup(permission):
            return
        with SqlAlchemyUnitOfWork(school_code=self.school_code) as uow:
            located = locate(ScopeResolver(uow._session))
        candidates = (located,) if isinstance(located, Target) else tuple(located)
        assert self.profile is not None  # `_settled_without_lookup` returned False
        if not any(self.profile.allows(permission, target) for target in candidates):
            raise _forbidden(permission)

    def narrow_all(
        self,
        permission: Permission,
        locate: Callable[[ScopeResolver], Sequence[Target]],
    ) -> None:
        """Like `narrow`, but every place found must be permitted.

        The difference is the request shape, not the rule. `narrow` asks about one thing
        that may sit in several places — a child in two classrooms — and any of them
        answering is enough. This asks about several *different* things in one request:
        a timetable posting thirty lessons across eight classes is eight separate
        permissions to check, and a supervisor holding seven of them may not write the
        eighth. Partial success is not on offer, because the write is one transaction.
        """
        if self._settled_without_lookup(permission):
            return
        with SqlAlchemyUnitOfWork(school_code=self.school_code) as uow:
            required = tuple(locate(ScopeResolver(uow._session)))
        assert self.profile is not None
        for target in required:
            if not self.profile.allows(permission, target):
                raise _forbidden(permission)

    def _settled_without_lookup(self, permission: Permission) -> bool:
        """Whether the answer is already known, so no query is needed. Raises on a refusal.

        Three cases end here: an integration (always allowed), somebody holding the
        permission globally, and somebody holding it over their own whole school — which
        is most of the staff room. Only a grant narrower than a school reaches the
        resolver, which is what keeps the cost on the requests that need it.
        """
        profile = self.profile
        if profile is None:
            return True  # The integration door. Unchanged, on purpose.

        widest = profile.widest_scope_for(permission)
        if widest is None:
            raise _forbidden(permission)
        if widest is ScopeType.GLOBAL:
            return True
        return profile.school_id is not None and profile.allows(
            permission, Target(school_id=profile.school_id)
        )


def require_permission(permission: Permission) -> Callable[..., Principal]:
    """The route gate. Admits an integration, or a person who holds `permission` somewhere.

    Replaces the pair of scope dependencies at a route's call site without removing them:
    an unauthenticated caller still goes through `_require_scopes`, which still admits
    everybody, so nothing that worked yesterday stops working. What is added is that a
    request carrying a session is judged by that session's roles.
    """

    def dependency(
        authorization: Annotated[str | None, Header(alias="Authorization")] = None,
        school_code: SchoolCodeDep = None,
    ) -> Principal:
        if not (authorization or "").strip().lower().startswith("bearer "):
            allowed = (
                (Scope.REGISTRAR, Scope.READER) if _is_read(permission) else (Scope.REGISTRAR,)
            )
            return Principal(caller=_require_scopes(*allowed)(), school_code=school_code)

        profile = get_access_profile(authorization, school_code)
        # The wide check: held anywhere, at any scope. `allows` with no target would ask
        # for a global grant and refuse every teacher in the building — see
        # `AccessProfile.holds`.
        if not profile.holds(permission):
            raise _forbidden(permission)
        return Principal(profile=profile, school_code=school_code)

    return dependency


#: The previous name for `require_permission`, kept so an out-of-tree router keeps working.
permission_or_integration = require_permission


def require_user_permission(permission: Permission) -> Callable[..., AccessProfile]:
    """A gate for routes only a signed-in person may call — managing roles, chiefly.

    No integration fallback: granting a role is an act by somebody, recorded against their
    name in `user_roles.granted_by`, and an anonymous caller has no name to record.
    """

    def dependency(profile: SessionProfile) -> AccessProfile:
        if not profile.holds(permission):
            raise _forbidden(permission)
        return profile

    return dependency


# ---------------------------------------------------------------------------
# Transactions and use cases
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Which school — and therefore which database
# ---------------------------------------------------------------------------


def get_school_code(
    x_school_code: Annotated[str | None, Header(alias=SCHOOL_HEADER)] = None,
) -> str | None:
    """The school this request is about, and the database it will be answered from.

    Schools are separated physically, so this header does not narrow a query — it
    chooses the connection. Everything downstream is then bound to one school's database
    and cannot reach another's rows at all, which is why no service or repository in this
    package takes a school argument.

    In single-school mode (`SIS_SCHOOLS` unset) the header is ignored entirely and this
    returns `None`, the process-wide database. That is what keeps a development laptop
    and the existing test suite working untouched.

    In multi-school mode the header is **required**, and an absent or unknown one is a
    refusal rather than a default. Both fallbacks that suggest themselves — "use the
    first school", "use the only school" — answer a request meant for one branch out of
    another branch's database the day a second school is added, which is the single
    failure this whole design exists to prevent.
    """
    registry = get_registry()
    if not registry.is_multi_school:
        return None

    presented = (x_school_code or "").strip()
    if not presented:
        raise MissingSchoolHeader(registry.codes)
    # Raises `UnknownSchool`, which `sis.api.errors` renders as a 404 — the same answer
    # an unknown school code gets anywhere else in the service.
    return registry.get(presented).code


SchoolCodeDep = Annotated[str | None, Depends(get_school_code)]
"""The resolved school for this request; `None` in single-school mode."""


def get_unit_of_work(school_code: SchoolCodeDep) -> Iterator[UnitOfWork]:
    """One entered transaction for the life of the request; rolled back unless committed.

    FastAPI throws the handler's exception back into this generator, so the `with`
    block's `__exit__` runs on the failure path too — a route that raises `TermClosed`
    after twelve writes leaves nothing behind, without having remembered to catch
    anything.

    The transaction is opened against the school named by `X-School-Code`. That is the
    whole of the isolation: a handler cannot read across schools because the connection
    it was handed does not reach them.
    """
    with SqlAlchemyUnitOfWork(school_code=school_code) as uow:
        yield uow


def get_unit_of_work_factory(school_code: SchoolCodeDep) -> Callable[[], UnitOfWork]:
    """A *factory*, for services whose steps are separate transactions.

    Preview and commit are two requests and two transactions, and every query wants its
    own so it cannot see another's half-written state. Those services therefore take a
    callable and open a unit of work per operation rather than being handed a live one.

    The school is bound into the factory here rather than passed to each service, so a
    service that opens five transactions opens all five against the same school without
    having to know that schools exist.
    """

    def factory() -> UnitOfWork:
        return SqlAlchemyUnitOfWork(school_code=school_code)

    return factory


UowFactoryDep = Annotated[Callable[[], UnitOfWork], Depends(get_unit_of_work_factory)]
"""A unit-of-work factory already bound to this request's school.

Every service provider below takes this rather than calling `get_unit_of_work_factory()`
itself. That is deliberate: called directly it is an ordinary function and FastAPI never
resolves its `X-School-Code` dependency, so the service would quietly compose against the
process-wide database and read the wrong school's rows.
"""


def get_max_upload_bytes() -> int:
    """The upload ceiling, injected so a test can shrink it without touching the env."""
    return get_settings().max_upload_bytes


def get_query_service(uow_factory: UowFactoryDep) -> QueryService:
    """Every read, and no writes at all."""
    return QueryService(uow_factory)


def get_timetable_service(uow_factory: UowFactoryDep) -> TimetableService:
    """The weekly plan. A factory, like every other service here.

    Its own service rather than more methods on the structure catalogue: the rules it
    enforces — the school's week, its period grid, stage 5's subject assignments — are read
    from three different places and are worth testing without a database.
    """
    return TimetableService(uow_factory)


def get_teacher_management_service(
    uow_factory: UowFactoryDep,
) -> TeacherManagementService:
    return TeacherManagementService(uow_factory)


def get_mark_sheet_service(uow_factory: UowFactoryDep) -> MarkSheetService:
    """One class's marks for one subject and term — the teacher's own write path.

    Separate from `GradeImportService`, which reads a spreadsheet and is the registrar's.
    A teacher entering eight figures and an office uploading six hundred are different
    jobs with different failure modes, and one service doing both would owe the teacher a
    batch report they never asked for.
    """
    return MarkSheetService(uow_factory)


def get_teaching_service(uow_factory: UowFactoryDep) -> TeachingService:
    """Which subject a teacher owns in which room — the one check a scope cannot make.

    Its own service rather than a method on teacher management, because that service
    *writes* assignments and this one reads them for authority. Keeping the grant and the
    check apart means a bug in either half cannot quietly widen the other.
    """
    return TeachingService(uow_factory)


def get_structure_service(uow_factory: UowFactoryDep) -> StructureGenerationService:
    """A factory, like every other service here.

    This previously depended on `get_unit_of_work` and handed over the transaction that
    dependency had already entered, which `generate` then re-entered -- so every call to
    POST /v1/structure/generate raised "This unit of work is already open" and returned
    500. The service opens and owns its own transaction, which is also what makes a
    generated ladder atomic.
    """
    return StructureGenerationService(uow_factory)


def get_roster_import_service(uow_factory: UowFactoryDep) -> RosterImportService:
    """Composed per request from configuration this layer alone is allowed to read.

    The TTL and the size ceiling are constructor arguments rather than values the
    service looks up, because a use case that reads `os.getenv` cannot be unit-tested
    without arranging the environment — and preview expiry cannot be tested at all
    without either injecting the TTL or sleeping for half an hour.
    """
    settings = get_settings()
    return RosterImportService(
        uow_factory,
        SpreadsheetFamilyRosterParser(
            default_country_code=settings.default_country_code
        ),
        preview_ttl=timedelta(minutes=settings.import_preview_ttl_minutes),
        max_upload_bytes=settings.max_upload_bytes,
    )


def get_guardian_import_service(uow_factory: UowFactoryDep) -> GuardianImportService:
    """Composed per request, like the roster importer above.

    `default_country_code` joins the TTL and the size ceiling as a constructor argument
    for the same reason: the parser has to turn `01001234567` into a number that can
    actually be dialled, and a parser that read the environment itself would parse one
    file two ways in two deployments with nothing on screen to say so.
    """
    settings = get_settings()
    return GuardianImportService(
        uow_factory,
        SpreadsheetGuardianParser(
            default_country_code=settings.default_country_code
        ),
        preview_ttl=timedelta(minutes=settings.import_preview_ttl_minutes),
        max_upload_bytes=settings.max_upload_bytes,
    )


def get_grade_import_service(uow_factory: UowFactoryDep) -> GradeImportService:
    """As above, for marks. Parser defaults stay unset: the request names the subject."""
    settings = get_settings()
    return GradeImportService(
        uow_factory,
        SpreadsheetGradeParser(),
        preview_ttl=timedelta(minutes=settings.import_preview_ttl_minutes),
        max_upload_bytes=settings.max_upload_bytes,
    )


# ---------------------------------------------------------------------------
# Adapters for the ports the routers state.
#
# `admin.py`, `imports.py` and `structure.py` each declare a Protocol for the one thing
# they need and leave the implementation to this file. These three are the whole of it.
# They live here rather than in `application/services` because none of them composes a
# use case: each is one or two repository calls and a transaction boundary, and inventing
# a service around that would be a class whose only behaviour is `uow.commit()`.
# ---------------------------------------------------------------------------


class ImportReports:
    """Reading a stored batch back: the header and one window of its rows.

    Both halves come from a single transaction, on purpose. Counted in one and listed in
    another, a registrar paging through a batch that is being committed underneath her
    reads a summary saying two rows were rejected beside a page listing three, and spends
    the afternoon deciding which is lying.
    """

    def __init__(self, uow_factory: Callable[[], UnitOfWork]) -> None:
        self._uow_factory = uow_factory

    def report(
        self,
        batch_id: str,
        *,
        page: PageRequest,
        outcomes: Collection[RowOutcome] | None = None,
    ) -> tuple[ImportBatch, Page[ImportRow]]:
        with self._uow_factory() as uow:
            batch = uow.imports.get(batch_id)
            if batch is None:
                # A 404 rather than an empty report: "no such batch" and "a batch with no
                # matching rows" look identical on screen and are opposite problems.
                raise ImportBatchNotFound(
                    f"no import batch {batch_id}", field="batch_id"
                )
            total = uow.imports.count_rows(batch_id, outcomes=outcomes)
            rows = uow.imports.list_rows(
                batch_id, outcomes=outcomes, offset=page.offset, limit=page.limit
            )
            return batch, Page.of(rows, total, page)


class StructureCatalogue:
    """Create-or-relabel one term or one subject, and say which of the two happened.

    An upsert rather than an insert because of invariant 6: the code is identity and the
    names are labels, so re-posting a term with a corrected Arabic name must rename it and
    detach not one grade. Forcing that correction through a separate PATCH is how a
    registrar ends up creating `T1-FIXED` beside the term everything already points at.
    """

    def __init__(self, uow_factory: Callable[[], UnitOfWork]) -> None:
        self._uow_factory = uow_factory

    def create_term(self, term: Term) -> bool:
        with self._uow_factory() as uow:
            created = uow.terms.upsert_many([term])
            uow.commit()
        return bool(created.get(str(term.code), False))

    def create_subject(self, subject: Subject) -> bool:
        """Store one subject in one academic year; `True` when this call created it.

        A subject that names a year with no row raises `UnknownReference` from the
        repository's code resolution, naming `academic_year_code` — which is what puts the
        message under the right field on the form rather than surfacing as an integrity
        error about a foreign key.
        """
        with self._uow_factory() as uow:
            created = uow.subjects.upsert_many([subject])
            uow.commit()
        return bool(created.get(str(subject.code), False))

    def subject_assignments(
        self, academic_year_code: AcademicYearCode
    ) -> Sequence[GradeSubjects]:
        """Every rung of the year that teaches something, and what it teaches.

        An unknown year is a refusal, not an empty board — the same rule the year's
        other listings follow, and for the same reason: a typo and a year nobody has
        set a timetable for look identical once the answer is `[]`.
        """
        with self._uow_factory() as uow:
            if uow.academic_years.get(academic_year_code) is None:
                raise UnknownReference(
                    f"no academic year {academic_year_code}", field="academic_year_code"
                )
            return uow.subjects.assignments_for_year(academic_year_code)

    def set_subject_assignment(
        self,
        academic_year_code: AcademicYearCode,
        subject_code: SubjectCode,
        year_level_code: YearCode,
        *,
        assigned: bool,
    ) -> bool:
        """Assign or un-assign one subject on one rung; `True` when this call assigned it.

        The school comes from the year rather than from the caller. It is the one fact
        that keeps the pair honest — a rung is only ever a rung *of a school*, and letting
        a client name the school here would let it pair the annexe's Secondary 1 with the
        main school's Physics.

        Both directions are idempotent, so a board that drops a subject twice, or fires a
        remove while a slow refresh is still in flight, does not have to distinguish
        "already true" from "failed".
        """
        with self._uow_factory() as uow:
            year = uow.academic_years.get(academic_year_code)
            if year is None:
                raise UnknownReference(
                    f"no academic year {academic_year_code}", field="academic_year_code"
                )
            if assigned:
                created = uow.subjects.assign_to_level(
                    subject_code, academic_year_code, year.school_code, year_level_code
                )
            else:
                uow.subjects.unassign_from_level(
                    subject_code, academic_year_code, year.school_code, year_level_code
                )
                created = False
            uow.commit()
            return created

    def create_class_section(self, section: ClassSection) -> bool:
        """Add one class to a year; `True` when this call created it.

        The generator exists for building a whole year at once and is the right tool for
        September. This is the other half of the job: the extra section a school opens in
        November because an intake arrived, which the generator cannot express without
        being told to rebuild the entire ladder — and which it would then report as
        forty-one items already present and one created.

        An upsert on `(year, level, code)`, so re-posting `3C` corrects its labels instead
        of failing, for the same reason terms and subjects upsert.
        """
        with self._uow_factory() as uow:
            created = uow.class_sections.upsert_many([section])
            uow.commit()
        return bool(created.get(section.identity, False))

    def rename_class_section(
        self,
        academic_year_code: AcademicYearCode,
        code: ClassCode,
        *,
        name_en: str | None = None,
        name_ar: str | None = None,
    ) -> ClassSection:
        """Relabel one class. The code cannot be reached from here — invariant 6.

        Renaming "3A" to "Falcons" is a label edit that detaches no student and no grade,
        and the repository enforces that by updating only the two name columns.
        """
        with self._uow_factory() as uow:
            section = uow.class_sections.rename(
                academic_year_code, code, name_en=name_en, name_ar=name_ar
            )
            uow.commit()
        return section
    def create_school(
        self, school: School, *, stated: Collection[str] = ()
    ) -> tuple[School, bool, tuple[TermPlan, ...]]:
        """Store one school; the row as written, and `True` when this call created it.

        The outermost act in the service, and an upsert like every other structural write:
        re-posting a code corrects its labels and detaches nothing, because everything below
        a school points at its surrogate id and not at the name on the sign.

        There is no delete. Closing a branch is `is_active: false`, and the RESTRICT on every
        year and rung pointing at it means the database refuses the alternative anyway — the
        registers taken and marks stated in the years it ran are still true.

        `stated` names the configuration fields the caller actually sent, and everything it
        does not name is carried over from the school already on file. That is what keeps the
        upsert safe now that a school carries a configuration as well as labels: closing a
        branch is a POST that says `is_active: false` and nothing about stages or terms, and
        without the carry-over it would quietly reset a primary-only school to the four
        stages, two terms and five working days that a NEW school defaults to. The read and
        the write share one unit of work, so nothing can land between them.

        **Changing the term count re-syncs the school's years**, and only then. That is what
        "the year structure follows the school's term count" has to mean once a school can
        be edited: a school moved from two terms to three has years on file that still hold
        two, and leaving them is a configuration screen that says three and a year screen
        that shows two. The sync is the same one year creation runs, so a term holding marks
        is kept rather than deleted — the returned plans say which. Nothing happens at all
        when the count is unchanged, so an ordinary rename does not walk the year list.
        """
        carry_over = SCHOOL_CONFIGURATION - set(stated)
        with self._uow_factory() as uow:
            existing = uow.schools.get(school.code)
            if carry_over and existing is not None:
                school = replace(
                    school, **{name: getattr(existing, name) for name in carry_over}
                )
            created = uow.schools.upsert_many([school])
            uow.schools.sync_tracks(school)
            plans: tuple[TermPlan, ...] = ()
            if existing is not None and existing.term_count != school.term_count:
                plans = tuple(
                    self._sync_terms(uow, year)
                    for year in uow.academic_years.list_all(school.code)
                )
            uow.commit()
        return school, bool(created.get(str(school.code), False)), plans

    def list_school_tracks(self, school_code: SchoolCode) -> list[AcademicTrack]:
        with self._uow_factory() as uow:
            if uow.schools.get(school_code) is None:
                raise UnknownReference(f"no school {school_code}", field="school_code")
            return list(uow.schools.list_tracks(school_code))

    def configured_grades(self, school_code: SchoolCode, track_code: str) -> list[dict[str, object]]:
        with self._uow_factory() as uow:
            school = uow.schools.get(school_code)
            if school is None:
                raise UnknownReference(f"no school {school_code}", field="school_code")
            tracks = {track.code for track in uow.schools.list_tracks(school_code)}
            track = track_code.strip().upper()
            if track not in tracks:
                raise ValidationError(f"track {track!r} is not active", field="track_code")
            prefixes = ((Stage.GARDEN, "KG", "KG"), (Stage.PRIMARY, "P", "Primary"),
                        (Stage.PREPARATORY, "PREP", "Preparatory"),
                        (Stage.SECONDARY, "SEC", "Secondary"))
            return [
                {"code": f"{track}-{prefix}{number}", "stage": stage.value,
                 "name_en": f"{label} {number}", "name_ar": f"{label} {number}",
                 "display_order": stage.order * 10 + number}
                for stage, prefix, label in prefixes
                for number in range(1, school.grade_count_for(stage) + 1)
            ]

    def create_configured_classes(
        self, academic_year_code: AcademicYearCode, track_code: str,
        counts: dict[str, int] | None, sequence: str, same_count: int | None = None,
    ) -> tuple[list[YearLevel], list[ClassSection]]:
        with self._uow_factory() as uow:
            year = uow.academic_years.get(academic_year_code)
            if year is None:
                raise UnknownReference(f"no academic year {academic_year_code}", field="academic_year_code")
            specs = self.configured_grades(year.school_code, track_code)
            expected = {str(spec["code"]) for spec in specs}
            if same_count is not None:
                counts = {code: same_count for code in expected}
            counts = counts or {}
            if set(counts) != expected:
                raise ValidationError("class counts must name every active grade and no others", field="classes_by_grade")
            if sequence not in {"numeric", "alphabetic"}:
                raise ValidationError("sequence must be numeric or alphabetic", field="sequence")
            for count in counts.values():
                if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 60:
                    raise ValidationError("class count must be between 0 and 60", field="classes_by_grade")
            levels = [YearLevel(school_code=year.school_code, track_code=track_code,
                code=str(spec["code"]), stage=str(spec["stage"]), name_en=str(spec["name_en"]),
                name_ar=str(spec["name_ar"]), display_order=int(spec["display_order"])) for spec in specs]
            uow.year_levels.upsert_many(levels)
            sections = []
            for level in levels:
                for index in range(1, counts[str(level.code)] + 1):
                    suffix = str(index) if sequence == "numeric" else chr(64 + index)
                    sections.append(ClassSection(code=f"{level.code}-{suffix}",
                        academic_year_code=academic_year_code, year_level_code=level.code,
                        name_en=suffix, name_ar=suffix))
            uow.class_sections.upsert_many(sections)
            uow.commit()
            return levels, sections

    def create_year_level(self, level: YearLevel) -> bool:
        """Add or relabel one rung of one school's ladder; `True` when created.

        The generator builds a whole ladder and is right for a new school. This is the rung
        added afterwards — a school opening a kindergarten, or classifying a rung into a
        stage it had left unspecified — and it is an upsert so the second act is the same
        call as the first.
        """
        with self._uow_factory() as uow:
            # A rung can only sit on a stage the school runs. The school states its stages
            # once, at creation; without this, a primary-only branch could still be given a
            # KG rung and the ladder would claim a stage the school does not teach.
            school = uow.schools.get(level.school_code)
            if school is None:
                raise UnknownReference(f"no school {level.school_code}", field="school_code")
            if not school.allows_stage(level.stage):
                raise ValidationError(
                    f"{level.stage.value} is not enabled for school {level.school_code}",
                    field="stage",
                )
            tracks = uow.schools.list_tracks(level.school_code)
            if level.track_code is None:
                # Old clients predate tracks. Keep their writes deterministic by choosing
                # Arabic first on a dual-track school; the Stage 3 UI always sends the
                # visibly selected track and new integrations can do the same.
                if tracks:
                    level = replace(level, track_code=tracks[0].code)
            elif level.track_code not in {track.code for track in tracks}:
                raise ValidationError(
                    f"track {level.track_code!r} is not active for school {level.school_code}",
                    field="track_code",
                )
            # The count selected when the school was created is both an enable switch and
            # the size of that stage.  Enforcing only the switch would let a school saved
            # with four primary grades acquire a fifth rung later, making the creation
            # settings untrue.  Relabelling an existing rung remains legal and changing its
            # stage consumes one place in the destination stage.
            existing = uow.year_levels.get(level.code, level.school_code)
            stage_levels = [
                item
                for item in uow.year_levels.list_for_school(level.school_code)
                if item.stage is level.stage
                and item.track_code == level.track_code
                and (existing is None or item.code != existing.code)
            ]
            allowed = school.grade_count_for(level.stage)
            if level.stage is not Stage.UNSPECIFIED and len(stage_levels) >= allowed:
                raise ValidationError(
                    f"{level.stage.value} is limited to {allowed} grade(s) for school "
                    f"{level.school_code}",
                    field="stage",
                )
            created = uow.year_levels.upsert_many([level])
            uow.commit()
        return bool(created.get(str(level.code), False))

    def create_academic_year(
        self, year: AcademicYear, *, make_current: bool
    ) -> tuple[bool, TermPlan]:
        """Store the year and bring its term sections in line; `True` when newly created.

        Every other structural row hangs off an academic year, and `generate` refuses
        without one -- so until this existed the whole structure workflow was unreachable
        from the UI, which had a "Create a new academic year" form posting at a route
        nobody had written.

        `make_current` is applied in the same transaction as the upsert. Two writes would
        leave a window in which the year exists and nothing is current, and the class
        dropdowns read the current year.

        **The terms are built here, in the same transaction, from the school's own term
        count.** A year that exists with no terms is a year nothing can be graded against,
        and asking a registrar to create Term 1, Term 2 and Term 3 by hand — right after
        answering "how many terms does this school run?" — is asking the same question
        twice and letting the two answers disagree. `_sync_terms` is idempotent, so
        re-posting a year to fix its Arabic name does not disturb the terms underneath it.
        """
        with self._uow_factory() as uow:
            created = uow.academic_years.upsert_many([year])
            if make_current:
                uow.academic_years.set_current(year.code)
            plan = self._sync_terms(uow, year)
            uow.commit()
        return bool(created.get(str(year.code), False)), plan

    @staticmethod
    def _sync_terms(uow: UnitOfWork, year: AcademicYear) -> TermPlan:
        """Make the year hold exactly the term sections its school says it runs.

        Three rules, and the second and third are the ones that matter:

        **Missing terms are created, undated.** `term_code_for` derives the code from the
        year, so a second run recognises what a first run made instead of creating a
        parallel set beside it. The dates are left `None` because nobody has been asked
        for them yet — that is the whole reason revision 0011 made them optional, and a
        default of "the year's own dates" would be three terms all claiming the full year.

        **A surplus term is only removed if nothing is stated against it.** Dropping a
        school from three terms to two must never take a term of marks with it, so the
        delete is conditional in SQL (`delete_if_unused`) and a term holding grades is
        kept and reported. The year then honestly shows three terms and the caller can
        say why.

        **Labels and dates already on file are never rewritten.** Only the terms this run
        creates get the default names; a term a registrar has renamed to "First Semester",
        or dated, keeps both. Upserting all of them would silently undo that edit on every
        subsequent save of the year.
        """
        school = uow.schools.get(SchoolCode(str(year.school_code)))
        if school is None:
            raise UnknownReference(
                f"no school {year.school_code}", field="school_code"
            )
        wanted = school.term_count
        existing = {term.sequence: term for term in uow.terms.list_for_year(year.code)}

        missing = [
            Term(
                code=term_code_for(str(year.code), sequence),
                academic_year_code=year.code,
                name_en=TERM_LABELS[sequence][0],
                name_ar=TERM_LABELS[sequence][1],
                sequence=sequence,
            )
            for sequence in range(1, wanted + 1)
            if sequence not in existing
        ]
        if missing:
            uow.terms.upsert_many(missing)

        removed: list[str] = []
        kept: list[str] = []
        for sequence, term in sorted(existing.items()):
            if sequence <= wanted:
                continue
            code = str(term.code)
            (removed if uow.terms.delete_if_unused(term.code) else kept).append(code)

        return TermPlan(
            academic_year_code=str(year.code),
            term_count=wanted,
            created=tuple(str(term.code) for term in missing),
            removed=tuple(removed),
            kept=tuple(kept),
        )

    def academic_year_detail(self, academic_year_code: AcademicYearCode) -> dict[str, object]:
        """One year and everything it is attached to, in a single read.

        The year has never been a standalone row — it belongs to a school, its rungs belong
        to academic tracks, and its classes hang off those rungs — but nothing said so in
        one place, so a screen wanting to show it issued four requests and stitched them
        together. Between the first and the last another registrar can add a rung, and the
        stitched picture is then of a school that never existed at any instant.

        Grouped by **track**, because that is the axis a bilingual school reads its year
        along: the Arabic section and the Languages section run the same year and share its
        terms, and share nothing else. A school with one track gets one group and a school
        whose rungs predate its tracks gets an unnamed group for them, rather than having
        them silently dropped.
        """
        with self._uow_factory() as uow:
            year = uow.academic_years.get(academic_year_code)
            if year is None:
                raise UnknownReference(
                    f"no academic year {academic_year_code}", field="academic_year_code"
                )
            school = uow.schools.get(SchoolCode(str(year.school_code)))
            if school is None:
                raise UnknownReference(
                    f"no school {year.school_code}", field="school_code"
                )
            terms = list(uow.terms.list_for_year(year.code))
            levels = list(uow.year_levels.list_for_school(year.school_code))
            sections = list(uow.class_sections.list_for_year(year.code))
            tracks = list(uow.schools.list_tracks(year.school_code))

        classes_per_level: dict[str, int] = {}
        for section in sections:
            key = str(section.year_level_code)
            classes_per_level[key] = classes_per_level.get(key, 0) + 1

        # `None` keys the group for rungs that belong to no track. Kept rather than
        # dropped: a rung nobody has placed in a section still teaches children.
        groups: dict[str | None, list[object]] = {}
        for level in levels:
            groups.setdefault(level.track_code, []).append(level)

        ordered: list[str | None] = [track.code for track in tracks if track.code in groups]
        ordered += [code for code in groups if code not in ordered]

        by_track = {track.code: track for track in tracks}
        return {
            "year": year,
            "school": school,
            "terms": terms,
            "tracks": [
                {
                    "track": by_track.get(code),
                    "track_code": code,
                    "levels": groups[code],
                    "classes_per_level": classes_per_level,
                }
                for code in ordered
            ],
            "class_count": len(sections),
        }

    def sync_year_terms(self, academic_year_code: AcademicYearCode) -> TermPlan:
        """Re-run the term sync for one year that is already on file.

        Exists so a school's term count can change after its years were built: the school
        write below calls this for each of the school's years, and the API exposes it so a
        registrar can see the effect without having to re-save the year.
        """
        with self._uow_factory() as uow:
            year = uow.academic_years.get(academic_year_code)
            if year is None:
                raise UnknownReference(
                    f"no academic year {academic_year_code}", field="academic_year_code"
                )
            plan = self._sync_terms(uow, year)
            uow.commit()
        return plan




class StudentDesk:
    """Single-student writes: the registrar correcting one child's file by hand.

    Every write in this service used to arrive as a spreadsheet. That is right for a
    September roster of nine hundred children and absurd for the two cases that actually
    fill a registrar's day — a misspelt name, and one child arriving in November — where it
    means building a one-row .xlsx, previewing it, and committing a batch to change a
    letter. This class is the direct path for those, and it deliberately does not replace
    the import: an import still owns anything touching more than one child, because that is
    where a per-row report earns its keep.

    What is preserved is the part that matters. Placement is still a dated membership
    (invariant 2), so a transfer is the open placement ended and a new one opened, never a
    class code rewritten in place — the repository will not do the latter, and this service
    does not ask it to.
    """

    def __init__(self, uow_factory: Callable[[], UnitOfWork]) -> None:
        self._uow_factory = uow_factory

    def save_student(self, student: Student) -> bool:
        """Create or correct one child by student number; `True` when created.

        An upsert, so the form that adds a child and the form that fixes her name are the
        same call. The repository leaves rows whose values already match out of its UPDATE,
        so saving a form nobody edited does not stamp `updated_at` and does not lie about
        when her details last changed.
        """
        with self._uow_factory() as uow:
            created = uow.students.upsert_many([student])
            uow.commit()
        return bool(created.get(str(student.student_number), False))

    def set_student_active(
        self, student_number: StudentNumber, *, is_active: bool
    ) -> Student:
        """Mark a child as having left the school, or having come back.

        There is no delete, and there should not be: her marks, her placements and her
        guardians are all still true statements about a term that happened. `is_active`
        takes her out of the pickers and leaves the record standing.
        """
        with self._uow_factory() as uow:
            student = uow.students.set_active(student_number, is_active=is_active)
            uow.commit()
        return student

    def place_student(self, enrolment: ClassEnrolment) -> bool:
        """Open a placement. `True` when this call created it.

        Refuses nothing itself: the partial unique index means a second *open* placement
        for the same child is rejected by the database, which is the guarantee worth having
        — it holds against two registrars clicking at once, and a check in Python would not.
        """
        with self._uow_factory() as uow:
            created = uow.enrolments.upsert_many([enrolment])
            uow.commit()
        return bool(next(iter(created.values()), False))

    def end_placement(
        self, student_number: StudentNumber, *, ends_on: date
    ) -> ClassEnrolment | None:
        """Close the child's open placement on her last day; `None` if she had none.

        `ends_on` is her **last day in the class**, not the day after. The distinction is
        the one thing about this route worth getting right: off by one, and a report card
        for the term that ended that week resolves to the wrong class.
        """
        with self._uow_factory() as uow:
            closed = uow.enrolments.close_open_enrolment(student_number, ends_on=ends_on)
            uow.commit()
        return closed

    def transfer_student(
        self,
        student_number: StudentNumber,
        *,
        academic_year_code: AcademicYearCode,
        to_class: ClassCode,
        on_date: date,
    ) -> tuple[ClassEnrolment | None, ClassEnrolment]:
        """Move a child to another class from `on_date`, in one transaction.

        This is the whole reason a transfer is not two API calls. Between "end 3A" and
        "start 3B" the child is in no class at all, and a marks upload landing in that
        window resolves no placement and rejects every one of her rows. One transaction, or
        a registrar's afternoon spent explaining why a child vanished from the register.

        The old placement ends the day *before* she starts in the new class, so the two
        windows do not both contain `on_date` — two placements covering the same day is
        exactly what `resolve_section_for_term` cannot answer, and it would make her Term
        marks ambiguous rather than wrong, which is harder to notice.
        """
        opened = ClassEnrolment(
            student_number=student_number,
            academic_year_code=academic_year_code,
            class_code=to_class,
            starts_on=on_date,
        )
        with self._uow_factory() as uow:
            closed = uow.enrolments.close_open_enrolment(
                student_number, ends_on=on_date - timedelta(days=1)
            )
            uow.enrolments.upsert_many([opened])
            uow.commit()
        return closed, opened



class ApiKeyMinter:
    """Generate a secret, hash it and store the record — one act, one transaction.

    Split across two calls there is a window where the secret exists in a response body
    and not in the database, or in the database under the hash of a different string. Both
    end the same way: an integration holding a key that has never worked, and nobody able
    to say why. The raw secret is returned to exactly one caller and stored nowhere.
    """

    def __init__(self, uow_factory: Callable[[], UnitOfWork]) -> None:
        self._uow_factory = uow_factory

    def mint(
        self, *, label: str, scope: Scope, expires_in_days: int | None = None
    ) -> tuple[ApiKey, str]:
        raw, prefix, key_hash = generate_api_key()
        now = datetime.now(UTC)
        key = ApiKey(
            prefix=prefix,
            key_hash=key_hash,
            label=label,
            scope=scope,
            is_active=True,
            # `None` means no expiry, which is a key revoked deliberately rather than one
            # that stops working under somebody at 3am.
            expires_at=(
                None if expires_in_days is None else now + timedelta(days=expires_in_days)
            ),
            created_at=now,
        )
        with self._uow_factory() as uow:
            stored = uow.api_keys.add(key)
            uow.commit()
        return stored, raw


def get_import_reports(uow_factory: UowFactoryDep) -> ImportReports:
    """Read-only; each report is its own transaction."""
    return ImportReports(uow_factory)


def get_structure_catalogue(uow_factory: UowFactoryDep) -> StructureCatalogue:
    """Term and subject upserts, committed per call."""
    return StructureCatalogue(uow_factory)


def get_today() -> date:
    """The calendar day the register rules are judged against.

    A dependency of its own rather than a `datetime.now` inside the service, so a test can
    pin it. It is deliberately NOT folded into `get_attendance_service`: overriding the
    service would swap out the wiring under test, and the point is to move the clock while
    everything else stays the code that actually ships.
    """
    return datetime.now(UTC).date()


TodayDep = Annotated[date, Depends(get_today)]


def get_attendance_service(
    uow_factory: UowFactoryDep, today: TodayDep
) -> AttendanceService:
    """The daily register. A factory, like every other service here."""
    return AttendanceService(uow_factory, today=lambda: today)


def get_student_desk(uow_factory: UowFactoryDep) -> StudentDesk:
    """Single-student and single-placement writes, committed per call."""
    return StudentDesk(uow_factory)


def get_api_key_minter(uow_factory: UowFactoryDep) -> ApiKeyMinter:
    """The only path that creates a credential."""
    return ApiKeyMinter(uow_factory)


# ---------------------------------------------------------------------------
# Annotated aliases. Routers depend on these names, not on the functions above.
# ---------------------------------------------------------------------------

RegistrarCaller = Annotated[Caller, Depends(require_registrar)]
ReaderCaller = Annotated[Caller, Depends(require_reader)]
ReadCaller = Annotated[Caller, Depends(require_read_access)]

UnitOfWorkDep = Annotated[UnitOfWork, Depends(get_unit_of_work)]
UnitOfWorkFactoryDep = Annotated[Callable[[], UnitOfWork], Depends(get_unit_of_work_factory)]
MaxUploadBytesDep = Annotated[int, Depends(get_max_upload_bytes)]

ApiKeyMinterDep = Annotated[ApiKeyMinter, Depends(get_api_key_minter)]
ImportReportsDep = Annotated[ImportReports, Depends(get_import_reports)]
StructureCatalogueDep = Annotated[StructureCatalogue, Depends(get_structure_catalogue)]
StudentDeskDep = Annotated[StudentDesk, Depends(get_student_desk)]
AttendanceServiceDep = Annotated[AttendanceService, Depends(get_attendance_service)]

QueryServiceDep = Annotated[QueryService, Depends(get_query_service)]
StructureServiceDep = Annotated[StructureGenerationService, Depends(get_structure_service)]
RosterImportServiceDep = Annotated[RosterImportService, Depends(get_roster_import_service)]
GradeImportServiceDep = Annotated[GradeImportService, Depends(get_grade_import_service)]
GuardianImportServiceDep = Annotated[
    GuardianImportService, Depends(get_guardian_import_service)
]


__all__ = [
    "API_KEY_HEADER",
    "ApiKeyMinter",
    "RequestId",
    "REQUEST_ID_HEADER",
    "ApiKeyMinterDep",
    "Caller",
    "GradeImportServiceDep",
    "GuardianImportServiceDep",
    "ImportReports",
    "ImportReportsDep",
    "MaxUploadBytesDep",
    "QueryServiceDep",
    "ReadCaller",
    "ReaderCaller",
    "RegistrarCaller",
    "RosterImportServiceDep",
    "StructureCatalogue",
    "StructureCatalogueDep",
    "StructureServiceDep",
    "UnitOfWorkDep",
    "UnitOfWorkFactoryDep",
    "generate_api_key",
    "get_api_key_minter",
    "get_grade_import_service",
    "get_guardian_import_service",
    "get_import_reports",
    "get_max_upload_bytes",
    "get_query_service",
    "get_roster_import_service",
    "get_structure_catalogue",
    "get_structure_service",
    "get_unit_of_work",
    "get_unit_of_work_factory",
    "hash_api_key",
    "key_prefix",
    "require_read_access",
    "require_reader",
    "require_registrar",
]
