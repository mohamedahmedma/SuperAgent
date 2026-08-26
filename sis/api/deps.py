"""Request-scoped wiring: who is calling, which transaction, and which use case.

Composition happens here. A router declares `caller: RegistrarCaller` and
`service: GradeImportServiceDep` and receives objects that are already authenticated and
already bound to a transaction; it never constructs a unit of work, never reads
`sis.config`, and never learns which repository implementation it is writing through.
That is what makes every service in this package testable with fakes: the only place
that knows an environment and a database exist is this file, and a test replaces it
wholesale through `app.dependency_overrides`.

**Two scopes, compared by exact equality.** `registrar` does not implicitly satisfy
`reader` — see `Scope.permits` for the incident that rule prevents. A route that
legitimately accepts either lists both, out loud, via `require_read_access`.

**A key is never logged, echoed or returned.** What may appear in a log line is the
`prefix`, which names a key so an operator can revoke the right one and is useless for
authenticating with it. The secret itself exists in this module only as a local variable
and a SHA-256 digest, and no error message ever quotes it back — an unauthenticated caller
learns that authentication failed and nothing further, because "your key is almost right"
is the one hint worth having.

**A key is found in the school it was minted for.** Authentication opens the same
per-school connection every other read uses, so a key issued at one branch does not exist
in another's database and cannot be made to work there by supplying its `X-School-Code`.
Physical separation reaches authentication rather than stopping just short of it.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
from collections.abc import Callable, Collection, Iterator
from dataclasses import dataclass
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
from sis.config import get_settings
from sis.domain.auth import PREFIX_LENGTH, ApiKey, Scope
from sis.domain.errors import ImportBatchNotFound, ValidationError
from sis.domain.imports import ImportBatch, ImportRow, RowOutcome
from sis.domain.people import ClassEnrolment, Student
from sis.domain.structure import (
    AcademicYear,
    ClassSection,
    School,
    Subject,
    Term,
    YearLevel,
)
from sis.domain.value_objects import (
    AcademicYearCode,
    ClassCode,
    SchoolCode,
    StudentNumber,
)
from sis.infrastructure.db.unit_of_work import SqlAlchemyUnitOfWork
from sis.infrastructure.parsers import (
    SpreadsheetGradeParser,
    SpreadsheetGuardianParser,
    SpreadsheetRosterParser,
)
from sis.tenancy import get_registry

logger = logging.getLogger(__name__)

_KEY_BYTES: Final[int] = 32

API_KEY_HEADER: Final[str] = "X-API-Key"

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


def _unauthorised(message: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"code": "not_authorized", "message": message},
        headers={"WWW-Authenticate": API_KEY_HEADER},
    )


def _matches_bootstrap(presented: str, configured: str | None) -> bool:
    """Constant-time comparison against `SIS_BOOTSTRAP_REGISTRAR_KEY`.

    Solves the chicken-and-egg of needing a registrar key to mint a registrar key. The
    value is compared, never stored: an empty database plus one environment variable is a
    working registrar, and unsetting the variable revokes it without a migration.

    Estate-wide rather than per-school, and deliberately so — it is the credential that
    exists before any school's database has a key in it. That is also why it should not
    survive first setup: see the warning `sis.app` logs while it is set.

    Encoded to bytes first because `secrets.compare_digest` raises `TypeError` on a `str`
    holding non-ASCII — and the presented value is a header a client controls completely,
    so the str form turns one hostile request into a 500.
    """
    if not configured:
        return False
    return secrets.compare_digest(presented.encode("utf-8"), configured.encode("utf-8"))


def _lookup_stored_key(presented: str, school_code: str | None) -> ApiKey | None:
    """Find and verify a stored key, in its own short transaction.

    **Scoped to the school the request names.** Keys live in the per-school database like
    every other row, so a key minted for one branch is not found in another's file at all.
    That is physical separation reaching authentication rather than stopping just short of
    it: a leaked key from one school cannot read a second school's data even if the
    attacker supplies its `X-School-Code`, because there is nothing to match against.

    Deliberately not the caller's unit of work. Authentication runs before the handler has
    written anything and must not be able to see — or contaminate — the transaction the
    handler will later roll back. The session is closed before the route body starts, so a
    request rejected here never holds a pooled connection for the length of the response.
    """
    with SqlAlchemyUnitOfWork(school_code=school_code) as uow:
        stored = uow.api_keys.get_by_prefix(key_prefix(presented))

    # Hash unconditionally, even when no row matched, so a wrong prefix and a wrong secret
    # do the same work. Skipping the comparison on a miss turns prefix enumeration into a
    # timing measurement, which is how an attacker learns which handles are real before
    # bothering to guess a secret.
    candidate = hash_api_key(presented)
    expected = stored.key_hash if stored is not None else _ABSENT_HASH
    if not hmac.compare_digest(candidate, expected):
        return None
    return stored


def _record_use(prefix: str, at: datetime, school_code: str | None) -> None:
    """Best effort, by contract. A failed touch must never fail an authorised request."""
    try:
        with SqlAlchemyUnitOfWork(school_code=school_code) as uow:
            uow.api_keys.touch(prefix, at=at)
            uow.commit()
    except Exception:  # noqa: BLE001 - the request is already authorised; do not fail it
        # The prefix identifies the key; it is not the secret. See the module docstring.
        logger.warning("Could not record last use of API key %s", prefix)


def _check_scope(caller: Caller, allowed: tuple[Scope, ...], required: str) -> None:
    if any(caller.scope.permits(scope) for scope in allowed):
        return
    logger.info(
        "Key %s refused: scope %s cannot call this route", caller.prefix, caller.scope.value
    )
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail={
            "code": "not_authorized",
            "message": f"This endpoint requires an API key with scope {required}.",
        },
    )


def _require_scopes(*allowed: Scope) -> Callable[..., Caller]:
    """Build the dependency that admits exactly the listed scopes and nothing else.

    Every allowed scope is spelled out at the call site. There is no ordering, no superset
    and no "registrar can do anything" shortcut, so reading a route's signature tells you
    precisely which credentials reach it.
    """

    allowed_scopes: tuple[Scope, ...] = allowed
    required = " or ".join(scope.value for scope in allowed_scopes)

    def dependency(
        school_code: SchoolCodeDep,
        x_api_key: Annotated[str | None, Header(alias=API_KEY_HEADER)] = None,
    ) -> Caller:
        presented = (x_api_key or "").strip()
        if not presented:
            raise _unauthorised(f"Missing {API_KEY_HEADER} header.")

        now = datetime.now(UTC)
        settings = get_settings()

        if _matches_bootstrap(presented, settings.bootstrap_registrar_key):
            caller = Caller(
                prefix=key_prefix(presented), scope=Scope.REGISTRAR, is_bootstrap=True
            )
            _check_scope(caller, allowed_scopes, required)
            return caller

        stored = _lookup_stored_key(presented, school_code)
        # Unknown prefix, wrong secret, revoked and expired all answer identically. A
        # caller who can tell them apart can enumerate which handles exist and learn that
        # a leaked key was noticed, both from error text alone.
        if stored is None or not stored.is_usable_at(now):
            raise _unauthorised("Invalid or expired API key.")

        caller = Caller(prefix=stored.prefix, scope=stored.scope)
        _check_scope(caller, allowed_scopes, required)
        _record_use(stored.prefix, now, school_code)
        return caller

    return dependency


require_registrar = _require_scopes(Scope.REGISTRAR)
"""Writes: structure generation, imports, key management."""

require_reader = _require_scopes(Scope.READER)
"""Read-only integrations. A registrar key is refused here, on purpose."""

require_read_access = _require_scopes(Scope.REGISTRAR, Scope.READER)
"""Both scopes, granted explicitly, for reads a registrar also legitimately performs.

This is what the parent-facing guardian routes use. `records/` holds a `reader` key and
the registrar console holds a `registrar` one, and both legitimately read a report card.
"""


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
        SpreadsheetRosterParser(),
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
    def create_school(self, school: School) -> bool:
        """Store one school; `True` when this call created it.

        The outermost act in the service, and an upsert like every other structural write:
        re-posting a code corrects its labels and detaches nothing, because everything below
        a school points at its surrogate id and not at the name on the sign.

        There is no delete. Closing a branch is `is_active: false`, and the RESTRICT on every
        year and rung pointing at it means the database refuses the alternative anyway — the
        registers taken and marks stated in the years it ran are still true.
        """
        with self._uow_factory() as uow:
            created = uow.schools.upsert_many([school])
            uow.commit()
        return bool(created.get(str(school.code), False))

    def create_year_level(self, level: YearLevel) -> bool:
        """Add or relabel one rung of one school's ladder; `True` when created.

        The generator builds a whole ladder and is right for a new school. This is the rung
        added afterwards — a school opening a kindergarten, or classifying a rung into a
        stage it had left unspecified — and it is an upsert so the second act is the same
        call as the first.
        """
        with self._uow_factory() as uow:
            created = uow.year_levels.upsert_many([level])
            uow.commit()
        return bool(created.get(str(level.code), False))

    def create_academic_year(self, year: AcademicYear, *, make_current: bool) -> bool:
        """Store the year; `True` when this call created it.

        Every other structural row hangs off an academic year, and `generate` refuses
        without one -- so until this existed the whole structure workflow was unreachable
        from the UI, which had a "Create a new academic year" form posting at a route
        nobody had written.

        `make_current` is applied in the same transaction as the upsert. Two writes would
        leave a window in which the year exists and nothing is current, and the class
        dropdowns read the current year.
        """
        with self._uow_factory() as uow:
            created = uow.academic_years.upsert_many([year])
            if make_current:
                uow.academic_years.set_current(year.code)
            uow.commit()
        return bool(created.get(str(year.code), False))




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


def get_attendance_service(uow_factory: UowFactoryDep) -> AttendanceService:
    """The daily register. A factory, like every other service here."""
    return AttendanceService(uow_factory)


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
