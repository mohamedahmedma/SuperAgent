"""Persistence interfaces, declared by the layer that *uses* them.

These Protocols live in `application/` and not in `infrastructure/` on purpose: the use
cases own the shape of the storage they need, and the SQLAlchemy repositories are
written to fit. Inverted the other way — services importing concrete repositories — a
unit test of "does a transfer close the old placement" needs a database, an engine and a
migration, so the test that should take a millisecond takes a fixture, and the one that
should assert a rule ends up asserting SQL.

`Protocol` rather than an abstract base class, because structural typing means a fake
repository in a test is a plain class with the right methods — it does not import this
module, does not inherit from anything, and cannot be broken by a base class gaining a
method it does not use. The type checker still catches an implementation that drifts.

Nothing here mentions a session, a transaction or a query. The transaction boundary
belongs to whoever composes the request; a repository that commits on its own turns an
import commit into "half the roster landed, then a row failed".

Three conventions run through every Protocol, so that implementers do not have to guess:

**Bulk is the default, not an optimisation.** A roster import carries hundreds of rows,
and a repository that exposes only `get`/`save` guarantees a service will loop over them:
one SELECT and one INSERT per child, a thousand round trips for one upload, and a
registrar watching a spinner. Every `*_many` method here is one statement's worth of
work, and the plural lookups exist so a service can load everything it needs to validate
a file before it writes anything.

**Every `upsert_many` returns `Mapping[key, bool]`, `True` meaning the row was created
and `False` meaning it already existed and was updated.** That single return type is what
makes invariant 3 both true and reportable: re-running "5 years x 8 classes" writes
nothing new and the caller can still say "40 already present" without re-reading the
table, and a per-year run ("year 1 has 3, year 2 has 5") reports through the same path.

**Keys in returned mappings are normalised code strings — `str(code)`, never the value
object.** A caller holding a `SubjectCode` and a caller holding the string it came from
must be able to look up the same entry, and `Mapping[SubjectCode, ...]` forces the second
one to rebuild a value object purely to index a dict. Arguments stay strict: a repository
takes `SubjectCode`, so the parsing failure surfaces at the boundary where the cell was
read rather than as an empty result three layers down.
"""
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol

from sis.domain.attendance import AttendanceMark
from sis.domain.access import AccessAttempt
from sis.domain.auth import ApiKey
from sis.domain.grades import SubjectGrade
from sis.domain.guardians import Guardian, StudentGuardian
from sis.domain.imports import ImportBatch, ImportRow, RowOutcome
from sis.domain.people import ClassEnrolment, Student
from sis.domain.structure import (
    AcademicTrack,
    AcademicYear,
    ClassSection,
    School,
    Subject,
    Term,
    YearLevel,
)
from sis.domain.timetable import TimetableEntry, TimetablePeriod, TimetableSlot
from sis.domain.staff import Teacher
from sis.domain.value_objects import (
    AcademicYearCode,
    ClassCode,
    Phone,
    SchoolCode,
    StudentNumber,
    SubjectCode,
    TermCode,
    YearCode,
)

# `(academic_year_code, class_code)` — `ClassSection.identity`. A class key is a pair
# because `3A` names a different room of children every September.
type ClassSectionKey = tuple[str, str]

# `(student_number, subject_code, term_code)` — `SubjectGrade.identity`, one stated
# figure per child, subject and term.
type GradeKey = tuple[str, str, str]

# `(student_number, academic_year_code, class_code, starts_on)`. The start date is part
# of the key: a child who returns to 3A after a term in 3B has two placements in that
# class, and they are different facts, not a duplicate.
type EnrolmentKey = tuple[str, str, str, date]

@dataclass(frozen=True, slots=True)
class GradeSubjects:
    """One rung and the subjects assigned to it — a read projection, not an entity.

    The track code rides along because every consumer of this list groups by it. It is
    the rung's own track, read through `year_levels.educational_system_id`: an assignment
    row has no track column of its own, and giving it one would let the two disagree.
    """

    year_level_code: str
    track_code: str | None
    subjects: Sequence[Subject]


# `(student_number, guardian_phone)` — `StudentGuardian.identity`. No date in this key,
# unlike `EnrolmentKey`: one adult holds one relationship to one child at a time, so a
# second row for the same pair is a correction of the first rather than a second fact.
type StudentGuardianKey = tuple[str, str]


class SchoolRepository(Protocol):
    """Schools: the outermost scope, and the boundary nothing but a child crosses.

    There is no `delete`. Closing a branch is `is_active = False` through `upsert_many`,
    because the years it ran, the registers taken in them and the marks stated against them
    are all still true — and the RESTRICT on every year and rung pointing at a school means
    the database refuses a delete regardless.
    """

    def get(self, code: SchoolCode) -> School | None:
        """The school, or `None` when no such code is on file."""

    def get_many(self, codes: Collection[SchoolCode]) -> Mapping[str, School]:
        """The schools that exist, keyed by code; absent codes are simply missing."""

    def list_all(self, *, include_inactive: bool = False) -> Sequence[School]:
        """Every school by code; closed branches only when asked for by name."""

    def upsert_many(self, schools: Sequence[School]) -> Mapping[str, bool]:
        """Insert or update by code; `True` marks the ones this call created."""

    def sync_tracks(self, school: School) -> None:
        """Ensure active tracks match the school's selected language type."""


@dataclass(frozen=True, slots=True)
class TeacherTeachingAssignment:
    """One valid subject/grade assignment, with optional concrete classes."""

    academic_year_code: str
    subject_code: str
    year_level_code: str
    track_code: str | None
    class_codes: Sequence[str] = ()


@dataclass(frozen=True, slots=True)
class TeacherRecord:
    teacher: Teacher
    school_code: str
    username: str | None
    email: str
    phone: str
    assignments: Sequence[TeacherTeachingAssignment]


class TeacherRepository(Protocol):
    """Teaching staff, their optional login, and their teaching scope."""

    def list_for_school(
        self, school_code: SchoolCode, *, year_level_code: YearCode | None = None
    ) -> Sequence[TeacherRecord]:
        """The school's teachers, or only those teaching on one grade.

        `year_level_code` narrows the record as well as the list: each returned teacher
        carries only their assignments on that grade. It is what a grade-scoped
        supervisor reads, and the filtering belongs here rather than in the caller so
        that no route can forget half of it.
        """

    def get(
        self,
        school_code: SchoolCode,
        staff_number: str,
        *,
        year_level_code: YearCode | None = None,
    ) -> TeacherRecord | None:
        """One teacher, or `None`. With a grade named, `None` unless they teach on it."""

    def save(
        self,
        *,
        school_code: SchoolCode,
        staff_number: str,
        full_name_en: str,
        full_name_ar: str,
        email: str,
        phone: str,
        is_active: bool,
        username: str | None,
        password_hash: str | None,
        assignments: Sequence[
            tuple[AcademicYearCode, SubjectCode, YearCode, Sequence[ClassCode]]
        ],
        assigned_by: str,
    ) -> TeacherRecord: ...

    def list_tracks(self, school_code: SchoolCode) -> Sequence[AcademicTrack]:
        """The school's active academic tracks, in display order."""


class AttendanceRepository(Protocol):
    """The daily register: one mark per child per day, and no row meaning "unmarked".

    A child with no row for a day is a child nobody marked, which is a different statement
    from every state this port can hold. That is why nothing here has an `unknown` state and
    why every count a caller builds from these reads has to carry how many days it counted:
    a rate divided by "school days" would be divided by a number this service does not hold.
    """

    def marks_for_class(
        self, class_section_id: int, on_date: date
    ) -> Mapping[str, AttendanceMark]:
        """What was recorded for one class on one day, keyed by student number.

        Keyed rather than listed because the caller is merging it into a register built from
        the enrolments: a child absent from this mapping is one nobody marked, and the screen
        must show that as blank rather than as present.
        """

    def marks_for_student(
        self,
        student_number: StudentNumber,
        *,
        from_date: date | None = None,
        to_date: date | None = None,
    ) -> Sequence[AttendanceMark]:
        """One child's marks, oldest first. Both bounds inclusive when given."""

    def marks_for_students(
        self,
        student_numbers: Collection[StudentNumber],
        *,
        from_date: date | None = None,
        to_date: date | None = None,
    ) -> Sequence[AttendanceMark]:
        """The same read for many children in one statement, for a class-wide summary."""

    def upsert_many(
        self, marks: Sequence[AttendanceMark], *, recorded_by: str = ""
    ) -> Mapping[tuple[str, date], bool]:
        """Write a day's register; `True` marks the entries this call created.

        Keyed on `(student number, day)` — the pair that is unique — so taking the register
        twice on one morning corrects it rather than writing a second, contradictory mark.
        """


class AcademicYearRepository(Protocol):
    """The school years everything else hangs from."""

    def get(self, code: AcademicYearCode) -> AcademicYear | None:
        """The year, or `None` when no such code is on file."""

    def get_many(
        self, codes: Collection[AcademicYearCode]
    ) -> Mapping[str, AcademicYear]:
        """The years that exist, keyed by code; absent codes are simply missing."""

    def list_all(self, school_code: SchoolCode | None = None) -> Sequence[AcademicYear]:
        """Every year, most recent start date first; one school's when asked for.

        The filter is optional rather than required because the school picker needs the
        unfiltered list to know which schools have any years at all.
        """

    def current(self, school_code: SchoolCode | None = None) -> AcademicYear | None:
        """The year flagged current, or `None` before a registrar has chosen one.

        Per school when asked: two branches each have a current year and they need not be
        the same one, because a school that starts a fortnight later is still mid-changeover
        while the other has moved on.
        """

    def set_current(self, code: AcademicYearCode) -> AcademicYear:
        """Make this the current year and clear the previous flag in the same write.

        One statement, not two, because between two there is a moment with either no
        current year or two of them, and every "this year's classes" screen answers
        wrongly in that window.
        """

    def upsert_many(
        self, years: Sequence[AcademicYear]
    ) -> Mapping[str, bool]:
        """Insert or update by code; `True` marks the ones this call created."""


class YearLevelRepository(Protocol):
    """The rungs of one school's ladder — `Y1`..`Y6` — shared by every academic year in it.

    Rungs are still not scoped to a *year*, which is the decision `YearLevel`'s docstring
    defends: "Year 3" is the same rung in 2025-2026 as in 2030-2031. They are scoped to a
    school, because "Year 3" at one branch is a different room of children from "Year 3" at
    another, and the two ladders are maintained separately.

    Every method therefore takes the school. Callers always have it: a rung is reached
    through an academic year, and a year names exactly one school.
    """

    def get(self, code: YearCode, school_code: SchoolCode) -> YearLevel | None:
        """The year level in that school, or `None`."""

    def get_many(
        self, codes: Collection[YearCode], school_code: SchoolCode
    ) -> Mapping[str, YearLevel]:
        """Bulk existence check within one school, keyed by code."""

    def list_for_school(self, school_code: SchoolCode) -> Sequence[YearLevel]:
        """One school's ladder, by stage then `display_order`, so `Y10` follows `Y9`."""

    def upsert_many(self, levels: Sequence[YearLevel]) -> Mapping[str, bool]:
        """Insert or update by `(school_code, code)`; `True` marks the new ones."""


class ClassSectionRepository(Protocol):
    """Class sections, unique per `(academic_year_code, code)` and never per term."""

    def get(
        self, academic_year_code: AcademicYearCode, code: ClassCode
    ) -> ClassSection | None:
        """One section within one year, or `None`."""

    def get_many(
        self, keys: Collection[ClassSectionKey]
    ) -> Mapping[ClassSectionKey, ClassSection]:
        """The sections that exist, so a whole roster validates in one query."""

    def list_for_year(
        self,
        academic_year_code: AcademicYearCode,
        *,
        year_level_code: YearCode | None = None,
    ) -> Sequence[ClassSection]:
        """Sections of a year, optionally narrowed to one level, in code order."""

    def upsert_many(
        self, sections: Sequence[ClassSection]
    ) -> Mapping[ClassSectionKey, bool]:
        """The one path structure generation writes through, uniform or per-year alike."""

    def rename(
        self,
        academic_year_code: AcademicYearCode,
        code: ClassCode,
        *,
        name_en: str | None = None,
        name_ar: str | None = None,
    ) -> ClassSection:
        """Change labels only. The code is identity and no method here may rewrite it.

        Invariant 6 as a signature: a rename implemented as "write a new code" leaves
        every enrolment and grade pointing at a code nothing resolves any more, and not
        one of those rows reports an error when it happens.
        """

    def ids_for(
        self, keys: Collection[ClassSectionKey]
    ) -> Mapping[ClassSectionKey, int]:
        """Surrogate ids for the sections named, absent keys omitted.

        `SubjectGrade.class_section_id` stores the surrogate, so a grade import has to
        resolve one per row; doing that per row is the query storm this method exists to
        collapse into a single statement.
        """


class TermRepository(Protocol):
    """Terms of an academic year.

    A term's dates are optional (revision 0011): a school states how many terms it runs
    before it has decided when they run, so `starts_on` and `ends_on` may both be `None`.
    `sequence` is what orders them, and always was.
    """

    def get(self, code: TermCode) -> Term | None:
        """The term, or `None`."""

    def get_many(self, codes: Collection[TermCode]) -> Mapping[str, Term]:
        """Bulk lookup for a grade file that names several terms."""

    def list_for_year(self, academic_year_code: AcademicYearCode) -> Sequence[Term]:
        """The year's terms in `sequence` order."""

    def upsert_many(self, terms: Sequence[Term]) -> Mapping[str, bool]:
        """Insert or update by code; `True` marks the ones this call created."""

    def set_closed(self, code: TermCode, *, is_closed: bool) -> Term:
        """Freeze or reopen a term; writes against a closed one are refused upstream."""

    def delete_if_unused(self, code: TermCode) -> bool:
        """Delete the term only if nothing is stated against it; `True` when deleted.

        The guard is the method. A school dropping from three terms to two must not take a
        term of marks with it, so "is it safe" and "delete it" cannot be two calls with a
        gap between them — another registrar's upload lands in that gap. One statement,
        conditional on the absence of grades, and the answer says which happened.

        A term that is in use is left exactly as it was and reported as `False`. That is a
        normal outcome, not a failure: the year genuinely still has that term.
        """


class TimetableRepository(Protocol):
    """The weekly plan: a school's periods, and one lesson per class per slot.

    One repository rather than two because the two tables are never asked about
    separately — every read of a timetable needs the period grid to draw it against, and
    every write of a lesson has to know the period exists. Splitting them would mean two
    repositories that no caller ever uses alone.

    Nothing here reads or writes attendance. A timetable is a plan; the register is a
    record, and this stage deliberately connects neither to the other.
    """

    def list_periods(self, school_code: SchoolCode) -> Sequence[TimetablePeriod]:
        """The school's day, in period order. Empty until a school lays one out."""

    def replace_periods(
        self, school_code: SchoolCode, periods: Sequence[TimetablePeriod]
    ) -> Sequence[TimetablePeriod]:
        """Set the school's whole day at once, returning it as stored.

        Whole-grid rather than per-period upsert, because "we run seven periods, not
        eight" is one decision and applying it as an upsert plus a guessed delete is how
        period 8 survives on some schools and not others. Removing a period that lessons
        are timetabled into is refused — see `DomainRuleViolation` from the service.
        """

    def list_entries(
        self,
        academic_year_code: AcademicYearCode,
        *,
        class_code: ClassCode | None = None,
        term_code: TermCode | None = None,
        year_level_code: YearCode | None = None,
    ) -> Sequence[TimetableEntry]:
        """The year's lessons, narrowed to one class, one term and/or one grade.

        Ordered by class, then day, then period — the order a grid is read in, so a caller
        can lay one out without sorting. Days sort by the school's own week rather than
        alphabetically; that ordering is applied by the caller that knows the school.
        """

    def upsert_entries(self, entries: Sequence[TimetableEntry]) -> Mapping[tuple, bool]:
        """Insert or update by slot; `True` marks the ones this call created.

        Keyed by `TimetableSlot.key`, because the slot *is* the identity: re-posting
        Sunday period 2 for 3A replaces what was there rather than adding a second lesson
        at the same moment. That is what makes laying out a grid idempotent, and it is the
        same property `uq_timetable_entries_slot` holds in the database.
        """

    def delete_entries(self, slots: Sequence[TimetableSlot]) -> int:
        """Clear these slots; the number of lessons actually removed.

        Clearing a slot is not the same as scheduling nothing into it. A row with no
        subject is a stated free period; no row at all is a slot nobody has planned. Both
        are reachable, and a caller has to be able to say which it means.
        """


class SubjectRepository(Protocol):
    """Subjects, scoped to the academic year that teaches them.

    Every method takes the year, and that is the whole shape of the decision: `(year, code)`
    is a subject's identity, so a lookup by code alone has no answer. A caller holding only
    a code has to say which year it means — usually the year of the term whose marks it is
    resolving, which is the only year the code can be about.
    """

    def get(
        self, code: SubjectCode, academic_year_code: AcademicYearCode
    ) -> Subject | None:
        """The subject as that year teaches it, or `None`."""

    def get_many(
        self, codes: Collection[SubjectCode], academic_year_code: AcademicYearCode
    ) -> Mapping[str, Subject]:
        """Resolve every subject column of one year's grade sheet, keyed by code.

        Keyed by code rather than by `(year, code)` because the year is the argument: the
        caller has already narrowed to one, and a two-part key would only be unpacked again
        at every call site.
        """

    def list_for_year(
        self,
        academic_year_code: AcademicYearCode,
        *,
        include_inactive: bool = False,
        year_level_code: YearCode | None = None,
    ) -> Sequence[Subject]:
        """The year's subjects in `display_order`; retired ones only when asked for.

        Naming a rung narrows the answer to the subjects assigned to it. That is a
        genuinely different question from "the year's catalogue": a school teaches
        Physics, but only Secondary sits it, and a Primary marks sheet offering it is
        how a mark ends up filed against a subject that rung never taught.
        """

    def upsert_many(self, subjects: Sequence[Subject]) -> Mapping[str, bool]:
        """Insert or update by `(academic_year_code, code)`; `True` marks the new ones.

        Keyed in the returned mapping by code alone, which is unambiguous because a single
        call is not expected to span years — and the service that writes one subject at a
        time is the only caller.
        """

    def assignments_for_year(
        self, academic_year_code: AcademicYearCode
    ) -> Sequence[GradeSubjects]:
        """Every assignment in the year, keyed by the rung code that carries it.

        Rungs belong to one academic track, so the Arabic and Languages sections of a
        bilingual school appear here as separate keys with separate subject lists —
        which is the whole of "separate assignments per track" in this schema.
        """

    def assign_to_level(
        self,
        code: SubjectCode,
        academic_year_code: AcademicYearCode,
        school_code: SchoolCode,
        year_level_code: YearCode,
    ) -> bool:
        """Assign once; `True` only when this call inserted a new association.

        Idempotent by design and by constraint: `uq_subject_year_levels_assignment` makes
        a duplicate impossible in the database, and re-assigning an already-assigned
        subject answers `False` rather than raising, so a double drop on the board is a
        no-op rather than an error a registrar has to read.
        """

    def unassign_from_level(
        self,
        code: SubjectCode,
        academic_year_code: AcademicYearCode,
        school_code: SchoolCode,
        year_level_code: YearCode,
    ) -> None:
        """Remove the association only.

        The subject row and every mark stated against it survive: un-assigning is a
        statement about next term's timetable, not a retraction of a mark a child was
        awarded. Retiring the subject itself remains `is_active=False`.
        """


class StudentRepository(Protocol):
    """Children, identified by the school's own student number."""

    def get(self, student_number: StudentNumber) -> Student | None:
        """The student, or `None`."""

    def get_many(
        self, student_numbers: Collection[StudentNumber]
    ) -> Mapping[str, Student]:
        """Which of these children are already on file — one query for a whole roster."""

    def search(
        self, query: str, *, limit: int = 50, include_inactive: bool = False
    ) -> Sequence[Student]:
        """Registrar type-ahead over student number and both name spellings."""

    def upsert_many(self, students: Sequence[Student]) -> Mapping[str, bool]:
        """Insert or update by student number; `True` marks the ones this call created."""

    def set_active(
        self, student_number: StudentNumber, *, is_active: bool
    ) -> Student:
        """Mark a child as left or returned; never deletes, because grades outlive them."""


class EnrolmentRepository(Protocol):
    """Time-bounded placements of children in classes — invariant 2 made storable."""

    def class_section_on(
        self, student_id: StudentNumber, on_date: date
    ) -> ClassSection | None:
        """Which class this child was in on `on_date`. The query invariant 2 exists for.

        A child who moves from 3A to 3B in March has two placements, both true: 3A from
        September to March, 3B from March onwards. Asked for a day in Term 1 this returns
        3A — after the transfer, forever — because that is where she sat and where those
        marks were earned. A `students.class_code` column cannot answer this at all: it
        holds one class, so the transfer either rewrites Term 1 into a room the child had
        never entered, or leaves her current class wrong. Both readings are wrong.

        The quieter failure is the one that costs a day of work. Single-column systems,
        asked for "3A, Term 1" after the transfer, find nothing and render the term as
        *no marks recorded* — a finished, published term shown as missing data, with no
        error anywhere — and the registrar re-enters grades that were never lost.

        `on_date` is passed in, never read from a clock: the caller answering "which
        class for Term 1" passes a day inside Term 1, and a repository that reached for
        `date.today()` would answer today's question to every historical report and be
        untestable besides.

        Returns `None` when no placement covers that day — a child who had left, or had
        not yet arrived. That is a real answer and must not be confused with "3A".
        """

    def class_sections_on(
        self, student_ids: Collection[StudentNumber], on_date: date
    ) -> Mapping[str, ClassSection]:
        """`class_section_on` for many children at once, keyed by student number.

        A grade import needs the class each mark was earned in, per row; without this it
        runs the transfer-aware lookup once per child and turns one upload into hundreds
        of queries. Children with no placement on that day are absent from the mapping,
        never mapped to a guessed class.
        """

    def open_enrolment(self, student_id: StudentNumber) -> ClassEnrolment | None:
        """The placement with no end date — where she is now — or `None`."""

    def list_for_student(self, student_id: StudentNumber) -> Sequence[ClassEnrolment]:
        """Her full placement history, earliest start first."""

    def list_for_students(
        self, student_ids: Collection[StudentNumber]
    ) -> Mapping[str, Sequence[ClassEnrolment]]:
        """Existing placements for many children, so overlap checks run in memory.

        The rule itself is `ClassEnrolment.conflicts_with`; the repository's job is to
        hand the service the rows, not to re-implement the rule in SQL where no unit test
        can reach it.
        """

    def roster_on(
        self,
        academic_year_code: AcademicYearCode,
        class_code: ClassCode,
        on_date: date,
    ) -> Sequence[ClassEnrolment]:
        """Who was in this class on that day — the register, as of any date."""

    def close_open_enrolment(
        self, student_id: StudentNumber, *, ends_on: date
    ) -> ClassEnrolment | None:
        """End the child's open placement on `ends_on`; `None` when she had none.

        A transfer is this call followed by a new enrolment, never an update of the old
        row's class: overwriting it is exactly the history rewrite invariant 2 forbids.
        """

    def upsert_many(
        self, enrolments: Sequence[ClassEnrolment]
    ) -> Mapping[EnrolmentKey, bool]:
        """Insert or update placements in bulk; `True` marks the ones this call created.

        Keyed including `starts_on`, so committing the same roster twice writes the same
        placements rather than stacking a second copy on top of the first.
        """


class GuardianRepository(Protocol):
    """The adults responsible for children, identified by phone number.

    Keyed on a phone rather than a code because a school issues no parent numbers for an
    importer to match on, so the number is the only stable handle a spreadsheet actually
    contains. Everything about `Phone`'s normalisation exists to make that key reliable.
    """

    def get(self, phone: Phone) -> Guardian | None:
        """The guardian reachable on this number, or `None`.

        Resolves through *every* number on file, not just each guardian's primary one. A
        mother who gave the school a mobile and a WhatsApp-only line is one person, and a
        later upload quoting her second number must find her rather than create a rival
        record holding half her children.
        """

    def get_many(self, phones: Collection[Phone]) -> Mapping[str, Guardian]:
        """Which of these numbers are already on file — one query for a whole upload.

        Keyed by the *queried* number, not by the guardian's primary one, so a caller
        holding the alternate number it asked about can find the answer it asked for.
        """

    def upsert_many(self, guardians: Sequence[Guardian]) -> Mapping[str, bool]:
        """Insert or update by primary phone; `True` marks the ones this call created.

        Additional numbers on a guardian are inserted alongside, never replacing the set:
        an upload that mentions only her mobile must not silently drop the second line an
        earlier one recorded.
        """

    def primary_phone_for(self, public_id: str) -> Phone | None:
        """The number that identifies this guardian, from her handle. The inverse of
        `public_id_for`.

        Exists because a caller who was given a handle has no other way back to her. The
        chat service holds one from the moment a parent signs in and never learns the
        number behind it, which is the property that keeps a parent's phone out of a
        process that talks to a language model.
        """

    def public_id_for(self, phone: Phone) -> str | None:
        """The guardian's stable external handle, or `None` when the number reaches nobody.

        Exists so another service can hold a reference to a parent without holding her
        phone number. `public_id` is opaque and permanent; a number is neither — she may
        add a second line or change carrier, and a phone in a URL is PII in every access
        log and browser history it passes through.

        Resolves through every number on file, like `get`, so a parent who verifies the
        second line she gave the school is the same person as one who verifies the first.
        """


class StudentGuardianRepository(Protocol):
    """Which adult is what to which child, and which of them may read her records.

    A first-class repository rather than a collection hanging off a student, for the same
    reason `enrolments` is: the query that matters most does not start from the child. "Which
    children may this guardian see" is what every parent-facing request asks first, and it
    is asked with a phone number in hand and no student number at all.
    """

    def list_for_student(
        self, student_number: StudentNumber
    ) -> Sequence[StudentGuardian]:
        """Every guardian link for one child."""

    def list_for_students(
        self, student_numbers: Collection[StudentNumber]
    ) -> Mapping[str, Sequence[StudentGuardian]]:
        """Existing links for many children, so an import validates in memory.

        Children with no guardians on file are absent from the mapping rather than mapped
        to an empty sequence, matching `list_for_students` on enrolments.
        """

    def list_students_for_guardian(
        self, phone: Phone, *, viewable_only: bool = True
    ) -> Sequence[StudentGuardian]:
        """Which children this guardian may ask about.

        `viewable_only` defaults to `True` because the parent-facing caller is the common
        one and the safe default is the narrow answer: a caller that forgets the argument
        gets the restricted set, never the full one. A registrar screen showing every link
        including the barred ones passes `False` deliberately.
        """

    def upsert_many(
        self, links: Sequence[StudentGuardian]
    ) -> Mapping[StudentGuardianKey, bool]:
        """Insert or update links in bulk; `True` marks the ones this call created."""

    def unlink(self, student_number: StudentNumber, phone: Phone) -> bool:
        """Remove one link. `False` when there was nothing to remove.

        A real delete, unlike `Student.set_active`, because a guardian link carries no
        history worth keeping once it is wrong: a link created against the wrong child is
        a mistake to erase, not a fact that stopped being true. Ending a *correct*
        relationship is `can_view_records`, which keeps the contact and removes the
        reading.
        """


class GradeRepository(Protocol):
    """Stated marks: one figure per student, subject and term. No aggregation lives here."""

    def get(
        self,
        student_number: StudentNumber,
        subject_code: SubjectCode,
        term_code: TermCode,
    ) -> SubjectGrade | None:
        """The mark on file, or `None` when nothing has been recorded."""

    def get_many(self, keys: Collection[GradeKey]) -> Mapping[GradeKey, SubjectGrade]:
        """Which of these marks already exist — how a preview reports restated figures."""

    def list_for_student(
        self, student_number: StudentNumber, *, term_code: TermCode | None = None
    ) -> Sequence[SubjectGrade]:
        """A child's marks, optionally one term's, in subject `display_order`."""

    def list_for_class(
        self,
        class_section_id: int,
        term_code: TermCode,
        *,
        subject_code: SubjectCode | None = None,
    ) -> Sequence[SubjectGrade]:
        """The grade sheet: one class, one term, optionally one subject."""

    def upsert_many(
        self, grades: Sequence[SubjectGrade]
    ) -> Mapping[GradeKey, bool]:
        """Write the stated figures in bulk; `True` marks the ones this call created.

        A `percentage` of `None` is written as SQL NULL — invariant 1: an ungraded child
        is ungraded, and storing 0.0 tells a parent their child scored zero. Whether a
        blank cell in a file should overwrite a mark already on file is the *service's*
        decision, taken where a human can see it; this method writes what it is handed.
        """


class ImportBatchRepository(Protocol):
    """Preview batches and their rows — one Protocol because a row has no life alone."""

    def add(self, batch: ImportBatch, rows: Sequence[ImportRow]) -> None:
        """Store a preview and every one of its rows in a single write."""

    def get(self, batch_id: str) -> ImportBatch | None:
        """The batch, or `None`; expiry is decided by the caller against its own clock."""

    def save(self, batch: ImportBatch) -> ImportBatch:
        """Persist a status transition — committed, expired — on an existing batch."""

    def list_rows(
        self,
        batch_id: str,
        *,
        outcomes: Collection[RowOutcome] | None = None,
        offset: int = 0,
        limit: int | None = None,
    ) -> Sequence[ImportRow]:
        """Rows in file order, filterable and paged.

        Paged because a registrar reviewing a two-thousand-row upload wants the rejected
        ones, and loading the batch whole to find forty of them is how the preview screen
        becomes slower than the import.
        """

    def count_rows(
        self, batch_id: str, *, outcomes: Collection[RowOutcome] | None = None
    ) -> int:
        """How many rows match, without fetching them."""

    def replace_rows(self, batch_id: str, rows: Sequence[ImportRow]) -> None:
        """Overwrite a batch's rows with the outcomes a commit actually produced."""

    def delete_expired(self, now: datetime) -> int:
        """Drop previews past their TTL; returns how many batches went."""


class ApiKeyRepository(Protocol):
    """Stored credentials, found by the public prefix of the presented key."""

    def get_by_prefix(self, prefix: str) -> ApiKey | None:
        """The key whose prefix this is, active or not — usability is the caller's call."""

    def list_all(self) -> Sequence[ApiKey]:
        """Every key, newest first, for the registrar's key screen."""

    def add(self, key: ApiKey) -> ApiKey:
        """Store a newly minted key. Only the hash is ever written."""

    def revoke(self, prefix: str) -> ApiKey | None:
        """Deactivate a key; `None` when no such prefix exists."""

    def touch(self, prefix: str, *, at: datetime) -> None:
        """Record last use. Best effort — a failed touch must never fail the request."""

    def has_any(self) -> bool:
        """Whether any key exists, so bootstrapping the first one stays a one-time act."""


class AccessAuditRepository(Protocol):
    """Access decisions: appended, and read back. **No update and no delete.**

    The append-only rule is enforced by the absence of the methods rather than by everyone
    remembering not to call them. A retention policy that genuinely has to expire rows
    should do it as a visible scheduled job against the table, not through a method sitting
    here waiting to be reached from a request handler.
    """

    def record(self, attempt: AccessAttempt) -> None:
        """Append one decision, allowed or refused."""

    def recent(
        self,
        *,
        guardian_public_id: str | None = None,
        student_number: str | None = None,
        allowed: bool | None = None,
        limit: int = 100,
    ) -> Sequence[AccessAttempt]:
        """Newest first. Filters are optional and compose; `allowed=False` is the
        alerting query."""
