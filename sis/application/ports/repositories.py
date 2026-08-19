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
from datetime import date, datetime
from typing import Protocol

from sis.domain.auth import ApiKey
from sis.domain.grades import SubjectGrade
from sis.domain.guardians import Guardian, StudentGuardian
from sis.domain.imports import ImportBatch, ImportRow, RowOutcome
from sis.domain.people import ClassEnrolment, Student
from sis.domain.structure import AcademicYear, ClassSection, Subject, Term, YearLevel
from sis.domain.value_objects import (
    AcademicYearCode,
    ClassCode,
    Phone,
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

# `(student_number, guardian_phone)` — `StudentGuardian.identity`. No date in this key,
# unlike `EnrolmentKey`: one adult holds one relationship to one child at a time, so a
# second row for the same pair is a correction of the first rather than a second fact.
type StudentGuardianKey = tuple[str, str]


class AcademicYearRepository(Protocol):
    """The school years everything else hangs from."""

    def get(self, code: AcademicYearCode) -> AcademicYear | None:
        """The year, or `None` when no such code is on file."""

    def get_many(
        self, codes: Collection[AcademicYearCode]
    ) -> Mapping[str, AcademicYear]:
        """The years that exist, keyed by code; absent codes are simply missing."""

    def list_all(self) -> Sequence[AcademicYear]:
        """Every year, most recent start date first."""

    def current(self) -> AcademicYear | None:
        """The year flagged current, or `None` before a registrar has chosen one."""

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
    """The rungs of the ladder — `Y1`..`Y6` — shared by every academic year."""

    def get(self, code: YearCode) -> YearLevel | None:
        """The year level, or `None`."""

    def get_many(self, codes: Collection[YearCode]) -> Mapping[str, YearLevel]:
        """Bulk existence check for an import, keyed by code."""

    def list_all(self) -> Sequence[YearLevel]:
        """Every level in `display_order`, so `Y10` follows `Y9` rather than preceding it."""

    def upsert_many(self, levels: Sequence[YearLevel]) -> Mapping[str, bool]:
        """Insert or update by code; `True` marks the ones this call created."""


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
    """Terms of an academic year, each a closed date range."""

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


class SubjectRepository(Protocol):
    """Subjects, global rather than scoped to a year level."""

    def get(self, code: SubjectCode) -> Subject | None:
        """The subject, or `None`."""

    def get_many(self, codes: Collection[SubjectCode]) -> Mapping[str, Subject]:
        """Resolve every subject column of a grade sheet in one query."""

    def list_all(self, *, include_inactive: bool = False) -> Sequence[Subject]:
        """Subjects in `display_order`; retired ones only when explicitly asked for."""

    def upsert_many(self, subjects: Sequence[Subject]) -> Mapping[str, bool]:
        """Insert or update by code; `True` marks the ones this call created."""


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
