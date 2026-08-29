"""The relational shape of the SIS — a *mapping* of the domain, never the domain itself.

These classes deliberately duplicate the field names in `sis/domain/`. The duplication
is the point: `sis.domain` imports no sqlalchemy, so a use case can be unit-tested
against fake repositories with no database and no engine. Mapping the domain dataclasses
directly (imperative mapping, or worse, making them declarative) would drag a `Session`
into every service test and make the domain unusable without one. Repositories translate
between the two, and that translation is the only place both vocabularies appear.

Two habits this file keeps throughout:

* **Codes are sized from the value objects.** Every code column's length is the matching
  `_Code.MAX_LENGTH`. If they drift, a code passes domain validation and is then
  truncated on write — MySQL historically did it silently, and a truncated student
  number matches no child.
* **School dates are `Date`, not `DateTime`.** A placement begins on a school day. Stored
  as an instant it acquires a timezone, and "which class was she in on 3 March" starts
  answering differently either side of midnight depending on the server's offset.
  Audit-ish timestamps (`created_at`, `last_used_at`) are the opposite case and are
  timezone-aware UTC.

Alembic owns this schema (decision 8). Nothing here calls `create_all`; the models are
the source Alembic autogenerates *from*, not a shortcut around it.
"""
from datetime import date, datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from sis.domain.access import REASON_LENGTH
from sis.domain.auth import Scope
from sis.domain.imports import ImportKind, ImportStatus, RowOutcome
from sis.infrastructure.db.base import Base

# Sizes mirror `sis.domain.value_objects.*.MAX_LENGTH`. See the module docstring.
_YEAR_CODE_LEN = 16
_LEVEL_CODE_LEN = 16
_CLASS_CODE_LEN = 32
_SUBJECT_CODE_LEN = 32
_TERM_CODE_LEN = 32
_STUDENT_NUMBER_LEN = 64
_NAME_LEN = 160
_SCHOOL_CODE_LEN = 16  # `SchoolCode.MAX_LENGTH`.
_STAGE_LEN = 16  # Longest `Stage` member is "preparatory".
_ATTENDANCE_STATE_LEN = 16  # Longest `AttendanceState` member is "present".
_ADDRESS_LEN = 500
_PHONE_LEN = 16  # `Phone.MAX_LENGTH`: '+' plus E.164's fifteen digits.
# Sized from the longest member of `sis.domain.people.Gender` with room to spare. No
# CHECK constraint beside it: the domain already degrades an unrecognised value to
# `unspecified`, and a constraint would turn that graceful loss into a failed import.
_GENDER_LEN = 16
_RELATIONSHIP_LEN = 16  # Longest `RelationshipType` member is "grandparent".
_PUBLIC_ID_LEN = 32


def _now() -> datetime:
    """Timezone-aware UTC; a naive timestamp cannot say which day an import happened."""
    return datetime.now(timezone.utc)


class School(Base):
    """One school. Every year, rung, class and mark below it belongs to exactly one.

    The service held a single school implicitly for its whole life: everything in the
    database was that school's and nothing had to say so. This table makes the boundary
    explicit, and the failure it exists to prevent is the quiet one — two branches each
    running a `3A`, and a register showing a child at a school she has never attended.

    **Students are deliberately not scoped here.** A child is a person rather than a
    school's property, so `students.student_number` stays globally unique and which school
    she attends follows from her placement: her class belongs to a year, and that year
    belongs to a school. That leaves the join key `records/` reads, the guardian tables and
    every stated mark untouched by this change, and it makes a child moving between branches
    a transfer rather than a second record of the same girl.
    """

    __tablename__ = "schools"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(_SCHOOL_CODE_LEN), unique=True, nullable=False)
    name_en: Mapped[str] = mapped_column(String(_NAME_LEN), default="", nullable=False)
    name_ar: Mapped[str] = mapped_column(String(_NAME_LEN), default="", nullable=False)
    # Closes a branch without deleting it: the registers and marks of the years it ran are
    # still true, and every row referencing it has to keep resolving.
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, nullable=False)


class AcademicYear(Base):
    """One school year — the scope everything else is qualified by."""

    __tablename__ = "academic_years"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # Immutable identity (decision 7); `name_*` are the renameable labels.
    #
    # Still globally unique, and that is a decision rather than an oversight. `2025-2026`
    # names exactly one year at exactly one school, so every route taking `?academic_year=`,
    # every import that names one, and `records/` reading through the facade keep working
    # unchanged and unambiguously. What it costs: two branches cannot both use the literal
    # code `2025-2026` and must distinguish them (`NC-2025-2026`). Making it unique per
    # school instead would force a school onto every one of those callers — a change to the
    # external contract, for no gain in safety.
    code: Mapped[str] = mapped_column(String(_YEAR_CODE_LEN), unique=True, nullable=False)
    school_id: Mapped[int] = mapped_column(
        ForeignKey("schools.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    name_en: Mapped[str] = mapped_column(String(_NAME_LEN), default="", nullable=False)
    name_ar: Mapped[str] = mapped_column(String(_NAME_LEN), default="", nullable=False)

    starts_on: Mapped[date] = mapped_column(Date, nullable=False)
    ends_on: Mapped[date] = mapped_column(Date, nullable=False)

    # Not unique: enforcing "exactly one current year" in the database would make the
    # rollover a two-statement operation with a window where the school has no current
    # year at all. The repository clears the old flag and sets the new one in one
    # transaction instead.
    is_current: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now, nullable=False
    )


class YearLevel(Base):
    """A rung of the ladder (`Y3`, `G10`), shared by every academic year.

    Global rather than per-year on purpose: a `Y3` that is re-created each September
    makes "how did this cohort do in Y3 over five years" a join across five surrogate
    ids that only agree by coincidence of spelling.
    """

    __tablename__ = "year_levels"
    __table_args__ = (
        # Identity, replacing the old global unique on `code`: the same rung code in a
        # different school is a different rung and has to be insertable.
        UniqueConstraint("school_id", "code", name="uq_year_levels_school_code"),
        # Serves the grouped ladder every school screen draws, and the generator's ordered
        # walk over one school's levels.
        Index("ix_year_levels_school_stage", "school_id", "stage", "display_order"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # Unique *within a school*, unlike an academic year code. "Year 1" genuinely exists at
    # every branch, and forcing a school prefix into the code a registrar types into class
    # codes and import headers all day would be the wrong trade here. It stays unambiguous
    # because a rung is only ever resolved alongside an academic year, and the year names
    # the school.
    code: Mapped[str] = mapped_column(String(_LEVEL_CODE_LEN), nullable=False)
    school_id: Mapped[int] = mapped_column(
        ForeignKey("schools.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    name_en: Mapped[str] = mapped_column(String(_NAME_LEN), default="", nullable=False)
    name_ar: Mapped[str] = mapped_column(String(_NAME_LEN), default="", nullable=False)

    # Explicit, because lexicographic order puts "Y10" before "Y9" and a school with
    # more than nine levels then prints its year list in an order no parent recognises.
    display_order: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # garden / primary / preparatory / secondary, for grouping a long ladder on screen.
    # A label carrying no rules: nothing is barred from a class, term or subject because of
    # it. `unspecified` is what every existing rung was before this column existed, which is
    # why it is the default rather than a guess.
    stage: Mapped[str] = mapped_column(
        String(_STAGE_LEN), default="unspecified", nullable=False
    )

    # Which section of the school this rung belongs to (revision 0007). Nullable because
    # every rung predating the column belongs to no stated section, and a school that does
    # not divide itself never fills it in — the naming then falls back to the stored label,
    # which is exactly what it did before.
    educational_system_id: Mapped[int | None] = mapped_column(
        ForeignKey("educational_systems.id", ondelete="SET NULL"), nullable=True, index=True
    )

    # The rung's number *within its own ladder*: 1 for "First Primary" in an Arabic
    # section, 3 for "Grade 3" in a language one. Stored rather than parsed out of the
    # name, because that is the fact the display name is generated from and a name is
    # something a registrar may retype. Nullable for the same reason as the column above.
    grade_number: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, nullable=False)


class ClassSection(Base):
    """One class of children, in one year level, in one academic year.

    The unique key is the triple `(academic_year_id, year_level_id, code)` rather than
    `(academic_year_id, code)` because uniform generation ("5 years x 8 classes") emits
    section codes like `A`..`H` under every level, and a per-year key would make the
    second level's `A` a duplicate. The triple is also exactly the key idempotent
    generation re-runs against (decision 3): the second run collides on it and reports
    "already present" instead of creating a parallel set of classes.

    The cost of that choice is that `(year, code)` alone is not guaranteed unique, so a
    repository resolving a bare "3A" from an import file must either rely on the school
    using level-qualified codes or qualify the lookup by year level. It must never take
    `.first()` of an ambiguous match — that silently enrols a child in the wrong room.
    """

    __tablename__ = "class_sections"
    __table_args__ = (
        UniqueConstraint(
            "academic_year_id", "year_level_id", "code", name="uq_class_section_year_level_code"
        ),
        # Serves the import path's "resolve this class code within this academic year",
        # run once per roster row, and the registrar's "list all classes this year".
        Index("ix_class_sections_year_code", "academic_year_id", "code"),
        # Serves "list the classes of year level N this year" — the structure screen and
        # the generator's existence check.
        Index("ix_class_sections_year_level", "academic_year_id", "year_level_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    academic_year_id: Mapped[int] = mapped_column(
        ForeignKey("academic_years.id", ondelete="RESTRICT"), nullable=False
    )
    year_level_id: Mapped[int] = mapped_column(
        ForeignKey("year_levels.id", ondelete="RESTRICT"), nullable=False
    )

    # Identity. Renaming happens on `name_*`; changing this detaches enrolments and
    # grades from the class they were earned in (decision 7).
    code: Mapped[str] = mapped_column(String(_CLASS_CODE_LEN), nullable=False)
    name_en: Mapped[str] = mapped_column(String(_NAME_LEN), default="", nullable=False)
    name_ar: Mapped[str] = mapped_column(String(_NAME_LEN), default="", nullable=False)

    # Nullable because "no stated capacity" is a real answer and 0 is not; a 0 here would
    # read as "this class is full" to any check that compares against it.
    capacity: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Which room on the rung — the second number in `1/2 ب` (revision 0007). Nullable
    # because a language section names rooms with letters and stores that in `name_en`
    # instead; the naming module reads whichever is present.
    section_number: Mapped[int | None] = mapped_column(Integer, nullable=True)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now, nullable=False
    )

    academic_year: Mapped["AcademicYear"] = relationship("AcademicYear", lazy="raise")
    year_level: Mapped["YearLevel"] = relationship("YearLevel", lazy="raise")


class Term(Base):
    """A dated slice of an academic year; grades hang off it and report cards close on it."""

    __tablename__ = "terms"
    __table_args__ = (
        CheckConstraint("ends_on >= starts_on", name="ck_terms_dates_ordered"),
        # Serves "list this year's terms in order", the header of every grade screen.
        Index("ix_terms_year_sequence", "academic_year_id", "sequence"),
        # Serves "which term covers this date" — how an enrolment window is matched to a
        # term when a grade names a term but the placement names days.
        Index("ix_terms_window", "starts_on", "ends_on"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(_TERM_CODE_LEN), unique=True, nullable=False)
    academic_year_id: Mapped[int] = mapped_column(
        ForeignKey("academic_years.id", ondelete="RESTRICT"), nullable=False
    )
    name_en: Mapped[str] = mapped_column(String(_NAME_LEN), default="", nullable=False)
    name_ar: Mapped[str] = mapped_column(String(_NAME_LEN), default="", nullable=False)

    starts_on: Mapped[date] = mapped_column(Date, nullable=False)
    ends_on: Mapped[date] = mapped_column(Date, nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    # Closing freezes the term. Writes against a closed term are refused in the service
    # (`TermClosed`), not here: the registrar must still be able to reopen one, and a
    # database that forbids the write forbids the correction too.
    is_closed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now, nullable=False
    )

    academic_year: Mapped["AcademicYear"] = relationship("AcademicYear", lazy="raise")


class Subject(Base):
    """A subject as taught in **one academic year**: `(academic_year_id, code)` is identity.

    This was global once, deliberately — one `MATH` row for the whole school, so a child's
    mathematics marks lined up from Year 3 to Year 4 and a report card could put them side
    by side. It is per-year now because a school sets its own catalogue each year, and the
    consequence of the change is worth stating in the file that carries it: `MATH` in
    2025-2026 and `MATH` in 2026-2027 are two rows and two subjects, so nothing downstream
    can treat a mark on one as comparable to a mark on the other. A cross-year view has to
    match on `code` itself and accept that two schools' worth of meaning can hide behind
    one string.

    What did *not* change is the reason `code` exists: it is still immutable for the life of
    the row within its year, so renaming "MATH" to "Mathematics" is a label edit that
    detaches no grade.
    """

    __tablename__ = "subjects"
    __table_args__ = (
        # Identity. Replaces the old global `uq_subjects_code`: the same code in a
        # different year is a different subject now, and must be insertable.
        UniqueConstraint("academic_year_id", "code", name="uq_subjects_year_code"),
        # Serves the grade sheet's column order and the subject picker, both of which are
        # now always asked for one year at a time.
        Index("ix_subjects_year_order", "academic_year_id", "display_order"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(_SUBJECT_CODE_LEN), nullable=False)
    # RESTRICT, as every other reference to a year: deleting a year out from under the
    # subjects its marks are stated against is not a thing the database will help with.
    academic_year_id: Mapped[int] = mapped_column(
        ForeignKey("academic_years.id", ondelete="RESTRICT"), nullable=False
    )
    name_en: Mapped[str] = mapped_column(String(_NAME_LEN), default="", nullable=False)
    name_ar: Mapped[str] = mapped_column(String(_NAME_LEN), default="", nullable=False)

    display_order: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # Retired subjects are deactivated, never deleted: the grades of the years they were
    # taught in still have to render a subject name.
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, nullable=False)


class Student(Base):
    """A child. Notably has **no class column** — placement lives in `class_enrolments`.

    Decision 2 in schema form. A `class_section_id` here would be the single most
    convenient column in the file and would destroy the history the service exists to
    keep: a child who moves 3A -> 3B in March would retroactively have "always" been in
    3B, and every Term 1 report card would reprint under the wrong class.
    """

    __tablename__ = "students"
    __table_args__ = (
        # Serves the registrar's name search, which is typed in either script.
        Index("ix_students_name_ar", "full_name_ar"),
        Index("ix_students_name_en", "full_name_en"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # The school's own number, and the join key `records/` matches on. Unique and
    # immutable; re-issuing one silently re-parents every grade and enrolment below it.
    student_number: Mapped[str] = mapped_column(
        String(_STUDENT_NUMBER_LEN), unique=True, nullable=False
    )

    full_name_ar: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    full_name_en: Mapped[str] = mapped_column(String(255), default="", nullable=False)

    # Left the school, but the transcript stays. Deletion is refused by the RESTRICT on
    # grades and enrolments, which is the intended behaviour, not an obstacle.
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Nullable, and no age column beside it: an age is right for one year and silently
    # wrong afterwards, so it is computed from this date at the moment of asking.
    date_of_birth: Mapped[date | None] = mapped_column(Date, nullable=True)

    # NOT NULL with a server default of "unspecified" rather than a nullable column, so
    # there is one spelling of "the school has not said" instead of two — `NULL` on the
    # rows that predate this and `""` on the ones that do not. Same reasoning as the
    # three contact columns below, and the same reason the server default is required:
    # this is added to a populated table.
    gender: Mapped[str] = mapped_column(
        String(_GENDER_LEN), default="unspecified", server_default="unspecified", nullable=False
    )

    # The child's own details, which are **not** her guardian's. A school holds both and
    # they differ; merging them is how a message meant for a parent reaches a nine-year-old.
    # Guardian numbers live in `guardians`/`guardian_phones`, where the permission to be
    # told about her marks lives with them.
    #
    # `server_default` as well as `default`, on all three, because these columns were added
    # to a populated table: a NOT NULL column with no server default cannot be added to
    # existing rows, and the alternative — nullable, so "no phone on file" is `None` on old
    # rows and `""` on new ones — would leave two spellings of empty for every reader to
    # handle. The model states the default the database actually holds.
    contact_phone: Mapped[str] = mapped_column(
        String(_PHONE_LEN), default="", server_default="", nullable=False
    )
    contact_email: Mapped[str] = mapped_column(
        String(255), default="", server_default="", nullable=False
    )
    address: Mapped[str] = mapped_column(
        String(_ADDRESS_LEN), default="", server_default="", nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now, nullable=False
    )


class Attendance(Base):
    """One mark per child per day: the daily register.

    `uq_attendance_student_day` is the load-bearing constraint. Taking the register twice on
    the same morning has to correct the mark rather than write a second, contradictory one —
    and with two registrars on two machines it is the database that has to guarantee that,
    not a check in Python.

    **There is no row meaning "unmarked".** A day with no row is a day nobody took the
    register, and that is different from every state this table can hold. An `unknown` row
    would make "days recorded" meaningless as a denominator, and that is the number every
    honest attendance figure has to be divided by.

    `class_section_id` is stored rather than resolved from the child at read time, exactly as
    it is on a grade and for the same reason: a child who moved 3A -> 3B in March was in 3A
    in October, and her October register has to keep saying so.
    """

    __tablename__ = "attendance"
    __table_args__ = (
        # One statement per child per day. A correction replaces it.
        UniqueConstraint("student_id", "on_date", name="uq_attendance_student_day"),
        # Serves "the register of this class on this day" — the read the class screen makes
        # every time a date is picked.
        Index("ix_attendance_section_day", "class_section_id", "on_date"),
        # Serves "this child's attendance across a range", for her card.
        Index("ix_attendance_student_day", "student_id", "on_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    student_id: Mapped[int] = mapped_column(
        ForeignKey("students.id", ondelete="RESTRICT"), nullable=False
    )
    class_section_id: Mapped[int] = mapped_column(
        ForeignKey("class_sections.id", ondelete="RESTRICT"), nullable=False
    )
    on_date: Mapped[date] = mapped_column(Date, nullable=False)

    # present / absent / late / excused. Text rather than an integer so a dump is readable
    # and so adding a state is not a renumbering of the ones already written.
    state: Mapped[str] = mapped_column(String(_ATTENDANCE_STATE_LEN), nullable=False)

    # Required by the domain for an excused absence: "excused by whom, for what" is the
    # entire content of that state, and an excused absence with no reason on file cannot be
    # told apart from a registrar clicking the wrong button.
    note: Mapped[str] = mapped_column(String(_ADDRESS_LEN), default="", nullable=False)

    # Who took the register, and when the row was last written. An attendance record is the
    # kind of statement queried months later by somebody who needs to know whether it was
    # taken that morning or corrected in June.
    recorded_by: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now, nullable=False
    )


class Guardian(Base):
    """An adult responsible for one or more children. **Has no phone column.**

    The numbers live in `guardian_phones`, one row each, and that split is the whole
    reason this table exists separately from the link below. A `phone` column here holds
    one number and loses the second, so a mother with a mobile and a WhatsApp-only line is
    either half-recorded or duplicated into two people holding half her children each.

    `public_id` rather than the row id or the phone in any URL a parent-facing service will
    one day use: ids are guessable and a phone number in a path is PII in every access log
    and browser history the request passes through.
    """

    __tablename__ = "guardians"
    __table_args__ = (
        # Serves the registrar's name search, typed in either script — the same pair of
        # indexes `students` carries, for the same screen.
        Index("ix_guardians_name_ar", "full_name_ar"),
        Index("ix_guardians_name_en", "full_name_en"),
        # The domain requires a name in at least one script; asserted here too so a row
        # written by a migration or a fixture cannot be nameless.
        CheckConstraint(
            "length(trim(full_name_ar)) > 0 OR length(trim(full_name_en)) > 0",
            name="ck_guardians_has_name",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    public_id: Mapped[str] = mapped_column(String(_PUBLIC_ID_LEN), unique=True, nullable=False)

    full_name_ar: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    full_name_en: Mapped[str] = mapped_column(String(255), default="", nullable=False)

    preferred_language: Mapped[str] = mapped_column(String(8), default="ar", nullable=False)
    # Left the school's contact list, but the links stay readable. Deletion is refused by
    # the RESTRICT on `student_guardians`, which is intended rather than an obstacle.
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now, nullable=False
    )


class GuardianPhone(Base):
    """One number that reaches one guardian. The unique constraint here is load-bearing.

    `phone` is unique **across the whole table**, not per guardian, and that is what makes
    "this number belongs to exactly one adult" a fact the database enforces rather than
    something the importer hopes. Everything downstream leans on it: the importer resolves
    a spreadsheet row to an existing person by this column, and a future parent login that
    verifies a number by sending it a code needs the answer to be unambiguous — two rows
    would mean a code sent to one family unlocking another's records.

    Numbers are stored in E.164 (`Phone`), never as typed, so `+201001234567` and
    `0100 123 4567` collide here instead of becoming two adults.
    """

    __tablename__ = "guardian_phones"
    __table_args__ = (
        # At most one PRIMARY number per guardian. A partial unique index, the same
        # portable idiom `class_enrolments` uses for its open placement: SQLite and
        # PostgreSQL both support it, so the constraint is real in the test suite too.
        Index(
            "uq_guardian_phones_primary",
            "guardian_id",
            unique=True,
            sqlite_where=text("is_primary = 1"),
            postgresql_where=text("is_primary"),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    guardian_id: Mapped[int] = mapped_column(
        ForeignKey("guardians.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    phone: Mapped[str] = mapped_column(String(_PHONE_LEN), unique=True, nullable=False)

    # Which number identifies this adult. The domain's `Guardian.phones[0]`.
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, nullable=False)


class StudentGuardian(Base):
    """Which adult is what to which child, and whether she may read the records.

    The many-to-many the feature exists for. Not a column on either side: a child
    ordinarily has more than one guardian and a guardian more than one child, so a
    `students.guardian_id` column records the father and silently loses the mother.

    `relationship_type` is per *link*, not per person, because the same man is `father` to
    one child on this roll and `guardian` to another he has custody of but is not related
    to. `can_view_records` is per link for the sharper version of the same reason: a
    grandmother may read one grandchild's report and be barred by a court order from
    another's.

    That flag defaults to **false at the database level** while the importer sets it true.
    The asymmetry is deliberate and documented in `sis.application.dto.guardians`: the
    default protects any code path that never considered the question, and the importer is
    a registrar who has, looking at a preview.
    """

    __tablename__ = "student_guardians"
    __table_args__ = (
        # One relationship per pair. A second upload of the same sheet updates this row
        # rather than stacking a duplicate that would double every contact list.
        UniqueConstraint("student_id", "guardian_id", name="uq_student_guardian_link"),
        # THE parent-facing query: "which children may this adult see", asked before any
        # record is returned. Leading `guardian_id` narrows to a handful of rows and the
        # flag resolves without touching the table.
        Index("ix_student_guardians_lookup", "guardian_id", "can_view_records"),
        # The registrar's direction: "who do we call about this child".
        Index("ix_student_guardians_student", "student_id"),
        # At most one primary contact per child — the number the office rings first. Same
        # partial-index idiom as above; without it two uploads can both claim the slot and
        # the school has no first call.
        Index(
            "uq_student_guardians_primary_contact",
            "student_id",
            unique=True,
            sqlite_where=text("is_primary_contact = 1"),
            postgresql_where=text("is_primary_contact"),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    student_id: Mapped[int] = mapped_column(
        ForeignKey("students.id", ondelete="RESTRICT"), nullable=False
    )
    guardian_id: Mapped[int] = mapped_column(
        ForeignKey("guardians.id", ondelete="RESTRICT"), nullable=False
    )

    relationship_type: Mapped[str] = mapped_column(
        String(_RELATIONSHIP_LEN), default="other", nullable=False
    )
    # The registrar's own words -- "big brother", "الجدة لأم". `Text` rather than a bounded
    # String because a label is prose attached to a fact, never parsed for a value, so
    # there is no length this service is entitled to reject.
    relationship_label: Mapped[str] = mapped_column(Text, default="", nullable=False)

    is_primary_contact: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    # Default deny at the storage layer. See the class docstring for why the importer
    # nonetheless grants.
    can_view_records: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default=text("0"), nullable=False
    )
    # Why access was restricted -- a court order reference, a school decision, a date.
    # Never rendered to a parent; it exists for the registrar and for an audit.
    restriction_note: Mapped[str] = mapped_column(Text, default="", nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now, nullable=False
    )


class ClassEnrolment(Base):
    """A time-bounded placement of one student in one class. `ends_on IS NULL` means open.

    **What the database enforces here, and what it cannot.**

    Enforced:

    * `ck_class_enrolments_dates_ordered` — a window may not end before it starts.
    * `uq_class_enrolments_open_per_student` — a *partial* unique index over
      `student_id WHERE ends_on IS NULL`. This is the one overlap the DB can express
      portably (SQLite and PostgreSQL both support partial indexes), and it is the
      common failure: a second roster import opens a new placement without closing the
      old one, and the child is live in two classes forever after.

    NOT enforced, and therefore the repository's job:

    * **Overlap between bounded windows.** 3A from 01-09 to 31-12 and 3B from 01-11 to
      31-03 are two closed rows that satisfy every constraint above and still put one
      child in two classes through November. Expressing this needs a range exclusion
      constraint (`EXCLUDE USING gist (student_id WITH =, daterange(...) WITH &&)`,
      PostgreSQL and `btree_gist` only); SQLite has no equivalent, so the schema would
      differ per dialect and the SQLite test suite would stop testing the rule.
    * **A bounded window overlapping the open one.** Same reason: the partial index sees
      only the open rows.
    * **Both checks are racy without a lock.** Two concurrent inserts each pass a
      `SELECT` that ran before the other's `INSERT`. The repository must take a write
      lock on the student row (`SELECT ... FOR UPDATE`, a no-op under SQLite's whole-file
      write lock) *before* the overlap query, inside the same transaction, and raise
      `OverlappingEnrolment` on a hit.

    Successive placements are the normal case and stay legal: closing 3A on 14-03 and
    opening 3B on 15-03 leaves exactly one open row and no overlapping window.
    """

    __tablename__ = "class_enrolments"
    __table_args__ = (
        CheckConstraint(
            "ends_on IS NULL OR ends_on >= starts_on", name="ck_class_enrolments_dates_ordered"
        ),
        # Idempotent re-import: the same placement uploaded twice collides here instead
        # of stacking duplicate rows that then look like an overlap.
        UniqueConstraint(
            "student_id", "class_section_id", "starts_on", name="uq_class_enrolment_placement"
        ),
        # At most one OPEN placement per student. Partial index — see the docstring for
        # what it does not cover.
        Index(
            "uq_class_enrolments_open_per_student",
            "student_id",
            unique=True,
            sqlite_where=text("ends_on IS NULL"),
            postgresql_where=text("ends_on IS NULL"),
        ),
        # THE hot query of this service: `class_section_on(student, date)` — "which class
        # was this child in on this day", asked for every grade row written and every
        # report card line rendered. Leading `student_id` narrows to a handful of rows;
        # the dates then resolve the window without touching the table.
        Index("ix_class_enrolments_student_on_date", "student_id", "starts_on", "ends_on"),
        # Serves "the roster of this class as of a date" — the class list screen.
        Index("ix_class_enrolments_section_window", "class_section_id", "starts_on", "ends_on"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    student_id: Mapped[int] = mapped_column(
        ForeignKey("students.id", ondelete="RESTRICT"), nullable=False
    )
    class_section_id: Mapped[int] = mapped_column(
        ForeignKey("class_sections.id", ondelete="RESTRICT"), nullable=False
    )

    starts_on: Mapped[date] = mapped_column(Date, nullable=False)
    # NULL means "still in this class, no end decided" — never a sentinel like 9999-12-31,
    # which sorts and prints like a real answer and eventually reaches a parent as one.
    ends_on: Mapped[date | None] = mapped_column(Date, nullable=True)

    # Why the child moved: "transfer", "correction", "initial". Free-ish text kept short;
    # a transfer and a fixed typo look identical in the dates alone.
    reason: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now, nullable=False
    )

    student: Mapped["Student"] = relationship("Student", lazy="raise")
    class_section: Mapped["ClassSection"] = relationship("ClassSection", lazy="raise")


class SubjectGrade(Base):
    """One stated mark: this student, this subject, this term.

    `class_section_id` is stored on the grade rather than resolved from the student at
    read time, because decision 2 means "her class" has different answers on different
    days. A mark earned in 3A must keep printing under 3A after a March transfer.

    No aggregation lives here (decision 5): no weighting, no drop-lowest, no computed
    term average column. A stored average is a figure nobody stated and that silently
    disagrees with the school's published policy the first time the policy changes.
    """

    __tablename__ = "subject_grades"
    __table_args__ = (
        UniqueConstraint("student_id", "subject_id", "term_id", name="uq_subject_grade_identity"),
        CheckConstraint(
            "percentage IS NULL OR (percentage >= 0 AND percentage <= 100)",
            name="ck_subject_grades_percentage_range",
        ),
        # Serves the grade-entry sheet and the import's existing-row lookup: "every grade
        # for this class, this subject, this term".
        Index("ix_subject_grades_sheet", "term_id", "class_section_id", "subject_id"),
        # Serves the report card: "every subject this child has in this term".
        Index("ix_subject_grades_student_term", "student_id", "term_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    student_id: Mapped[int] = mapped_column(
        ForeignKey("students.id", ondelete="RESTRICT"), nullable=False
    )
    subject_id: Mapped[int] = mapped_column(
        ForeignKey("subjects.id", ondelete="RESTRICT"), nullable=False
    )
    term_id: Mapped[int] = mapped_column(ForeignKey("terms.id", ondelete="RESTRICT"), nullable=False)
    # The class the mark was earned in, frozen at write time. See the class docstring.
    class_section_id: Mapped[int] = mapped_column(
        ForeignKey("class_sections.id", ondelete="RESTRICT"), nullable=False
    )

    # NULLABLE, and this is the single most important nullability in the schema
    # (decision 1). NULL means "not graded yet"; 0.0 means a child sat the assessment and
    # scored nothing. They are different sentences to a parent, and a NOT NULL column
    # with a `server_default="0"` would convert every ungraded subject in the school into
    # a zero the moment the table is created — a fabricated failing mark, produced by a
    # schema decision, that no teacher ever wrote. There is no default here on purpose:
    # a row that omits the percentage stores NULL, which is the truth.
    percentage: Mapped[float | None] = mapped_column(Float, nullable=True)

    # The raw figures when the teacher stated them ("17 out of 20"). Kept beside the
    # percentage, never re-derived from it in either direction: the derivation would
    # invent precision the teacher did not write, and a correction could no longer be
    # audited against the original.
    points: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_points: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Free text the teacher attached to this mark; shown as-is, never parsed for a value.
    remark: Mapped[str] = mapped_column(Text, default="", nullable=False)

    # Which api key / registrar wrote it, and when. A grade is the kind of figure someone
    # will eventually dispute, and "who entered this" must not require reading a log file.
    recorded_by: Mapped[str] = mapped_column(String(120), default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now, nullable=False
    )

    student: Mapped["Student"] = relationship("Student", lazy="raise")
    subject: Mapped["Subject"] = relationship("Subject", lazy="raise")
    term: Mapped["Term"] = relationship("Term", lazy="raise")
    class_section: Mapped["ClassSection"] = relationship("ClassSection", lazy="raise")


class ImportBatch(Base):
    """One upload, previewed and then committed — the header of decision 4's two-step flow.

    A batch is written at *preview* time, before anything is applied. That is what lets
    the registrar read the per-row outcomes and decide, and what makes commit checkable:
    `content_hash` is compared against the file presented at commit, so a client that
    re-uploads between the two steps is refused rather than applying rows no human saw.

    Rows outlive their usefulness quickly, so `expires_at` is a real deadline, not a
    cleanup hint: a preview taken against last week's class list must not be committable
    today, after the structure it validated against has changed.
    """

    __tablename__ = "import_batches"
    __table_args__ = (
        # Serves the expiry sweep and the "your preview timed out" check on commit.
        Index("ix_import_batches_status_expiry", "status", "expires_at"),
        # Serves the registrar's "my recent uploads" list.
        Index("ix_import_batches_actor_time", "actor", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # The opaque id handed to the client at preview and quoted back at commit. Separate
    # from `id` so a caller cannot enumerate other registrars' batches by counting.
    batch_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)

    # Stored as the StrEnum's own text (`"roster"`, `"grades"`, `"previewed"`), not as a
    # database enum type: a new member would otherwise need a migration on PostgreSQL and
    # be unreadable in a SQLite dump.
    kind: Mapped[str] = mapped_column(String(16), default=ImportKind.ROSTER.value, nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), default=ImportStatus.PREVIEWED.value, nullable=False
    )

    # Digest of the uploaded bytes. The guard against a swapped file between the steps.
    content_hash: Mapped[str] = mapped_column(String(128), default="", nullable=False)
    filename: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    actor: Mapped[str] = mapped_column(String(120), default="", nullable=False)

    # Per-outcome tallies ({"created": 12, "rejected": 3}). JSON rather than five columns
    # because `RowOutcome` is allowed to grow without a migration, and this value is only
    # ever read whole.
    counts: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    committed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    rows: Mapped[list["ImportRow"]] = relationship(
        "ImportRow", back_populates="batch", cascade="all, delete-orphan", lazy="raise"
    )


class ImportRow(Base):
    """The outcome of one line of one upload. Kept for every line, good or bad.

    Decision 4 in schema form: one bad row must never discard the good ones, so the unit
    of failure is this row and not the batch. A rejected row carries the domain error's
    `code`, `message` and `field` verbatim, which is what lets the registrar UI point at
    the offending cell instead of saying "import failed".
    """

    __tablename__ = "import_rows"
    __table_args__ = (
        # One outcome per source line; also makes a retried preview idempotent.
        UniqueConstraint("batch_id", "line_number", name="uq_import_row_line"),
        # Serves the preview screen's default view: "show me the rejected rows of this
        # batch", and the ordered full listing behind it.
        Index("ix_import_rows_batch_outcome", "batch_id", "outcome", "line_number"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    batch_id: Mapped[int] = mapped_column(
        ForeignKey("import_batches.id", ondelete="CASCADE"), nullable=False
    )

    # The line number in the *file* the registrar is looking at, header included, not an
    # index into the parsed rows. Telling someone "row 12" about their row 14 is worse
    # than telling them nothing.
    line_number: Mapped[int] = mapped_column(Integer, nullable=False)

    # The parsed cells, kept so the preview can be rendered and a commit can be replayed
    # without re-reading the upload the client no longer holds.
    payload: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    outcome: Mapped[str] = mapped_column(String(16), default=RowOutcome.CREATED.value, nullable=False)
    # `SisError.code` for a rejection, the outcome's own value otherwise. Same closed
    # vocabulary as the API returns, so renaming one is a breaking change.
    code: Mapped[str] = mapped_column(String(40), default=RowOutcome.CREATED.value, nullable=False)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Which column was at fault; NULL when the failure is not about one cell.
    field: Mapped[str | None] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, nullable=False)

    batch: Mapped["ImportBatch"] = relationship("ImportBatch", back_populates="rows", lazy="raise")


class ApiKey(Base):
    """A caller's credential: public prefix in clear, secret as a hash, never the key.

    The full secret is shown once at creation and is unrecoverable afterwards. `prefix`
    exists so a key can be *named* — in a log line, in an audit row, in the sentence
    "revoke the one ending in 4f2a" — without the log becoming a place secrets live.

    A key authenticates a system, not a person, and `scope` is checked by exact equality:
    a `registrar` key does not implicitly satisfy a `reader` check. See
    `sis.domain.auth.Scope.permits` for why the tempting ordering is a security bug.
    """

    __tablename__ = "api_keys"
    __table_args__ = (
        # Serves authentication on every single request: look the key up by its prefix,
        # then verify the hash. Unique because the prefix must resolve to one key or none
        # — a prefix collision would make the verify step decide identity by luck.
        Index("ix_api_keys_prefix_active", "prefix", "is_active"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prefix: Mapped[str] = mapped_column(String(16), unique=True, nullable=False)
    key_hash: Mapped[str] = mapped_column(String(255), nullable=False)

    label: Mapped[str] = mapped_column(String(120), default="", nullable=False)
    scope: Mapped[str] = mapped_column(String(16), default=Scope.READER.value, nullable=False)

    # Revocation is a flag, not a delete: a deleted key leaves audit lines referring to a
    # prefix nothing resolves, and "who did this" becomes unanswerable.
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, nullable=False)


class AccessAudit(Base):
    """Append-only: every attempt to read a child's record on a guardian's behalf.

    See `sis.domain.access` for why this lives here rather than in the facade in front of
    it, and why the refusals are the interesting rows.

    **External identifiers, not foreign keys.** A guardian handle and a student number are
    stored as text, and deliberately: an audit row has to survive the deletion of the
    guardian it refers to, and a cascade would erase precisely the history somebody is most
    likely to want after an account is removed. It also means this table can be copied out
    of the database for retention without dragging six others with it.
    """

    __tablename__ = "access_audit"
    __table_args__ = (
        # The two questions ever asked of this table: "everything about this child" and
        # "everything this guardian was told". Both are time-ordered, because the answer
        # is always a period rather than a total.
        Index("ix_access_audit_student_time", "student_number", "created_at"),
        Index("ix_access_audit_guardian_time", "guardian_public_id", "created_at"),
        # The alerting query: a run of refusals is somebody probing.
        Index("ix_access_audit_allowed_time", "allowed", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    guardian_public_id: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    student_number: Mapped[str] = mapped_column(String(64), default="", nullable=False)

    allowed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    reason: Mapped[str] = mapped_column(String(REASON_LENGTH), default="", nullable=False)

    # The API key prefix that asked. Names a caller; cannot authenticate as one.
    actor: Mapped[str] = mapped_column(String(16), default="", nullable=False)
    request_id: Mapped[str] = mapped_column(String(64), default="", nullable=False, index=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )


# ---------------------------------------------------------------------------
# Sections of the school, the people who work in it, and what they may do
#
# Everything below arrived with revision 0007. Two groups: `educational_systems` is the
# structural half (a school runs an Arabic section and a language section side by side,
# and they name their rungs differently), and the rest is the staff half — accounts,
# roles, scoped grants, teachers and their assignments.
#
# The `api_keys` table above is untouched by all of it. Machine callers keep
# authenticating exactly as they did; a person signing in is a second, separate door.
# ---------------------------------------------------------------------------

_ROLE_CODE_LEN = 32
_PERMISSION_CODE_LEN = 48
_SCOPE_TYPE_LEN = 16  # Longest `ScopeType` member is "class_section".
_SYSTEM_KIND_LEN = 16  # Longest `EducationalSystemKind` member is "unspecified".
_USERNAME_LEN = 64  # `sis.domain.staff.USERNAME_MAX_LENGTH`.
_STAFF_NUMBER_LEN = 32
_SETTING_KEY_LEN = 64


class EducationalSystem(Base):
    """One section of a school: the Arabic section, the language section.

    A school runs both at once, and that is the case this table exists for. They are not
    two schools — one head, one building, one roll — but they count their rungs
    differently, name their rooms differently and cannot share a `YearLevel`.

    `kind` drives naming (`sis/domain/naming.py`); `code` and the names are the school's
    own words. Keeping them apart is what lets a school call this "National Section"
    while the service still knows to render `1/2 ب`.
    """

    __tablename__ = "educational_systems"
    __table_args__ = (
        UniqueConstraint("school_id", "code", name="uq_educational_systems_school_code"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    school_id: Mapped[int] = mapped_column(
        ForeignKey("schools.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    code: Mapped[str] = mapped_column(String(_LEVEL_CODE_LEN), nullable=False)
    # arabic / language / unspecified. Not a CHECK constraint: the domain refuses an
    # unknown value on the way in, and a constraint here would turn adding a third kind
    # into a table rebuild on every school's database.
    kind: Mapped[str] = mapped_column(
        String(_SYSTEM_KIND_LEN), default="unspecified", nullable=False
    )
    name_en: Mapped[str] = mapped_column(String(_NAME_LEN), default="", nullable=False)
    name_ar: Mapped[str] = mapped_column(String(_NAME_LEN), default="", nullable=False)
    display_order: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )


class Role(Base):
    """A named bundle of permissions. Data, so a new role is an insert, not a deploy.

    `is_builtin` marks the seven the service ships with (`sis.domain.rbac.BUILT_IN_ROLES`).
    They are re-synced on every seed run and must not be deleted — a school that removed
    "Teacher" would have a hundred `user_roles` rows pointing at nothing.
    """

    __tablename__ = "roles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(_ROLE_CODE_LEN), unique=True, nullable=False)
    name_en: Mapped[str] = mapped_column(String(_NAME_LEN), default="", nullable=False)
    name_ar: Mapped[str] = mapped_column(String(_NAME_LEN), default="", nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    # The scope the assignment UI offers first. Advisory, never enforced — a school may
    # legitimately want a school-wide attendance supervisor.
    default_scope: Mapped[str] = mapped_column(
        String(_SCOPE_TYPE_LEN), default="school", nullable=False
    )
    is_builtin: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )


class PermissionRow(Base):
    """The catalogue of verbs. Named `PermissionRow` so it cannot be mistaken for the enum.

    A table as well as an enum because an administrator's screen has to *list* the
    permissions a role carries, with a label, in two languages — and because a foreign key
    from `role_permissions` is what stops a typo becoming a permission nobody holds.
    """

    __tablename__ = "permissions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(
        String(_PERMISSION_CODE_LEN), unique=True, nullable=False
    )
    name_en: Mapped[str] = mapped_column(String(_NAME_LEN), default="", nullable=False)
    name_ar: Mapped[str] = mapped_column(String(_NAME_LEN), default="", nullable=False)


class RolePermission(Base):
    """Which verbs a role carries. Cascades, because a deleted role's rows mean nothing."""

    __tablename__ = "role_permissions"
    __table_args__ = (
        UniqueConstraint("role_id", "permission_id", name="uq_role_permissions_role_permission"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    role_id: Mapped[int] = mapped_column(
        ForeignKey("roles.id", ondelete="CASCADE"), nullable=False, index=True
    )
    permission_id: Mapped[int] = mapped_column(
        ForeignKey("permissions.id", ondelete="CASCADE"), nullable=False, index=True
    )


class User(Base):
    """An account. One per person, however many roles they hold.

    `school_id` is nullable for exactly one reason: the system administrator belongs to no
    school. Everybody else is bound to one, and that binding is a second wall behind the
    role scopes rather than a substitute for them.

    The lockout columns are on the account rather than in a side table because they are
    read on every login attempt and written on most of them; a join for two integers on
    the hottest path in the auth flow is a cost with nothing on the other side of it.
    """

    __tablename__ = "users"
    __table_args__ = (
        # Serves the login lookup, which is the one query that runs before anything is
        # authenticated and therefore the one an unauthenticated caller can force.
        Index("ix_users_username", "username", unique=True),
        Index("ix_users_school", "school_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(_USERNAME_LEN), nullable=False)
    # The verifier. Never returned by any route, never logged — see sis/domain/staff.py.
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    full_name_en: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    full_name_ar: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    preferred_language: Mapped[str] = mapped_column(String(8), default="en", nullable=False)
    school_id: Mapped[int | None] = mapped_column(
        ForeignKey("schools.id", ondelete="RESTRICT"), nullable=True
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    failed_attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    locked_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_login_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now, nullable=False
    )


class UserRole(Base):
    """A role granted to a user, bounded to a scope. The additive unit.

    The unique key is the whole grant — user, role, scope type, scope id — so the same
    role at two different rungs is two rows and granting the same one twice is idempotent.
    `scope_id` is nullable only because a `global` scope names nothing.

    Nothing here can express a denial. See `sis/domain/rbac.py` for why.
    """

    __tablename__ = "user_roles"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "role_id",
            "scope_type",
            "scope_id",
            name="uq_user_roles_user_role_scope",
        ),
        # Serves the one query every authenticated request makes: "what does this user hold".
        Index("ix_user_roles_user", "user_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    role_id: Mapped[int] = mapped_column(
        ForeignKey("roles.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    scope_type: Mapped[str] = mapped_column(
        String(_SCOPE_TYPE_LEN), default="school", nullable=False
    )
    # No foreign key, deliberately: the id means a different table depending on
    # `scope_type`, and there is no single column a constraint could point at. The service
    # resolves it, and an assignment to a deleted rung simply covers nothing.
    scope_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # A username rather than an id, so an audit line reads without a join.
    granted_by: Mapped[str] = mapped_column(String(_USERNAME_LEN), default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )


class UserSession(Base):
    """A signed-in browser. The token is hashed; the raw value exists only in the client.

    Revocation is a timestamp rather than a delete, so "this session ended at 16:04"
    stays a fact after the fact. Expired rows are pruned on login rather than by a job:
    the table is small, the prune is one indexed delete, and a cron nobody notices has
    stopped is how a session table becomes a million rows.
    """

    __tablename__ = "user_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    token_hash: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    client_ip: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )


class Teacher(Base):
    """A member of teaching staff. Points at a user; is not one.

    Nullable `user_id` because the two come apart in both directions — a teacher recorded
    on the timetable before IT creates their account is a real state, and so is a teacher
    whose account is disabled while their marks stay attached to their name.
    """

    __tablename__ = "teachers"
    __table_args__ = (
        UniqueConstraint("school_id", "staff_number", name="uq_teachers_school_staff_number"),
        Index("ix_teachers_user", "user_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    staff_number: Mapped[str] = mapped_column(String(_STAFF_NUMBER_LEN), nullable=False)
    school_id: Mapped[int] = mapped_column(
        ForeignKey("schools.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    full_name_en: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    full_name_ar: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    email: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    phone: Mapped[str] = mapped_column(String(_PHONE_LEN), default="", nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )


class TeacherSubject(Base):
    """What a teacher teaches. The principal's record.

    Subjects are year-scoped in this schema, so this is per academic year — which is
    correct rather than incidental: a teacher moves from Science to Physics between years,
    and last year's marks must stay attached to the subject they were actually awarded in.
    """

    __tablename__ = "teacher_subjects"
    __table_args__ = (
        UniqueConstraint("teacher_id", "subject_id", name="uq_teacher_subjects_teacher_subject"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    teacher_id: Mapped[int] = mapped_column(
        ForeignKey("teachers.id", ondelete="CASCADE"), nullable=False, index=True
    )
    subject_id: Mapped[int] = mapped_column(
        ForeignKey("subjects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Denormalised from the subject so "who teaches in 2025-2026" is one indexed read
    # rather than a join every screen repeats.
    academic_year_id: Mapped[int] = mapped_column(
        ForeignKey("academic_years.id", ondelete="CASCADE"), nullable=False, index=True
    )
    is_primary: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )


class TeacherYearLevel(Base):
    """Which rungs a teacher teaches their subject on. Also the principal's record.

    This is the outer boundary a year supervisor works inside: the supervisor picks rooms,
    but only rooms on a rung that appears here. See `sis.domain.staff.assignable_classes`.
    """

    __tablename__ = "teacher_year_levels"
    __table_args__ = (
        UniqueConstraint(
            "teacher_id",
            "year_level_id",
            "subject_id",
            name="uq_teacher_year_levels_teacher_level_subject",
        ),
        Index("ix_teacher_year_levels_level", "year_level_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    teacher_id: Mapped[int] = mapped_column(
        ForeignKey("teachers.id", ondelete="CASCADE"), nullable=False, index=True
    )
    year_level_id: Mapped[int] = mapped_column(
        ForeignKey("year_levels.id", ondelete="CASCADE"), nullable=False
    )
    subject_id: Mapped[int] = mapped_column(
        ForeignKey("subjects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )


class TeacherClassSection(Base):
    """Which rooms a teacher stands in. The year supervisor's record.

    `assigned_by` is a username rather than an id for the same reason `UserRole` keeps
    one: "who put this teacher in 4B" is a question asked in a corridor, and the answer
    should not need a join.
    """

    __tablename__ = "teacher_class_sections"
    __table_args__ = (
        UniqueConstraint(
            "teacher_id",
            "class_section_id",
            "subject_id",
            name="uq_teacher_class_sections_teacher_class_subject",
        ),
        # Serves "who teaches this class", which the class screen asks once per subject.
        Index("ix_teacher_class_sections_class", "class_section_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    teacher_id: Mapped[int] = mapped_column(
        ForeignKey("teachers.id", ondelete="CASCADE"), nullable=False, index=True
    )
    class_section_id: Mapped[int] = mapped_column(
        ForeignKey("class_sections.id", ondelete="CASCADE"), nullable=False
    )
    subject_id: Mapped[int] = mapped_column(
        ForeignKey("subjects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    assigned_by: Mapped[str] = mapped_column(
        String(_USERNAME_LEN), default="", nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )


class SystemSetting(Base):
    """Estate-wide switches the administrator can turn, one row per key.

    A key/value table rather than a column per setting: these are read rarely, written
    rarely, and adding one must not be a migration against every school's database. The
    only one that exists today is `system.status`.
    """

    __tablename__ = "system_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(_SETTING_KEY_LEN), unique=True, nullable=False)
    value: Mapped[str] = mapped_column(Text, default="", nullable=False)
    # Free text the administrator types when pausing: "upgrading to 0.2, back at 18:00".
    # Shown to everyone who is refused, because a refusal with no reason generates a
    # phone call to the same administrator.
    note: Mapped[str] = mapped_column(Text, default="", nullable=False)
    updated_by: Mapped[str] = mapped_column(
        String(_USERNAME_LEN), default="", nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now, nullable=False
    )


__all__ = [
    "AcademicYear",
    "AccessAudit",
    "ApiKey",
    "ClassEnrolment",
    "ClassSection",
    "EducationalSystem",
    "Guardian",
    "GuardianPhone",
    "ImportBatch",
    "ImportRow",
    "PermissionRow",
    "Role",
    "RolePermission",
    "Student",
    "StudentGuardian",
    "Subject",
    "SubjectGrade",
    "SystemSetting",
    "Teacher",
    "TeacherClassSection",
    "TeacherSubject",
    "TeacherYearLevel",
    "Term",
    "User",
    "UserRole",
    "UserSession",
    "YearLevel",
]
