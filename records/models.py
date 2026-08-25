"""Tables the records facade owns.

The split this file encodes: **the system of record owns academic facts, the facade
owns who may see them.** Grades, attendance and enrolments are never copied here — they
are read through the LMS adapter at request time. What lives here is the part a
gradebook has no good answer for:

  * the guardian relationship, including the custody restrictions that make it an
    authorisation model rather than a contact list,
  * terms, which a gradebook typically does not model at all,
  * the course-naming convention that maps a flat course list back onto
    "subject x section x term",
  * an append-only record of every student record an agent read,
  * report cards, which must be frozen snapshots rather than live recomputations.

Every table here is small and slow-changing. The hot read path touches only
`guardian_students` and `api_keys`, both of which are indexed for exactly that.
"""
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from records.db import Base


def _now() -> datetime:
    """Timezone-aware UTC.

    Naive timestamps are a liability in the two places this schema cares about most:
    an audit row has to survive a subpoena, and a report card has to say which term
    it closed in. Both are answered wrong by a server whose local clock drifts.
    """
    return datetime.now(timezone.utc)


class Term(Base):
    """One academic term — the entity Moodle does not have.

    Terms are what make grades comparable and report cards possible. `is_closed` is
    the important flag: once a term closes, its report cards are frozen and no live
    recomputation may replace them. A policy change next year must not silently
    rewrite what last year's report card said.
    """

    __tablename__ = "terms"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # Stable external key, e.g. "2026-T1". Appears in course idnumbers and in the
    # agent-facing contract, so it must never be renumbered.
    code: Mapped[str] = mapped_column(String(32), unique=True, index=True, nullable=False)
    name_en: Mapped[str] = mapped_column(String(120), default="", nullable=False)
    name_ar: Mapped[str] = mapped_column(String(120), default="", nullable=False)

    academic_year: Mapped[str] = mapped_column(String(16), default="", nullable=False, index=True)
    starts_on: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ends_on: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Closed terms serve report cards from snapshots; open terms compute live.
    is_closed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, nullable=False)


class Student(Base):
    """A student, as the facade knows them — an identity and an LMS join key.

    Deliberately thin. Nothing academic lives here: no grades, no attendance, no
    enrolment history. Those are Moodle's, and duplicating them is how two systems
    start disagreeing about a child's transcript.

    `lms_user_id` is the whole integration. It is the Moodle user id, and it is what
    every adapter call is keyed on.
    """

    __tablename__ = "students"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # The school's own student number — what a parent would read off a letter.
    external_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    # Moodle's numeric user id. Nullable because a student may be registered here
    # before their LMS account exists; reads degrade to "no LMS record" rather than 500.
    lms_user_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)

    # Both scripts, always. A 90%-Arabic school still needs the Latin form for
    # transcripts and external correspondence, and picking one at write time means
    # the other is gone.
    full_name_ar: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    full_name_en: Mapped[str] = mapped_column(String(255), default="", nullable=False)

    grade_level: Mapped[str] = mapped_column(String(16), default="", nullable=False)
    section: Mapped[str] = mapped_column(String(16), default="", nullable=False)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, nullable=False)

    guardians = relationship("GuardianStudent", back_populates="student", cascade="all, delete-orphan")


class Guardian(Base):
    """A parent or legal guardian — the subject the agent acts on behalf of.

    Guardians never hold an LMS account. The agent is their entire interface, which
    is precisely why Moodle's awkward mentor role can be skipped: nothing here needs
    to exist on the Moodle side.
    """

    __tablename__ = "guardians"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    external_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)

    full_name_ar: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    full_name_en: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    # Contact fields are how the chat front end recognises a returning parent; they
    # are not credentials and must never be accepted as proof of identity.
    phone: Mapped[str] = mapped_column(String(32), default="", nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(255), default="", nullable=False)

    preferred_language: Mapped[str] = mapped_column(String(8), default="ar", nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, nullable=False)

    students = relationship("GuardianStudent", back_populates="guardian", cascade="all, delete-orphan")


class GuardianStudent(Base):
    """Which guardian may see which student — the authorisation model itself.

    This is the single most security-sensitive table in the system, and the reason
    the facade exists as its own service rather than a Moodle plugin.

    `can_view_records` is separate from the link existing on purpose. Custody
    arrangements produce guardians who are legitimately on file — they receive
    emergency contact, they attend events — and who are legally barred from seeing
    academic records. Modelling that as "delete the row" loses the contact
    relationship; modelling it as a flag keeps both facts, and makes the restriction
    something a court order can be attached to via `restriction_note`.

    The flag is checked server-side on every read. It is never sent to the model,
    never included in a prompt, and never used as a filter the agent could be talked
    out of applying.
    """

    __tablename__ = "guardian_students"
    __table_args__ = (
        UniqueConstraint("guardian_id", "student_id", name="uq_guardian_student"),
        # The hot path: "which students may this guardian see", asked on every single
        # agent request before any LMS call is made.
        Index("ix_guardian_students_lookup", "guardian_id", "can_view_records"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    guardian_id: Mapped[int] = mapped_column(
        ForeignKey("guardians.id", ondelete="CASCADE"), nullable=False, index=True
    )
    student_id: Mapped[int] = mapped_column(
        ForeignKey("students.id", ondelete="CASCADE"), nullable=False, index=True
    )

    relationship_type: Mapped[str] = mapped_column(String(32), default="parent", nullable=False)

    # Default deny. A link created without an explicit grant reveals nothing, so a
    # half-finished import or a buggy sync cannot leak a record.
    can_view_records: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    # Free text for the human reason — court order reference, school decision, date.
    # Never rendered to the parent; it exists for the audit and the registrar.
    restriction_note: Mapped[str] = mapped_column(Text, default="", nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now, nullable=False)

    guardian = relationship("Guardian", back_populates="students")
    student = relationship("Student", back_populates="guardians")


class ApiKey(Base):
    """Service credential for a calling system — the agent, an admin script, a sync job.

    Prefix plus hash, never the key itself. The full secret is returned exactly once
    at creation and is unrecoverable afterwards; `prefix` is what makes a key
    identifiable in logs and revocable by a human reading an audit row.

    Crucially, a key authenticates a *system*, not a person. It says "this caller is
    the school assistant", never "this caller is Ahmed's mother". Holding a key grants
    the ability to ask on behalf of a guardian; it does not grant that guardian's
    access. That separation is what keeps a leaked key from being a student-body dump.
    """

    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # First few characters of the key, stored in clear so a key can be named in logs.
    prefix: Mapped[str] = mapped_column(String(12), unique=True, index=True, nullable=False)
    key_hash: Mapped[str] = mapped_column(String(128), nullable=False)

    label: Mapped[str] = mapped_column(String(120), default="", nullable=False)
    # "agent" reads records on behalf of a guardian. "admin" manages links and keys
    # and may never read a student record through the agent-facing routes.
    scope: Mapped[str] = mapped_column(String(16), default="agent", nullable=False)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, nullable=False)


class AccessAudit(Base):
    """Append-only log of every attempt to read a student record.

    Append-only by contract: nothing in this service updates or deletes a row here,
    and the admin routes expose read only. When the school is audited — and a school
    holding minors' records will be — this table is the answer to "who saw my child's
    grades, and when".

    Denials are logged as loudly as successes. A run of `allowed=False` rows against
    one guardian is the signal that someone is probing, and it is invisible if only
    successful reads are recorded.
    """

    __tablename__ = "access_audit"
    __table_args__ = (
        # The two questions ever asked of this table: "everything about this child"
        # and "everything this guardian did".
        Index("ix_access_audit_student_time", "student_external_id", "created_at"),
        Index("ix_access_audit_guardian_time", "guardian_external_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # External ids, not foreign keys. An audit row must survive the deletion of the
    # guardian it refers to, and a cascade would erase precisely the history someone
    # is most likely to want after an account is removed.
    guardian_external_id: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    student_external_id: Mapped[str] = mapped_column(String(64), default="", nullable=False)

    api_key_prefix: Mapped[str] = mapped_column(String(12), default="", nullable=False, index=True)
    endpoint: Mapped[str] = mapped_column(String(160), default="", nullable=False)

    allowed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    # Machine-readable outcome: "ok", "no_link", "records_restricted", "unknown_student",
    # "lms_unavailable". Aggregating on free text does not work at audit time.
    reason: Mapped[str] = mapped_column(String(40), default="", nullable=False)

    # Correlates a records read back to the chat turn that caused it, so a complaint
    # about one conversation can be traced without scanning by timestamp.
    request_id: Mapped[str] = mapped_column(String(64), default="", nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, nullable=False)


class CourseBinding(Base):
    """Maps one Moodle course onto "subject x section x term".

    Moodle's course list is flat and its idnumber is free text, so without this table
    a course called `2026-T1-G7A-MATH` means whatever the person who typed it meant.
    Binding it explicitly is what stops the naming convention from becoming a parser
    scattered across the codebase — the convention is data, and a course that has not
    been bound is simply not visible to parents yet.

    That last property is a feature: a teacher's sandbox course cannot leak into a
    report card by accident.
    """

    __tablename__ = "course_bindings"
    __table_args__ = (
        UniqueConstraint("lms_course_id", name="uq_course_binding_lms_course"),
        Index("ix_course_bindings_term_section", "term_id", "grade_level", "section"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    lms_course_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    # The Moodle course idnumber, kept for reconciliation against a re-import.
    lms_idnumber: Mapped[str] = mapped_column(String(120), default="", nullable=False)

    term_id: Mapped[int] = mapped_column(ForeignKey("terms.id", ondelete="RESTRICT"), nullable=False, index=True)
    subject_code: Mapped[str] = mapped_column(String(32), default="", nullable=False)
    subject_name_ar: Mapped[str] = mapped_column(String(120), default="", nullable=False)
    subject_name_en: Mapped[str] = mapped_column(String(120), default="", nullable=False)

    grade_level: Mapped[str] = mapped_column(String(16), default="", nullable=False)
    section: Mapped[str] = mapped_column(String(16), default="", nullable=False)

    # Excluded from parent-visible rollups without deleting the binding — the switch
    # a registrar flips while a term's grades are still being entered.
    is_published: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, nullable=False)


class ReportCard(Base):
    """A frozen, published snapshot of one student's term result.

    The whole point is that it does not recompute. `payload` holds the fully rendered
    figures as they stood when the registrar published them — subject rows, weights,
    letter grades, attendance totals. If a teacher corrects a grade next month, or the
    school changes its rounding policy next year, this row does not move; a correction
    produces a *new* version, and the old one stays readable.

    `version` plus `superseded_at` is what makes that auditable rather than silent. A
    parent who asks "why did my child's grade change" gets a real answer.
    """

    __tablename__ = "report_cards"
    __table_args__ = (
        UniqueConstraint("student_id", "term_id", "version", name="uq_report_card_version"),
        Index("ix_report_cards_student_term", "student_id", "term_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("students.id", ondelete="CASCADE"), nullable=False, index=True)
    term_id: Mapped[int] = mapped_column(ForeignKey("terms.id", ondelete="RESTRICT"), nullable=False, index=True)

    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, nullable=False)
    published_by: Mapped[str] = mapped_column(String(120), default="", nullable=False)
    # Set when a later version replaces this one. Never deleted.
    superseded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
