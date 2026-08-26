"""Tables the records facade owns — three, and none of them about a child.

This file used to encode the split "the system of record owns academic facts, the facade
owns who may see them." It held guardians, students, the guardian-student link and its
custody flags, terms, and published report cards.

**It owns none of that now, and the deletions are the point.** Who a child's parents are
is the registrar's fact, amended by court orders and audited in `sis/`; a second copy here
would disagree with it the first time one of them was corrected, and the copy that was
wrong would be the one answering a parent. The academic calendar went the same way. The
guardian tables had already stopped being read — `records.guardian_directory` asks SIS on
every request — and this removes the schema that made it look as though they had not.

What is left is the part that genuinely belongs to a facade and to no system of record:

  * `api_keys` — which *system* may call this service. Not which parent; that is proved
    by a signed token this service cannot mint.
  * `access_audit` — an append-only record of every attempt to read a student record,
    including the refusals, correlated back to the chat turn that caused it.
  * `course_bindings` — the naming convention that maps a flat course list back onto
    "subject x section x term", for a backend whose course titles are whatever a teacher
    typed. Unused on the SIS path, which reports its own subjects, and kept precisely
    because it is what a *different* system of record would need.

Report cards are gone rather than moved. The read route had been broken since the guardian
tables stopped being populated — it looked up `student.id` on an object that has no `id` —
and no test covered it, which is a fair measure of how much the feature was worth in this
service. Freezing a published term is a real requirement; it belongs where the marks are.

Every table here is small and slow-changing, and the hot read path touches only
`api_keys`.
"""
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from records.db import Base


def _now() -> datetime:
    """Timezone-aware UTC.

    Naive timestamps are a liability in the place this schema cares about most: an audit
    row has to survive a subpoena, and a server whose local clock drifts answers that
    wrong.
    """
    return datetime.now(timezone.utc)


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
    parent-visible rollup by accident.
    """

    __tablename__ = "course_bindings"
    __table_args__ = (
        UniqueConstraint("lms_course_id", name="uq_course_binding_lms_course"),
        Index("ix_course_bindings_term_section", "term_code", "grade_level", "section"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    lms_course_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    # The backend's own course identifier, kept for reconciliation against a re-import.
    lms_idnumber: Mapped[str] = mapped_column(String(120), default="", nullable=False)

    # The school calendar's term code, not a local row id. There is no `terms` table here
    # any more: the calendar is SIS's, and a foreign key to a copy of it was the last
    # reason this service had to keep one.
    term_code: Mapped[str] = mapped_column(String(32), default="", nullable=False, index=True)
    subject_code: Mapped[str] = mapped_column(String(32), default="", nullable=False)
    subject_name_ar: Mapped[str] = mapped_column(String(120), default="", nullable=False)
    subject_name_en: Mapped[str] = mapped_column(String(120), default="", nullable=False)

    grade_level: Mapped[str] = mapped_column(String(16), default="", nullable=False)
    section: Mapped[str] = mapped_column(String(16), default="", nullable=False)

    # Excluded from parent-visible rollups without deleting the binding — the switch
    # a registrar flips while a term's grades are still being entered.
    is_published: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, nullable=False)
