"""A child this guardian may be told about.

Deliberately thin, and thinner than the ORM row it replaced. It carries what a parent's
question needs answered about — a name to greet her by and the year group that narrows a
general question like "what are the fees for my son?" — and nothing about her record.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PermittedStudent:
    """A child this guardian may be told about, as far as the system of record is concerned.

    Deliberately thin, and thinner than the ORM row it replaces. `grade_level` and
    `section` began as Moodle's: the course-binding lookup keys on them, and the SIS path
    left both empty because it keys grades on the student number alone.

    `grade_level` is no longer empty on the SIS path. It carries the child's year group —
    the school's own label, "Year 4" or "الصف الرابع" — because a parent asking a general
    question ("what are the fees for my son?") is asking it about one year, and a fee
    table covers every year in the school. Without this the answer is the whole table.

    `section` stays empty there, deliberately: which ROOM a child sits in narrows nothing
    a parent asks about, and it changes mid-year in a way a year group does not.
    """

    student_id: str
    full_name_ar: str = ""
    full_name_en: str = ""
    grade_level: str = ""
    section: str = ""
    #: "male" | "female" | "unspecified", as the system of record states it. Relayed, not
    #: interpreted: this service holds no data and forms no view about a child, and the
    #: only reason it carries this at all is that the chat service needs it to understand
    #: a parent who writes "my son" rather than a name.
    gender: str = "unspecified"

    @property
    def external_id(self) -> str:
        """The name the previous ORM row used. Kept so call sites did not all have to move."""
        return self.student_id


__all__ = ["PermittedStudent"]
