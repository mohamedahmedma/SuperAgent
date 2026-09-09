"""The people who log in, and the teachers among them.

Two types, kept apart on purpose.

A **User** is an account: a username, a verifier, and whether it is allowed to sign in.
Every role in the school is a user — the owner, the principal, the supervisor, the
attendance clerk. Nothing about teaching lives here.

A **Teacher** is a member of staff who stands in front of a class. It carries the
assignments: which subject, which rungs, which rooms. It points at a user rather than
being one, because the two genuinely come apart in both directions — a teacher on
maternity leave keeps her assignments while her account is disabled, and a principal has
an account and teaches nothing.

Splitting them is also what makes "Teacher **and** Supervisor" expressible without a
second account. The roles are `user_roles` rows; the teaching is a `teachers` row; a
person who does both has one login.

**Who assigns what**, because it is the part that gets muddled:

    principal        -> subject, and the rungs the teacher teaches it on
    year supervisor  -> which rooms on their own rung, for teachers already on it

So `TeacherSubject` and `TeacherYearLevel` are the principal's records, and
`TeacherClassSection` is the supervisor's. The supervisor can only narrow what the
principal granted — `assignable_classes` is the rule, and it lives here rather than in a
handler so it can be unit-tested without a database.

The domain never reads the clock. Lockout and expiry are answered against a `now` the
caller supplies.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Final
from enum import StrEnum

from sis.domain.errors import ValidationError

# Long enough that a stolen laptop is not an open session tomorrow, short enough that a
# registrar working a full day is not asked to sign in twice.
SESSION_LIFETIME: Final[timedelta] = timedelta(hours=12)

# Five wrong passwords, then fifteen minutes. Slows an online guess to a few hundred
# attempts a day without giving anyone a way to lock a colleague out for the afternoon.
MAX_FAILED_ATTEMPTS: Final[int] = 5
LOCKOUT: Final[timedelta] = timedelta(minutes=15)

USERNAME_MAX_LENGTH: Final[int] = 64
# The shortest password this service will store. Stated as a constant so the seed, the
# API schema and the change-password route cannot disagree about it.
PASSWORD_MIN_LENGTH: Final[int] = 8


class StaffAttendanceState(StrEnum):
    PRESENT = "present"
    ABSENT = "absent"
    LATE = "late"
    LEAVE = "leave"


@dataclass(frozen=True, slots=True)
class User:
    """One account. Frozen: a failed login writes a new row state, it does not mutate this.

    `school_id` is `None` for an account that belongs to no single school — the system
    administrator, and only them. Everyone else is bound to one, and that binding is a
    second wall behind the role scopes: a principal whose grants were somehow widened
    still reads their own school's database and no other.
    """

    id: int
    username: str
    password_hash: str = field(repr=False)  # verifier; kept out of every accidental log
    full_name_en: str = ""
    full_name_ar: str = ""
    email: str = ""
    preferred_language: str = "en"
    school_id: int | None = None
    is_active: bool = True
    failed_attempts: int = 0
    locked_until: datetime | None = None
    last_login_at: datetime | None = None

    def __post_init__(self) -> None:
        if not str(self.username).strip():
            raise ValidationError("a user needs a username", field="username")
        if len(str(self.username)) > USERNAME_MAX_LENGTH:
            raise ValidationError(
                f"username is longer than {USERNAME_MAX_LENGTH} characters", field="username"
            )
        if not str(self.password_hash).strip():
            raise ValidationError("a user needs a password", field="password_hash")
        for name in ("locked_until", "last_login_at"):
            moment: datetime | None = getattr(self, name)
            # A naive datetime beside an aware `now` raises on comparison, which would turn
            # every login into a 500 rather than a refusal. Refuse it where it is built.
            if moment is not None and moment.tzinfo is None:
                raise ValidationError(f"{name} must be timezone-aware", field=name)

    def is_locked_at(self, now: datetime) -> bool:
        return self.locked_until is not None and self.locked_until > now

    def can_sign_in_at(self, now: datetime) -> bool:
        """Disabled and locked-out answer the same way to a caller, on purpose."""
        return self.is_active and not self.is_locked_at(now)

    def display_name(self, language: str = "en") -> str:
        """The name to show, falling back to the other language and then the username."""
        arabic = str(language).lower().startswith("ar")
        preferred = self.full_name_ar if arabic else self.full_name_en
        return preferred or self.full_name_en or self.full_name_ar or self.username

    def __str__(self) -> str:
        return self.username


@dataclass(frozen=True, slots=True)
class Teacher:
    """A member of teaching staff. Identified by a staff number for their whole career.

    The staff number is the immutable handle, like a student number: a teacher who marries
    and changes name is the same teacher, and every mark they recorded stays attached.
    """

    id: int
    staff_number: str
    school_id: int
    user_id: int | None = None
    full_name_en: str = ""
    full_name_ar: str = ""
    is_active: bool = True

    def __post_init__(self) -> None:
        if not str(self.staff_number).strip():
            raise ValidationError("a teacher needs a staff number", field="staff_number")

    def display_name(self, language: str = "en") -> str:
        arabic = str(language).lower().startswith("ar")
        preferred = self.full_name_ar if arabic else self.full_name_en
        return preferred or self.full_name_en or self.full_name_ar or self.staff_number


@dataclass(frozen=True, slots=True)
class TeacherAssignment:
    """What one teacher may teach, as the two authorities between them decided.

    Read as: *this teacher teaches `subject_id`; on the rungs in `year_level_ids`; in the
    rooms in `class_section_ids`.* The rooms are always a subset of the rungs — see
    `assignable_classes`, which is the rule that keeps a supervisor inside their remit.
    """

    teacher_id: int
    subject_id: int
    year_level_ids: tuple[int, ...] = ()
    class_section_ids: tuple[int, ...] = ()

    def teaches_class(self, class_section_id: int) -> bool:
        return class_section_id in self.class_section_ids

    def teaches_rung(self, year_level_id: int) -> bool:
        return year_level_id in self.year_level_ids


def assignable_classes(
    *,
    assignment: TeacherAssignment,
    supervisor_year_level_ids: Iterable[int],
    classes_by_year_level: dict[int, tuple[int, ...]],
) -> tuple[int, ...]:
    """The rooms a supervisor may put this teacher into. The intersection of three facts.

    A supervisor may only touch their own rungs; the teacher may only be placed on rungs
    the principal put them on; and a room only counts if it is on such a rung. Computing
    the intersection here rather than filtering in a handler means the boundary is one
    testable function instead of three `if`s that each have to be remembered.

    Returned in the order the rungs are given, so a screen listing them is stable.
    """
    permitted_rungs = [
        rung for rung in supervisor_year_level_ids if rung in assignment.year_level_ids
    ]
    rooms: list[int] = []
    for rung in permitted_rungs:
        for class_section_id in classes_by_year_level.get(rung, ()):
            if class_section_id not in rooms:
                rooms.append(class_section_id)
    return tuple(rooms)


@dataclass(frozen=True, slots=True)
class Session:
    """A signed-in browser. The token itself is never stored — only its hash.

    Same reasoning as the API key: what reaches the database cannot be replayed if the
    database leaks. Revocation is a timestamp rather than a delete so "this session was
    ended at 16:04" survives as a fact.
    """

    id: int
    user_id: int
    token_hash: str = field(repr=False)
    expires_at: datetime | None = None
    revoked_at: datetime | None = None
    created_at: datetime | None = None

    def is_usable_at(self, now: datetime) -> bool:
        if self.revoked_at is not None:
            return False
        return self.expires_at is None or self.expires_at > now


__all__ = [
    "LOCKOUT",
    "MAX_FAILED_ATTEMPTS",
    "PASSWORD_MIN_LENGTH",
    "SESSION_LIFETIME",
    "USERNAME_MAX_LENGTH",
    "Session",
    "Teacher",
    "TeacherAssignment",
    "User",
    "assignable_classes",
]
