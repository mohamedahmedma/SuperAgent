"""A child's room, what it studies, and who teaches it.

Values, not rows, like [marks.py](marks.py) and [timetable.py](timetable.py): this service
asks, reshapes, and forgets. Three decisions shape it.

**One value for three questions, because they share one fact.** "Which class is she in",
"what does she study" and "who teaches her" all hang off the room she was placed in for the
term, and the system of record resolves that once. Asked separately they can disagree — one
room's subjects beside another room's staff — so they arrive together and the facade
projects them into three parent-facing reads afterwards.

**Three empties, three meanings, and none of them is an error.** `NO_CLASS` says no
placement covered the term: she had left, or had not yet joined, and there is no room to ask
about. With a room present, an empty `subjects` says nobody has curated that rung's subject
board, and an empty `teachers` says nobody has entered the room's staffing. The first is a
fact about the child; the other two are facts about the school's admin. Flattened together
they become "your daughter has no teachers", which is false in two cases out of three and
alarming in all of them.

**A teacher carries a name and a subject, and there is nothing else on the type to leak.**
The system of record holds an email, a phone number and a username for every member of
staff. None of them is here, and that is the point: the projection is the privacy boundary,
not a filter somebody downstream has to remember to apply. A parent needs to know who
teaches their child, not how to reach a member of staff directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ClassroomStatus(str, Enum):
    """Whether there is a room to talk about at all.

    Only two members, deliberately. The other two ways this answer can be empty — no
    subject board, no staffing — are properties of the two lists rather than of the
    envelope, because a room can have either, both or neither and one enum cannot say so.
    See the module docstring.
    """

    #: A placement covered the term. The lists may still be empty; that is about the school.
    OK = "ok"
    #: No placement covered the term. There is no room, so nothing is missing.
    NO_CLASS = "no_class"


@dataclass(frozen=True)
class StudySubject:
    """One subject on the rung's board — something the child is taught.

    The rung's ASSIGNED subjects, not the year's whole catalogue: a school teaches Physics,
    but only Secondary sits it. Retired subjects are absent, because a dropped subject is
    not one she studies — though it still resolves for marks already stated against it.

    Both names travel. A subject rendered with no name at all is worse than one rendered in
    the wrong language, and this service holds no translation to invent the missing side
    with — the same rule the grade assembler follows for a subject with one spelling.
    """

    code: str
    name_ar: str = ""
    name_en: str = ""


@dataclass(frozen=True)
class ClassTeacher:
    """One teacher in the room, and the subject they teach in it.

    One entry per (teacher, subject). A teacher taking two subjects for the class appears
    twice; a subject taught by two teachers has two entries. Both are states the system of
    record permits — its table is unique on (teacher, class, subject) and only one of its
    two write paths guards against a second teacher — so a consumer must treat this as a
    list and never as one name per subject.
    """

    full_name_ar: str = ""
    full_name_en: str = ""
    subject_code: str = ""
    subject_name_ar: str = ""
    subject_name_en: str = ""


@dataclass(frozen=True)
class StudentClassroom:
    """One child's room for one term: its name, its subjects and its staff.

    `class_name_ar`/`class_name_en` are what the school itself calls the room — "Primary 1
    Class 1", "3/1" — and are the answer to "which class is my child in". `class_code` is
    the school's internal key for it ("3A", "P1-01"), carried for correlation rather than
    for reading out to a family.

    **`teachers` is present tense.** The term resolves which room she sat in; the staffing
    table carries no term, so who teaches that room can only be reported as of now. A
    consumer must not describe it as who taught her last November.
    """

    status: ClassroomStatus = ClassroomStatus.NO_CLASS
    class_code: str = ""
    class_name_ar: str = ""
    class_name_en: str = ""
    year_level_code: str = ""
    year_level_name_ar: str = ""
    year_level_name_en: str = ""
    subjects: tuple[StudySubject, ...] = ()
    teachers: tuple[ClassTeacher, ...] = ()

    @property
    def has_class(self) -> bool:
        """Whether a placement covered the term at all."""
        return self.status is ClassroomStatus.OK


__all__ = [
    "ClassTeacher",
    "ClassroomStatus",
    "StudentClassroom",
    "StudySubject",
]
