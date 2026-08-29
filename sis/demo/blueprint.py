"""The demo school, declared as data rather than as a script.

Everything the seeder writes is described here first: the sections, the rungs, the rooms,
the subjects, the staff and who they are. The seeder walks this and does the inserting.
The split is what makes the demo reviewable — a person asking "which classes exist and
what will 4/1 ب be called" reads a table instead of following a loop.

**One school, both sections.** `DEMO` runs an Arabic section and a language section side
by side, because that is the arrangement the service has to support and a demo with only
one of them proves nothing about the other.

**Nothing here is a real school, a real child or a real member of staff.** The names come
from `names.py`, which is a pool of invented combinations, and the school itself is
fictional. The demo passwords are weak on purpose and are documented as such: this data is
for a development database and the seeder refuses to write it anywhere else without being
told to (see `sis/demo/seeder.py`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Final

from sis.domain.naming import EducationalSystemKind
from sis.domain.rbac import RoleCode, ScopeType
from sis.domain.structure import Stage

# The school every demo row hangs off. `reset` deletes by this code and nothing else, so
# a database holding both demo and real schools loses only the demo one.
SCHOOL_CODE: Final[str] = "DEMO"
SCHOOL_NAME_EN: Final[str] = "Nile International School (Demo)"
SCHOOL_NAME_AR: Final[str] = "مدرسة النيل الدولية (تجريبية)"

YEAR_CODE: Final[str] = "2025-2026"
YEAR_STARTS: Final[date] = date(2025, 9, 1)
YEAR_ENDS: Final[date] = date(2026, 6, 30)

# The password every demo account shares. One value, so the account table is readable at a
# glance and a tester does not have to look up a different string per role. It is
# deliberately not a secret, and the seeder says so out loud when it runs.
DEMO_PASSWORD: Final[str] = "Demo#2026"


@dataclass(frozen=True, slots=True)
class SectionSpec:
    """One educational system: the Arabic section or the language section."""

    code: str
    kind: EducationalSystemKind
    name_en: str
    name_ar: str
    display_order: int


SECTIONS: Final[tuple[SectionSpec, ...]] = (
    SectionSpec(
        code="AR",
        kind=EducationalSystemKind.ARABIC,
        name_en="Arabic Section",
        name_ar="القسم العربي",
        display_order=1,
    ),
    SectionSpec(
        code="LANG",
        kind=EducationalSystemKind.LANGUAGE,
        name_en="Language Section",
        name_ar="قسم اللغات",
        display_order=2,
    ),
)


@dataclass(frozen=True, slots=True)
class RungSpec:
    """One year level, and the rooms on it.

    `code` is the machine identity and obeys `YearCode` — uppercase, no spaces. The name a
    person reads is generated from `stage` + `grade_number` by `sis/domain/naming.py`, and
    `name_en` / `name_ar` here are what gets stored so existing screens have something to
    print without calling the generator.

    `rooms` is a list of `(section_number, label_en, label_ar)`. A language section leaves
    `section_number` at `None` and names its rooms with letters; an Arabic section numbers
    them, and the number is the second half of `1/2 ب`.
    """

    section: str
    code: str
    stage: Stage
    grade_number: int
    name_en: str
    name_ar: str
    display_order: int
    rooms: tuple[tuple[int | None, str, str], ...] = ()


# -- The Arabic section ----------------------------------------------------------------
#
# Rungs count from one inside each stage, which is what an Egyptian school does: "الصف
# الأول الابتدائي" and "الصف الأول الإعدادي" are both "first", of different stages. That is
# why `grade_number` restarts at 1 for preparatory and secondary below, and why the stage
# letter is part of the room's written name.

_ARABIC_RUNGS: Final[tuple[RungSpec, ...]] = (
    RungSpec(
        section="AR", code="AR-KG1", stage=Stage.GARDEN, grade_number=1,
        name_en="KG 1", name_ar="KG 1", display_order=1,
        rooms=((1, "Class 1", "1/1 ر"),),
    ),
    RungSpec(
        section="AR", code="AR-KG2", stage=Stage.GARDEN, grade_number=2,
        name_en="KG 2", name_ar="KG 2", display_order=2,
        rooms=((1, "Class 1", "2/1 ر"),),
    ),
    RungSpec(
        section="AR", code="AR-P1", stage=Stage.PRIMARY, grade_number=1,
        name_en="First Primary", name_ar="الصف الأول الابتدائي", display_order=11,
        # Two rooms, so the "several classes on one rung" case is testable.
        rooms=((1, "Class 1", "1/1 ب"), (2, "Class 2", "1/2 ب")),
    ),
    RungSpec(
        section="AR", code="AR-P2", stage=Stage.PRIMARY, grade_number=2,
        name_en="Second Primary", name_ar="الصف الثاني الابتدائي", display_order=12,
        rooms=((1, "Class 1", "2/1 ب"),),
    ),
    RungSpec(
        section="AR", code="AR-P3", stage=Stage.PRIMARY, grade_number=3,
        name_en="Third Primary", name_ar="الصف الثالث الابتدائي", display_order=13,
        rooms=((1, "Class 1", "3/1 ب"),),
    ),
    RungSpec(
        section="AR", code="AR-P4", stage=Stage.PRIMARY, grade_number=4,
        name_en="Fourth Primary", name_ar="الصف الرابع الابتدائي", display_order=14,
        # Two rooms: this is the rung the demo supervisor owns, and a supervisor with one
        # room cannot demonstrate "assign a teacher to A but not B".
        rooms=((1, "Class 1", "4/1 ب"), (2, "Class 2", "4/2 ب")),
    ),
    RungSpec(
        section="AR", code="AR-PR1", stage=Stage.PREPARATORY, grade_number=1,
        name_en="First Preparatory", name_ar="الصف الأول الإعدادي", display_order=21,
        rooms=((1, "Class 1", "1/1 ع"),),
    ),
    RungSpec(
        section="AR", code="AR-PR2", stage=Stage.PREPARATORY, grade_number=2,
        name_en="Second Preparatory", name_ar="الصف الثاني الإعدادي", display_order=22,
        rooms=((1, "Class 1", "2/1 ع"),),
    ),
    RungSpec(
        section="AR", code="AR-PR3", stage=Stage.PREPARATORY, grade_number=3,
        name_en="Third Preparatory", name_ar="الصف الثالث الإعدادي", display_order=23,
        rooms=((1, "Class 1", "3/1 ع"),),
    ),
    RungSpec(
        section="AR", code="AR-S1", stage=Stage.SECONDARY, grade_number=1,
        name_en="First Secondary", name_ar="الصف الأول الثانوي", display_order=31,
        rooms=((1, "Class 1", "1/1 ث"),),
    ),
    RungSpec(
        section="AR", code="AR-S2", stage=Stage.SECONDARY, grade_number=2,
        name_en="Second Secondary", name_ar="الصف الثاني الثانوي", display_order=32,
        rooms=((1, "Class 1", "2/1 ث"),),
    ),
    RungSpec(
        section="AR", code="AR-S3", stage=Stage.SECONDARY, grade_number=3,
        name_en="Third Secondary", name_ar="الصف الثالث الثانوي", display_order=33,
        rooms=((1, "Class 1", "3/1 ث"),),
    ),
)

# -- The language section --------------------------------------------------------------
#
# One continuous count, Grade 3 upward, with kindergarten counted separately as KG 1 and
# KG 2. Rooms are letters, except in kindergarten where they are names the school chose —
# which is the whole point of storing the label rather than deriving it.

_LANGUAGE_RUNGS: Final[tuple[RungSpec, ...]] = (
    RungSpec(
        section="LANG", code="LG-KG1", stage=Stage.GARDEN, grade_number=1,
        name_en="KG 1", name_ar="KG 1", display_order=1,
        rooms=((None, "Simba Class", "فصل سيمبا"), (None, "Princess Class", "فصل الأميرات")),
    ),
    RungSpec(
        section="LANG", code="LG-KG2", stage=Stage.GARDEN, grade_number=2,
        name_en="KG 2", name_ar="KG 2", display_order=2,
        rooms=((None, "Jellyfish Class", "فصل قنديل البحر"),),
    ),
    RungSpec(
        section="LANG", code="LG-G3", stage=Stage.PRIMARY, grade_number=3,
        name_en="Grade 3", name_ar="الصف الثالث", display_order=13,
        rooms=((None, "A", "أ"), (None, "B", "ب")),
    ),
    RungSpec(
        section="LANG", code="LG-G4", stage=Stage.PRIMARY, grade_number=4,
        name_en="Grade 4", name_ar="الصف الرابع", display_order=14,
        rooms=((None, "A", "أ"), (None, "B", "ب")),
    ),
    RungSpec(
        section="LANG", code="LG-G5", stage=Stage.PRIMARY, grade_number=5,
        name_en="Grade 5", name_ar="الصف الخامس", display_order=15,
        rooms=((None, "A", "أ"),),
    ),
    RungSpec(
        section="LANG", code="LG-G6", stage=Stage.PRIMARY, grade_number=6,
        name_en="Grade 6", name_ar="الصف السادس", display_order=16,
        rooms=((None, "A", "أ"),),
    ),
    RungSpec(
        section="LANG", code="LG-G9", stage=Stage.PREPARATORY, grade_number=9,
        name_en="Grade 9", name_ar="الصف التاسع", display_order=29,
        rooms=((None, "A", "أ"), (None, "B", "ب")),
    ),
    RungSpec(
        section="LANG", code="LG-G10", stage=Stage.PREPARATORY, grade_number=10,
        name_en="Grade 10", name_ar="الصف العاشر", display_order=30,
        rooms=((None, "A", "أ"),),
    ),
    RungSpec(
        section="LANG", code="LG-G11", stage=Stage.PREPARATORY, grade_number=11,
        name_en="Grade 11", name_ar="الصف الحادي عشر", display_order=31,
        rooms=((None, "A", "أ"),),
    ),
    RungSpec(
        section="LANG", code="LG-G12", stage=Stage.SECONDARY, grade_number=12,
        name_en="Grade 12", name_ar="الصف الثاني عشر", display_order=42,
        rooms=((None, "A", "أ"),),
    ),
    RungSpec(
        section="LANG", code="LG-G13", stage=Stage.SECONDARY, grade_number=13,
        name_en="Grade 13", name_ar="الصف الثالث عشر", display_order=43,
        rooms=((None, "A", "أ"),),
    ),
    RungSpec(
        section="LANG", code="LG-G14", stage=Stage.SECONDARY, grade_number=14,
        name_en="Grade 14", name_ar="الصف الرابع عشر", display_order=44,
        rooms=((None, "A", "أ"),),
    ),
)

RUNGS: Final[tuple[RungSpec, ...]] = _ARABIC_RUNGS + _LANGUAGE_RUNGS


@dataclass(frozen=True, slots=True)
class SubjectSpec:
    code: str
    name_en: str
    name_ar: str
    display_order: int


SUBJECTS: Final[tuple[SubjectSpec, ...]] = (
    SubjectSpec("AR", "Arabic", "اللغة العربية", 1),
    SubjectSpec("EN", "English", "اللغة الإنجليزية", 2),
    SubjectSpec("MA", "Mathematics", "الرياضيات", 3),
    SubjectSpec("SC", "Science", "العلوم", 4),
    SubjectSpec("SS", "Social Studies", "الدراسات الاجتماعية", 5),
    SubjectSpec("CS", "Computer Science", "الحاسب الآلي", 6),
)


@dataclass(frozen=True, slots=True)
class TermSpec:
    code: str
    name_en: str
    name_ar: str
    starts_on: date
    ends_on: date
    sequence: int


TERMS: Final[tuple[TermSpec, ...]] = (
    TermSpec(
        "2025-2026-T1", "Term 1", "الفصل الدراسي الأول",
        date(2025, 9, 1), date(2026, 1, 15), 1,
    ),
    TermSpec(
        "2025-2026-T2", "Term 2", "الفصل الدراسي الثاني",
        date(2026, 1, 25), date(2026, 6, 30), 2,
    ),
)


@dataclass(frozen=True, slots=True)
class RoleGrant:
    """One row of `user_roles`, written in terms a reader of this file can check.

    `scope_ref` is the *code* of the thing the role is bounded to — a rung code, a class
    key — resolved to an id by the seeder. Writing ids here would make the blueprint
    unreadable and would break the moment the demo is seeded into a database that already
    has rows.
    """

    role: RoleCode
    scope_type: ScopeType
    # None for a global grant; the school code, a rung code, or `rung/room` for a class.
    scope_ref: str | None = None


@dataclass(frozen=True, slots=True)
class StaffSpec:
    """One demo account, and the teaching record behind it when there is one."""

    username: str
    full_name_en: str
    full_name_ar: str
    email: str
    language: str
    roles: tuple[RoleGrant, ...]
    # Set when this person also stands in front of a class. The staff number is their
    # permanent handle, like a student number.
    staff_number: str = ""
    # Subject code the principal assigned, and the rungs they teach it on.
    subject: str = ""
    rungs: tuple[str, ...] = ()
    # Rooms the year supervisor put them in, as `rung/room-label`.
    rooms: tuple[str, ...] = ()
    # A one-line explanation of what this account is for, printed by the credentials
    # table. Written here so the documentation cannot drift from the data.
    purpose: str = ""


# The scope reference for a class: the rung code and the room's English label, joined.
# Spelled once so the blueprint and the seeder cannot disagree about the separator.
def room_ref(rung_code: str, label_en: str) -> str:
    return f"{rung_code}/{label_en}"


STAFF: Final[tuple[StaffSpec, ...]] = (
    # -- The estate ---------------------------------------------------------------------
    StaffSpec(
        username="sysadmin",
        full_name_en="System Administrator",
        full_name_ar="مدير النظام",
        email="sysadmin@demo.school",
        language="en",
        roles=(RoleGrant(RoleCode.SYSTEM_ADMIN, ScopeType.GLOBAL),),
        purpose="Everything, every school. Can pause the system and restore it.",
    ),
    # -- The school -----------------------------------------------------------------------
    StaffSpec(
        username="owner",
        full_name_en="Hala Farouk",
        full_name_ar="هالة فاروق",
        email="owner@demo.school",
        language="ar",
        roles=(RoleGrant(RoleCode.SCHOOL_OWNER, ScopeType.SCHOOL, SCHOOL_CODE),),
        purpose="Reads every screen of the demo school. Writes nothing, anywhere.",
    ),
    StaffSpec(
        username="principal",
        full_name_en="Sameh Abdelaziz",
        full_name_ar="سامح عبد العزيز",
        email="principal@demo.school",
        language="ar",
        roles=(RoleGrant(RoleCode.PRINCIPAL, ScopeType.SCHOOL, SCHOOL_CODE),),
        purpose=(
            "Reads the whole school, grants roles, and decides which subject and rungs "
            "each teacher gets."
        ),
    ),
    # -- Supervisors ----------------------------------------------------------------------
    StaffSpec(
        username="supervisor.p4",
        full_name_en="Nagwa Serageldin",
        full_name_ar="نجوى سراج الدين",
        email="supervisor.p4@demo.school",
        language="ar",
        roles=(RoleGrant(RoleCode.YEAR_SUPERVISOR, ScopeType.YEAR_LEVEL, "AR-P4"),),
        purpose=(
            "Fourth Primary only. Sees that rung entirely and puts its teachers into "
            "4/1 ب and 4/2 ب. Every other rung is refused."
        ),
    ),
    StaffSpec(
        username="supervisor.g3",
        full_name_en="Peter Ghattas",
        full_name_ar="بيتر غطاس",
        email="supervisor.g3@demo.school",
        language="en",
        roles=(RoleGrant(RoleCode.YEAR_SUPERVISOR, ScopeType.YEAR_LEVEL, "LG-G3"),),
        purpose="Grade 3 of the language section only — the same role on the other ladder.",
    ),
    StaffSpec(
        username="attendance",
        full_name_en="Mona Kamal",
        full_name_ar="منى كمال",
        email="attendance@demo.school",
        language="ar",
        roles=(
            RoleGrant(RoleCode.ATTENDANCE_SUPERVISOR, ScopeType.CLASS_SECTION,
                      room_ref("AR-P1", "Class 1")),
            RoleGrant(RoleCode.ATTENDANCE_SUPERVISOR, ScopeType.CLASS_SECTION,
                      room_ref("AR-P1", "Class 2")),
            RoleGrant(RoleCode.ATTENDANCE_SUPERVISOR, ScopeType.CLASS_SECTION,
                      room_ref("AR-P4", "Class 1")),
            RoleGrant(RoleCode.ATTENDANCE_SUPERVISOR, ScopeType.CLASS_SECTION,
                      room_ref("LG-G3", "A")),
        ),
        purpose=(
            "Takes the register for 1/1 ب, 1/2 ب, 4/1 ب and Grade 3 A. Cannot open any "
            "other class, and cannot see a mark anywhere."
        ),
    ),
    # -- Teachers -------------------------------------------------------------------------
    StaffSpec(
        username="t.arabic",
        full_name_en="Ahmed Selim",
        full_name_ar="أحمد سليم",
        email="ahmed.selim@demo.school",
        language="ar",
        roles=(
            RoleGrant(RoleCode.TEACHER, ScopeType.CLASS_SECTION, room_ref("AR-P1", "Class 1")),
            RoleGrant(RoleCode.TEACHER, ScopeType.CLASS_SECTION, room_ref("AR-P1", "Class 2")),
            RoleGrant(RoleCode.TEACHER, ScopeType.CLASS_SECTION, room_ref("AR-P2", "Class 1")),
            RoleGrant(RoleCode.TEACHER, ScopeType.CLASS_SECTION, room_ref("AR-P3", "Class 1")),
        ),
        staff_number="T-001",
        subject="AR",
        rungs=("AR-P1", "AR-P2", "AR-P3"),
        rooms=(
            room_ref("AR-P1", "Class 1"),
            room_ref("AR-P1", "Class 2"),
            room_ref("AR-P2", "Class 1"),
            room_ref("AR-P3", "Class 1"),
        ),
        purpose="Arabic across First to Third Primary, four rooms. The plain teacher case.",
    ),
    StaffSpec(
        username="t.maths",
        full_name_en="Randa Wagdy",
        full_name_ar="رندا وجدي",
        email="randa.wagdy@demo.school",
        language="ar",
        roles=(
            RoleGrant(RoleCode.TEACHER, ScopeType.CLASS_SECTION, room_ref("AR-P4", "Class 1")),
            RoleGrant(RoleCode.TEACHER, ScopeType.CLASS_SECTION, room_ref("AR-P4", "Class 2")),
            RoleGrant(RoleCode.TEACHER, ScopeType.CLASS_SECTION, room_ref("LG-G5", "A")),
        ),
        staff_number="T-002",
        subject="MA",
        rungs=("AR-P4", "LG-G5"),
        rooms=(
            room_ref("AR-P4", "Class 1"),
            room_ref("AR-P4", "Class 2"),
            room_ref("LG-G5", "A"),
        ),
        purpose="Mathematics, and the one teacher who works in both sections at once.",
    ),
    StaffSpec(
        username="t.english",
        full_name_en="Sherine Bishara",
        full_name_ar="شيرين بشارة",
        email="sherine.bishara@demo.school",
        language="en",
        roles=(
            RoleGrant(RoleCode.TEACHER, ScopeType.CLASS_SECTION, room_ref("LG-G3", "A")),
            RoleGrant(RoleCode.TEACHER, ScopeType.CLASS_SECTION, room_ref("LG-G3", "B")),
            RoleGrant(RoleCode.TEACHER, ScopeType.CLASS_SECTION, room_ref("LG-G4", "A")),
            RoleGrant(RoleCode.TEACHER, ScopeType.CLASS_SECTION, room_ref("LG-G4", "B")),
        ),
        staff_number="T-003",
        subject="EN",
        rungs=("LG-G3", "LG-G4"),
        rooms=(
            room_ref("LG-G3", "A"),
            room_ref("LG-G3", "B"),
            room_ref("LG-G4", "A"),
            room_ref("LG-G4", "B"),
        ),
        purpose="English in the language section, Grades 3 and 4.",
    ),
    StaffSpec(
        username="t.science",
        full_name_en="Tamer Halim",
        full_name_ar="تامر حليم",
        email="tamer.halim@demo.school",
        language="en",
        roles=(
            RoleGrant(RoleCode.TEACHER, ScopeType.CLASS_SECTION, room_ref("LG-G9", "A")),
            RoleGrant(RoleCode.TEACHER, ScopeType.CLASS_SECTION, room_ref("LG-G9", "B")),
            RoleGrant(RoleCode.TEACHER, ScopeType.CLASS_SECTION, room_ref("LG-G10", "A")),
            # And the supervisor of the rung he teaches on. Two roles, one login: this is
            # the additive-permissions case, and it is the reason this account exists.
            RoleGrant(RoleCode.YEAR_SUPERVISOR, ScopeType.YEAR_LEVEL, "LG-G9"),
        ),
        staff_number="T-004",
        subject="SC",
        rungs=("LG-G9", "LG-G10"),
        rooms=(
            room_ref("LG-G9", "A"),
            room_ref("LG-G9", "B"),
            room_ref("LG-G10", "A"),
        ),
        purpose=(
            "Teacher AND Academic Year Supervisor of Grade 9. Marks his own classes, and "
            "reads all of Grade 9 including classes he does not teach."
        ),
    ),
    StaffSpec(
        username="t.social",
        full_name_en="Iman Abdelhady",
        full_name_ar="إيمان عبد الهادي",
        email="iman.abdelhady@demo.school",
        language="ar",
        roles=(
            RoleGrant(RoleCode.TEACHER, ScopeType.CLASS_SECTION, room_ref("AR-PR1", "Class 1")),
            RoleGrant(RoleCode.TEACHER, ScopeType.CLASS_SECTION, room_ref("AR-PR2", "Class 1")),
            RoleGrant(RoleCode.TEACHER, ScopeType.CLASS_SECTION, room_ref("AR-PR3", "Class 1")),
            # Teacher AND attendance supervisor — the second additive case, and a
            # different pair of roles from the one above.
            RoleGrant(RoleCode.ATTENDANCE_SUPERVISOR, ScopeType.CLASS_SECTION,
                      room_ref("AR-PR1", "Class 1")),
            RoleGrant(RoleCode.ATTENDANCE_SUPERVISOR, ScopeType.CLASS_SECTION,
                      room_ref("AR-PR2", "Class 1")),
        ),
        staff_number="T-005",
        subject="SS",
        rungs=("AR-PR1", "AR-PR2", "AR-PR3"),
        rooms=(
            room_ref("AR-PR1", "Class 1"),
            room_ref("AR-PR2", "Class 1"),
            room_ref("AR-PR3", "Class 1"),
        ),
        purpose=(
            "Teacher AND Attendance Supervisor for two of her three rooms. Can mark the "
            "register in 1/1 ع and 2/1 ع but not in 3/1 ع, where she only teaches."
        ),
    ),
    StaffSpec(
        username="t.computer",
        full_name_en="Bassem Nashaat",
        full_name_ar="باسم نشأت",
        email="bassem.nashaat@demo.school",
        language="en",
        roles=(
            RoleGrant(RoleCode.TEACHER, ScopeType.CLASS_SECTION, room_ref("LG-G12", "A")),
            RoleGrant(RoleCode.TEACHER, ScopeType.CLASS_SECTION, room_ref("LG-G13", "A")),
            RoleGrant(RoleCode.TEACHER, ScopeType.CLASS_SECTION, room_ref("AR-S1", "Class 1")),
            RoleGrant(RoleCode.SUBJECT_COORDINATOR, ScopeType.SCHOOL, SCHOOL_CODE),
        ),
        staff_number="T-006",
        subject="CS",
        rungs=("LG-G12", "LG-G13", "AR-S1"),
        rooms=(
            room_ref("LG-G12", "A"),
            room_ref("LG-G13", "A"),
            room_ref("AR-S1", "Class 1"),
        ),
        purpose=(
            "Computer Science in both sections, plus Subject Coordinator — a third "
            "combination, and the one that reads across the whole school."
        ),
    ),
    StaffSpec(
        username="t.arabic2",
        full_name_en="Ghada Lotfy",
        full_name_ar="غادة لطفي",
        email="ghada.lotfy@demo.school",
        language="ar",
        roles=(
            RoleGrant(RoleCode.TEACHER, ScopeType.CLASS_SECTION, room_ref("AR-P4", "Class 1")),
            RoleGrant(RoleCode.TEACHER, ScopeType.CLASS_SECTION, room_ref("AR-P4", "Class 2")),
        ),
        staff_number="T-007",
        subject="AR",
        rungs=("AR-P4",),
        rooms=(room_ref("AR-P4", "Class 1"), room_ref("AR-P4", "Class 2")),
        purpose=(
            "A second Arabic teacher on the supervisor's own rung, so 'two teachers, one "
            "subject, one rung' is testable."
        ),
    ),
    StaffSpec(
        username="t.unassigned",
        full_name_en="Nabil Amer",
        full_name_ar="نبيل عامر",
        email="nabil.amer@demo.school",
        language="ar",
        roles=(),
        staff_number="T-008",
        subject="MA",
        rungs=("AR-P4",),
        # No rooms: the principal has said what he teaches and on which rung, and the
        # supervisor has not yet put him anywhere. This is the state the assignment
        # screen exists to resolve, and a demo without it cannot exercise that screen.
        rooms=(),
        purpose=(
            "Given a subject and a rung by the principal, placed in no room yet. Log in "
            "as supervisor.p4 and assign him."
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class ClassPlan:
    """How many children a room gets, and whether its register and marks are filled in.

    Not every class is populated to the same depth, on purpose. A demo where all
    twenty-four rooms hold twenty children and a full term of registers is slow to seed,
    slow to reset, and no more informative than one where the interesting rooms are deep
    and the rest are shallow. `register_days` and `graded` mark the interesting ones.
    """

    students: int = 12
    register_days: int = 0
    graded: bool = False


# Rooms named here get more children, a register, and marks. Everything else gets the
# default: a dozen children and no history, which is enough to see the class on a screen.
CLASS_PLANS: Final[dict[str, ClassPlan]] = {
    room_ref("AR-P1", "Class 1"): ClassPlan(students=18, register_days=12, graded=True),
    room_ref("AR-P1", "Class 2"): ClassPlan(students=16, register_days=12, graded=True),
    room_ref("AR-P2", "Class 1"): ClassPlan(students=15, register_days=8, graded=True),
    room_ref("AR-P3", "Class 1"): ClassPlan(students=14, register_days=8, graded=True),
    room_ref("AR-P4", "Class 1"): ClassPlan(students=17, register_days=12, graded=True),
    room_ref("AR-P4", "Class 2"): ClassPlan(students=15, register_days=12, graded=True),
    room_ref("AR-PR1", "Class 1"): ClassPlan(students=16, register_days=8, graded=True),
    room_ref("AR-PR2", "Class 1"): ClassPlan(students=14, register_days=8, graded=True),
    room_ref("AR-PR3", "Class 1"): ClassPlan(students=13, register_days=0, graded=True),
    room_ref("LG-G3", "A"): ClassPlan(students=18, register_days=12, graded=True),
    room_ref("LG-G3", "B"): ClassPlan(students=17, register_days=12, graded=True),
    room_ref("LG-G4", "A"): ClassPlan(students=16, register_days=8, graded=True),
    room_ref("LG-G4", "B"): ClassPlan(students=15, register_days=8, graded=True),
    room_ref("LG-G5", "A"): ClassPlan(students=16, register_days=8, graded=True),
    room_ref("LG-G9", "A"): ClassPlan(students=15, register_days=8, graded=True),
    room_ref("LG-G9", "B"): ClassPlan(students=14, register_days=8, graded=True),
    room_ref("LG-G10", "A"): ClassPlan(students=13, register_days=0, graded=True),
    room_ref("LG-G12", "A"): ClassPlan(students=12, register_days=0, graded=True),
}

DEFAULT_PLAN: Final[ClassPlan] = ClassPlan(students=12, register_days=0, graded=False)


__all__ = [
    "CLASS_PLANS",
    "DEFAULT_PLAN",
    "DEMO_PASSWORD",
    "RUNGS",
    "SCHOOL_CODE",
    "SCHOOL_NAME_AR",
    "SCHOOL_NAME_EN",
    "SECTIONS",
    "STAFF",
    "SUBJECTS",
    "TERMS",
    "YEAR_CODE",
    "YEAR_ENDS",
    "YEAR_STARTS",
    "ClassPlan",
    "RoleGrant",
    "RungSpec",
    "SectionSpec",
    "StaffSpec",
    "SubjectSpec",
    "TermSpec",
    "room_ref",
]
