"""Large, deterministic Arabic-school showcase dataset.

Run only against a disposable/demo SIS database.  The loader is additive and refuses to
run twice: it never deletes or rewrites another school's records.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import date, time, timedelta
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from sis.application.services.access import sync_builtin_rbac
from sis.infrastructure.crypto import hash_password
from sis.infrastructure.db import models as m

SCHOOL_CODE = "ARABIC-DEMO"
YEAR_CODE = "ARABIC-2025-2026"
PASSWORD = "Demo@2026"
START = date(2025, 9, 21)
END = date(2026, 6, 30)

FIRST_M = ("محمد", "أحمد", "عمر", "يوسف", "آدم", "ياسين", "محمود", "مصطفى", "زياد", "حمزة", "علي", "كريم")
FIRST_F = ("فاطمة", "ليلى", "مريم", "ملك", "نور", "سارة", "جنى", "سلمى", "فرح", "هدى", "يارا", "منة")
FATHERS = ("أحمد", "محمود", "محمد", "خالد", "حسن", "إبراهيم", "طارق", "وليد", "أيمن", "شريف", "عمرو", "هشام")
FAMILIES = ("عبد الرحمن", "السيد", "حسن", "علي", "إبراهيم", "عثمان", "منصور", "رشاد", "فؤاد", "النجار", "الشاذلي", "الرفاعي")

SUBJECTS = {
    "AR": ("Arabic", "اللغة العربية"), "MATH": ("Mathematics", "الرياضيات"),
    "EN": ("English", "اللغة الإنجليزية"), "CS": ("Computer", "الكمبيوتر"),
    "ART": ("Art", "الرسم"), "PE": ("Physical Education", "الألعاب"),
    "RE": ("Religion", "التربية الدينية"), "SS": ("Social Studies", "الدراسات الاجتماعية"),
    "SCI": ("Science", "العلوم"), "BIO": ("Biology", "الأحياء"),
    "PHIL": ("Philosophy", "الفلسفة"), "L2": ("Second Language", "اللغة الثانية"),
    "HIST": ("History", "التاريخ"), "GEO": ("Geography", "الجغرافيا"),
    "PHY": ("Physics", "الفيزياء"), "CHEM": ("Chemistry", "الكيمياء"),
    "MATH1": ("Mathematics 1", "رياضة 1"), "MATH2": ("Mathematics 2", "رياضة 2"),
    "LOGIC": ("Logic", "المنطق"), "PURE": ("Pure Mathematics", "رياضة بحتة"),
    "APPLIED": ("Applied Mathematics", "رياضة تطبيقية"), "GEOL": ("Geology", "الجيولوجيا"),
}
BASE = ("AR", "MATH", "EN", "CS", "ART", "PE", "RE")


@dataclass(frozen=True)
class Level:
    code: str
    en: str
    ar: str
    stage: str
    order: int
    classes: int
    students: int
    subjects: tuple[str, ...]
    gender_split: bool = False


LEVELS = (
    Level("KG1", "KG 1", "KG 1", "garden", 1, 10, 25, BASE),
    Level("KG2", "KG 2", "KG 2", "garden", 2, 10, 25, BASE),
    *tuple(Level(f"P{i}", f"Primary {i}", f"الصف {i} الابتدائي", "primary", 10+i, 15, 35,
                 BASE if i < 4 else BASE + ("SS", "SCI")) for i in range(1, 7)),
    *tuple(Level(f"PREP{i}", f"Preparatory {i}", f"الصف {i} الإعدادي", "preparatory", 20+i,
                 20, 30, BASE + ("SS", "SCI"), True) for i in range(1, 4)),
    Level("SEC1", "Secondary 1", "الصف الأول الثانوي", "secondary", 31, 10, 30,
          ("AR", "MATH", "EN", "BIO", "PHIL", "L2", "HIST", "GEO", "PHY", "CHEM"), True),
    Level("SEC2-SCI", "Secondary 2 Scientific", "الصف الثاني الثانوي علمي", "secondary", 32, 5, 30,
          ("AR", "EN", "MATH1", "L2", "PHY", "CHEM", "BIO", "MATH2"), True),
    Level("SEC2-LIT", "Secondary 2 Literary", "الصف الثاني الثانوي أدبي", "secondary", 33, 5, 30,
          ("AR", "EN", "MATH1", "L2", "HIST", "GEO", "PHIL", "LOGIC"), True),
    Level("SEC3-MATH", "Secondary 3 Mathematics", "الصف الثالث الثانوي علمي رياضة", "secondary", 34, 4, 30,
          ("AR", "EN", "L2", "CHEM", "PHY", "PURE", "APPLIED"), True),
    Level("SEC3-SCI", "Secondary 3 Science", "الصف الثالث الثانوي علمي علوم", "secondary", 35, 3, 30,
          ("AR", "EN", "L2", "BIO", "CHEM", "PHY", "GEOL"), True),
    Level("SEC3-LIT", "Secondary 3 Literary", "الصف الثالث الثانوي أدبي", "secondary", 36, 3, 30,
          ("AR", "EN", "L2", "HIST", "GEO", "PHIL", "LOGIC"), True),
)


def _phone(n: int) -> str:
    return f"+2011{n:08d}"


def _person_name(n: int, female: bool) -> tuple[str, str]:
    first = (FIRST_F if female else FIRST_M)[n % 12]
    father = FATHERS[(n // 12) % 12]
    family = FAMILIES[(n // 144) % 12]
    ar = f"{first} {father} {family}"
    return ar, f"Demo Person {n:04d}"


def _class_name(level: Level, index: int) -> tuple[str, str, str]:
    if level.gender_split:
        half = math.ceil(level.classes / 2)
        female = index > half
        local = index - half if female else index
        gender_ar, gender_en = ("بنات", "Girls") if female else ("بنين", "Boys")
        return f"{level.code}-{index:02d}", f"{level.en} {gender_en} {local}", f"{level.ar} {gender_ar} {local}"
    return f"{level.code}-{index:02d}", f"{level.en} Class {index}", f"{level.ar} فصل {index}"


def _account(session: Session, school_id: int, role_id: int, scope_type: str,
             scope_id: int | None, username: str, ar_name: str, purpose: str,
             credentials: list[tuple[str, str, str, str]], password_hash: str) -> m.User:
    user = m.User(username=username, password_hash=password_hash,
                  email=f"{username}@arabic-demo.school", full_name_en=purpose,
                  full_name_ar=ar_name, preferred_language="ar", school_id=school_id,
                  is_active=True)
    session.add(user); session.flush()
    session.add(m.UserRole(user_id=user.id, role_id=role_id, scope_type=scope_type,
                           scope_id=scope_id, granted_by="arabic.admin"))
    credentials.append((purpose, ar_name, username, PASSWORD))
    return user


def load(session: Session) -> dict[str, int]:
    """Create the complete school atomically. Refuses an existing school code."""
    if session.scalar(select(m.School.id).where(m.School.code == SCHOOL_CODE)):
        raise RuntimeError(f"{SCHOOL_CODE} already exists; the showcase was not changed")
    sync_builtin_rbac(session)
    school = m.School(code=SCHOOL_CODE, name_en="Al Rowad Arabic School",
                      name_ar="مدرسة الرواد العربية المتكاملة", language_type="arabic",
                      kg_grade_count=2, primary_grade_count=6, preparatory_grade_count=3,
                      secondary_grade_count=3, term_count=2)
    session.add(school); session.flush()
    system = m.EducationalSystem(school_id=school.id, code="AR", kind="arabic",
                                 name_en="Arabic Education", name_ar="التعليم العربي",
                                 display_order=1, is_active=True)
    session.add(system)
    year = m.AcademicYear(code=YEAR_CODE, school_id=school.id, name_en="2025/2026",
                          name_ar="العام الدراسي ٢٠٢٥/٢٠٢٦", starts_on=START, ends_on=END,
                          is_current=True)
    session.add(year); session.flush()
    terms = [
        m.Term(code=f"{YEAR_CODE}-T1", academic_year_id=year.id, name_en="First Term",
               name_ar="الفصل الدراسي الأول", starts_on=START, ends_on=date(2026, 1, 15), sequence=1),
        m.Term(code=f"{YEAR_CODE}-T2", academic_year_id=year.id, name_en="Second Term",
               name_ar="الفصل الدراسي الثاني", starts_on=date(2026, 2, 8), ends_on=END, sequence=2),
    ]
    session.add_all(terms)
    subjects = {code: m.Subject(code=code, academic_year_id=year.id, name_en=names[0],
                                name_ar=names[1], display_order=i, is_active=True)
                for i, (code, names) in enumerate(SUBJECTS.items(), 1)}
    session.add_all(subjects.values()); session.flush()

    levels: dict[str, m.YearLevel] = {}
    classes: dict[str, list[m.ClassSection]] = {}
    for spec in LEVELS:
        row = m.YearLevel(code=spec.code, school_id=school.id, name_en=spec.en, name_ar=spec.ar,
                          display_order=spec.order, stage=spec.stage,
                          educational_system_id=system.id, grade_number=spec.order % 10 or 1)
        session.add(row); session.flush(); levels[spec.code] = row
        session.add_all(m.SubjectYearLevel(subject_id=subjects[s].id, year_level_id=row.id)
                        for s in spec.subjects)
        rooms = []
        for i in range(1, spec.classes + 1):
            code, en, ar = _class_name(spec, i)
            room = m.ClassSection(academic_year_id=year.id, year_level_id=row.id, code=code,
                                  name_en=en, name_ar=ar, capacity=spec.students,
                                  section_number=i, is_active=True)
            session.add(room); rooms.append(room)
        session.flush(); classes[spec.code] = rooms

    # Nine lessons plus a real break between periods four and five.
    cursor = time(7, 45)
    for number in range(1, 11):
        is_break = number == 5
        minutes = 30 if is_break else 45
        end_minutes = cursor.hour * 60 + cursor.minute + minutes
        end_at = time(end_minutes // 60, end_minutes % 60)
        session.add(m.TimetablePeriod(school_id=school.id, period_number=number,
                    name_en="Break" if is_break else f"Period {number if number < 5 else number-1}",
                    name_ar="الفسحة" if is_break else f"الحصة {number if number < 5 else number-1}",
                    starts_at=cursor, ends_at=end_at, is_teaching=not is_break))
        cursor = end_at

    roles = {row.code: row.id for row in session.scalars(select(m.Role)).all()}
    credentials: list[tuple[str, str, str, str]] = []
    shared_hash = hash_password(PASSWORD)
    _account(session, school.id, roles["principal"], "school", school.id, "arabic.manager",
             "أحمد عبد الحميد", "مدير المدرسة", credentials, shared_hash)

    # Grade/floor supervisors and attendance supervisors are deliberately different people.
    attendance_user: dict[str, str] = {}
    for n, spec in enumerate(LEVELS, 1):
        _account(session, school.id, roles["year_supervisor"], "year_level", levels[spec.code].id,
                 f"floor.{spec.code.lower()}", _person_name(7000+n, n % 2 == 0)[0],
                 f"مشرف دور {spec.ar}", credentials, shared_hash)
        att = _account(session, school.id, roles["attendance_supervisor"], "year_level",
                       levels[spec.code].id, f"attendance.{spec.code.lower()}",
                       _person_name(7100+n, n % 2 == 1)[0], f"مشرف غياب {spec.ar}",
                       credentials, shared_hash)
        attendance_user[spec.code] = att.username

    # Three teacher teams for large grades, two for KG/secondary. Each gets its own account.
    teachers: dict[tuple[str, str], list[m.Teacher]] = {}
    teacher_number = 1
    for spec in LEVELS:
        team_count = max(2, math.ceil(spec.classes / 5))
        for subject_code in spec.subjects:
            team: list[m.Teacher] = []
            for team_no in range(1, team_count + 1):
                username = f"teacher.{subject_code.lower()}.{spec.code.lower()}.{team_no}"
                ar_name = _person_name(8000 + teacher_number, teacher_number % 3 == 0)[0]
                user = _account(session, school.id, roles["teacher"], "year_level",
                                levels[spec.code].id, username, ar_name,
                                f"مدرس {SUBJECTS[subject_code][1]} - {spec.ar}", credentials, shared_hash)
                teacher = m.Teacher(staff_number=f"AR-T-{teacher_number:04d}", school_id=school.id,
                                    user_id=user.id, full_name_en=f"Teacher {teacher_number:04d}",
                                    full_name_ar=ar_name, email=f"{username}@arabic-demo.school",
                                    phone=_phone(70000000 + teacher_number), is_active=True)
                session.add(teacher); session.flush()
                session.add(m.TeacherSubject(teacher_id=teacher.id, subject_id=subjects[subject_code].id,
                                              academic_year_id=year.id, is_primary=True))
                session.add(m.TeacherYearLevel(teacher_id=teacher.id, year_level_id=levels[spec.code].id,
                                               subject_id=subjects[subject_code].id))
                team.append(teacher); teacher_number += 1
            teachers[(spec.code, subject_code)] = team

    # Students, their guardian records, placements, recent attendance and two assessments.
    student_number = 1
    rng = random.Random(20250921)
    special = {"SEC2-SCI": ("فاطمة محمد أبو الحسن", "Fatma Mohamed Aboulhassan", "+201093887199", "محمد أبو الحسن"),
               "P2": ("ليلى عمر أبو الحسن", "Layla Omar Aboulhassan", "+201024066401", "عمر أبو الحسن")}
    school_days = [date(2025, 11, 16) + timedelta(days=i) for i in range(5)]
    student_rows: list[tuple[m.Student, m.ClassSection, Level]] = []
    for spec in LEVELS:
        for class_index, room in enumerate(classes[spec.code], 1):
            for seat in range(1, spec.students + 1):
                female = class_index > math.ceil(spec.classes / 2) if spec.gender_split else (seat % 2 == 0)
                ar_name, en_name = _person_name(student_number, female)
                guardian_phone, guardian_ar = _phone(student_number), f"{FATHERS[student_number % 12]} {FAMILIES[(student_number // 12) % 12]}"
                if spec.code in special and class_index == 1 and seat == 1:
                    ar_name, en_name, guardian_phone, guardian_ar = special.pop(spec.code)
                    female = True
                if spec.stage == "garden":
                    age = 4 + spec.order
                elif spec.stage == "primary":
                    age = 5 + int(spec.code[1:])
                elif spec.stage == "preparatory":
                    age = 11 + int(spec.code[-1:])
                else:
                    age = 14 + int(spec.code[3])
                student = m.Student(student_number=f"AR-S-{student_number:06d}", full_name_ar=ar_name,
                                    full_name_en=en_name, gender="female" if female else "male",
                                    date_of_birth=date(2025 - age, 1 + student_number % 12,
                                                       1 + student_number % 27),
                                    contact_phone="", contact_email=f"student{student_number}@demo.school",
                                    address=f"شارع {1 + student_number % 80}، مدينة نصر، القاهرة")
                session.add(student); session.flush()
                guardian = m.Guardian(public_id=f"AR-G-{student_number:06d}", full_name_ar=guardian_ar,
                                      full_name_en=f"Guardian {student_number:06d}", preferred_language="ar")
                session.add(guardian); session.flush()
                session.add_all([
                    m.GuardianPhone(guardian_id=guardian.id, phone=guardian_phone, is_primary=True),
                    m.StudentGuardian(student_id=student.id, guardian_id=guardian.id,
                                      relationship_type="father", relationship_label="والد الطالب",
                                      is_primary_contact=True, can_view_records=True),
                    m.ClassEnrolment(student_id=student.id, class_section_id=room.id,
                                     starts_on=START, ends_on=None, reason="initial"),
                ])
                for day in school_days:
                    roll = rng.random()
                    state = "present" if roll < .90 else "late" if roll < .95 else "absent" if roll < .985 else "excused"
                    session.add(m.Attendance(student_id=student.id, class_section_id=room.id,
                                             on_date=day, state=state,
                                             note="عذر طبي موثق" if state == "excused" else "",
                                             recorded_by=attendance_user[spec.code]))
                for subject_code in spec.subjects:
                    for term, maximum, label in ((terms[0], 20.0, "امتحان الشهر"), (terms[1], 10.0, "النموذج التدريبي")):
                        points = round(maximum * rng.uniform(.48, .99), 1)
                        session.add(m.SubjectGrade(student_id=student.id, subject_id=subjects[subject_code].id,
                                                   term_id=term.id, class_section_id=room.id,
                                                   points=points, max_points=maximum,
                                                   percentage=round(points / maximum * 100, 1),
                                                   remark=label, recorded_by="arabic.manager"))
                student_rows.append((student, room, spec)); student_number += 1

    # Assign teachers to rooms and produce 45 weekly teaching slots (9 lessons x 5 days).
    teaching_periods = (1, 2, 3, 4, 6, 7, 8, 9, 10)
    busy: set[tuple[int, str, int]] = set()
    for spec in LEVELS:
        for class_index, room in enumerate(classes[spec.code]):
            for subject_code in spec.subjects:
                team = teachers[(spec.code, subject_code)]
                teacher = team[class_index % len(team)]
                session.add(m.TeacherClassSection(teacher_id=teacher.id, class_section_id=room.id,
                                                   subject_id=subjects[subject_code].id,
                                                   assigned_by=f"floor.{spec.code.lower()}"))
            slots = [(day, period) for day in ("sunday", "monday", "tuesday", "wednesday", "thursday")
                     for period in teaching_periods]
            for slot_no, (day, period) in enumerate(slots):
                subject_code = spec.subjects[(slot_no + class_index) % len(spec.subjects)]
                choices = teachers[(spec.code, subject_code)]
                teacher = next((candidate for candidate in choices
                                if (candidate.id, day, period) not in busy), None)
                if teacher: busy.add((teacher.id, day, period))
                session.add(m.TimetableEntry(class_section_id=room.id, academic_year_id=year.id,
                            term_id=terms[0].id, day_of_week=day, period_number=period,
                            subject_id=subjects[subject_code].id,
                            teacher_id=teacher.id if teacher else None))

    # Realistic teacher attendance for the same working week.
    for team in teachers.values():
        for teacher in team:
            for day in school_days:
                state = "present" if rng.random() < .96 else "late"
                session.add(m.TeacherAttendance(teacher_id=teacher.id, school_id=school.id,
                                                on_date=day, state=state, note="",
                                                recorded_by="arabic.manager"))

    _write_credentials(credentials)
    return {"levels": len(LEVELS), "classes": sum(x.classes for x in LEVELS),
            "students": student_number - 1, "guardians": student_number - 1,
            "teachers": teacher_number - 1, "accounts": len(credentials),
            "attendance": (student_number - 1) * len(school_days),
            "grades": sum(x.classes * x.students * len(x.subjects) * 2 for x in LEVELS),
            "timetable_entries": sum(x.classes for x in LEVELS) * 45}


def validate(session: Session) -> list[str]:
    """Return presentation-critical facts read back from the live database."""
    school_id = session.scalar(select(m.School.id).where(m.School.code == SCHOOL_CODE))
    if school_id is None:
        raise RuntimeError(f"{SCHOOL_CODE} is not loaded")
    from sqlalchemy import func
    special = session.execute(
        select(m.Student.full_name_ar, m.GuardianPhone.phone, m.YearLevel.code)
        .join(m.StudentGuardian, m.StudentGuardian.student_id == m.Student.id)
        .join(m.Guardian, m.Guardian.id == m.StudentGuardian.guardian_id)
        .join(m.GuardianPhone, m.GuardianPhone.guardian_id == m.Guardian.id)
        .join(m.ClassEnrolment, m.ClassEnrolment.student_id == m.Student.id)
        .join(m.ClassSection, m.ClassSection.id == m.ClassEnrolment.class_section_id)
        .join(m.YearLevel, m.YearLevel.id == m.ClassSection.year_level_id)
        .where(m.GuardianPhone.phone.in_(("+201093887199", "+201024066401")))
    ).all()
    year_id = session.scalar(select(m.AcademicYear.id).where(m.AcademicYear.code == YEAR_CODE))
    unassigned = session.scalar(select(func.count()).select_from(m.TimetableEntry).where(
        m.TimetableEntry.academic_year_id == year_id, m.TimetableEntry.teacher_id.is_(None)))
    accounts = session.scalar(select(func.count()).select_from(m.User).where(m.User.school_id == school_id))
    return [f"special guardian rows: {special}", f"unassigned timetable lessons: {unassigned}",
            f"school accounts: {accounts}"]


def _write_credentials(rows: list[tuple[str, str, str, str]]) -> None:
    groups: dict[str, list[tuple[str, str, str]]] = {}
    for purpose, name, username, password in rows:
        key = "الإدارة والمشرفون" if not purpose.startswith("مدرس ") else purpose.split(" - ")[0]
        groups.setdefault(key, []).append((name, username, password))
    lines = ["حسابات مدرسة الرواد العربية المتكاملة", "=" * 44,
             "هذه حسابات عرض تجريبية وليست للاستخدام الإنتاجي.", ""]
    for group, accounts in groups.items():
        lines.extend((f"[{group}]", "الاسم | اسم المستخدم | كلمة المرور"))
        lines.extend(f"{name} | {username} | {password}" for name, username, password in accounts)
        lines.append("")
    try:
        Path("اداره.txt").write_text("\n".join(lines), encoding="utf-8")
    except PermissionError:
        # The production image runs as an unprivileged user in read-only /app. The host-side
        # generation already created the hand-off file; credentials must never make the
        # database transaction fail merely because the container cannot duplicate it.
        pass
