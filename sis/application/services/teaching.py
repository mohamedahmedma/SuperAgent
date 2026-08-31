"""What a teacher teaches, and the one authorisation question the scope model cannot answer.

`sis/domain/rbac.py` bounds a grant to a *place*: a school, a rung, a classroom. That is the
right shape for almost everything a school does, and it is the wrong shape for exactly one
rule — **a teacher may record marks for their own subject and no other.**

The reason it cannot be expressed as a scope is worth stating, because the temptation to try
is strong. A teacher's grant is a classroom, and `grades.write` on 4/1 covers every question
that names 4/1. Physics and Arabic are both taught in that room, so a scope check that
passes for one passes for the other: the Arabic teacher of 4/1 could write its Physics marks
and every check in the service would agree they were entitled to. Narrowing the grant to
`ScopeType.SUBJECT` does not fix it either — a subject-scoped grant says "Arabic, wherever
you find it", which is a claim over every Arabic class in the school including the rooms
this teacher has never stood in. The rule needs *both* halves at once, and a `Scope` holds
one id.

So the pair is read from where the school already records it. `teacher_class_sections` is
written by the principal (Stage 11) and by the grade supervisor (Stage 12), and it says
which teacher stands in which room for which subject. This module turns that table into an
answer to "may this person write this mark", and nothing else consults it for authority.

**Who this restricts, and who it does not.** A person with a `teachers` row is bounded by
their assignments. A person without one — a registrar, a principal, an integration holding
an API key — is bounded only by their scope, as before. That asymmetry is the school's own:
"which subject do you teach" is a question about a member of teaching staff and is
meaningless asked of the office. Stated as a rule rather than inferred from a role code,
because a teacher who is *also* a supervisor holds two roles and is still one person who
teaches Arabic.

**It fails closed.** A teacher whose assignments have not been set up yet owns nothing and
may write nothing. That is the safe direction: the fix is a supervisor assigning them a
class, which is a screen that exists, and the failure is a refusal rather than a mark
written into somebody else's subject.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from sis.application.ports.unit_of_work import UnitOfWork
from sis.infrastructure.db import models as m

__all__ = ["TeachingAssignment", "TeachingService"]


@dataclass(frozen=True, slots=True)
class TeachingAssignment:
    """One room, one subject, and the rung the room sits on.

    The unit a teacher's authority is actually granted in, and the unit the mark-entry
    screen is driven by: a teacher picks one of these and then sees a mark sheet. The grade
    rides along because a teacher of four rooms across three rungs needs them grouped, and
    grouping them client-side from a second call is how the two lists drift apart.
    """

    class_section_id: int
    class_code: str
    class_name_en: str
    class_name_ar: str
    subject_id: int
    subject_code: str
    subject_name_en: str
    subject_name_ar: str
    year_level_code: str
    year_level_name_en: str
    year_level_name_ar: str
    track_code: str | None
    academic_year_code: str


class TeachingService:
    """Reads teaching assignments, and answers whether one covers a mark.

    Takes a unit-of-work factory like every other service here. It reads three tables by
    key and writes none of them: assigning a teacher to a room is Stage 11's and Stage 12's
    work, and this module is deliberately only the reader — a service that could both grant
    and check its own authority is one an error in either half compromises.
    """

    def __init__(self, uow_factory: Callable[[], UnitOfWork]) -> None:
        self._uow_factory = uow_factory

    def assignments_for_user(
        self, user_id: int | None, *, academic_year_code: str | None = None
    ) -> Sequence[TeachingAssignment]:
        """Every (room, subject) this person teaches, ordered as a school reads its ladder.

        Empty for somebody who is not teaching staff, which is not the same as somebody who
        may do nothing: a registrar has no assignments and writes marks anywhere their
        scope reaches. Callers asking "what may I record" want this; callers asking "may I
        record *this*" want `may_record`, which knows the difference.
        """
        if user_id is None:
            return ()
        with self._uow_factory() as uow:
            return self._assignments(uow._session, user_id, academic_year_code)

    def is_teaching_staff(self, user_id: int | None) -> bool:
        """Whether a `teachers` row is linked to this account.

        The question that decides *whether the subject rule applies at all*, kept apart
        from whether they hold any assignment. A teacher appointed this morning and not yet
        given a class is teaching staff who may write nothing; a registrar with no teacher
        row is not teaching staff and is bounded only by scope. Collapsing the two would
        silently promote every unassigned teacher to a registrar.
        """
        if user_id is None:
            return False
        with self._uow_factory() as uow:
            return (
                uow._session.scalar(
                    select(m.Teacher.id).where(m.Teacher.user_id == int(user_id))
                )
                is not None
            )

    def may_record(
        self, user_id: int | None, *, class_section_id: int, subject_id: int
    ) -> bool:
        """The rule: teaching staff record their own subject, in their own rooms.

        Answers `True` for a caller who is not teaching staff — their boundary is their
        scope, checked by the route before this is ever consulted. This method narrows a
        teacher further; it never widens anybody.
        """
        if user_id is None:
            return True
        with self._uow_factory() as uow:
            session = uow._session
            teacher_id = session.scalar(
                select(m.Teacher.id).where(m.Teacher.user_id == int(user_id))
            )
            if teacher_id is None:
                return True
            return (
                session.scalar(
                    select(m.TeacherClassSection.id).where(
                        m.TeacherClassSection.teacher_id == teacher_id,
                        m.TeacherClassSection.class_section_id == int(class_section_id),
                        m.TeacherClassSection.subject_id == int(subject_id),
                    )
                )
                is not None
            )

    def may_record_by_code(
        self,
        user_id: int | None,
        *,
        academic_year_code: str,
        class_code: str,
        subject_code: str,
    ) -> bool:
        """`may_record`, addressed the way a route holds it: by code rather than by id.

        A route has a class code and a subject code out of the URL and the body; the
        assignment table holds surrogates. Resolving here rather than in the router keeps
        the join in one place and means a route cannot accidentally check the wrong pair.

        A code that resolves to nothing answers `False` for teaching staff. That is the
        fail-closed direction: an unresolvable subject is not one anybody is assigned to
        teach, and the route's own lookup will refuse it with a better message anyway.
        """
        if user_id is None:
            return True
        with self._uow_factory() as uow:
            session = uow._session
            teacher_id = session.scalar(
                select(m.Teacher.id).where(m.Teacher.user_id == int(user_id))
            )
            if teacher_id is None:
                return True
            return (
                session.scalar(
                    select(m.TeacherClassSection.id)
                    .join(
                        m.ClassSection,
                        m.TeacherClassSection.class_section_id == m.ClassSection.id,
                    )
                    .join(
                        m.AcademicYear,
                        m.ClassSection.academic_year_id == m.AcademicYear.id,
                    )
                    .join(m.Subject, m.TeacherClassSection.subject_id == m.Subject.id)
                    .where(
                        m.TeacherClassSection.teacher_id == teacher_id,
                        m.AcademicYear.code == str(academic_year_code),
                        m.ClassSection.code == str(class_code),
                        m.Subject.code == str(subject_code),
                    )
                )
                is not None
            )

    def subject_ids_in_class(
        self, user_id: int | None, *, class_section_id: int
    ) -> set[int] | None:
        """Which subjects this person may record in one room, or `None` for "not bounded".

        `None` and `set()` are different answers and the caller must keep them apart:
        nobody-in-particular may record anything, an unassigned teacher may record nothing.
        Used to mark a report card up — every line the reader owns is editable, and the
        rest are shown and not offered.
        """
        if user_id is None:
            return None
        with self._uow_factory() as uow:
            session = uow._session
            teacher_id = session.scalar(
                select(m.Teacher.id).where(m.Teacher.user_id == int(user_id))
            )
            if teacher_id is None:
                return None
            return set(
                session.scalars(
                    select(m.TeacherClassSection.subject_id).where(
                        m.TeacherClassSection.teacher_id == teacher_id,
                        m.TeacherClassSection.class_section_id == int(class_section_id),
                    )
                ).all()
            )

    def subject_codes_in_class(
        self, user_id: int | None, *, class_section_id: int
    ) -> set[str] | None:
        """Subject codes visible to this person in one classroom.

        ``None`` means the person is not teaching staff and remains governed by their
        ordinary RBAC scope. An empty set means they are teaching staff but have no
        subject assignment in this room. Keeping those cases distinct prevents an
        unassigned teacher from inheriting office-wide behaviour.
        """
        subject_ids = self.subject_ids_in_class(
            user_id, class_section_id=class_section_id
        )
        if subject_ids is None:
            return None
        if not subject_ids:
            return set()
        with self._uow_factory() as uow:
            return set(
                uow._session.scalars(
                    select(m.Subject.code).where(m.Subject.id.in_(subject_ids))
                ).all()
            )

    # -- Internals ----------------------------------------------------------------

    @staticmethod
    def _assignments(
        session: Session, user_id: int, academic_year_code: str | None
    ) -> tuple[TeachingAssignment, ...]:
        statement = (
            select(
                m.ClassSection,
                m.Subject,
                m.YearLevel,
                m.EducationalSystem,
                m.AcademicYear,
            )
            .join(
                m.TeacherClassSection,
                m.TeacherClassSection.class_section_id == m.ClassSection.id,
            )
            .join(m.Teacher, m.TeacherClassSection.teacher_id == m.Teacher.id)
            .join(m.Subject, m.TeacherClassSection.subject_id == m.Subject.id)
            .join(m.YearLevel, m.ClassSection.year_level_id == m.YearLevel.id)
            .join(m.AcademicYear, m.ClassSection.academic_year_id == m.AcademicYear.id)
            .outerjoin(
                m.EducationalSystem,
                m.YearLevel.educational_system_id == m.EducationalSystem.id,
            )
            .where(m.Teacher.user_id == int(user_id))
            # The school's own order, not alphabetical: a teacher of First, Second and
            # Third Primary reads them in that order and nowhere else.
            .order_by(
                m.YearLevel.display_order,
                m.YearLevel.code,
                m.ClassSection.code,
                m.Subject.display_order,
                m.Subject.code,
            )
        )
        if academic_year_code is not None:
            statement = statement.where(m.AcademicYear.code == str(academic_year_code))

        return tuple(
            TeachingAssignment(
                class_section_id=section.id,
                class_code=section.code,
                class_name_en=section.name_en,
                class_name_ar=section.name_ar,
                subject_id=subject.id,
                subject_code=subject.code,
                subject_name_en=subject.name_en,
                subject_name_ar=subject.name_ar,
                year_level_code=level.code,
                year_level_name_en=level.name_en,
                year_level_name_ar=level.name_ar,
                track_code=None if track is None else track.code,
                academic_year_code=year.code,
            )
            for section, subject, level, track, year in session.execute(statement).all()
        )
