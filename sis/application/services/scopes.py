"""Turning the ids a route knows into the scopes a grant is checked against.

`sis/domain/rbac.py` answers "does this grant cover this target" and deliberately knows
nothing about where a target comes from. This module is the other half: a route holds a
class code, and a grant is held on a *rung* or a *school*, and something has to walk the
ladder between them.

**Why it cannot be skipped.** `Scope.covers` matches on ids the target actually names, and
it does not go looking. A supervisor granted `attendance.write` on Grade 4 asking about
class `4B` is refused unless the target says `year_level_id=<Grade 4>` — the grant is real,
the request is legitimate, and the answer is still no. The failure is silent and looks like
a permissions bug in the console. So every route that names a narrow thing resolves it
here first, and the resolution is a fact about the school's structure rather than about
the caller.

**It fails closed, twice.** A code that resolves to nothing yields a target naming only
what was proven — usually the school, sometimes nothing at all — so a narrow grant simply
does not match and the request is refused. And an *ambiguous* class code (the same code
under two rungs of one year, which the schema permits) resolves to the school and no
further, rather than to whichever row came back first: guessing there would hand a teacher
of `3A` the register of a different `3A`.

**Nothing here is cached across requests.** The structure moves — a class is created, a
rung is re-parented — and an authorisation decision made against last week's ladder is the
one kind of stale read that must not happen. Within a single request the memo below is
safe, because a request is answered inside one transaction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from sis.domain.rbac import Target
from sis.infrastructure.db import models as m


@dataclass(slots=True)
class ScopeResolver:
    """Locates a thing in the school's ladder. One instance per request.

    Takes a `Session` rather than a unit of work for the same reason `access.py` does:
    every method here is one read by key against a table that has no repository, and five
    classes of pass-through would buy nothing.
    """

    session: Session
    _memo: dict[tuple[str, ...], Target] = field(default_factory=dict, repr=False)
    #: Lookups that answer with several places at once — see `for_student`.
    _rooms: dict[tuple[str, ...], tuple[Target, ...]] = field(
        default_factory=dict, repr=False
    )

    # -- The ladder ---------------------------------------------------------------

    def for_class(self, *, academic_year_code: str, class_code: str) -> Target:
        """Everything true of one classroom: its school, track, rung and itself.

        The pair is the key rather than the class code alone because a class code is only
        unique within `(year, rung)` — see `ClassSection` — so `3A` on its own is not a
        room. Two rows back means the school reuses section letters across rungs, and the
        honest answer is "somewhere in this school" rather than a coin flip.
        """
        key = ("class", str(academic_year_code), str(class_code))
        if key in self._memo:
            return self._memo[key]

        rows = self.session.execute(
            select(
                m.ClassSection.id,
                m.YearLevel.id,
                m.YearLevel.educational_system_id,
                m.AcademicYear.school_id,
            )
            .join(m.AcademicYear, m.ClassSection.academic_year_id == m.AcademicYear.id)
            .join(m.YearLevel, m.ClassSection.year_level_id == m.YearLevel.id)
            .where(
                m.AcademicYear.code == str(academic_year_code),
                m.ClassSection.code == str(class_code),
            )
        ).all()

        if len(rows) == 1:
            class_id, level_id, track_id, school_id = rows[0]
            target = Target(
                school_id=school_id,
                track_id=track_id,
                year_level_id=level_id,
                class_section_id=class_id,
            )
        else:
            # Nothing matched, or too much did. Either way the only proven fact is the
            # school the year belongs to, and a class-scoped grant must not match it.
            target = Target(school_id=self._school_of_year(str(academic_year_code)))

        self._memo[key] = target
        return target

    def for_class_id(self, class_section_id: int | None) -> Target:
        """The same, from a surrogate id — what the role-assignment routes hold."""
        if class_section_id is None:
            return Target()
        key = ("class_id", str(class_section_id))
        if key in self._memo:
            return self._memo[key]

        row = self.session.execute(
            select(
                m.YearLevel.id,
                m.YearLevel.educational_system_id,
                m.AcademicYear.school_id,
            )
            .join(m.ClassSection, m.ClassSection.year_level_id == m.YearLevel.id)
            .join(m.AcademicYear, m.ClassSection.academic_year_id == m.AcademicYear.id)
            .where(m.ClassSection.id == int(class_section_id))
        ).one_or_none()

        target = (
            Target()
            if row is None
            else Target(
                school_id=row[2],
                track_id=row[1],
                year_level_id=row[0],
                class_section_id=int(class_section_id),
            )
        )
        self._memo[key] = target
        return target

    def for_year_level(self, *, school_id: int | None, year_level_code: str) -> Target:
        """A rung, and the track and school above it.

        Scoped by school because rung codes are unique per school and `Y3` genuinely
        exists at every branch — an unscoped lookup would authorise against another
        branch's Year 3.
        """
        key = ("level", str(school_id), str(year_level_code))
        if key in self._memo:
            return self._memo[key]

        statement = select(m.YearLevel.id, m.YearLevel.educational_system_id, m.YearLevel.school_id).where(
            m.YearLevel.code == str(year_level_code)
        )
        if school_id is not None:
            statement = statement.where(m.YearLevel.school_id == int(school_id))

        rows = self.session.execute(statement).all()
        if len(rows) == 1:
            level_id, track_id, owning_school = rows[0]
            target = Target(
                school_id=owning_school, track_id=track_id, year_level_id=level_id
            )
        else:
            target = Target(school_id=school_id)

        self._memo[key] = target
        return target

    def for_year_level_in_school(
        self, *, school_code: str, year_level_code: str
    ) -> Target:
        """Resolve the requested grade inside the requested school.

        Using the caller's school id here would authorize their school's grade and then
        let the route load a same-coded grade from a different school.
        """
        school_id = self.session.scalar(
            select(m.School.id).where(m.School.code == str(school_code))
        )
        return self.for_year_level(
            school_id=school_id, year_level_code=year_level_code
        )

    def for_subject(self, *, academic_year_code: str, subject_code: str) -> Target:
        """A subject within one year. Names the school too, so a wider grant still covers it."""
        key = ("subject", str(academic_year_code), str(subject_code))
        if key in self._memo:
            return self._memo[key]

        row = self.session.execute(
            select(m.Subject.id, m.AcademicYear.school_id)
            .join(m.AcademicYear, m.Subject.academic_year_id == m.AcademicYear.id)
            .where(
                m.AcademicYear.code == str(academic_year_code),
                m.Subject.code == str(subject_code),
            )
        ).one_or_none()

        target = (
            Target(school_id=self._school_of_year(str(academic_year_code)))
            if row is None
            else Target(school_id=row[1], subject_id=row[0])
        )
        self._memo[key] = target
        return target

    def for_student(
        self, *, academic_year_code: str, student_number: str
    ) -> tuple[Target, ...]:
        """Every room a child was placed in during one academic year, each fully located.

        A child is not a scope — nothing is ever granted "on Fatima" — so this answers with
        the rooms she sits in and everything above them. That is what makes a teacher's
        class-scoped grant reach her report card: the mark is hers, the boundary is the
        classroom she earned it in.

        **Several, not one, and ended enrolments count.** A child who moved from 3A to 3B
        in March was in both, and her Term 1 marks were earned in 3A — an answer naming
        only today's class would refuse her Term 1 teacher the marks he wrote himself.
        Returning them all is safe because each id is a room she was genuinely placed in,
        so a grant on any of them is a grant over part of her year; the caller passes if
        any one matches.
        """
        key = ("student", str(academic_year_code), str(student_number))
        if key in self._rooms:
            return self._rooms[key]

        rows = self.session.execute(
            select(
                m.ClassSection.id,
                m.YearLevel.id,
                m.YearLevel.educational_system_id,
                m.AcademicYear.school_id,
            )
            .join(m.ClassEnrolment, m.ClassEnrolment.class_section_id == m.ClassSection.id)
            .join(m.Student, m.ClassEnrolment.student_id == m.Student.id)
            .join(m.AcademicYear, m.ClassSection.academic_year_id == m.AcademicYear.id)
            .join(m.YearLevel, m.ClassSection.year_level_id == m.YearLevel.id)
            .where(
                m.Student.student_number == str(student_number),
                m.AcademicYear.code == str(academic_year_code),
            )
        ).all()

        if rows:
            found = tuple(
                Target(
                    school_id=school_id,
                    track_id=track_id,
                    year_level_id=level_id,
                    class_section_id=class_id,
                )
                for class_id, level_id, track_id, school_id in rows
            )
        else:
            # Not enrolled anywhere this year, or no such child. Either way the only
            # proven fact is the school, and a class-scoped grant must not match it.
            found = (Target(school_id=self._school_of_year(str(academic_year_code))),)

        self._rooms[key] = found
        return found

    def for_student_in_term(
        self, *, term_code: str, student_number: str
    ) -> tuple[Target, ...]:
        """The same, for a route that names a term instead of a year.

        A term belongs to exactly one academic year, so this is that lookup followed by
        `for_student`. It is a separate method rather than something a route works out for
        itself because "which year is 2026-T1 in" is a fact about the school's calendar,
        and a caller deriving it from the string would be right until a school numbers its
        terms differently.
        """
        year_code = self.session.scalar(
            select(m.AcademicYear.code)
            .join(m.Term, m.Term.academic_year_id == m.AcademicYear.id)
            .where(m.Term.code == str(term_code))
        )
        if year_code is None:
            return (Target(),)
        return self.for_student(
            academic_year_code=year_code, student_number=student_number
        )

    def for_school(self, school_code: str | None) -> Target:
        """Just the school. What a listing route that names nothing narrower asks with."""
        if not school_code:
            return Target()
        key = ("school", str(school_code))
        if key not in self._memo:
            found = self.session.scalar(
                select(m.School.id).where(m.School.code == str(school_code))
            )
            self._memo[key] = Target(school_id=found)
        return self._memo[key]

    def for_year(self, academic_year_code: str | None) -> Target:
        """The school an academic year belongs to. A year is not itself a scope."""
        if not academic_year_code:
            return Target()
        return Target(school_id=self._school_of_year(str(academic_year_code)))

    # -- Internals ----------------------------------------------------------------

    def _school_of_year(self, academic_year_code: str) -> int | None:
        key = ("year", academic_year_code)
        if key not in self._memo:
            found = self.session.scalar(
                select(m.AcademicYear.school_id).where(
                    m.AcademicYear.code == academic_year_code
                )
            )
            self._memo[key] = Target(school_id=found)
        return self._memo[key].school_id


__all__ = ["ScopeResolver"]
