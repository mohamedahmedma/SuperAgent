"""The read model: what the registrar UI, the API and the `records/` adapter ask for.

Reads live in their own service because they have a different shape of correctness from
writes. A write has to be refused when it is wrong; a read has to be *unambiguous* when
it is empty. Most of the care in this module is spent on that second thing — an unknown
class raises `UnknownReference` rather than returning an empty roster, because "no such
class" and "a class with nobody in it" render identically on a screen and send the
registrar to look for children who were never missing.

The other reason this module exists is `resolve_sections_for_term`. Invariant 2 says a
placement is time-bounded, so "which class was she in for Term 1" is a lookup against a
date rather than a column — and the *choice of date* is a real decision that must be made
in exactly one place. `GradeImportService` files a mark under the class this function
resolves, and the report read back through `student_term_grades` resolves it the same
way. Two implementations of that rule would eventually disagree by one day, and the
symptom is a Term 1 mark filed under 3A that prints under 3B.

Ports only: no sqlalchemy, no fastapi, no `sis.config`. Every method here is exercisable
against fake repositories.
"""
import logging
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime

from sis.application.ports.repositories import EnrolmentRepository
from sis.application.ports.unit_of_work import UnitOfWork
from sis.domain.access import AccessAttempt, AccessReason
from sis.domain.errors import UnknownReference
from sis.domain.grades import SubjectGrade
from sis.domain.guardians import Guardian, StudentGuardian
from sis.domain.people import ClassEnrolment, Student
from sis.domain.structure import (
    AcademicYear,
    ClassSection,
    School,
    Subject,
    Term,
    YearLevel,
)
from sis.domain.value_objects import (
    AcademicYearCode,
    ClassCode,
    Phone,
    SchoolCode,
    StudentNumber,
    SubjectCode,
    TermCode,
    YearCode,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ClassRosterEntry",
    "GradeLine",
    "GuardianIdentity",
    "GuardianLink",
    "QueryService",
    "StudentTermGrades",
    "resolve_section_for_term",
    "resolve_sections_for_term",
]


def resolve_sections_for_term(
    enrolments: EnrolmentRepository,
    student_numbers: Collection[StudentNumber],
    term: Term,
) -> Mapping[str, ClassSection]:
    """Which class each of these children was in *as at* `term`, keyed by student number.

    The whole of invariant 2 in one function, and the anchor dates are the decision worth
    stating. A term is a period, not a day, and a child can be in two classes across it,
    so "the class for Term 1" has to name a day. This asks about the **last** day of term
    first, because that is where a term's reporting is cut and where a mid-term transfer
    should already have taken effect: a child who moved 3A->3B in March gets her Term 3
    marks under 3B, which is where she sat when they were given.

    The **first** day of term is the fallback, and it is not a nicety. A child who leaves
    the school in November has no placement on the last day of Term 1, and resolving her
    to nothing would reject a term of marks she genuinely earned — or, worse in a report,
    render a finished term as blank. Asking again at the start of term files those marks
    under the class she actually sat in.

    Children with no placement covering either day are simply absent from the result. That
    is a real answer — she had left, or had not yet arrived — and callers must treat it as
    one rather than substituting a class, which is how a grade sheet acquires a child who
    was never in the room.
    """
    if not student_numbers:
        return {}
    resolved = dict(enrolments.class_sections_on(student_numbers, term.ends_on))
    missing = [s for s in student_numbers if s.value not in resolved]
    # Second query only for the children the first one could not place. A file of 600
    # marks for a settled class costs one query, not two.
    if missing:
        for number, section in enrolments.class_sections_on(
            missing, term.starts_on
        ).items():
            resolved.setdefault(number, section)
    return resolved


def resolve_section_for_term(
    enrolments: EnrolmentRepository, student_number: StudentNumber, term: Term
) -> ClassSection | None:
    """One child's class as at a term. Same rule as the bulk form, never a second one."""
    return resolve_sections_for_term(enrolments, [student_number], term).get(
        student_number.value
    )


@dataclass(frozen=True, slots=True)
class ClassRosterEntry:
    """One child in a class register, with the placement window she is listed under.

    `student` is optional so a placement whose student row cannot be loaded still appears,
    carrying its number. Dropping it would shorten the register silently, and a register
    that is quietly one child short is worse than one showing a number without a name.
    """

    student_number: str
    enrolment: ClassEnrolment
    student: Student | None = None

    @property
    def display_name_ar(self) -> str:
        return self.student.full_name_ar if self.student else ""

    @property
    def display_name_en(self) -> str:
        return self.student.full_name_en if self.student else ""


@dataclass(frozen=True, slots=True)
class GuardianIdentity:
    """A guardian named by her stable handle rather than by her phone number.

    What another service holds when it needs to refer to a parent later — an
    authentication service that has just proved she controls a number, for instance. The
    handle is the point: `public_id` is opaque and permanent, while a phone is neither.
    She may add a second line or change carrier, and a number in a URL is PII in every
    access log it passes through, which is the reason `guardians.public_id` exists at all.

    Carries her names and preferred language so the caller can greet her without a second
    round trip, and nothing else. It deliberately does not carry her children: whether she
    may see any given child is a separate question with a separate answer, and bundling it
    here would invite a caller to treat "I resolved her" as "she may read this".
    """

    public_id: str
    full_name_ar: str = ""
    full_name_en: str = ""
    preferred_language: str = "ar"

    @property
    def display_name(self) -> str:
        """Her name in whichever script the school recorded, Arabic first."""
        return self.full_name_ar or self.full_name_en


@dataclass(frozen=True, slots=True)
class GuardianLink:
    """One guardian beside the link that says what she is to a child.

    Both sides are optional for the reason `ClassRosterEntry.student` is: a link whose
    guardian row cannot be loaded still appears, carrying the number it names. Dropping it
    would shorten the list silently, and a contact list that is quietly one adult short is
    worse than one showing a number without a name — the number still reaches somebody.

    Serves both directions. `student` is filled when the question started from a guardian
    ("which children may this number ask about") and left `None` when it started from a
    child, where every entry shares one student and repeating her would be noise.
    """

    link: StudentGuardian
    guardian: Guardian | None = None
    student: Student | None = None
    #: Where she is *today*, when the caller asked for it and she has a placement.
    #: `None` covers three different real states — the caller did not ask, she has no
    #: open placement (a child enrolled for next September has none today), or her
    #: section could not be loaded — and none of them is an error.
    class_section: ClassSection | None = None
    #: The rung she stands on, resolved from that section. Separate from it because a
    #: parent thinks in year groups ("Year 4") and a registrar thinks in rooms ("4B"),
    #: and the corpus a parent's question is answered from is written in the first.
    year_level: YearLevel | None = None

    @property
    def phone(self) -> str:
        return str(self.link.guardian_phone)

    @property
    def year_label(self) -> str:
        """What to call her year group, Arabic first, or `""` when nothing is known.

        Falls back to the class section's own name rather than to the year code: "4B"
        is something a parent recognises and `Y4` is an internal string. Empty rather
        than a guess — a wrong year narrows a fee table to the wrong row.
        """
        if self.year_level is not None:
            return self.year_level.name_ar or self.year_level.name_en
        if self.class_section is not None:
            return self.class_section.name_ar or self.class_section.name_en
        return ""

    @property
    def phones(self) -> tuple[str, ...]:
        """Every number that reaches this adult, primary first."""
        return tuple(str(p) for p in self.guardian.phones) if self.guardian else (self.phone,)

    @property
    def student_number(self) -> str:
        return str(self.link.student_number)

    @property
    def display_name_ar(self) -> str:
        return self.guardian.full_name_ar if self.guardian else ""

    @property
    def display_name_en(self) -> str:
        return self.guardian.full_name_en if self.guardian else ""


@dataclass(frozen=True, slots=True)
class GradeLine:
    """A stated mark beside the subject it belongs to — one line of a report card.

    `subject` may be `None` for the same reason as above: a mark whose subject row has
    gone missing still has to be shown, because a report card that quietly drops a line
    is a report card nobody can tell is incomplete.

    Nothing here totals, averages or ranks (invariant 5). A caller wanting an average is
    a caller stating a policy, and it must do so where a human can see it.
    """

    grade: SubjectGrade
    subject: Subject | None = None

    @property
    def is_graded(self) -> bool:
        """Delegates on purpose: no caller re-derives null-vs-zero (invariant 1)."""
        return self.grade.is_graded


@dataclass(frozen=True, slots=True)
class StudentTermGrades:
    """A child's marks for one term, under the class she was in *for that term*.

    `class_section` is resolved through `resolve_sections_for_term` rather than read from
    her current placement, which is what keeps a Term 1 report printing 3A after a March
    move to 3B. `None` means no placement covered that term at all — she had left, or had
    not yet joined — and is deliberately distinguishable from "her current class".
    """

    student: Student
    term: Term
    class_section: ClassSection | None
    lines: tuple[GradeLine, ...] = ()

    @property
    def graded_count(self) -> int:
        """Subjects with a stated figure. The rest are awaiting a mark, not scoring zero."""
        return sum(1 for line in self.lines if line.is_graded)


class QueryService:
    """Every read the API and the `records/` adapter need, and no writes at all.

    Takes a factory rather than a live unit of work so each query runs in its own
    transaction and none of them can see another's half-written state. Nothing here
    commits: leaving the context manager rolls back, which for a read is exactly right and
    means a query can never leave a lock or a stray write behind.
    """

    def __init__(self, uow_factory: Callable[[], UnitOfWork]) -> None:
        self._uow_factory = uow_factory

    # -- Structure ---------------------------------------------------------

    def list_schools(self, *, include_inactive: bool = False) -> Sequence[School]:
        """Every school, by code. Closed branches only when asked for by name."""
        with self._uow_factory() as uow:
            return tuple(uow.schools.list_all(include_inactive=include_inactive))

    def get_school(self, school_code: SchoolCode) -> School:
        """One school, or a refusal. Unknown code is never an empty shell."""
        with self._uow_factory() as uow:
            school = uow.schools.get(school_code)
            if school is None:
                raise UnknownReference(
                    f"no school {school_code}", field="school_code"
                )
            return school

    def list_academic_years(
        self, school_code: SchoolCode | None = None
    ) -> Sequence[AcademicYear]:
        """Every school year, most recent first; one school's when asked for.

        The school is optional here and required on `list_year_levels` below, and the
        asymmetry is not an oversight: the year list is what the school picker itself is
        built from, so it has to be answerable before a school has been chosen. A ladder is
        only ever asked for inside a school.
        """
        with self._uow_factory() as uow:
            if school_code is not None:
                self._require_school(uow, school_code)
            return tuple(uow.academic_years.list_all(school_code))

    def current_academic_year(
        self, school_code: SchoolCode | None = None
    ) -> AcademicYear | None:
        """The year the registrar has marked current, or `None` before one is chosen."""
        with self._uow_factory() as uow:
            return uow.academic_years.current(school_code)

    def list_year_levels(self, school_code: SchoolCode) -> Sequence[YearLevel]:
        """One school's ladder, grouped by stage and ordered within it.

        Grouped rather than flat because a fourteen-rung ladder is unreadable as a list:
        a registrar looks for the garden block, then primary. The grouping is a property of
        each rung (`YearLevel.stage`) rather than a structure returned here, so a caller
        that wants a flat list still has one.
        """
        with self._uow_factory() as uow:
            self._require_school(uow, school_code)
            return tuple(uow.year_levels.list_for_school(school_code))

    def list_classes(
        self,
        academic_year_code: AcademicYearCode,
        *,
        year_level_code: YearCode | None = None,
    ) -> Sequence[ClassSection]:
        """A year's sections, optionally one rung's.

        The year is required rather than defaulted to the current one: a class code names
        a different room of children every September, so a caller that forgot to say which
        year would silently be answered about whichever year happens to be flagged.
        """
        with self._uow_factory() as uow:
            self._require_year(uow, academic_year_code)
            return tuple(
                uow.class_sections.list_for_year(
                    academic_year_code, year_level_code=year_level_code
                )
            )

    def list_terms(self, academic_year_code: AcademicYearCode) -> Sequence[Term]:
        """The year's terms in `sequence` order — chronological without parsing codes."""
        with self._uow_factory() as uow:
            self._require_year(uow, academic_year_code)
            return tuple(uow.terms.list_for_year(academic_year_code))

    def list_subjects(
        self, academic_year_code: AcademicYearCode, *, include_inactive: bool = False
    ) -> Sequence[Subject]:
        """One year's subjects in `display_order`; retired ones only when asked for.

        The year is required rather than defaulted to the current one. A subject catalogue
        is per-year now, so "the subjects" is not a question with an answer — and a silent
        default would put next year's catalogue on screen for a registrar who is working
        through last year's marks, with nothing on the page to say which they were shown.
        """
        with self._uow_factory() as uow:
            self._require_year(uow, academic_year_code)
            return tuple(
                uow.subjects.list_for_year(
                    academic_year_code, include_inactive=include_inactive
                )
            )

    # -- People ------------------------------------------------------------

    def get_student(self, student_number: StudentNumber) -> Student:
        """One child's own record. Unknown number is a refusal, never an empty shell.

        No placement and no marks here on purpose: those are separate questions with
        separate answers per year and per term, and folding "her class" into a student
        record is precisely the column invariant 2 exists to keep out of this service.
        """
        with self._uow_factory() as uow:
            student = uow.students.get(student_number)
            if student is None:
                raise UnknownReference(
                    f"no student {student_number}", field="student_number"
                )
            return student

    def search_students(
        self, query: str, *, limit: int = 50, include_inactive: bool = False
    ) -> Sequence[Student]:
        """Type-ahead over number and both spellings of the name.

        A blank query returns nothing rather than the whole school. The screen behind this
        is a search box: answering an empty box with nine hundred children is a page of
        noise, and it is also the request a registrar makes by accident most often.

        Inactive children are excluded by default. A child who left in March must be
        findable — her marks and her guardians are still true — but she should not appear in
        the picker a registrar uses to place somebody in a class today.
        """
        with self._uow_factory() as uow:
            return tuple(
                uow.students.search(
                    query, limit=limit, include_inactive=include_inactive
                )
            )

    def student_placements(
        self, student_number: StudentNumber
    ) -> Sequence[ClassEnrolment]:
        """Every placement this child has ever had, so the history is visible as history.

        This is invariant 2 made legible. A child who moved 3A -> 3B in March has two rows
        here, both true, and a screen that shows only the open one cannot explain why her
        Term 1 report card says 3A.
        """
        with self._uow_factory() as uow:
            if uow.students.get(student_number) is None:
                raise UnknownReference(
                    f"no student {student_number}", field="student_number"
                )
            return tuple(uow.enrolments.list_for_student(student_number))

    def class_roster(
        self,
        academic_year_code: AcademicYearCode,
        class_code: ClassCode,
        on_date: date,
    ) -> Sequence[ClassRosterEntry]:
        """Who was in this class on `on_date` — the register as of any day.

        `on_date` is a required argument and never `date.today()`. A register is a
        statement about a day, and a default of "now" would answer today's question to a
        caller printing last term's attendance sheet, with nothing on screen to say so.
        """
        with self._uow_factory() as uow:
            self._require_year(uow, academic_year_code)
            if uow.class_sections.get(academic_year_code, class_code) is None:
                # Refused rather than answered with an empty list: "no such class" and
                # "a class with nobody in it" are indistinguishable once rendered, and
                # the first is a typo the caller can fix in seconds.
                raise UnknownReference(
                    f"no class {class_code} in {academic_year_code}", field="class_code"
                )
            enrolments = uow.enrolments.roster_on(
                academic_year_code, class_code, on_date
            )
            students = uow.students.get_many(
                [
                    e.student_number
                    for e in enrolments
                    if isinstance(e.student_number, StudentNumber)
                ]
            )
        return tuple(
            ClassRosterEntry(
                student_number=str(enrolment.student_number),
                enrolment=enrolment,
                student=students.get(str(enrolment.student_number)),
            )
            for enrolment in enrolments
        )

    # -- Guardians ---------------------------------------------------------

    def student_guardians(
        self, student_number: StudentNumber
    ) -> Sequence[GuardianLink]:
        """Every adult on file for one child, with what each is to her.

        An unknown student raises rather than returning an empty sequence, for the reason
        `class_roster` refuses an unknown class: "no such child" and "a child with no
        guardians recorded" render identically, and only one of them is a typo the caller
        can fix in seconds. The second is a real and common answer — a roster is uploaded
        before any guardian file is — so it must stay distinguishable.
        """
        with self._uow_factory() as uow:
            if uow.students.get(student_number) is None:
                raise UnknownReference(
                    f"no student {student_number}", field="student_number"
                )
            links = uow.student_guardians.list_for_student(student_number)
            guardians = uow.guardians.get_many([link.guardian_phone for link in links])
        return tuple(
            GuardianLink(link=link, guardian=guardians.get(str(link.guardian_phone)))
            for link in links
        )

    def resolve_guardian(self, phone: Phone) -> GuardianIdentity | None:
        """Who does this number reach? `None` when it reaches nobody on file.

        `None` rather than a raise, unlike `student_guardians` and `guardian_students`
        below. Those answer a question about somebody the caller already names, so "no
        such person" is a mistake worth reporting; this one *is* the question, and a
        number the school has never seen is an ordinary answer that a caller has to
        handle either way.

        The caller that matters is an authentication service asking "has this verified
        number been entered by a registrar" — which must be able to say no without an
        exception, and must learn nothing about a number that answers no.
        """
        with self._uow_factory() as uow:
            public_id = uow.guardians.public_id_for(phone)
            if public_id is None:
                return None
            guardian = uow.guardians.get(phone)
        if guardian is None:
            return None
        return GuardianIdentity(
            public_id=public_id,
            full_name_ar=guardian.full_name_ar,
            full_name_en=guardian.full_name_en,
            preferred_language=guardian.preferred_language,
        )

    def _audit(self, attempt: AccessAttempt) -> None:
        """Append one decision, in its own transaction, and never fail the request for it.

        **Committed separately from whatever the request goes on to do.** An audit rolled
        back alongside a refusal records only the accesses that succeeded, which is exactly
        backwards — the refusals are the interesting rows. So this opens its own unit of
        work and commits it before the caller raises.

        Best effort, deliberately. A school whose audit table is briefly unwritable should
        still be able to tell a parent her daughter's marks; the alternative is an outage
        caused by bookkeeping. The failure is logged loudly, which is the compromise: a
        silent gap in an audit is the one thing worse than a noisy one.
        """
        try:
            with self._uow_factory() as uow:
                uow.access_audit.record(attempt)
                uow.commit()
        except Exception:  # noqa: BLE001 - never fail a read because the audit failed
            logger.exception(
                "could not record an access attempt: guardian=%s student=%s reason=%s",
                attempt.guardian_public_id,
                attempt.student_number,
                attempt.reason.value,
            )

    def require_guardian_may_see(
        self,
        public_id: str,
        student_number: StudentNumber,
        *,
        actor: str = "",
        request_id: str = "",
    ) -> None:
        """Raise unless the school says this guardian may be told about this child.

        **The authorisation decision for every parent-facing read, in one place.** It is
        made here rather than by the caller, and it is made on *this* request, even though
        every caller is expected to have listed the children first and picked one from that
        list. The reason is what the caller *is*: a chat service running a language model
        over text a stranger can write. Its filtering is a convenience for the model, not a
        security boundary, and a prompt that talks the model into naming another child must
        meet a server that says no rather than one that trusts the id it was handed.

        It is also why the decision belongs here and not in a token or an upstream service.
        The link it reads is the registrar's own, amended the minute a court order arrives;
        anything cached, copied or signed elsewhere keeps answering with the old family
        until it expires.

        The refusal is deliberately the same `UnknownReference` a genuinely missing child
        produces. A caller that could tell "not your child" from "no such child" could walk
        student numbers and learn which ones exist — and one that could tell either from
        "her access was restricted" could detect a custody order from outside the school.

        **Every outcome is audited, including the successful one.** The reason recorded is
        the real one — `no_children` for a handle that reaches nobody, `no_link` for a real
        parent naming a child who is not hers — even though the caller is told the same
        thing either way. That asymmetry is the point of keeping the record: the difference
        between somebody probing with a junk handle and somebody walking student numbers
        against a real parent's is invisible in the response and obvious in this table.
        """
        try:
            permitted = {
                str(entry.link.student_number)
                for entry in self.guardian_students_by_id(public_id, viewable_only=True)
            }
        except UnknownReference:
            # No such guardian handle. Audited under the same reason as a guardian whose
            # every link is restricted, because those two are one fact from outside — see
            # `guardian_students_by_id` — and then re-raised unchanged.
            self._audit(
                AccessAttempt(
                    guardian_public_id=public_id or "unknown",
                    student_number=str(student_number),
                    reason=AccessReason.NO_CHILDREN,
                    at=datetime.now(UTC),
                    actor=actor,
                    request_id=request_id,
                )
            )
            raise

        if str(student_number) in permitted:
            reason = AccessReason.OK
        elif not permitted:
            reason = AccessReason.NO_CHILDREN
        else:
            reason = AccessReason.NO_LINK

        self._audit(
            AccessAttempt(
                guardian_public_id=public_id,
                student_number=str(student_number),
                reason=reason,
                at=datetime.now(UTC),
                actor=actor,
                request_id=request_id,
            )
        )

        if reason is not AccessReason.OK:
            raise UnknownReference(
                "no such student for this guardian", field="student_number"
            )

    def guardian_student_term_grades(
        self,
        public_id: str,
        student_number: StudentNumber,
        term_code: TermCode,
        *,
        actor: str = "",
        request_id: str = "",
    ) -> StudentTermGrades:
        """One child's marks for a term, for a caller who is only a guardian handle."""
        self.require_guardian_may_see(
            public_id, student_number, actor=actor, request_id=request_id
        )
        return self.student_term_grades(student_number, term_code)

    def guardian_students_by_id(
        self, public_id: str, *, viewable_only: bool = True, on_date: date | None = None
    ) -> Sequence[GuardianLink]:
        """The same answer as `guardian_students`, asked with a handle instead of a number.

        This is the parent-facing question as the chat service asks it. That service is
        handed a handle when a parent signs in and is deliberately never told the number
        behind it: it is the process running a language model over untrusted input, and a
        parent's phone number is the one piece of PII it has no reason to hold.

        Raises for an unknown handle, like its sibling, so "that is not a guardian" stays
        distinguishable from "that guardian may see no children" — the second is what a
        custody restriction looks like, and it must not read as a broken token.
        """
        with self._uow_factory() as uow:
            phone = uow.guardians.primary_phone_for(public_id)
        if phone is None:
            raise UnknownReference(
                "no guardian is on file under that reference", field="guardian_id"
            )
        return self.guardian_students(
            phone, viewable_only=viewable_only, on_date=on_date
        )

    def guardian_students(
        self, phone: Phone, *, viewable_only: bool = True, on_date: date | None = None
    ) -> Sequence[GuardianLink]:
        """Which children this number may ask about — the parent-facing question.

        Raises when the number reaches nobody, so a caller can tell "not a guardian here"
        from "a guardian barred from every child on file". Those need different answers:
        the first is a wrong number, the second is a custody restriction working.

        `on_date` asks for each child's year group as of that day, and is optional
        because most callers do not need it and it costs two more queries. Passed in
        rather than read from a clock, like every other date-sensitive question in this
        service: "which class is she in" has a different answer in June and in September,
        and a query that reads `date.today()` cannot be tested for either.
        """
        with self._uow_factory() as uow:
            guardian = uow.guardians.get(phone)
            if guardian is None:
                raise UnknownReference(f"no guardian on {phone}", field="phone")
            links = uow.student_guardians.list_students_for_guardian(
                phone, viewable_only=viewable_only
            )
            numbers = [
                link.student_number
                for link in links
                if isinstance(link.student_number, StudentNumber)
            ]
            students = uow.students.get_many(numbers)
            sections, levels = self._placements(uow, numbers, on_date)
        return tuple(
            GuardianLink(
                link=link,
                guardian=guardian,
                student=students.get(str(link.student_number)),
                class_section=sections.get(str(link.student_number)),
                year_level=levels.get(str(link.student_number)),
            )
            for link in links
        )

    @staticmethod
    def _placements(
        uow: UnitOfWork,
        numbers: Sequence[StudentNumber],
        on_date: date | None,
    ) -> tuple[Mapping[str, ClassSection], Mapping[str, YearLevel]]:
        """Each child's class today, and the year group that class sits on.

        Three bulk queries for the whole family rather than three per child, using the
        transfer-aware `class_sections_on` — a child who moved 4A -> 4B in March resolves
        to the room she is in now, not to whichever placement was written first.

        The year level takes two hops because the schema refuses to shortcut them, and
        both refusals are deliberate: a class is scoped to an academic year (so `3A` of
        2025-2026 is a different group of children from `3A` of 2026-2027), while a year
        LEVEL is scoped to a school and not to a year (so "Year 3" is the same rung every
        year, and cross-year comparisons are a filter rather than a text match). Getting
        from one to the other therefore goes through the academic year, which is the only
        thing that names the school.

        Every step degrades to "not known" rather than raising. A child with no open
        placement is an ordinary state — one enrolled for next September has none today —
        and so is a school whose ladder was never uploaded.
        """
        if not numbers or on_date is None:
            return {}, {}
        sections = uow.enrolments.class_sections_on(numbers, on_date)
        if not sections:
            return {}, {}

        years = uow.academic_years.get_many(
            {section.academic_year_code for section in sections.values()}
        )
        # Year levels are keyed by (school, code), so they are fetched a school at a
        # time. A family is almost always in one school; the loop exists for the branch
        # transfer that is not.
        wanted: dict[str, set] = {}
        for section in sections.values():
            year = years.get(str(section.academic_year_code))
            if year is not None:
                wanted.setdefault(str(year.school_code), set()).add(section.year_level_code)

        by_school: dict[str, Mapping[str, YearLevel]] = {
            school: uow.year_levels.get_many(codes, school)
            for school, codes in wanted.items()
        }

        levels: dict[str, YearLevel] = {}
        for number, section in sections.items():
            year = years.get(str(section.academic_year_code))
            if year is None:
                continue
            found = by_school.get(str(year.school_code), {}).get(str(section.year_level_code))
            if found is not None:
                levels[number] = found
        return sections, levels

    # -- Grades ------------------------------------------------------------

    def student_term_grades(
        self, student_number: StudentNumber, term_code: TermCode
    ) -> StudentTermGrades:
        """A child's marks for one term, with the class she was in for that term.

        Returns the grades exactly as stated (invariant 5) and in the order the repository
        contract defines — subject `display_order`. They are not re-sorted here, because a
        re-sort would need every `Subject` row and would quietly reshuffle a report card
        whenever one of them failed to load.
        """
        with self._uow_factory() as uow:
            student = uow.students.get(student_number)
            if student is None:
                raise UnknownReference(
                    f"no student {student_number}", field="student_number"
                )
            term = uow.terms.get(term_code)
            if term is None:
                raise UnknownReference(f"no term {term_code}", field="term_code")
            grades = uow.grades.list_for_student(student_number, term_code=term_code)
            # Resolved within the term's own year, which is the only year these marks can
            # be about. A lookup by code alone would have to guess, and the guess it would
            # most naturally make — the current year — is wrong for every report card a
            # registrar reprints after September.
            subjects = uow.subjects.get_many(
                [
                    grade.subject_code
                    for grade in grades
                    if isinstance(grade.subject_code, SubjectCode)
                ],
                term.academic_year_code,
            )
            section = resolve_section_for_term(uow.enrolments, student_number, term)
        return StudentTermGrades(
            student=student,
            term=term,
            class_section=section,
            lines=tuple(
                GradeLine(grade=grade, subject=subjects.get(str(grade.subject_code)))
                for grade in grades
            ),
        )

    def subject_codes_for_grades(
        self, student_number: StudentNumber, term_code: TermCode
    ) -> Sequence[SubjectCode]:
        """The subjects a child has a *row* for this term, graded or not.

        Separate from `student_term_grades` because the `records/` adapter asks only which
        subjects are on file, and pulling a full report card to answer that is a read of
        every mark for a question about headings.
        """
        return tuple(
            SubjectCode(str(line.grade.subject_code))
            for line in self.student_term_grades(student_number, term_code).lines
        )

    @staticmethod
    def _require_school(uow: UnitOfWork, code: SchoolCode) -> None:
        """One unknown-school check, so every listing fails the same way it succeeds."""
        if uow.schools.get(code) is None:
            raise UnknownReference(f"no school {code}", field="school_code")

    @staticmethod
    def _require_year(uow: UnitOfWork, code: AcademicYearCode) -> None:
        """One unknown-year check, so every listing fails the same way it succeeds."""
        if uow.academic_years.get(code) is None:
            raise UnknownReference(
                f"no academic year {code}", field="academic_year_code"
            )
