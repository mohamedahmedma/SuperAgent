"""The student-record tools — thin relays to the records facade.

ONE TOOL PER CAPABILITY, and per question a parent actually asks: grades for the term, one
subject's breakdown, attendance, the weekly timetable, which class the child is in, what
they study, who teaches them, and who teaches them one named subject. That is the whole
structure of this file.

The last four sit over THREE facade endpoints rather than four, because three of those
questions are answers about the same room and the facade resolves it once — see
`records/ports/classroom.py`. The tool layer still gets one name per question, which is
what the planner selects on.

They were a single `get_student_records(record_type=…)` until the planner learned to
dispatch a SET of tools rather than pick one, at which point the merged shape stopped
being able to say what a turn needed. A planner that has decided a turn is about
attendance cannot express that as a tool NAME while the tool is chosen by an argument —
it has to guess the argument too, and a guessed `record_type` is a wrong lookup the
model then pays a round trip to correct. Split, the same decision is `needed_tools`,
which is already how every other tool is selected, and a question about marks AND
absences dispatches both at once instead of serialising two calls through one name.

Adding a capability is therefore: a builder here, a line in `TOOL_BUILDERS`, and a line
in the profile's `tool_selection`. No argument enum, no branch in a dispatcher, nothing
that has to know about the others. `get_student_timetable` is the one that proved it: it
slotted in beside the other three when the facade grew the endpoint, and nothing in the
planner, the narrowing or the dispatcher had to learn it existed.

The last five are the ones whose subject is not the child. Marks and attendance are facts
about the child; a week, a class name, a subject board and a staff list all belong to the
ROOM they are placed in. That resolution stays behind the facade — this file sends a
student number and no class code, because a placement changes mid-year and a class code
cached up here would eventually name a room the child has left.

The cost is honest and worth stating: eight tool schemas reach the model every turn
instead of one, and each keeps its own call budget rather than sharing one ceiling. The
audited-read ceiling is NOT one of those budgets — `_resolve_student` takes a slot from a
single per-turn allowance shared by every tool here, so splitting a capability into two
names never widens how much of a minor's record one turn may read.

This file is deliberately the smallest thing that could work. All the judgement lives
elsewhere: authorisation in the records facade, identity in the identity service,
grade arithmetic in `records/grading.py`. What is left here is an HTTP call, a
mapping from response to outcome, and nothing else. If this module starts growing
policy, the policy is in the wrong place.

**The model cannot name a guardian.** There is no tool argument for one. The guardian
id and the bearer token come from `ctx`, which the HTTP layer sets from the caller's
verified session. No amount of prompt injection reaches them, because they are not
parameters the model can reach — the worst a hostile message achieves is asking about
a student the signed token already authorises.

What this module SAYS back to the model lives in
backend/prompts/templates/tools/records_result.j2, matching the split used by
search_knowledge_base: this file decides which outcome occurred, the template renders
it. Every branch depends on what the call actually returned, so the system prompt
cannot state any of it in advance.
"""
import logging
import os

import requests
from langchain_core.tools import tool

from backend.chat.child_resolution import resolve_child
from backend.chat.child_roster import ChildOption, forget, load_roster
from backend.chat.request_context import ChatRequestContext
from backend.env import records_api_key, records_base_url
from backend.prompts import render as render_prompt
from backend.text_matching import name_key

logger = logging.getLogger(__name__)

# Resolved in `backend.env`, which `backend.chat.child_roster` reads too. Both talk to the
# same facade with the same credentials, and two copies of the default is how they end up
# talking to two different ones — the marks from the configured facade, the child roster
# from wherever the other copy pointed.
BASE_URL = records_base_url()
API_KEY = records_api_key()
# Short. This sits inside a chat turn a parent is waiting on, and a slow answer that
# arrives is worse than a fast "records are unavailable" they can act on.
TIMEOUT_SECONDS = float(os.getenv("RECORDS_TIMEOUT_SECONDS") or 8)


def _get(path: str, ctx: ChatRequestContext, params: dict | None = None) -> tuple[str, dict]:
    """One GET against the facade. Returns `(outcome, payload)`.

    Every failure mode collapses to an outcome string here so the tool body never
    branches on HTTP status codes. Note that a timeout and a 503 produce the same
    `unavailable` — to the parent they are the same event, and the template's job is
    to stop the model inventing a figure either way.
    """
    headers = {
        "X-API-Key": API_KEY,
        # The parent's own identity, relayed unchanged. This service does not mint it
        # and cannot alter whose records it authorises.
        "Authorization": f"Bearer {ctx.guardian_token}",
        # Ties the facade's audit row back to this chat turn.
        "X-Request-Id": ctx.session_id or "",
    }

    try:
        response = requests.get(
            f"{BASE_URL}{path}", headers=headers, params=params or {}, timeout=TIMEOUT_SECONDS
        )
    except requests.RequestException as exc:
        logger.warning("records facade unreachable: %s", exc)
        return "unavailable", {}

    if response.status_code in (401, 403):
        # The session is not authorised for this guardian. Not a "no records" answer —
        # saying "she has no grades" to a parent whose token expired is a lie.
        logger.warning("records facade rejected identity: %s", response.status_code)
        return "not_authorized", {}
    if response.status_code == 404:
        return "no_records", {}
    if response.status_code >= 500:
        logger.warning("records facade error: %s", response.status_code)
        return "unavailable", {}
    if response.status_code != 200:
        return "unavailable", {}

    try:
        return "ok", response.json()
    except ValueError:
        return "unavailable", {}


def _match_student(
    students: list[ChildOption], student_name: str, *, ctx=None
) -> ChildOption | None:
    """Pick the child the parent meant, using the turn's one resolver.

    A thin adapter over `backend.chat.child_resolution.resolve_child` rather than a
    second route table. There used to be one here and another in the planner, and two
    matchers with different rules drift — which here means the prompt naming one child
    while the tool answers about a different one, in the same turn.

    Returns None when the answer is not unambiguous. The caller then asks rather than
    guessing, since guessing means showing one child's grades while naming another.

    A name the model supplied still wins over the pin, so "and how is Omar?" moves the
    conversation on even when the previous question was about his sister — that is
    route 1 of the shared resolver, reached by passing `reference="named"`.

    ## The planner's answer wins over both

    When the turn planner already resolved a child, that is who this reads, and the
    `student_name` argument is not consulted at all. Both ends of that trade are
    measured:

    What is given up is nothing. The planner reaches its answer through the SAME
    resolver, on the same roster, from a classifier that reports a name the message
    actually contains — so "and how is Omar?" arrives here as a resolved Omar, by route
    1, exactly as it did before. The one case the argument used to cover on its own is
    the case the planner now covers first.

    What is bought is the failure this closes. `student_name` is the model's
    transcription of a name it read once, and «ليلى أحمد» came back mis-spelled often
    enough to matter: the roster matcher then found nobody, the tool asked which child,
    and a parent who had named their daughter in plain words was asked to name her
    again. The planner's answer is a roster row — it cannot be mis-spelled, because it
    was never re-typed.

    This never widens what may be read. The id came from a roster fetched under this
    turn's guardian token, and the facade re-checks that token on the read itself.
    """
    if not students:
        return None
    planned = getattr(ctx, "planned_child_id", "")
    if planned:
        resolved = next((s for s in students if s.student_id == planned), None)
        if resolved is not None:
            return resolved
        # On the roster the planner read and not on this one. A child withdrawn
        # mid-conversation, or two reads either side of a change. Fall through and
        # resolve from what is actually here rather than answering about nobody.
        logger.info("the planner's child is not on the roster this call read")
    pin = getattr(ctx, "child", None)
    found = resolve_child(
        reference="named" if student_name else "context",
        child_name=student_name,
        roster=students,
        pin=pin,
    )
    if not found.resolved:
        return None
    return next((s for s in students if s.student_id == found.student_id), None)


#: The tools this module builds, in the order a profile would naturally list them.
#: Named here so that policy and profile code can talk about "the record tools" without
#: importing this module, which reaches the request context and the HTTP layer.
GRADES_TOOL = "get_student_grades"
ATTENDANCE_TOOL = "get_student_attendance"
TIMETABLE_TOOL = "get_student_timetable"
CLASS_TOOL = "get_student_class"
SUBJECTS_TOOL = "get_student_subjects"
TEACHERS_TOOL = "get_student_teachers"


def _reporter(ctx: ChatRequestContext, tool_name: str):
    """Render one outcome, and tell the turn which one it was.

    Every return from every tool below goes through this, so the string the model reads
    and the string the turn records are produced from the same variable and cannot come
    to disagree. The recorded half is what lets something downstream know that a record
    WAS retrieved — see `ChatRequestContext.note_tool_outcome` for why a call count on
    its own could not.

    The OUTCOME names are shared across the three tools and unchanged by the split:
    `service.RECORDS_RETRIEVED` reads outcomes rather than tool names, so the check that
    catches an answer denying the record it just read keeps working without needing to
    know how many tools can produce one.
    """

    def _result(outcome: str, **context) -> str:
        ctx.note_tool_outcome(tool_name, outcome)
        if outcome in _PRESENTED:
            # Rendered from the SAME context the model's copy is rendered from, so the
            # grid the parent reads and the grid the model was shown cannot drift. The
            # data beside it is built from that context too, for the same reason.
            try:
                ctx.note_answer_block(
                    render_prompt("tools/records_block.j2", outcome=outcome, **context),
                    kind=outcome,
                    data=_block_data(outcome, context),
                )
            except Exception:  # pragma: no cover - a block must never break a turn
                logger.warning("could not render the %s block", outcome, exc_info=True)
        return render_prompt("tools/records_result.j2", outcome=outcome, **context)

    return _result


def _refused(ctx: ChatRequestContext, outcome: str, result) -> str:
    """A refusal is evidence the cached roster is stale; an outage is not.

    The pin is dropped only here, and deliberately not on `unavailable`: a hint the
    reader re-checks anyway is not worth discarding over a timeout, and doing so would
    re-ask the parent for a reason they could never see.
    """
    if outcome == "not_authorized":
        forget(ctx)
        ctx.forget_child()
    return result(outcome)


def _resolve_student(ctx: ChatRequestContext, student_name: str, result):
    """Everything that has to be true before any record can be read.

    Returns `(refusal, student)` with exactly one of them set. Shared by all three tools
    rather than repeated in each, because these are not conveniences — they are the
    budget, the identity check and the roster match, and three copies of them is three
    places for one of the checks to go missing.

    The budget slot is taken HERE, so it is shared ACROSS the record tools rather than
    held per tool. That is deliberate: it exists because every call is an audited read of
    a minor's records, and splitting one tool into three must not silently triple how
    many of those a single turn can perform.
    """
    if not ctx.acquire_records_tool_slot():
        return result("call_limit"), None

    # No verified guardian on this session. Staff, a test, or a signed-out user.
    if not ctx.guardian_token or not ctx.guardian_id:
        return result("not_a_parent"), None

    # One cached read per conversation, shared with anything else in the turn that needs
    # to know who this parent's children are.
    roster_outcome, students = load_roster(ctx)
    # The roster's outcome names are the template's outcome names, so a refusal or an
    # outage relays unchanged and keeps its own careful wording.
    if roster_outcome in ("unavailable", "not_authorized"):
        return _refused(ctx, roster_outcome, result), None
    if not students:
        return result("no_students"), None

    student = _match_student(students, student_name, ctx=ctx)
    if student is None:
        # Either several children and no name, or a name matching more than one. Both
        # are questions for the parent, never a coin flip.
        return result("which_student", options=[s.label for s in students]), None

    # Pinned for the rest of the conversation, so a parent who answered once is not
    # asked again. Recorded only after a child has actually been resolved, so a turn
    # that failed to identify one leaves nothing wrong pinned behind it.
    #
    # The label rides along because the pin is durable: a later turn that wants to say
    # which child it is answering about would otherwise have only an opaque student
    # number to show a parent.
    ctx.remember_child(student.student_id, label=student.label, gender=student.gender)
    return None, student


def _student_path(ctx: ChatRequestContext, student_id: str) -> str:
    return f"/v1/guardians/{ctx.guardian_id}/students/{student_id}"


#: Appended to every tool docstring below. Stated once rather than written out three
#: times, because it is the same guarantee each time and a copy that drifts is a copy
#: that tells the model something untrue about whose records it is reading.
_STUDENT_NAME_NOTE = """
        student_name: leave this empty. Which child the question is about has already
        been worked out from the school's own list of this parent's children, and that
        answer is used. Fill it in only if you are starting a subject the conversation
        has not mentioned at all.

        You never supply the parent's identity; it is taken from their signed-in
        session. Which child is settled the same way. If the tool asks you to clarify
        which child, put that question to the parent rather than choosing one.
        """


def make_get_student_grades(ctx: ChatRequestContext):
    @tool(GRADES_TOOL)
    def get_student_grades(student_name: str = "", subject: str = "") -> str:
        """Read one child's marks for the current term, across every subject or one.

        Use this for how a child is doing overall, their results or their report card,
        and equally for their mark in one named subject — what she got in Arabic. For
        absences use get_student_attendance. Do not use either for school policies, fees
        or general information; those come from the knowledge base.

        subject: OPTIONAL, and the only argument that is yours to supply — it comes from
        the message itself. Leave it empty whenever the question names no subject; naming
        one adds that subject's assignment-by-assignment detail, and a name matching none
        is answered with the subjects the child actually takes, so a wrong guess costs
        nothing.
        """
        result = _reporter(ctx, GRADES_TOOL)
        refusal, student = _resolve_student(ctx, student_name, result)
        if refusal:
            return refusal

        path = _student_path(ctx, student.student_id)
        # The term's rollup first, whichever question was asked. Without a subject it IS
        # the answer; with one it is how a subject named in words becomes the course id
        # the facade addresses a breakdown by.
        outcome, data = _get(f"{path}/grades", ctx)
        if outcome != "ok":
            return _refused(ctx, outcome, result)

        wanted = (subject or "").strip()
        if not wanted:
            return result("grades", **_render_context(student.label, data))

        course = _match_course(data, wanted)
        if course is None:
            return result(
                "which_subject",
                student=student,
                options=[
                    c.get("subject_name_ar") or c.get("subject_name_en")
                    for c in data.get("courses") or []
                ],
            )

        detail_outcome, detail = _get(f"{path}/grades/{course.get('course_id')}", ctx)
        if detail_outcome != "ok":
            return _refused(ctx, detail_outcome, result)
        return result("subject", **_render_context(student.label, detail))

    get_student_grades.description += _STUDENT_NAME_NOTE
    return get_student_grades


def make_get_student_attendance(ctx: ChatRequestContext):
    @tool(ATTENDANCE_TOOL)
    def get_student_attendance(student_name: str = "") -> str:
        """Read one child's attendance: the days they were present, absent or late.

        Use this for absences, lateness and attendance. For marks use
        get_student_grades.
        """
        result = _reporter(ctx, ATTENDANCE_TOOL)
        refusal, student = _resolve_student(ctx, student_name, result)
        if refusal:
            return refusal

        outcome, data = _get(f"{_student_path(ctx, student.student_id)}/attendance", ctx)
        if outcome != "ok":
            return _refused(ctx, outcome, result)
        return result("attendance", **_render_context(student.label, data))

    get_student_attendance.description += _STUDENT_NAME_NOTE
    return get_student_attendance


def make_get_student_timetable(ctx: ChatRequestContext):
    @tool(TIMETABLE_TOOL)
    def get_student_timetable(student_name: str = "") -> str:
        """Read one child's weekly class timetable: which lessons fall on which day.

        Use this for their schedule, their timetable, what lessons they have on a given
        day, when a subject is taught, or what time the school day runs to. For marks use
        get_student_grades; for absences use get_student_attendance. Do not use it for term
        dates, holidays or exam schedules — those are school-wide and come from the
        knowledge base.
        """
        result = _reporter(ctx, TIMETABLE_TOOL)
        refusal, student = _resolve_student(ctx, student_name, result)
        if refusal:
            return refusal

        # No class code is sent, and there is none to send. A timetable belongs to a room,
        # the room is the child's placement for the term, and the facade resolves it
        # through SIS — which is the only thing that knows a placement changed in March.
        outcome, data = _get(f"{_student_path(ctx, student.student_id)}/timetable", ctx)
        if outcome != "ok":
            return _refused(ctx, outcome, result)
        return result(
            "timetable",
            **_timetable_context(student.label, data, getattr(ctx, "language", "")),
        )

    get_student_timetable.description += _STUDENT_NAME_NOTE
    return get_student_timetable


#: Week day names for the block a PARENT reads. The model used to translate these on its
#: way past; now that the grid reaches the reader untouched, the presentation is ours to
#: do. A parent asking in Arabic and being shown «sunday» is the cost of rendering
#: verbatim, and this is where that cost is paid rather than handed back to the model.
_DAY_NAMES_AR = {
    "sunday": "الأحد", "monday": "الاثنين", "tuesday": "الثلاثاء",
    "wednesday": "الأربعاء", "thursday": "الخميس", "friday": "الجمعة",
    "saturday": "السبت",
}


def _clock(value: str) -> str:
    """`07:45:00` as `07:45`. Seconds are never what a school means by a bell time."""
    text = str(value or "").strip()
    parts = text.split(":")
    return f"{parts[0]}:{parts[1]}" if len(parts) >= 2 else text


def _day_label(day: str, language: str) -> str:
    """The day as the reader says it, falling back to the school's own spelling."""
    key = str(day or "").strip().lower()
    if str(language or "").startswith("ar"):
        return _DAY_NAMES_AR.get(key, str(day or ""))
    return str(day or "").capitalize()


def _period_label(row: dict, language: str) -> str:
    """A period's name ("فسحة", "Break") in the reader's language, by `_day_label`'s rule."""
    if str(language or "").startswith("ar"):
        return _label(row, "name_ar", "name_en")
    return _label(row, "name_en", "name_ar")


def _timetable_context(student_label: str, data: dict, language: str = "") -> dict:
    """Flatten a week into the day-by-day shape the template renders.

    Grouped here rather than in Jinja for the reason `_render_context` gives: a filter
    chain over rows that may be missing a key is one optional field away from raising
    mid-turn under StrictUndefined. And the grouping itself is a decision — a parent asks
    "what does she have on Sunday", so the model is handed days rather than a flat list it
    would have to sort, in the school's own week order rather than any order Python would
    pick.

    `status` arrives from the facade already saying which of the three answers this is, so
    nothing here infers "no lessons" from an empty list — that is the conflation the whole
    contract is shaped to prevent.
    """
    lessons = data.get("lessons") or []
    periods = {
        int(row.get("period_number") or 0): row for row in data.get("periods") or []
    }

    def _slot(lesson: dict) -> dict:
        period = periods.get(int(lesson.get("period_number") or 0)) or {}
        return {
            "period_number": lesson.get("period_number"),
            "subject": _label(lesson, "subject_name_ar", "subject_name_en"),
            "is_free": not (lesson.get("subject_code") or ""),
            "starts_at": period.get("starts_at") or "",
            "ends_at": period.get("ends_at") or "",
            # Trimmed copies for the parent-facing block. Kept BESIDE the raw values
            # rather than replacing them: the model-facing render quotes the school's
            # own strings, and the grounding check reads that text.
            "shows_from": _clock(period.get("starts_at") or ""),
            "shows_to": _clock(period.get("ends_at") or ""),
        }

    days = [
        {
            "name": day,
            "shows_as": _day_label(day, language),
            "slots": [
                _slot(lesson)
                for lesson in lessons
                if str(lesson.get("day_of_week") or "").lower() == str(day).lower()
            ],
        }
        for day in data.get("days") or []
    ]

    return {
        "student_label": student_label,
        "term_label": _label(data.get("term") or {}, "name_ar", "name_en", "term_id"),
        # The year the term belongs to, rendered beside it. A parent naturally hears
        # "which year is this?" in a record answer, and a model supplying it from its
        # own head states a figure that is in no evidence — which the grounding check
        # discards the whole answer over. Written here, quoting it is grounded.
        "academic_year_label": str((data.get("term") or {}).get("academic_year") or ""),
        "status": str(data.get("status") or ""),
        "class_label": _label(data, "class_name_ar", "class_name_en", "class_code"),
        "days": days,
        # The breaks, so the model can say when the day ends and when the break falls
        # without having to read them out of the lessons, which never contain them.
        "breaks": [
            {
                "name": _label(row, "name_ar", "name_en"),
                "starts_at": row.get("starts_at") or "",
                "ends_at": row.get("ends_at") or "",
            }
            for row in data.get("periods") or []
            if not row.get("is_teaching", True)
        ],
        # The school's whole day, breaks included, in its own order — for a client that
        # draws the week rather than reading it out. The facade is explicit that a client
        # drawing the day must show a break, so they stay in place here rather than being
        # set aside the way the model's copy above sets them aside.
        "periods": [
            {
                "number": int(row.get("period_number") or 0),
                "label": _period_label(row, language),
                "shows_from": _clock(row.get("starts_at") or ""),
                "shows_to": _clock(row.get("ends_at") or ""),
                "is_teaching": bool(row.get("is_teaching", True)),
            }
            for row in sorted(
                data.get("periods") or [], key=lambda row: int(row.get("period_number") or 0)
            )
        ],
        "data": data,
    }


# --- the room she sits in: her class, her subjects, her teachers -------------------
#
# Three facade endpoints and three tools over them, one each. A parent who names a
# subject wants one answer rather than a list to read through, and that used to be its
# own tool — but a filter is not a different record: the classifier had to tell two
# near-identical descriptions apart every turn, and a tool whose argument comes from the
# message can never be pre-dispatched. It is an optional `subject` on the parent tool now.
#
# None of them sends a class code, and none of them could: the facade resolves the room
# from the child's placement for the term. A class code held up here would be one that
# keeps naming 3A after she has moved to 3B.


def make_get_student_class(ctx: ChatRequestContext):
    @tool(CLASS_TOOL)
    def get_student_class(student_name: str = "") -> str:
        """Read which class one child is in — the class name the school uses for it.

        Use this when the question asks which class, section or room a child is in, or
        what their class is called. For the lessons that class sits use
        get_student_timetable; for the subjects it studies use get_student_subjects.
        """
        result = _reporter(ctx, CLASS_TOOL)
        refusal, student = _resolve_student(ctx, student_name, result)
        if refusal:
            return refusal

        outcome, data = _get(f"{_student_path(ctx, student.student_id)}/class", ctx)
        if outcome != "ok":
            return _refused(ctx, outcome, result)
        return result(
            "class",
            student_label=student.label,
            status=str(data.get("status") or ""),
            # The NAME, and only the name: the code is an internal key and a parent has
            # never seen it. `_label` falls back to the other script rather than to blank.
            class_label=_label(data, "class_name_ar", "class_name_en"),
            year_label=_label(data, "year_level_name_ar", "year_level_name_en"),
        )

    get_student_class.description += _STUDENT_NAME_NOTE
    return get_student_class


def make_get_student_subjects(ctx: ChatRequestContext):
    @tool(SUBJECTS_TOOL)
    def get_student_subjects(student_name: str = "") -> str:
        """Read the list of subjects one child studies this year.

        Use this for which subjects a child takes, what they study, or their curriculum.
        For their marks in those subjects use get_student_grades; for when each one is
        taught use get_student_timetable; for who teaches them use get_student_teachers.
        """
        result = _reporter(ctx, SUBJECTS_TOOL)
        refusal, student = _resolve_student(ctx, student_name, result)
        if refusal:
            return refusal

        outcome, data = _get(f"{_student_path(ctx, student.student_id)}/subjects", ctx)
        if outcome != "ok":
            return _refused(ctx, outcome, result)
        return result(
            "subjects",
            student_label=student.label,
            status=str(data.get("status") or ""),
            class_label=_label(data, "class_name_ar", "class_name_en"),
            subjects=[
                _label(row, "name_ar", "name_en", "code")
                for row in data.get("subjects") or []
            ],
        )

    get_student_subjects.description += _STUDENT_NAME_NOTE
    return get_student_subjects


def make_get_student_teachers(ctx: ChatRequestContext):
    @tool(TEACHERS_TOOL)
    def get_student_teachers(student_name: str = "", subject: str = "") -> str:
        """Read who teaches one child, for every subject or for one named subject.

        Use this for who a child's teachers are, and equally for who teaches them one
        subject — who their maths teacher is, who gives them Arabic. Do not use it for
        how to contact a teacher or for parent-meeting times; those come from the
        knowledge base.

        subject: OPTIONAL, and the only argument that is yours to supply — it comes from
        the message itself. Leave it empty whenever the question names no subject; a name
        matching none is answered with the subjects the child actually has a teacher for,
        so a wrong guess costs nothing.
        """
        result = _reporter(ctx, TEACHERS_TOOL)
        refusal, student = _resolve_student(ctx, student_name, result)
        if refusal:
            return refusal

        # One call either way. The teacher rows carry their own subject names, so a named
        # subject is matched against what came back rather than resolved through a second
        # endpoint first — which is what the grades tool has to do only because a
        # subject's detail lives at its own URL.
        outcome, data = _get(f"{_student_path(ctx, student.student_id)}/teachers", ctx)
        if outcome != "ok":
            return _refused(ctx, outcome, result)

        wanted = (subject or "").strip()
        if not wanted:
            return result("teachers", **_teachers_context(student.label, data))

        rows = data.get("teachers") or []
        matched = _match_subject_rows(rows, wanted)
        if not matched:
            return result(
                "which_subject",
                student=student,
                options=sorted(
                    {_label(row, "subject_name_ar", "subject_name_en", "subject_code")
                     for row in rows}
                ),
            )
        # Every teacher of that subject, never the first: two teachers for one subject is a
        # state the school's records permit, and dropping one hides a real person from the
        # parent who asked.
        return result(
            "subject_teacher",
            student_label=student.label,
            status=str(data.get("status") or ""),
            subject_label=_label(
                matched[0], "subject_name_ar", "subject_name_en", "subject_code"
            ),
            teachers=[_label(row, "full_name_ar", "full_name_en") for row in matched],
        )

    get_student_teachers.description += _STUDENT_NAME_NOTE
    return get_student_teachers


def _match_subject_rows(rows: list, subject: str) -> list:
    """The teacher rows whose subject is the one the parent named.

    Folded, not casefolded, for the reason `_match_course` gives: a parent writes
    «الرياضيات» however their keyboard produced it and the subject table spells it one
    fixed way. Matched against both scripts and the code, so "maths", "Math" and the
    school's own «الرياضيات» all land.
    """
    needle = name_key(subject)
    if not needle:
        return []
    return [
        row
        for row in rows
        if needle in name_key(row.get("subject_name_ar") or "")
        or needle in name_key(row.get("subject_name_en") or "")
        or needle in name_key(row.get("subject_code") or "")
    ]


def _teachers_context(student_label: str, data: dict) -> dict:
    """Flatten the staff list into one entry per subject, teachers grouped under it.

    Grouped here rather than in Jinja for the reason `_render_context` gives, and grouped
    at all because the payload is one row per (teacher, subject): rendered flat, a subject
    with two teachers reads as two separate facts, and a teacher taking two subjects reads
    as two people. Grouping also makes the co-teacher case say what it is — "maths: A and
    B" rather than two lines a model may summarise into one.

    Order is preserved from the payload, which is the school's own subject order.
    """
    groups: list[dict] = []
    seen: dict[str, dict] = {}
    for row in data.get("teachers") or []:
        code = str(row.get("subject_code") or "")
        label = _label(row, "subject_name_ar", "subject_name_en", "subject_code")
        key = code or label
        group = seen.get(key)
        if group is None:
            group = {"subject": label, "teachers": []}
            seen[key] = group
            groups.append(group)
        name = _label(row, "full_name_ar", "full_name_en")
        if name:
            group["teachers"].append(name)

    return {
        "student_label": student_label,
        "status": str(data.get("status") or ""),
        "class_label": _label(data, "class_name_ar", "class_name_en"),
        # A subject whose every teacher row has no name on file is dropped rather than
        # rendered as "maths: " with nothing after it. The school can create a teacher
        # with neither name spelling filled in, and a blank beside a subject reads to a
        # parent as a missing answer rather than as missing data — if that empties the
        # list entirely, the template's "not recorded yet" branch says so honestly.
        "subject_groups": [group for group in groups if group["teachers"]],
    }


def _match_course(grades: dict, subject: str):
    """The course a parent meant, matched by name against what the child actually takes.

    Folded, not casefolded: a parent asks about «الرياضيات» however their keyboard
    produced it, and the subject table spells it one fixed way. Same class of failure as
    the child-name matcher above, and the same fix. Folded rather than stemmed because a
    subject name is a proper noun — see backend/text_matching.py.
    """
    needle = name_key(subject)
    if not needle:
        return None
    return next(
        (
            c
            for c in grades.get("courses") or []
            if needle in name_key(c.get("subject_name_ar") or "")
            or needle in name_key(c.get("subject_name_en") or "")
        ),
        None,
    )


def _label(record: dict, *keys: str) -> str:
    """First non-empty value among `keys`, Arabic-first at every call site."""
    for key in keys:
        value = (record or {}).get(key)
        if value:
            return str(value)
    return ""


def _render_context(student_label: str, data: dict) -> dict:
    """Flatten the payload into what the template renders.

    Labels and the two summary flags are resolved here rather than in Jinja for the
    reason stated in the template header: deciding them is this module's job, and a
    filter chain over rows that may be missing a key is one optional field away from
    raising mid-turn under StrictUndefined.
    """
    courses = data.get("courses") or []
    course = data.get("course") or {}

    return {
        "student_label": student_label,
        "term_label": _label(data.get("term") or {}, "name_ar", "name_en", "term_id"),
        "academic_year_label": str((data.get("term") or {}).get("academic_year") or ""),
        "courses": courses,
        # Precomputed so the template states each caveat only when it is true, and
        # never pays for the wording when it is not.
        "any_in_progress": any(not c.get("is_complete") for c in courses),
        "any_excused": any(c.get("excused_count") for c in courses),
        "subject_label": _label(course, "subject_name_ar", "subject_name_en"),
        "subject_percentage": course.get("computed_percentage"),
        "subject_letter": course.get("letter_grade") or "",
        "assignments": data.get("assignments") or [],
        "data": data,
    }


# --- the same tables, as data ---------------------------------------------------------
#
# A table outcome reaches the reader twice: as the markdown `records_block.j2` renders,
# which is what gets stored and what any client shows, and as the structure below, which
# a client that draws tables uses instead — see `AnswerBlock` in backend/schemas/chat.py.
# Both are built from the one context the tool produced, so they cannot disagree about a
# figure; each builder returns None exactly when the markdown block renders empty.


def _timetable_block(context: dict) -> dict | None:
    """The week, for a client that draws it rather than printing it.

    The same days the markdown block shows, and only those: a day with no lesson is left
    out of both, because drawn empty it reads to a parent as a day off, and the school has
    said it is not one. None when there is no week at all — `no_class` and `no_timetable`
    arrive through this same outcome, and an empty grid is not a table.
    """
    days = [
        {
            "day": str(day.get("name") or ""),
            "label": str(day.get("shows_as") or day.get("name") or ""),
            "slots": [
                {
                    "period": int(slot.get("period_number") or 0),
                    "subject": "" if slot.get("is_free") else str(slot.get("subject") or ""),
                    "is_free": bool(slot.get("is_free")),
                }
                for slot in day.get("slots") or []
            ],
        }
        for day in context.get("days") or []
        if day.get("slots")
    ]
    if not days:
        return None
    return {
        "class_label": str(context.get("class_label") or ""),
        "term_label": str(context.get("term_label") or ""),
        "periods": [
            {
                "number": period["number"],
                "label": period["label"],
                "starts_at": period["shows_from"],
                "ends_at": period["shows_to"],
                "is_teaching": period["is_teaching"],
            }
            for period in context.get("periods") or []
        ],
        "days": days,
    }


def _grades_block(context: dict) -> dict | None:
    """The term's marks, for a client that draws them.

    The rows the markdown block prints, carrying the same two facts it states: a blank
    grade stays None rather than becoming a zero, and a mark still in progress says so.
    """
    courses = [
        {
            "subject": _label(course, "subject_name_ar", "subject_name_en"),
            "percentage": course.get("computed_percentage"),
            "letter": str(course.get("letter_grade") or ""),
            "missing_count": int(course.get("missing_count") or 0),
            "in_progress": not course.get("is_complete"),
        }
        for course in context.get("courses") or []
    ]
    if not courses:
        return None
    return {"term_label": str(context.get("term_label") or ""), "courses": courses}


#: Outcomes whose payload is a TABLE, and which therefore go to the reader as a rendered
#: block instead of being retyped by the model — each with the builder for its data. One
#: map rather than a set and a dispatch, so a table outcome cannot be presented without
#: saying what its data is. See `ChatRequestContext.note_answer_block`.
_PRESENTED = {
    "timetable": _timetable_block,
    "grades": _grades_block,
}


def _block_data(outcome: str, context: dict) -> dict | None:
    """The data for one presented outcome, or None — never an exception.

    Guarded separately from the markdown render: a structure that fails to build costs
    the reader the drawn table and nothing else, because the text block still goes out
    and every client can show that.
    """
    try:
        return _PRESENTED[outcome](context)
    except Exception:
        logger.warning("could not build the %s block's data", outcome, exc_info=True)
        return None
