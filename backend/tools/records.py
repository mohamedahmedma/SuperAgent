"""The student-record tools — thin relays to the records facade.

ONE TOOL PER CAPABILITY, and per endpoint the facade actually exposes: grades for the
term, one subject's breakdown, attendance. That is the whole structure of this file.

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
that has to know about the others. A `get_student_timetable` would slot in beside these
three the day the facade grows the endpoint for it.

The cost is honest and worth stating: three tool schemas reach the model every turn
instead of one, and each keeps its own call budget rather than sharing one ceiling.

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
SUBJECT_TOOL = "get_subject_grades"
ATTENDANCE_TOOL = "get_student_attendance"


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
    def get_student_grades(student_name: str = "") -> str:
        """Read one child's marks for the current term, across every subject.

        Use this for how a child is doing overall, their results, or their report card,
        and for their marks when the question names no single subject. For one subject's
        assignment-by-assignment detail use get_subject_grades; for absences use
        get_student_attendance. Do not use any of them for school policies, fees or
        general information — those come from the knowledge base.
        """
        result = _reporter(ctx, GRADES_TOOL)
        refusal, student = _resolve_student(ctx, student_name, result)
        if refusal:
            return refusal

        outcome, data = _get(f"{_student_path(ctx, student.student_id)}/grades", ctx)
        if outcome != "ok":
            return _refused(ctx, outcome, result)
        return result("grades", **_render_context(student.label, data))

    get_student_grades.description += _STUDENT_NAME_NOTE
    return get_student_grades


def make_get_subject_grades(ctx: ChatRequestContext):
    @tool(SUBJECT_TOOL)
    def get_subject_grades(subject: str, student_name: str = "") -> str:
        """Read one child's assignment-by-assignment breakdown in a single subject.

        Use this when the question names a subject. For every subject at once use
        get_student_grades instead.

        subject: the subject name as the parent said it, in Arabic or English. This one
        IS yours to supply — it comes from the message itself, and a name matching no
        subject is answered with the subjects the child actually takes.
        """
        result = _reporter(ctx, SUBJECT_TOOL)
        refusal, student = _resolve_student(ctx, student_name, result)
        if refusal:
            return refusal

        path = _student_path(ctx, student.student_id)
        # The course list first, because the facade addresses one subject by course id
        # and the parent named it in words.
        grades_outcome, grades = _get(f"{path}/grades", ctx)
        if grades_outcome != "ok":
            return _refused(ctx, grades_outcome, result)

        course = _match_course(grades, subject)
        if course is None:
            return result(
                "which_subject",
                student=student,
                options=[
                    c.get("subject_name_ar") or c.get("subject_name_en")
                    for c in grades.get("courses") or []
                ],
            )

        outcome, data = _get(f"{path}/grades/{course.get('course_id')}", ctx)
        if outcome != "ok":
            return _refused(ctx, outcome, result)
        return result("subject", **_render_context(student.label, data))

    get_subject_grades.description += _STUDENT_NAME_NOTE
    return get_subject_grades


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
