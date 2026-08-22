"""The get_student_records tool — a thin relay to the records facade.

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

from backend.chat.child_roster import ChildOption, forget, load_roster
from backend.chat.request_context import ChatRequestContext
from backend.prompts import render as render_prompt

logger = logging.getLogger(__name__)

BASE_URL = os.getenv("RECORDS_BASE_URL", "http://localhost:8100").rstrip("/")
API_KEY = os.getenv("RECORDS_API_KEY", "")
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
    students: list[ChildOption], student_name: str, *, remembered: str = ""
) -> ChildOption | None:
    """Pick the child the parent meant.

    Substring match across both scripts, because a parent writing in Arabic and a record
    stored with a Latin transliteration are the normal case in this school, not the
    exception. Returns None when the answer is not unambiguous — the caller then asks
    rather than guessing, since guessing means showing one child's grades while naming
    another.

    Three routes to a child, in order of how firmly the parent stated it:

    1. **A name they just used.** Always wins, so "and how is Omar?" moves the
       conversation on even when the previous question was about his sister.
    2. **An only child.** Nothing to disambiguate, so they are never asked at all.
    3. **The child already under discussion.** A parent who answered "Layla" once is not
       asked again on their next question.

    A name matching nobody falls through to `None` rather than quietly keeping the
    remembered child: the parent named somebody, and answering about a different child
    while they watch is worse than one more question.
    """
    if not students:
        return None
    if not student_name:
        if len(students) == 1:
            return students[0]
        if remembered:
            found = next((s for s in students if s.student_id == remembered), None)
            if found is not None:
                return found
        return None

    needle = student_name.strip().casefold()
    matches = [
        s
        for s in students
        if needle in s.label.casefold() or needle == s.student_id.casefold()
    ]
    return matches[0] if len(matches) == 1 else None


def make_get_student_records(ctx: ChatRequestContext):
    @tool("get_student_records")
    def get_student_records(
        record_type: str = "grades", student_name: str = "", subject: str = ""
    ) -> str:
        """Look up the signed-in parent's own child's school records.

        Use this for anything about a specific student's academic record: grades,
        marks, results, how they are doing in a subject, absences, attendance, or a
        report card. Do not use it for school policies, fees or general information —
        those come from the knowledge base.

        record_type: "grades" for all subjects, "subject" for one subject's
            assignment-by-assignment breakdown, or "attendance".
        student_name: the child's name, in Arabic or English. Leave empty if the
            parent has only one child or has not said which.
        subject: required when record_type is "subject" — the subject name as the
            parent said it.

        You never supply the parent's identity; it is taken from their signed-in
        session. If the tool asks you to clarify which child, put that question to the
        parent rather than choosing one.
        """
        if not ctx.acquire_records_tool_slot():
            return render_prompt("tools/records_result.j2", outcome="call_limit")

        # No verified guardian on this session. Staff, a test, or a signed-out user.
        if not ctx.guardian_token or not ctx.guardian_id:
            return render_prompt("tools/records_result.j2", outcome="not_a_parent")

        def _refused(outcome: str) -> str:
            """A refusal is evidence the cached roster is stale; an outage is not.

            The pin is dropped only here too, and deliberately not on `unavailable`: a
            hint the reader re-checks anyway is not worth discarding over a timeout, and
            doing so would re-ask the parent for a reason they could never see.
            """
            if outcome == "not_authorized":
                forget(ctx)
                ctx.forget_child()
            return render_prompt("tools/records_result.j2", outcome=outcome)

        base = f"/v1/guardians/{ctx.guardian_id}"

        # One cached read per conversation, shared with anything else in the turn that
        # needs to know who this parent's children are. This was a fresh HTTP round trip
        # on every one of the tool's four permitted calls.
        roster_outcome, students = load_roster(ctx)
        # The roster's outcome names are the template's outcome names, so a refusal or
        # an outage relays unchanged and keeps its own careful wording.
        if roster_outcome in ("unavailable", "not_authorized"):
            return _refused(roster_outcome)
        if not students:
            return render_prompt("tools/records_result.j2", outcome="no_students")

        student = _match_student(
            students, student_name, remembered=ctx.remembered_child
        )
        if student is None:
            # Either several children and no name, or a name matching more than one.
            # Both are questions for the parent, never a coin flip.
            return render_prompt(
                "tools/records_result.j2",
                outcome="which_student",
                options=[s.label for s in students],
            )

        student_id = student.student_id
        # Pinned for the rest of the conversation, so a parent who answered "Layla" once
        # is not asked again. Recorded only after a child has actually been resolved, so a
        # turn that failed to identify one leaves nothing wrong pinned behind it.
        #
        # The label rides along because the pin is now durable: a later turn that wants to
        # say which child it is answering about would otherwise have only an opaque
        # student number to show a parent.
        ctx.remember_child(student_id, label=student.label, gender=student.gender)
        kind = (record_type or "grades").strip().lower()

        if kind == "attendance":
            outcome, data = _get(f"{base}/students/{student_id}/attendance", ctx)
            template_outcome = "attendance"
        elif kind == "subject":
            grades_outcome, grades = _get(f"{base}/students/{student_id}/grades", ctx)
            if grades_outcome != "ok":
                return _refused(grades_outcome)

            needle = (subject or "").strip().casefold()
            course = next(
                (
                    c
                    for c in grades.get("courses") or []
                    if needle
                    and (
                        needle in (c.get("subject_name_ar") or "").casefold()
                        or needle in (c.get("subject_name_en") or "").casefold()
                    )
                ),
                None,
            )
            if course is None:
                return render_prompt(
                    "tools/records_result.j2",
                    outcome="which_subject",
                    student=student,
                    options=[
                        c.get("subject_name_ar") or c.get("subject_name_en")
                        for c in grades.get("courses") or []
                    ],
                )

            outcome, data = _get(
                f"{base}/students/{student_id}/grades/{course.get('course_id')}", ctx
            )
            template_outcome = "subject"
        else:
            outcome, data = _get(f"{base}/students/{student_id}/grades", ctx)
            template_outcome = "grades"

        if outcome != "ok":
            return _refused(outcome)

        return render_prompt(
            "tools/records_result.j2",
            outcome=template_outcome,
            **_render_context(student.label, data),
        )

    return get_student_records


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
