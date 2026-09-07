"""The get_student_records tool.

These assert the two things the agent side is responsible for: that the model can
never name a guardian, and that every failure produces a refusal rather than an
invented figure. The authorisation itself is tested in `records/tests` — this file
covers the relay and the wording it hands back to the model.
"""
import pytest
import requests

from backend.chat.caller_identity import CallerIdentity
from backend.chat.request_context import ChatRequestContext
from backend.tools.records import (
    make_get_student_attendance,
    make_get_student_class,
    make_get_student_grades,
    make_get_student_subjects,
    make_get_student_teachers,
    make_get_student_timetable,
    make_get_subject_grades,
    make_get_subject_teacher,
)

PARENT_TOKEN = "signed.identity.token"


@pytest.fixture(autouse=True)
def no_roster_cache(monkeypatch):
    """Every case in this file drives the facade through a canned `requests.get`.

    The roster now sits behind a short cache, and a cache shared with whatever Redis
    happens to be running on the machine would carry one case's children into the next
    — which is exactly how this suite started reporting "which child?" for a test about
    a 500. Turning the TTL off is the supported way to say "read it fresh".
    """
    monkeypatch.setenv("CHILD_ROSTER_TTL_SECONDS", "0")


class _Response:
    def __init__(self, status_code: int, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload


def _ctx(guardian_id: str = "G-1", token: str = PARENT_TOKEN) -> ChatRequestContext:
    return ChatRequestContext(
        user_id="user-1",
        session_id="turn-1",
        caller=CallerIdentity(
            user_id="user-1", guardian_id=guardian_id, guardian_token=token
        ),
    )


def _route(responses: dict):
    """Serve canned payloads by URL suffix."""

    def fake_get(url, headers=None, params=None, timeout=None):
        for suffix, response in responses.items():
            if url.endswith(suffix):
                return response
        return _Response(404)

    return fake_get


ONE_CHILD = _Response(
    200,
    {"guardian_id": "G-1", "students": [{"student_id": "S-1", "full_name_ar": "ليلى", "full_name_en": "Layla"}]},
)
TWO_CHILDREN = _Response(
    200,
    {
        "guardian_id": "G-1",
        "students": [
            {"student_id": "S-1", "full_name_ar": "ليلى", "full_name_en": "Layla"},
            {"student_id": "S-2", "full_name_ar": "عمر", "full_name_en": "Omar"},
        ],
    },
)


def test_the_model_cannot_supply_a_guardian_id():
    """Structural, not behavioural: there is no argument to inject into.

    The guardian comes from the session. If a future edit adds it as a parameter,
    this test fails and the whole prompt-injection defence is gone.
    """
    builders = (make_get_student_grades, make_get_subject_grades, make_get_student_attendance)
    for build in builders:
        args = set(build(_ctx()).args.keys())
        # Every record tool takes the child and nothing else that could name a person;
        # `subject` names a school subject, not anybody.
        assert args <= {"student_name", "subject"}, args
        assert not args & {"guardian_id", "guardian", "student_id", "token"}


def test_no_parent_session_refuses_without_asking_for_identifiers(monkeypatch):
    """A signed-out user must not be invited to type a student ID instead."""
    tool = make_get_student_grades(_ctx(guardian_id="", token=""))
    result = tool.invoke({})

    assert "NOT_A_PARENT_SESSION" in result
    assert "signing in" in result


def test_unreachable_facade_forbids_inventing_a_figure(monkeypatch):
    def boom(*args, **kwargs):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "get", boom)
    result = make_get_student_grades(_ctx()).invoke({})

    assert "RECORDS_UNAVAILABLE" in result
    assert "Do NOT state, estimate or infer" in result


def test_server_error_is_unavailable_not_no_records(monkeypatch):
    """A 500 must never render as "your child has no grades"."""
    monkeypatch.setattr(requests, "get", _route({"/students": _Response(503)}))
    result = make_get_student_grades(_ctx()).invoke({})

    assert "RECORDS_UNAVAILABLE" in result
    assert "NO_RECORDS" not in result


def test_expired_identity_is_not_reported_as_missing_records(monkeypatch):
    monkeypatch.setattr(requests, "get", _route({"/students": _Response(401)}))
    result = make_get_student_grades(_ctx()).invoke({})

    assert "NOT_AUTHORIZED" in result
    assert "sign in again" in result


def test_no_linked_students_does_not_name_anyone(monkeypatch):
    monkeypatch.setattr(
        requests, "get", _route({"/students": _Response(200, {"guardian_id": "G-1", "students": []})})
    )
    result = make_get_student_grades(_ctx()).invoke({})

    assert "NO_STUDENTS_LINKED" in result


def test_two_children_and_no_name_asks_which(monkeypatch):
    """Guessing here means showing one child's grades while naming another."""
    monkeypatch.setattr(requests, "get", _route({"/students": TWO_CHILDREN}))
    result = make_get_student_grades(_ctx()).invoke({})

    assert "NEEDS_STUDENT_CHOICE" in result
    assert "ليلى" in result and "عمر" in result


def test_a_named_child_is_matched_in_arabic(monkeypatch):
    monkeypatch.setattr(
        requests,
        "get",
        _route(
            {
                "/students": TWO_CHILDREN,
                "/grades": _Response(
                    200,
                    {
                        "term": {"term_id": "2026-T1", "name_ar": "الفصل الأول"},
                        "courses": [
                            {
                                "course_id": "9001",
                                "subject_name_ar": "الرياضيات",
                                "subject_name_en": "Mathematics",
                                "computed_percentage": 90.0,
                                "letter_grade": "A",
                                "excused_count": 0,
                                "missing_count": 0,
                                "is_complete": True,
                            }
                        ],
                    },
                ),
            }
        ),
    )
    result = make_get_student_grades(_ctx()).invoke({"student_name": "ليلى"})

    assert "STUDENT_GRADES" in result
    assert "90.0%" in result


def test_grades_forbid_recalculation_and_explain_excused(monkeypatch):
    """The model must not average subjects or call excused work a bad mark."""
    monkeypatch.setattr(
        requests,
        "get",
        _route(
            {
                "/students": ONE_CHILD,
                "/grades": _Response(
                    200,
                    {
                        "term": {"term_id": "2026-T1"},
                        "courses": [
                            {
                                "course_id": "9001",
                                "subject_name_en": "Mathematics",
                                "computed_percentage": 90.0,
                                "letter_grade": "A",
                                "excused_count": 2,
                                "missing_count": 0,
                                "is_complete": True,
                            }
                        ],
                    },
                ),
            }
        ),
    )
    result = make_get_student_grades(_ctx()).invoke({})

    assert "Do not recalculate" in result
    assert "not counted against the student and is not a zero" in result


def test_in_progress_subject_is_flagged_as_not_final(monkeypatch):
    monkeypatch.setattr(
        requests,
        "get",
        _route(
            {
                "/students": ONE_CHILD,
                "/grades": _Response(
                    200,
                    {
                        "term": {"term_id": "2026-T1"},
                        "courses": [
                            {
                                "course_id": "9001",
                                "subject_name_en": "Mathematics",
                                "computed_percentage": 72.0,
                                "letter_grade": "C",
                                "excused_count": 0,
                                "missing_count": 0,
                                "is_complete": False,
                            }
                        ],
                    },
                ),
            }
        ),
    )
    result = make_get_student_grades(_ctx()).invoke({})

    assert "still in progress" in result
    assert "not a final grade" in result


def test_empty_term_is_not_reported_as_failing(monkeypatch):
    monkeypatch.setattr(
        requests,
        "get",
        _route(
            {
                "/students": ONE_CHILD,
                "/grades": _Response(200, {"term": {"term_id": "2026-T1"}, "courses": []}),
            }
        ),
    )
    result = make_get_student_grades(_ctx()).invoke({})

    assert "nothing is recorded yet" in result
    assert "not that the student is failing" in result


def test_attendance_separates_excused_from_unexcused(monkeypatch):
    monkeypatch.setattr(
        requests,
        "get",
        _route(
            {
                "/students": ONE_CHILD,
                "/attendance": _Response(
                    200,
                    {
                        "term": {"term_id": "2026-T1"},
                        "present_count": 40,
                        "absent_count": 2,
                        "late_count": 1,
                        "excused_count": 3,
                        "total_sessions": 46,
                        "attendance_rate": 95.65,
                    },
                ),
            }
        ),
    )
    result = make_get_student_attendance(_ctx()).invoke({})

    assert "ATTENDANCE" in result
    assert "rather than adding them together" in result


# --- the timetable ---------------------------------------------------------------
#
# The fourth record tool, and the only one whose subject is a room rather than the child.
# Two things are worth testing here and nowhere else in this suite: that the wording keeps
# the three answers apart — a class with no published week must never be reported as a
# child with no lessons — and that the request names no class, because the room is the
# facade's to resolve from a placement that changes mid-year.


def _week(status: str = "ok", **overrides) -> _Response:
    payload = {
        "student": {"student_id": "S-1"},
        "term": {"term_id": "2026-T1", "name_ar": "الفصل الأول"},
        "status": status,
        "class_code": "3A",
        "class_name_ar": "الثالث أ",
        "class_name_en": "Year 3 A",
        # Saturday-first deliberately: a week this order proves nothing was re-sorted
        # into a Monday-first one on the way through.
        "days": ["saturday", "sunday", "monday"],
        "periods": [
            {
                "period_number": 1,
                "name_ar": "حصة ١",
                "name_en": "Period 1",
                "starts_at": "08:00",
                "ends_at": "08:45",
                "is_teaching": True,
            },
            {
                "period_number": 2,
                "name_ar": "فسحة",
                "name_en": "Break",
                "starts_at": "08:45",
                "ends_at": "09:05",
                "is_teaching": False,
            },
        ],
        "lessons": [
            {
                "day_of_week": "saturday",
                "period_number": 1,
                "subject_code": "MATH",
                "subject_name_ar": "الرياضيات",
                "subject_name_en": "Mathematics",
            },
            {
                "day_of_week": "sunday",
                "period_number": 1,
                "subject_code": "",
                "subject_name_ar": "",
                "subject_name_en": "",
            },
        ],
        "teaching_slots": 3,
    }
    payload.update(overrides)
    return _Response(200, payload)


def test_the_week_is_reported_day_by_day_in_the_school_s_own_order(monkeypatch):
    """A parent asks "what does she have on Sunday", so the model is handed days.

    The school's order is preserved and the model is told not to re-sort it: the week
    begins on Saturday here, and a model that helpfully rewrote it to start on Monday
    would be describing a week the school does not run.
    """
    monkeypatch.setattr(
        requests, "get", _route({"/students": ONE_CHILD, "/timetable": _week()})
    )
    result = make_get_student_timetable(_ctx()).invoke({})

    assert "TIMETABLE" in result
    assert "الثالث أ" in result
    assert "الرياضيات" in result
    # Saturday before Sunday, as the school stated it.
    assert result.index("saturday") < result.index("sunday")
    assert "do not reorder them" in result
    # A day the school opens with nothing planned is not a day off.
    assert "monday: nothing timetabled" in result
    assert "does not mean the child is off school" in result


def test_a_break_is_not_reported_as_a_lesson(monkeypatch):
    """The break is in the grid and never in the lessons, so it arrives separately.

    Folded in, it becomes a subject the child does not take; dropped entirely, the model
    cannot answer "when is her break", which is a question parents actually ask.
    """
    monkeypatch.setattr(
        requests, "get", _route({"/students": ONE_CHILD, "/timetable": _week()})
    )
    result = make_get_student_timetable(_ctx()).invoke({})

    assert "Break periods" in result
    assert "فسحة" in result
    # A stated free period stays a free period rather than becoming a nameless subject.
    assert "free" in result


def test_no_class_this_term_is_not_reported_as_having_no_lessons(monkeypatch):
    """The first of the two empty answers, and a fact about the child.

    "The school has not placed her in a class yet" and "she has no lessons" are different
    sentences, and only one of them is true here.
    """
    monkeypatch.setattr(
        requests,
        "get",
        _route(
            {
                "/students": ONE_CHILD,
                "/timetable": _week(status="no_class", class_code="", lessons=[], days=[]),
            }
        ),
    )
    result = make_get_student_timetable(_ctx()).invoke({})

    assert "NO_CLASS_THIS_TERM" in result
    assert "not placed their child in a class" in result
    assert "Do NOT say the child has no lessons" in result


def test_an_unpublished_timetable_names_the_class_and_blames_nobody(monkeypatch):
    """The second empty answer, and a fact about the school rather than the child.

    She has a room; its week is not out yet. Reported as "she has no lessons", this is
    the false sentence the whole three-way status exists to prevent.
    """
    monkeypatch.setattr(
        requests,
        "get",
        _route(
            {
                "/students": ONE_CHILD,
                "/timetable": _week(status="no_timetable", lessons=[]),
            }
        ),
    )
    result = make_get_student_timetable(_ctx()).invoke({})

    assert "TIMETABLE_NOT_PUBLISHED" in result
    assert "الثالث أ" in result
    assert "do not invent a schedule" in result
    assert "Do NOT say the child has no lessons" in result


def test_the_request_never_names_a_class(monkeypatch):
    """The property the design rests on: the room is resolved behind the facade.

    A class code sent from here would be one this process had to hold — and a held class
    code is one that keeps naming 3A after a child has moved to 3B.
    """
    seen: list[tuple[str, dict]] = []

    def fake_get(url, headers=None, params=None, timeout=None):
        seen.append((url, params or {}))
        return ONE_CHILD if url.endswith("/students") else _week()

    monkeypatch.setattr(requests, "get", fake_get)
    make_get_student_timetable(_ctx()).invoke({})

    url, params = seen[-1]
    assert url.endswith("/students/S-1/timetable")
    assert "class" not in url
    assert params == {}


def test_an_unreachable_facade_forbids_inventing_a_week(monkeypatch):
    """Shares the `unavailable` branch with the other record tools, deliberately.

    One failure vocabulary across the family: a second one is what a caller eventually
    handles wrong.
    """
    monkeypatch.setattr(
        requests, "get", _route({"/students": ONE_CHILD, "/timetable": _Response(503)})
    )
    ctx = _ctx()
    result = make_get_student_timetable(ctx).invoke({})

    assert "RECORDS_UNAVAILABLE" in result
    assert ctx.tool_outcomes == [("get_student_timetable", "unavailable")]


def test_the_timetable_tool_reports_its_own_outcome(monkeypatch):
    """So the turn can tell that a record WAS retrieved — see `note_tool_outcome`."""
    monkeypatch.setattr(
        requests, "get", _route({"/students": ONE_CHILD, "/timetable": _week()})
    )
    ctx = _ctx()
    make_get_student_timetable(ctx).invoke({})

    assert ctx.tool_outcomes == [("get_student_timetable", "timetable")]


def test_the_timetable_cannot_be_read_without_a_parent_session(monkeypatch):
    """Same gate as every other record tool: no verified guardian, no read."""
    monkeypatch.setattr(requests, "get", _route({}))
    result = make_get_student_timetable(_ctx(token="")).invoke({})

    assert "NOT_A_PARENT_SESSION" in result


def test_two_children_and_no_name_asks_which_before_showing_a_week(monkeypatch):
    """A timetable names a child as surely as a mark does, so the same question comes first."""
    monkeypatch.setattr(
        requests, "get", _route({"/students": TWO_CHILDREN, "/timetable": _week()})
    )
    result = make_get_student_timetable(_ctx()).invoke({})

    assert "NEEDS_STUDENT_CHOICE" in result


# --- the room she sits in: class, subjects, teachers -------------------------------
#
# Four tools over three facade endpoints. What is asserted here and nowhere else is the
# wording that keeps three different empties apart, and the two staff rules: every teacher
# of a subject is named, and no contact detail is ever offered.

ROOM = {
    "student": {"student_id": "S-1"},
    "term": {"term_id": "2026-T1"},
    "status": "ok",
    # Code and name differ on purpose: a tool that read the code by mistake must fail,
    # not pass by coincidence.
    "class_code": "3A",
    "class_name_ar": "الثالث ١",
    "class_name_en": "Primary 3 Class 1",
    "year_level_name_ar": "الصف الثالث",
    "subjects": [
        {"code": "MATH", "name_ar": "الرياضيات", "name_en": "Mathematics"},
        {"code": "SCI", "name_ar": "العلوم", "name_en": "Science"},
    ],
    # Both name scripts, as the facade actually sends them — the subject matcher reads
    # either, so a fixture carrying only one would not exercise an English question.
    "teachers": [
        {
            "full_name_ar": "أ. سامي",
            "full_name_en": "Mr Sami",
            "subject_code": "MATH",
            "subject_name_ar": "الرياضيات",
            "subject_name_en": "Mathematics",
        },
        {
            "full_name_ar": "أ. هدى",
            "full_name_en": "Ms Huda",
            "subject_code": "SCI",
            "subject_name_ar": "العلوم",
            "subject_name_en": "Science",
        },
        {
            "full_name_ar": "أ. منى",
            "full_name_en": "Ms Mona",
            "subject_code": "SCI",
            "subject_name_ar": "العلوم",
            "subject_name_en": "Science",
        },
    ],
}


def _room(**overrides) -> _Response:
    payload = dict(ROOM)
    payload.update(overrides)
    return _Response(200, payload)


def _rooms(**overrides):
    """Serve the same room from all three classroom endpoints."""
    response = _room(**overrides)
    return _route(
        {
            "/students": ONE_CHILD,
            "/class": response,
            "/subjects": response,
            "/teachers": response,
        }
    )


def test_the_class_answer_is_the_name_and_not_the_code(monkeypatch):
    """What the parent asked for. `3A` is an internal key they have never seen.

    The model is also told not to reformat it: "Primary 3 Class 1" and "3/1" are whatever
    the registrar typed, and a helpful renumbering would be a different class.
    """
    monkeypatch.setattr(requests, "get", _rooms())
    result = make_get_student_class(_ctx()).invoke({})

    assert "CLASS for" in result
    assert "الثالث ١" in result
    assert "الصف الثالث" in result
    assert "3A" not in result
    assert "Do not translate it, renumber it" in result


def test_a_child_with_no_placement_is_not_given_a_guessed_class(monkeypatch):
    monkeypatch.setattr(
        requests,
        "get",
        _rooms(status="no_class", class_code="", class_name_ar="", class_name_en=""),
    )
    result = make_get_student_class(_ctx()).invoke({})

    assert "NO_CLASS_THIS_TERM" in result
    assert "not placed their child in a class" in result


def test_the_subject_list_is_reported_in_the_school_s_order(monkeypatch):
    monkeypatch.setattr(requests, "get", _rooms())
    result = make_get_student_subjects(_ctx()).invoke({})

    assert "SUBJECTS for" in result
    assert result.index("الرياضيات") < result.index("العلوم")
    assert "Do not add subjects the parent mentioned" in result


def test_an_uncurated_subject_board_is_not_a_child_who_studies_nothing(monkeypatch):
    """The second of the three empties, and a fact about the school's admin.

    Reported as "she studies no subjects" this is false and alarming; the branch names the
    class and says the list is not published.
    """
    monkeypatch.setattr(requests, "get", _rooms(subjects=[]))
    result = make_get_student_subjects(_ctx()).invoke({})

    assert "SUBJECTS_NOT_PUBLISHED" in result
    assert "الثالث ١" in result
    assert "Do NOT say the child studies no subjects" in result


def test_teachers_are_grouped_by_subject_and_none_is_dropped(monkeypatch):
    """Grouping is what makes the co-teacher case legible.

    The payload is one row per (teacher, subject). Rendered flat, science reads as two
    separate facts and a model may summarise it to one name — which is the person who
    disappears.
    """
    monkeypatch.setattr(requests, "get", _rooms())
    result = make_get_student_teachers(_ctx()).invoke({})

    assert "TEACHERS for" in result
    assert "الرياضيات: أ. سامي" in result
    assert "العلوم: أ. هدى، أ. منى" in result
    assert "both teach it" in result


def test_a_class_with_no_staffing_recorded_is_not_a_child_with_no_teachers(monkeypatch):
    """The third empty. Same shape of lie as the subjects one, same fix."""
    monkeypatch.setattr(requests, "get", _rooms(teachers=[]))
    result = make_get_student_teachers(_ctx()).invoke({})

    assert "TEACHERS_NOT_ASSIGNED" in result
    assert "Do NOT say the child has no teachers" in result


def test_a_teacher_with_no_name_on_file_is_not_rendered_as_a_blank(monkeypatch):
    """The school can create a teacher with neither name spelling filled in.

    Rendered straight through that becomes "maths: " with nothing after it, which a parent
    reads as the assistant failing rather than as the school's record being incomplete.
    Dropping the row leaves the honest "not recorded yet" answer instead.
    """
    monkeypatch.setattr(
        requests,
        "get",
        _rooms(
            teachers=[
                {
                    "full_name_ar": "",
                    "full_name_en": "",
                    "subject_code": "MATH",
                    "subject_name_ar": "الرياضيات",
                }
            ]
        ),
    )
    result = make_get_student_teachers(_ctx()).invoke({})

    assert "TEACHERS_NOT_ASSIGNED" in result
    assert "الرياضيات:" not in result


def test_no_teacher_contact_details_are_ever_offered(monkeypatch):
    """The facade sends none, and the model is told where to send the parent instead.

    Without that instruction a helpful model invents an email from the school's domain,
    which is worse than saying it does not have one.
    """
    monkeypatch.setattr(requests, "get", _rooms())
    listed = make_get_student_teachers(_ctx()).invoke({})
    one = make_get_subject_teacher(_ctx()).invoke({"subject": "العلوم"})

    for result in (listed, one):
        assert "school office" in result
        assert "phone number" in result or "contact details" in result


def test_a_named_subject_returns_only_that_subject_s_teachers(monkeypatch):
    """The whole point of the narrow tool: one answer, not a list to read through.

    And every teacher of it — science has two, and both are named.
    """
    monkeypatch.setattr(requests, "get", _rooms())
    result = make_get_subject_teacher(_ctx()).invoke({"subject": "العلوم"})

    assert "SUBJECT_TEACHER" in result
    assert "أ. هدى" in result and "أ. منى" in result
    # The maths teacher is not part of this answer.
    assert "أ. سامي" not in result
    assert "all of them teach it" in result


def test_a_subject_named_in_english_matches_the_arabic_board(monkeypatch):
    """A parent writes in either language and the board spells it one fixed way."""
    monkeypatch.setattr(requests, "get", _rooms())
    result = make_get_subject_teacher(_ctx()).invoke({"subject": "Science"})

    assert "SUBJECT_TEACHER" in result
    assert "أ. هدى" in result


def test_a_subject_nobody_teaches_her_asks_rather_than_guessing(monkeypatch):
    """Answered with the subjects she actually has a teacher for, never a nearest guess."""
    monkeypatch.setattr(requests, "get", _rooms())
    result = make_get_subject_teacher(_ctx()).invoke({"subject": "الموسيقى"})

    assert "NEEDS_SUBJECT_CHOICE" in result
    assert "الرياضيات" in result and "العلوم" in result


def test_the_classroom_tools_never_name_a_class_in_the_request(monkeypatch):
    """The room is resolved behind the facade, from the child's placement.

    A class code sent from here would be one this process had to hold, and a held class
    code keeps naming 3A after she has moved to 3B.
    """
    seen: list[tuple[str, dict]] = []

    def fake_get(url, headers=None, params=None, timeout=None):
        seen.append((url, params or {}))
        return ONE_CHILD if url.endswith("/students") else _room()

    monkeypatch.setattr(requests, "get", fake_get)
    make_get_student_class(_ctx()).invoke({})
    make_get_student_subjects(_ctx()).invoke({})
    make_get_student_teachers(_ctx()).invoke({})

    asked = [(url, params) for url, params in seen if not url.endswith("/students")]
    assert [url.rsplit("/", 1)[-1] for url, _ in asked] == [
        "class",
        "subjects",
        "teachers",
    ]
    for url, params in asked:
        assert "3A" not in url
        assert params == {}


@pytest.mark.parametrize(
    "builder,args,tool_name,outcome",
    [
        (make_get_student_class, {}, "get_student_class", "class"),
        (make_get_student_subjects, {}, "get_student_subjects", "subjects"),
        (make_get_student_teachers, {}, "get_student_teachers", "teachers"),
        (
            make_get_subject_teacher,
            {"subject": "العلوم"},
            "get_subject_teacher",
            "subject_teacher",
        ),
    ],
)
def test_each_classroom_tool_reports_its_own_outcome(
    monkeypatch, builder, args, tool_name, outcome
):
    """So the turn can tell a record WAS retrieved.

    Every one of these names has to appear in `backend.chat.service.RECORDS_RETRIEVED` too,
    or the check that catches an answer denying the record it just read cannot fire — which
    is exactly how `timetable` was missed.
    """
    monkeypatch.setattr(requests, "get", _rooms())
    ctx = _ctx()
    builder(ctx).invoke(args)

    assert ctx.tool_outcomes == [(tool_name, outcome)]

    from backend.chat.service import RECORDS_RETRIEVED

    assert outcome in RECORDS_RETRIEVED


@pytest.mark.parametrize(
    "builder,args",
    [
        (make_get_student_class, {}),
        (make_get_student_subjects, {}),
        (make_get_student_teachers, {}),
        (make_get_subject_teacher, {"subject": "العلوم"}),
    ],
)
def test_an_unreachable_facade_forbids_inventing_a_room(monkeypatch, builder, args):
    """Shares the `unavailable` branch with every other record tool, deliberately."""
    monkeypatch.setattr(
        requests,
        "get",
        _route(
            {
                "/students": ONE_CHILD,
                "/class": _Response(503),
                "/subjects": _Response(503),
                "/teachers": _Response(503),
            }
        ),
    )
    result = builder(_ctx()).invoke(args)

    assert "RECORDS_UNAVAILABLE" in result


def test_the_turn_budget_stops_a_loop(monkeypatch):
    monkeypatch.setenv("RECORDS_MAX_CALLS_PER_TURN", "2")
    monkeypatch.setattr(requests, "get", _route({"/students": ONE_CHILD}))

    ctx = _ctx()
    tool = make_get_student_grades(ctx)
    for _ in range(2):
        tool.invoke({})

    assert "TOOL_CALL_LIMIT_REACHED" in tool.invoke({})


def test_session_id_is_sent_so_the_facade_can_correlate_its_audit(monkeypatch):
    seen = {}

    def capture(url, headers=None, params=None, timeout=None):
        seen.update(headers or {})
        return ONE_CHILD if url.endswith("/students") else _Response(404)

    monkeypatch.setattr(requests, "get", capture)
    make_get_student_grades(_ctx()).invoke({})

    assert seen.get("X-Request-Id") == "turn-1"
    assert seen.get("Authorization") == f"Bearer {PARENT_TOKEN}"


@pytest.mark.parametrize("status", [500, 502, 504])
def test_all_server_errors_collapse_to_unavailable(monkeypatch, status):
    monkeypatch.setattr(requests, "get", _route({"/students": _Response(status)}))
    result = make_get_student_grades(_ctx()).invoke({})
    assert "RECORDS_UNAVAILABLE" in result


# --- which child the tool reads -------------------------------------------------
#
# The model's transcription of a name it read once is the least reliable link in a chain
# whose other end is the school's own roster. The planner resolved this turn's child
# against that roster before the agent ran; these pin that its answer is the one used.


def _grades_for(percentage: float = 91.0):
    return _Response(
        200,
        {
            "term": {"term_id": "2026-T1", "name_ar": "الفصل الأول"},
            "courses": [
                {
                    "course_id": "9001",
                    "subject_name_ar": "الرياضيات",
                    "subject_name_en": "Mathematics",
                    "computed_percentage": percentage,
                    "letter_grade": "A",
                    "excused_count": 0,
                    "missing_count": 0,
                    "is_complete": True,
                }
            ],
        },
    )


def test_the_planners_child_is_read_and_not_the_name_the_model_typed(monkeypatch):
    """Two children, and a name that matches the WRONG one.

    Under the old rule the argument won and the tool answered about عمر. The planner had
    already settled on ليلى from the roster, which is the answer that has actually been
    checked against a list of real children.
    """
    read = {}

    def fake_get(url, headers=None, params=None, timeout=None):
        if url.endswith("/students"):
            return TWO_CHILDREN
        read["url"] = url
        return _grades_for()

    monkeypatch.setattr(requests, "get", fake_get)
    ctx = _ctx()
    ctx.note_turn_plan([], [], child_id="S-1", child_label="ليلى")
    result = make_get_student_grades(ctx).invoke({"student_name": "عمر"})

    assert "/students/S-1/grades" in read["url"]
    assert "STUDENT_GRADES" in result


def test_a_planner_child_no_longer_on_the_roster_falls_back_to_resolving(monkeypatch):
    """A child withdrawn mid-conversation, or two reads either side of a change.

    Answering about nobody would be worse than asking, so the ordinary resolver runs.
    """
    monkeypatch.setattr(requests, "get", _route({"/students": TWO_CHILDREN}))
    ctx = _ctx()
    ctx.note_turn_plan([], [], child_id="S-99", child_label="مين ده")
    result = make_get_student_grades(ctx).invoke({})

    assert "NEEDS_STUDENT_CHOICE" in result


def test_half_a_planned_child_is_no_planned_child(monkeypatch):
    """An id with no label names a child nothing can call by name, and a label with no
    id names one nothing can read. Either half alone is not a state to handle."""
    ctx = _ctx()
    ctx.note_turn_plan([], [], child_id="S-1")
    assert ctx.planned_child_id == ""


# --- what the turn knows about what happened -------------------------------------


def test_the_tool_reports_which_outcome_it_produced(monkeypatch):
    """A call count says the tool ran. Only the outcome says it found anything, and only
    that can contradict an answer claiming it did not."""
    monkeypatch.setattr(
        requests, "get", _route({"/students": ONE_CHILD, "/grades": _grades_for(87.5)})
    )
    ctx = _ctx()
    make_get_student_grades(ctx).invoke({})

    assert ctx.tool_outcomes == [("get_student_grades", "grades")]


def test_a_failure_reports_its_own_outcome_too(monkeypatch):
    monkeypatch.setattr(requests, "get", _route({"/students": _Response(503)}))
    ctx = _ctx()
    make_get_student_grades(ctx).invoke({})

    assert ctx.tool_outcomes == [("get_student_grades", "unavailable")]
