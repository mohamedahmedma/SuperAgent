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
from backend.tools.records import make_get_student_records

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
    tool = make_get_student_records(_ctx())
    assert set(tool.args.keys()) == {"record_type", "student_name", "subject"}


def test_no_parent_session_refuses_without_asking_for_identifiers(monkeypatch):
    """A signed-out user must not be invited to type a student ID instead."""
    tool = make_get_student_records(_ctx(guardian_id="", token=""))
    result = tool.invoke({"record_type": "grades"})

    assert "NOT_A_PARENT_SESSION" in result
    assert "signing in" in result


def test_unreachable_facade_forbids_inventing_a_figure(monkeypatch):
    def boom(*args, **kwargs):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "get", boom)
    result = make_get_student_records(_ctx()).invoke({"record_type": "grades"})

    assert "RECORDS_UNAVAILABLE" in result
    assert "Do NOT state, estimate or infer" in result


def test_server_error_is_unavailable_not_no_records(monkeypatch):
    """A 500 must never render as "your child has no grades"."""
    monkeypatch.setattr(requests, "get", _route({"/students": _Response(503)}))
    result = make_get_student_records(_ctx()).invoke({"record_type": "grades"})

    assert "RECORDS_UNAVAILABLE" in result
    assert "NO_RECORDS" not in result


def test_expired_identity_is_not_reported_as_missing_records(monkeypatch):
    monkeypatch.setattr(requests, "get", _route({"/students": _Response(401)}))
    result = make_get_student_records(_ctx()).invoke({"record_type": "grades"})

    assert "NOT_AUTHORIZED" in result
    assert "sign in again" in result


def test_no_linked_students_does_not_name_anyone(monkeypatch):
    monkeypatch.setattr(
        requests, "get", _route({"/students": _Response(200, {"guardian_id": "G-1", "students": []})})
    )
    result = make_get_student_records(_ctx()).invoke({"record_type": "grades"})

    assert "NO_STUDENTS_LINKED" in result


def test_two_children_and_no_name_asks_which(monkeypatch):
    """Guessing here means showing one child's grades while naming another."""
    monkeypatch.setattr(requests, "get", _route({"/students": TWO_CHILDREN}))
    result = make_get_student_records(_ctx()).invoke({"record_type": "grades"})

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
    result = make_get_student_records(_ctx()).invoke(
        {"record_type": "grades", "student_name": "ليلى"}
    )

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
    result = make_get_student_records(_ctx()).invoke({"record_type": "grades"})

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
    result = make_get_student_records(_ctx()).invoke({"record_type": "grades"})

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
    result = make_get_student_records(_ctx()).invoke({"record_type": "grades"})

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
    result = make_get_student_records(_ctx()).invoke({"record_type": "attendance"})

    assert "ATTENDANCE" in result
    assert "rather than adding them together" in result


def test_the_turn_budget_stops_a_loop(monkeypatch):
    monkeypatch.setenv("RECORDS_MAX_CALLS_PER_TURN", "2")
    monkeypatch.setattr(requests, "get", _route({"/students": ONE_CHILD}))

    ctx = _ctx()
    tool = make_get_student_records(ctx)
    for _ in range(2):
        tool.invoke({"record_type": "grades"})

    assert "TOOL_CALL_LIMIT_REACHED" in tool.invoke({"record_type": "grades"})


def test_session_id_is_sent_so_the_facade_can_correlate_its_audit(monkeypatch):
    seen = {}

    def capture(url, headers=None, params=None, timeout=None):
        seen.update(headers or {})
        return ONE_CHILD if url.endswith("/students") else _Response(404)

    monkeypatch.setattr(requests, "get", capture)
    make_get_student_records(_ctx()).invoke({"record_type": "grades"})

    assert seen.get("X-Request-Id") == "turn-1"
    assert seen.get("Authorization") == f"Bearer {PARENT_TOKEN}"


@pytest.mark.parametrize("status", [500, 502, 504])
def test_all_server_errors_collapse_to_unavailable(monkeypatch, status):
    monkeypatch.setattr(requests, "get", _route({"/students": _Response(status)}))
    result = make_get_student_records(_ctx()).invoke({"record_type": "grades"})
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
    result = make_get_student_records(ctx).invoke(
        {"record_type": "grades", "student_name": "عمر"}
    )

    assert "/students/S-1/grades" in read["url"]
    assert "STUDENT_GRADES" in result


def test_a_planner_child_no_longer_on_the_roster_falls_back_to_resolving(monkeypatch):
    """A child withdrawn mid-conversation, or two reads either side of a change.

    Answering about nobody would be worse than asking, so the ordinary resolver runs.
    """
    monkeypatch.setattr(requests, "get", _route({"/students": TWO_CHILDREN}))
    ctx = _ctx()
    ctx.note_turn_plan([], [], child_id="S-99", child_label="مين ده")
    result = make_get_student_records(ctx).invoke({"record_type": "grades"})

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
    make_get_student_records(ctx).invoke({"record_type": "grades"})

    assert ctx.tool_outcomes == [("get_student_records", "grades")]


def test_a_failure_reports_its_own_outcome_too(monkeypatch):
    monkeypatch.setattr(requests, "get", _route({"/students": _Response(503)}))
    ctx = _ctx()
    make_get_student_records(ctx).invoke({"record_type": "grades"})

    assert ctx.tool_outcomes == [("get_student_records", "unavailable")]
