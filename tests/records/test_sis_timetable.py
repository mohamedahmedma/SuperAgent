"""SIS's weekly plan, mapped onto the parent-facing timetable contract.

The mapping is small — SIS already serves a child's week from one route — so almost every
assertion here is about one of two decisions rather than about a reshaping:

**Three answers, two of which are empty and mean different things.** A child with no
placement for the term has no room to ask about; a class nobody has timetabled yet has a
room and no grid. `status` names which, and the tests below are what stop the two collapsing
into "she has no lessons" — the false sentence a parent would otherwise be told in the second
case.

**The class never crosses the wire into this service.** SIS resolves the room from the
placement; this adapter asks about a student. A regression that added a class argument would
be invisible from the data and would put a code that goes stale mid-year into a cache.

Mocked at the adapter's own `_get`, which is where `records/`'s other adapter tests sit — no
network, and no extra dependency to keep current.
"""
from unittest.mock import patch

import pytest

from records.adapters.sis.timetable import SisTimetableAdapter
from records.domain.errors import TimetableUnavailable
from records.domain.timetable import TimetableStatus

#: One SIS `StudentWeekOut`, as the route actually answers it: a four-day school, a break in
#: the middle of the day, one maths lesson and one stated free period.
WEEK = {
    "student_number": "S-1001",
    "term_code": "2026-T1",
    "academic_year_code": "2025-2026",
    "class_code": "3A",
    "class_name_ar": "الثالث أ",
    "class_name_en": "Year 3 A",
    "days": ["saturday", "sunday", "monday", "tuesday"],
    "periods": [
        {
            "period_number": 1,
            "name_ar": "حصة ١",
            "name_en": "Period 1",
            "starts_at": "08:00:00",
            "ends_at": "08:45:00",
            "is_teaching": True,
            "is_timed": True,
        },
        {
            "period_number": 2,
            "name_ar": "فسحة",
            "name_en": "Break",
            "starts_at": None,
            "ends_at": None,
            "is_teaching": False,
            "is_timed": False,
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
            "subject_code": None,
            "subject_name_ar": "",
            "subject_name_en": "",
        },
    ],
    "teaching_slots": 4,
}

NO_CLASS = {
    "student_number": "S-1001",
    "term_code": "2026-T1",
    "academic_year_code": None,
    "class_code": None,
    "class_name_ar": None,
    "class_name_en": None,
    "days": [],
    "periods": [],
    "lessons": [],
    "teaching_slots": 0,
}


@pytest.fixture()
def adapter() -> SisTimetableAdapter:
    return SisTimetableAdapter(
        base_url="http://sis.test", api_key="reader", timeout_seconds=5.0
    )


def test_a_week_arrives_with_its_grid_and_its_names(adapter: SisTimetableAdapter) -> None:
    """The grid travels with the lessons, and the subject arrives named.

    Read apart, the two can disagree and a client draws a lesson in a row that is not
    there. And `MATH` is the school's filing key: the reader at the end of this contract is
    a parent, so the names are what has to survive the mapping.
    """
    with patch.object(SisTimetableAdapter, "_get", return_value=WEEK):
        week = adapter.get_timetable(
            student_ref="S-1001", term="2026-T1", guardian_ref="G-1"
        )

    assert week.status is TimetableStatus.OK
    assert week.class_code == "3A"
    assert week.class_name_ar == "الثالث أ"
    # The school's own week order, never sorted — this one starts on Saturday.
    assert week.days == ("saturday", "sunday", "monday", "tuesday")
    assert week.teaching_slots == 4

    assert [(l.day_of_week, l.period_number) for l in week.lessons] == [
        ("saturday", 1),
        ("sunday", 1),
    ]
    assert week.lessons[0].subject_name_ar == "الرياضيات"
    # A stated free period is a fact, distinguishable from a slot nobody planned.
    assert week.lessons[1].is_free is True
    assert week.lessons[0].is_free is False


def test_the_break_stays_a_break(adapter: SisTimetableAdapter) -> None:
    """`is_teaching` is read from the wire, never defaulted true.

    A break that arrived as a teaching period would be counted as a lesson the child does
    not have, and drawn as one.
    """
    with patch.object(SisTimetableAdapter, "_get", return_value=WEEK):
        week = adapter.get_timetable(
            student_ref="S-1001", term="2026-T1", guardian_ref="G-1"
        )

    lesson_slot, break_slot = week.periods
    assert lesson_slot.is_teaching is True
    assert break_slot.is_teaching is False


def test_the_bell_is_trimmed_to_minutes_and_an_unfixed_one_stays_empty(
    adapter: SisTimetableAdapter,
) -> None:
    """A parent reads a time, not a timestamp — and never an invented one.

    The seconds on a bell schedule are always zero. An unfixed boundary stays empty rather
    than acquiring a plausible hour, because a school agrees how many periods it runs long
    before it agrees when each one rings, and an invented 08:00 cannot afterwards be told
    from an agreed one.
    """
    with patch.object(SisTimetableAdapter, "_get", return_value=WEEK):
        week = adapter.get_timetable(
            student_ref="S-1001", term="2026-T1", guardian_ref="G-1"
        )

    lesson_slot, break_slot = week.periods
    assert (lesson_slot.starts_at, lesson_slot.ends_at) == ("08:00", "08:45")
    assert lesson_slot.is_timed is True
    assert (break_slot.starts_at, break_slot.ends_at) == ("", "")
    assert break_slot.is_timed is False


def test_no_class_and_no_timetable_are_different_answers(
    adapter: SisTimetableAdapter,
) -> None:
    """The whole reason `status` is on the contract.

    A child no placement covered has no room to ask about; a class nobody has laid out has a
    room and no grid. Both are empty, and only the second one means the school still has
    typing to do. Told apart nowhere else, a parent hears "she has no lessons" for both.
    """
    with patch.object(SisTimetableAdapter, "_get", return_value=NO_CLASS):
        absent = adapter.get_timetable(
            student_ref="S-1001", term="2026-T1", guardian_ref="G-1"
        )
    assert absent.status is TimetableStatus.NO_CLASS
    assert absent.class_code == ""
    assert absent.has_class is False

    with patch.object(
        SisTimetableAdapter, "_get", return_value={**WEEK, "lessons": []}
    ):
        untimetabled = adapter.get_timetable(
            student_ref="S-1001", term="2026-T1", guardian_ref="G-1"
        )
    assert untimetabled.status is TimetableStatus.NO_TIMETABLE
    assert untimetabled.class_code == "3A"
    assert untimetabled.has_class is True
    # The grid survives, so a client can draw an empty week rather than nothing at all.
    assert len(untimetabled.periods) == 2


def test_nothing_on_file_is_an_answer_and_not_an_outage(
    adapter: SisTimetableAdapter,
) -> None:
    """SIS's `unknown_reference` covers an unknown child, one that is not hers, and an
    unknown term. `records/` deliberately makes those indistinguishable, so all three
    arrive here as "no class on file" rather than as three different failures."""
    with patch.object(SisTimetableAdapter, "_get", return_value=None):
        week = adapter.get_timetable(
            student_ref="S-1001", term="2026-T1", guardian_ref="G-1"
        )

    assert week.status is TimetableStatus.NO_CLASS
    assert week.lessons == ()


def test_lessons_on_a_day_are_answerable_without_a_caller_matching_strings(
    adapter: SisTimetableAdapter,
) -> None:
    """A day with no lessons is a real answer a renderer has to be able to ask for."""
    with patch.object(SisTimetableAdapter, "_get", return_value=WEEK):
        week = adapter.get_timetable(
            student_ref="S-1001", term="2026-T1", guardian_ref="G-1"
        )

    assert [l.subject_code for l in week.lessons_on("saturday")] == ["MATH"]
    assert week.lessons_on("Monday") == ()


# ---------------------------------------------------------------------------
# Which URL is actually asked
# ---------------------------------------------------------------------------
#
# Same reasoning as the marks adapter's own block: a regression here is invisible from
# every other suite, because the data comes back identical and only the second
# authorisation check disappears.


def _path_asked(adapter: SisTimetableAdapter, **kwargs) -> tuple[str, dict]:
    seen: list[tuple[str, dict]] = []

    def _capture(self, path, params):  # noqa: ANN001 - patching an instance method
        seen.append((path, params))
        return WEEK

    with patch.object(SisTimetableAdapter, "_get", _capture):
        adapter.get_timetable(**kwargs)
    return seen[0]


def test_the_week_is_read_through_the_guardian_scoped_route(
    adapter: SisTimetableAdapter,
) -> None:
    """The guardian travels with the request, so SIS re-checks the link on this read.

    A route that named no parent would throw the second refusal away at the last hop and
    still return the right week.
    """
    path, params = _path_asked(
        adapter, student_ref="S-1001", term="2026-T1", guardian_ref="G-1"
    )
    assert path == "/v1/guardians/by-id/G-1/students/S-1001/timetable"
    assert params == {"term": "2026-T1"}


def test_no_class_is_ever_asked_for(adapter: SisTimetableAdapter) -> None:
    """The room is resolved by the system of record that owns the placement.

    Pinned because the tempting shortcut — have the caller pass a class code — is one that
    works until a child moves and then asks for the week of a room she has left.
    """
    path, params = _path_asked(
        adapter, student_ref="S-1001", term="2026-T1", guardian_ref="G-1"
    )
    assert "class" not in path
    assert "class_code" not in params


def test_a_handle_with_a_slash_cannot_rewrite_the_path(
    adapter: SisTimetableAdapter,
) -> None:
    """A handle is opaque and comes off a token. Quoting it is not optional.

    Unquoted, `../..` would climb out of the guardian prefix — the one place a crafted
    value turns a scoped read into an unscoped one.
    """
    path, _ = _path_asked(
        adapter, student_ref="S-1001", term="2026-T1", guardian_ref="../.."
    )
    assert path == "/v1/guardians/by-id/..%2F../students/S-1001/timetable"
    assert "/v1/students/" not in path


def test_an_unreachable_sis_is_never_an_empty_week() -> None:
    """"Could not ask" is never "the answer is no".

    A timetable read that failed soft would tell a parent her daughter has no lessons
    because a service was briefly down. `TimetableUnavailable` is an `UpstreamUnavailable`,
    so the facade answers 503 with `lms_unavailable` and the agent says records are
    unavailable rather than inventing a week.
    """
    adapter = SisTimetableAdapter(
        base_url="http://sis.test", api_key="reader", timeout_seconds=5.0
    )

    class _Boom:
        def get(self, *_args, **_kwargs):
            import httpx

            raise httpx.ConnectError("refused")

    with patch.object(adapter._pool, "get", return_value=_Boom()):
        with pytest.raises(TimetableUnavailable):
            adapter.get_timetable(
                student_ref="S-1001", term="2026-T1", guardian_ref="G-1"
            )


def test_a_base_url_is_required() -> None:
    """A misconfigured deployment fails at startup rather than at the first question."""
    with pytest.raises(RuntimeError):
        SisTimetableAdapter(base_url="", timeout_seconds=5.0)
