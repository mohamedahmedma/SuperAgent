"""SIS's daily register, mapped onto the per-subject attendance contract.

The two systems keep attendance differently and the mapping is a decision, not a
translation: Moodle records a register per subject, SIS records one mark per day for the
whole school day. Every assertion here pins a choice that was made rather than a behaviour
that fell out.

Mocked at the adapter's own `_get`, which is where `records/`'s other adapter tests sit —
no network, and no extra dependency to keep current.
"""
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from records.calendar import FakeSchoolCalendar, SchoolTerm
from records.sis_adapter import SisAdapter

TERM = SchoolTerm(
    code="2026-T1",
    name_en="Term 1",
    name_ar="الفصل الأول",
    academic_year="2025-2026",
    starts_on=datetime(2025, 9, 1, tzinfo=timezone.utc),
    ends_on=datetime(2025, 12, 15, tzinfo=timezone.utc),
)


def _tally(present=0, absent=0, late=0, excused=0):
    recorded = present + absent + late + excused
    return {
        "counts": {
            "present": present,
            "absent": absent,
            "late": late,
            "excused": excused,
            "recorded": recorded,
            "in_the_room": present + late,
            "away": absent + excused,
        }
    }


@pytest.fixture()
def adapter() -> SisAdapter:
    return SisAdapter(
        base_url="http://sis.test",
        api_key="reader",
        calendar=FakeSchoolCalendar([TERM]),
    )


def test_a_term_of_register_days_becomes_one_entry(adapter: SisAdapter) -> None:
    """SIS has no per-subject register, so there is nothing to aggregate across.

    One entry standing for the term is not a subject pretending to be one — it is the
    register SIS actually keeps, reported at the granularity it is kept, and summing it
    gives the contract's term figures exactly.
    """
    with patch.object(SisAdapter, "_get", return_value=_tally(present=7, absent=1, late=1, excused=1)):
        subjects = adapter.get_subject_attendance(student_ref="S001", term="2026-T1")

    assert len(subjects) == 1
    assert subjects[0].taken_sessions == 10


def test_an_excused_day_counts_as_attended(adapter: SisAdapter) -> None:
    """The contract's rule, and the one place the two services genuinely disagree.

    SIS's `in_the_room` is present-plus-late and treats an excused day as missed. This
    contract does not: `AttendanceAssembler.PRESENT_LIKE` counts excused as present and
    the parent-facing template says so out loud. Mapping SIS's narrower figure through
    would show a child with a doctor's note as having missed school.
    """
    with patch.object(SisAdapter, "_get", return_value=_tally(present=7, absent=1, late=1, excused=1)):
        (subject,) = adapter.get_subject_attendance(student_ref="S001", term="2026-T1")

    # 9 of 10 — everything except the unexcused absence.
    assert subject.percentage == 90.0
    assert subject.points == 9.0
    assert subject.max_points == 10.0


def test_a_term_nobody_marked_reports_nothing_rather_than_zero(adapter: SisAdapter) -> None:
    """A child cannot be absent from classes nobody recorded.

    Empty, so the contract publishes `attendance_rate: null`. Returning a zero-session
    entry would render as 0% attendance for a term the register was simply never taken in.
    """
    with patch.object(SisAdapter, "_get", return_value=_tally()):
        assert adapter.get_subject_attendance(student_ref="S001", term="2026-T1") == []


def test_the_register_is_asked_for_over_the_term_s_own_dates(adapter: SisAdapter) -> None:
    """SIS's register is addressed by dates and knows nothing about term codes."""
    with patch.object(SisAdapter, "_get", return_value=_tally(present=1)) as get:
        adapter.get_subject_attendance(student_ref="S001", term="2026-T1")

    path, params = get.call_args[0]
    assert path == "/v1/students/S001/attendance"
    assert params == {"from": "2025-09-01", "to": "2025-12-15"}


def test_an_unknown_term_reports_nothing_rather_than_failing(adapter: SisAdapter) -> None:
    """A term the school has not configured has no register — a real answer, not an outage."""
    with patch.object(SisAdapter, "_get") as get:
        assert adapter.get_subject_attendance(student_ref="S001", term="2099-T9") == []
    get.assert_not_called()


def test_without_a_calendar_attendance_is_simply_absent() -> None:
    """A deployment that wired no calendar cannot resolve a term to dates.

    Empty rather than an exception: reading grades must keep working for a caller that
    never asked for attendance.
    """
    bare = SisAdapter(base_url="http://sis.test", api_key="reader")

    with patch.object(SisAdapter, "_get") as get:
        assert bare.get_subject_attendance(student_ref="S001", term="2026-T1") == []
    get.assert_not_called()


def test_the_statuses_a_parent_is_shown_are_carried_through(adapter: SisAdapter) -> None:
    """The contract counts by description, so the four have to arrive named."""
    with patch.object(SisAdapter, "_get", return_value=_tally(present=7, absent=1, late=1, excused=1)):
        (subject,) = adapter.get_subject_attendance(student_ref="S001", term="2026-T1")

    counts = {row["description"]: row["count"] for row in subject.by_status}
    assert counts == {"Present": 7, "Absent": 1, "Late": 1, "Excused": 1}
