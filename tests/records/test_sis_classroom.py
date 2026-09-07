"""SIS's class, subject board and staff list, mapped onto the parent-facing contract.

The mapping is thin — SIS answers all three from one route — so almost every assertion here
pins one of three decisions:

**The class NAME survives, distinctly from its code.** "Which class is my child in" is
answered by what the school calls the room, and a mapping that dropped the name and kept
the key would answer `3A` to a parent who has only ever seen "Primary 3 Class 1".

**A co-teacher survives.** Two teachers on one subject is a state the system of record
permits; anything here that collapsed to one name per subject would hide a real person.

**"No class" is a status, not an emptiness.** SIS states a null class for a child no
placement covered, and empty lists for a room whose board or staffing nobody has entered.
The first becomes a word a consumer can branch on; the other two stay as emptiness, because
a room can lack either, both or neither.

Mocked at the adapter's own `_get`, which is where `records/`'s other adapter tests sit — no
network, and no extra dependency to keep current.
"""
from unittest.mock import patch

import pytest

from records.adapters.sis.classroom import SisClassroomAdapter
from records.domain.classroom import ClassroomStatus
from records.domain.errors import ClassroomUnavailable

#: One SIS `StudentClassroomOut`, as the route actually answers it. The code and the name
#: differ on purpose, and SCI is taught by two people.
ROOM = {
    "student_number": "S-1001",
    "term_code": "2026-T1",
    "academic_year_code": "2025-2026",
    "class_code": "3A",
    "class_name_ar": "الثالث ١",
    "class_name_en": "Primary 3 Class 1",
    "year_level_code": "AR-P3",
    "year_level_name_ar": "الصف الثالث",
    "year_level_name_en": "Year 3",
    "subjects": [
        {"code": "MATH", "name_ar": "الرياضيات", "name_en": "Mathematics"},
        {"code": "SCI", "name_ar": "العلوم", "name_en": "Science"},
    ],
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

NO_CLASS = {
    "student_number": "S-1001",
    "term_code": "2026-T1",
    "academic_year_code": None,
    "class_code": None,
    "class_name_ar": None,
    "class_name_en": None,
    "year_level_code": None,
    "year_level_name_ar": None,
    "year_level_name_en": None,
    "subjects": [],
    "teachers": [],
}


@pytest.fixture()
def adapter() -> SisClassroomAdapter:
    return SisClassroomAdapter(
        base_url="http://sis.test", api_key="reader", timeout_seconds=5.0
    )


def test_the_room_arrives_named_and_whole(adapter: SisClassroomAdapter) -> None:
    with patch.object(SisClassroomAdapter, "_get", return_value=ROOM):
        room = adapter.get_classroom(
            student_ref="S-1001", term="2026-T1", guardian_ref="G-1"
        )

    assert room.status is ClassroomStatus.OK
    assert room.has_class is True
    # The name is the answer; the code is a key that happens to differ from it.
    assert room.class_name_ar == "الثالث ١"
    assert room.class_name_en == "Primary 3 Class 1"
    assert room.class_code == "3A"
    assert room.year_level_name_ar == "الصف الثالث"


def test_the_subject_board_keeps_the_school_s_order(adapter: SisClassroomAdapter) -> None:
    """Never re-sorted: alphabetical differs between the two scripts this estate renders,
    so a sort here would make one board read in two orders."""
    with patch.object(SisClassroomAdapter, "_get", return_value=ROOM):
        room = adapter.get_classroom(
            student_ref="S-1001", term="2026-T1", guardian_ref="G-1"
        )

    assert [s.code for s in room.subjects] == ["MATH", "SCI"]
    assert [s.name_ar for s in room.subjects] == ["الرياضيات", "العلوم"]


def test_a_co_teacher_is_not_collapsed_away(adapter: SisClassroomAdapter) -> None:
    """One entry per (teacher, subject), so SCI has two.

    The system of record's table is unique on (teacher, class, subject) and only one of its
    two write paths guards against a second teacher, so this is a real state — and a shape
    that held one teacher per subject would silently drop somebody.
    """
    with patch.object(SisClassroomAdapter, "_get", return_value=ROOM):
        room = adapter.get_classroom(
            student_ref="S-1001", term="2026-T1", guardian_ref="G-1"
        )

    assert len(room.teachers) == 3
    science = sorted(t.full_name_ar for t in room.teachers if t.subject_code == "SCI")
    assert science == ["أ. منى", "أ. هدى"]


def test_no_placement_is_a_status_rather_than_three_empty_lists(
    adapter: SisClassroomAdapter,
) -> None:
    """A fact about the child, and the one emptiness that must be branchable.

    With a room present, an empty board or an empty staff list means the school has typing
    to do. With no room at all there is nothing to have typed, and a consumer has to be
    able to tell those apart or it tells a parent her daughter studies nothing.
    """
    with patch.object(SisClassroomAdapter, "_get", return_value=NO_CLASS):
        room = adapter.get_classroom(
            student_ref="S-1001", term="2026-T1", guardian_ref="G-1"
        )

    assert room.status is ClassroomStatus.NO_CLASS
    assert room.has_class is False
    assert room.class_code == ""
    assert room.class_name_ar == ""
    assert room.subjects == ()
    assert room.teachers == ()


def test_a_room_with_nothing_entered_yet_still_has_a_class(
    adapter: SisClassroomAdapter,
) -> None:
    """The other two empties: she has a room, the school has not filled it in.

    Status stays `ok`, so a consumer can say "her class is Primary 3 Class 1 and the school
    has not published its subjects yet" rather than "she has no class".
    """
    with patch.object(
        SisClassroomAdapter, "_get", return_value={**ROOM, "subjects": [], "teachers": []}
    ):
        room = adapter.get_classroom(
            student_ref="S-1001", term="2026-T1", guardian_ref="G-1"
        )

    assert room.status is ClassroomStatus.OK
    assert room.class_name_en == "Primary 3 Class 1"
    assert room.subjects == ()
    assert room.teachers == ()


def test_nothing_on_file_is_an_answer_and_not_an_outage(
    adapter: SisClassroomAdapter,
) -> None:
    """SIS's `unknown_reference` covers an unknown child, one that is not hers, and an
    unknown term. `records/` deliberately makes those indistinguishable."""
    with patch.object(SisClassroomAdapter, "_get", return_value=None):
        room = adapter.get_classroom(
            student_ref="S-1001", term="2026-T1", guardian_ref="G-1"
        )

    assert room.status is ClassroomStatus.NO_CLASS


# ---------------------------------------------------------------------------
# Which URL is actually asked
# ---------------------------------------------------------------------------


def _path_asked(adapter: SisClassroomAdapter, **kwargs) -> tuple[str, dict]:
    seen: list[tuple[str, dict]] = []

    def _capture(self, path, params):  # noqa: ANN001 - patching an instance method
        seen.append((path, params))
        return ROOM

    with patch.object(SisClassroomAdapter, "_get", _capture):
        adapter.get_classroom(**kwargs)
    return seen[0]


def test_the_room_is_read_through_the_guardian_scoped_route(
    adapter: SisClassroomAdapter,
) -> None:
    """The guardian travels with the request, so SIS re-checks the link on this read."""
    path, params = _path_asked(
        adapter, student_ref="S-1001", term="2026-T1", guardian_ref="G-1"
    )
    assert path == "/v1/guardians/by-id/G-1/students/S-1001/classroom"
    assert params == {"term": "2026-T1"}


def test_no_class_is_ever_asked_for(adapter: SisClassroomAdapter) -> None:
    """The room is resolved by the system of record that owns the placement.

    Pinned because the tempting shortcut — pass the class code the last response carried —
    works until a child moves and then asks about a room she has left.
    """
    path, params = _path_asked(
        adapter, student_ref="S-1001", term="2026-T1", guardian_ref="G-1"
    )
    assert "class" not in params
    assert "3A" not in path


def test_a_handle_with_a_slash_cannot_rewrite_the_path(
    adapter: SisClassroomAdapter,
) -> None:
    """A handle is opaque and comes off a token. Quoting it is not optional."""
    path, _ = _path_asked(
        adapter, student_ref="S-1001", term="2026-T1", guardian_ref="../.."
    )
    assert path == "/v1/guardians/by-id/..%2F../students/S-1001/classroom"
    assert "/v1/students/" not in path


def test_an_unreachable_sis_is_never_a_child_with_no_teachers() -> None:
    """"Could not ask" is never "the answer is no".

    Failing soft here would tell a parent her daughter has no class and no teachers because
    a service was briefly down.
    """
    adapter = SisClassroomAdapter(
        base_url="http://sis.test", api_key="reader", timeout_seconds=5.0
    )

    class _Boom:
        def get(self, *_args, **_kwargs):
            import httpx

            raise httpx.ConnectError("refused")

    with patch.object(adapter._pool, "get", return_value=_Boom()):
        with pytest.raises(ClassroomUnavailable):
            adapter.get_classroom(
                student_ref="S-1001", term="2026-T1", guardian_ref="G-1"
            )


def test_a_base_url_is_required() -> None:
    """A misconfigured deployment fails at startup rather than at the first question."""
    with pytest.raises(RuntimeError):
        SisClassroomAdapter(base_url="", timeout_seconds=5.0)
