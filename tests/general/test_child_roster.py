"""The parent's children, read once and cached.

The cases that earn their place are the ones about what must NOT be cached. A cached
outage tells a parent for ninety seconds that the school has no record of their child;
a cached refusal does the same for a sign-in that has merely expired. Both are worse
than the extra HTTP call the cache exists to save.
"""
import pytest

import backend.chat.child_roster as child_roster
from backend.chat.caller_identity import CallerIdentity
from backend.chat.child_roster import (
    NONE,
    NOT_AUTHORIZED,
    OK,
    UNAVAILABLE,
    ChildOption,
    load_roster,
)
from backend.chat.request_context import ChatRequestContext

ROWS = [
    {"student_id": "S-1", "full_name_ar": "ليلى", "full_name_en": "Layla"},
    {"student_id": "S-2", "full_name_ar": "", "full_name_en": "Omar"},
]


def _ctx(guardian_id="G-1", token="tok"):
    return ChatRequestContext(
        user_id="u",
        session_id="turn-1",
        caller=CallerIdentity(user_id="u", guardian_id=guardian_id, guardian_token=token),
    )


def _fetch(outcome, rows):
    calls = []

    def fake(guardian_id, token, request_id):
        calls.append((guardian_id, token, request_id))
        return outcome, rows

    fake.calls = calls
    return fake


@pytest.fixture(autouse=True)
def no_cache(monkeypatch):
    monkeypatch.setenv("CHILD_ROSTER_TTL_SECONDS", "0")


class TestReading:
    def test_children_come_back_labelled_arabic_first(self):
        outcome, children = load_roster(_ctx(), fetch=_fetch(OK, ROWS))

        assert outcome == OK
        assert [c.label for c in children] == ["ليلى", "Omar"]
        assert [c.student_id for c in children] == ["S-1", "S-2"]

    def test_a_child_with_no_name_at_all_falls_back_to_the_number(self):
        rows = [{"student_id": "S-9", "full_name_ar": "", "full_name_en": ""}]
        _, children = load_roster(_ctx(), fetch=_fetch(OK, rows))
        assert children[0].label == "S-9"

    def test_a_row_with_no_student_id_is_dropped_rather_than_labelled_blank(self):
        rows = [{"full_name_ar": "مجهول"}, *ROWS]
        _, children = load_roster(_ctx(), fetch=_fetch(OK, rows))
        assert len(children) == 2

    def test_gender_defaults_to_unknown_which_is_the_day_one_state(self):
        """Nothing has a gender until a registrar uploads one, and `unknown` must never
        be able to select a child on its own."""
        _, children = load_roster(_ctx(), fetch=_fetch(OK, ROWS))
        assert {c.gender for c in children} == {"unknown"}

    def test_a_non_parent_session_asks_nobody_anything(self):
        ctx = ChatRequestContext(user_id="staff", session_id="s")
        probe = _fetch(OK, ROWS)

        outcome, children = load_roster(ctx, fetch=probe)

        assert (outcome, children) == (NONE, [])
        assert probe.calls == []


class TestOutcomesStayDistinct:
    def test_an_outage_is_not_an_empty_family(self):
        assert load_roster(_ctx(), fetch=_fetch(UNAVAILABLE, [])) == (UNAVAILABLE, [])

    def test_an_expired_sign_in_is_not_an_outage(self):
        """Telling a parent whose token expired that records are down sends them away to
        wait for a service that is working perfectly."""
        assert load_roster(_ctx(), fetch=_fetch(NOT_AUTHORIZED, [])) == (NOT_AUTHORIZED, [])

    def test_a_genuinely_empty_roster_reports_none_not_ok(self):
        outcome, children = load_roster(_ctx(), fetch=_fetch(OK, []))
        assert (outcome, children) == (NONE, [])


class TestCaching:
    @pytest.fixture(autouse=True)
    def spy_cache(self, monkeypatch):
        writes = {}
        monkeypatch.setenv("CHILD_ROSTER_TTL_SECONDS", "90")
        monkeypatch.setattr(child_roster.cache, "get_json", lambda key: writes.get(key))
        monkeypatch.setattr(
            child_roster.cache,
            "set_json",
            lambda key, value, ttl=None: writes.__setitem__(key, value),
        )
        monkeypatch.setattr(child_roster.cache, "delete", lambda key: writes.pop(key, None))
        self.writes = writes

    def test_a_second_read_in_the_conversation_costs_nothing(self):
        probe = _fetch(OK, ROWS)

        load_roster(_ctx(), fetch=probe)
        outcome, children = load_roster(_ctx(), fetch=probe)

        assert len(probe.calls) == 1
        assert outcome == OK and len(children) == 2

    def test_an_outage_is_never_written(self):
        """Ninety seconds of telling a parent nothing is there, from a three-second blip."""
        load_roster(_ctx(), fetch=_fetch(UNAVAILABLE, []))
        assert self.writes == {}

    def test_a_refusal_is_never_written(self):
        load_roster(_ctx(), fetch=_fetch(NOT_AUTHORIZED, []))
        assert self.writes == {}

    def test_an_empty_list_is_never_written(self):
        load_roster(_ctx(), fetch=_fetch(OK, []))
        assert self.writes == {}

    def test_two_guardians_do_not_share_an_entry(self):
        load_roster(_ctx(guardian_id="G-1"), fetch=_fetch(OK, ROWS))
        probe = _fetch(OK, [{"student_id": "S-9", "full_name_ar": "سارة"}])

        _, children = load_roster(_ctx(guardian_id="G-2"), fetch=probe)

        assert [c.student_id for c in children] == ["S-9"]
        assert len(probe.calls) == 1

    def test_a_guardian_id_cannot_smuggle_a_separator_into_the_key(self):
        load_roster(_ctx(guardian_id="G-1:evil"), fetch=_fetch(OK, ROWS))
        assert all(":evil" not in key for key in self.writes)

    def test_forgetting_drops_the_entry_so_the_next_read_is_fresh(self):
        ctx = _ctx()
        load_roster(ctx, fetch=_fetch(OK, ROWS))

        child_roster.forget(ctx)
        probe = _fetch(OK, ROWS)
        load_roster(ctx, fetch=probe)

        assert len(probe.calls) == 1

    def test_the_switch_turns_the_cache_off_entirely(self, monkeypatch):
        monkeypatch.setenv("CHILD_ROSTER_TTL_SECONDS", "0")
        probe = _fetch(OK, ROWS)

        load_roster(_ctx(), fetch=probe)
        load_roster(_ctx(), fetch=probe)

        assert len(probe.calls) == 2
        assert self.writes == {}
