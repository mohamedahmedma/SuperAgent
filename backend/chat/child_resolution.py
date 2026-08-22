"""Which of the caller's children this turn is about.

The sibling of `backend/chat/resolution.py`, and deliberately shaped like it. That module
turns an under-specified message into a standalone question; this one turns an
under-specified reference into a particular child. Both run once per turn, before
anything expensive; both distinguish "nobody looked" from "something looked and found
nothing" through a `resolved` flag; both abstain on every failure path rather than
denying the turn.

They stay separate modules because they need different things. `resolve_question` reads
words and history and is identity-blind — it can be handed to a model. This reads the
caller's own roster, which is a list of real children's names, and must never cost a
model call at all.

## Pure, and that is the point

No I/O, no clock, no model. The roster is passed in. Everything here is a function of its
arguments, so every rule below is a unit test rather than a live conversation somebody
has to reproduce.

## A hint, never an authority

Resolving a child selects *which* of this parent's own children a vague question is
about. It can never widen who may be read: the roster came from the records facade under
this turn's guardian, and every subsequent read is re-checked there. The worst a wrong
answer here produces is a refusal, or an answer about the wrong one of the caller's own
children — which is why the rules below degrade toward asking rather than guessing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Tuple

from backend.chat.child_context import SessionChild
from backend.chat.child_roster import ChildOption

#: Reference kinds that name a sex. `unknown` on a child matches BOTH of these, so a
#: half-filled gender column can never select a child by virtue of a blank cell.
_GENDERED = {"son": "male", "daughter": "female"}


@dataclass(frozen=True)
class ResolvedChild:
    """One turn's answer to "which child", with how it was reached.

    `resolved` and `ask` are mutually exclusive and both may be false — that third
    state is "this turn is not about a child", which is most turns.
    """

    student_id: str = ""
    label: str = ""
    year_level: str = ""
    #: The children to offer when `ask` is set. A subset when the message narrowed it:
    #: after "my son", asking "Ali, Ahmed or Layla?" is a worse question than asking
    #: "Ali or Ahmed?" — it ignores what the parent just said.
    options: Tuple[ChildOption, ...] = ()
    resolved: bool = False
    ask: bool = False
    #: named | only_child | gender | pin — how firmly the parent stated it.
    source: str = ""
    reason: str = ""

    @property
    def option_labels(self) -> list:
        return [child.label for child in self.options]


def no_child(reason: str) -> ResolvedChild:
    """Nothing to say about a child this turn. Every abstention returns this."""
    return ResolvedChild(reason=reason)


def _ask(options: Sequence[ChildOption], reason: str) -> ResolvedChild:
    return ResolvedChild(options=tuple(options), ask=True, reason=reason)


def _found(child: ChildOption, source: str, reason: str) -> ResolvedChild:
    return ResolvedChild(
        student_id=child.student_id,
        label=child.label,
        year_level=child.year_level,
        resolved=True,
        source=source,
        reason=reason,
    )


def _named(roster: Sequence[ChildOption], name: str) -> list:
    """Children whose label contains `name`, casefolded.

    Substring across the whole label because a parent writing "Ali" means the child
    stored as "Ali Osman", and a school in this country stores a full patronymic. That
    is also why a match is only trusted when it is UNIQUE: «أحمد» is inside both «علي
    أحمد حسن» and «أحمد أحمد حسن», and picking the first would show one child's marks
    while the parent was asking about the other.
    """
    needle = (name or "").strip().casefold()
    if not needle:
        return []
    return [
        child
        for child in roster
        if needle in child.label.casefold() or needle == child.student_id.casefold()
    ]


def resolve_child(
    *,
    reference: str,
    child_name: str = "",
    roster: Sequence[ChildOption] = (),
    pin: SessionChild | None = None,
) -> ResolvedChild:
    """Pick the child, or say that the parent has to be asked.

    The routes are ordered by how firmly the parent stated it, which is the same
    ordering `records.py`'s matcher already argued for, plus the two this feature adds.
    """
    roster = list(roster or [])
    pin = pin or SessionChild()

    if not roster:
        # Never ask, never hint. "This parent has no readable children" is a fact the
        # records tool has careful wording for, and only it should say it.
        return no_child("no readable children")

    # 1. A name in the message. Always wins, so "and how is Omar?" moves the
    #    conversation on even when the previous question was about his sister.
    if reference == "named":
        matches = _named(roster, child_name)
        if len(matches) == 1:
            return _found(matches[0], "named", f"the message names {matches[0].label}")
        if len(matches) > 1:
            # Ask between the ones that matched, not the whole family.
            return _ask(matches, f"{len(matches)} children match that name")
        # A name matching nobody falls through to asking rather than quietly using the
        # pin: the parent named somebody, and answering about a different child while
        # they watch is worse than one more question.
        return _ask(roster, "the name matches none of this parent's children")

    # 2. An only child. Nothing to disambiguate, so they are never asked at all.
    if len(roster) == 1:
        return _found(roster[0], "only_child", "an only child")

    # 3. Plural. Never narrows and never asks — collapsing "all of them" to one child is
    #    worse than not helping, and the tool can still read the whole roster.
    if reference == "plural":
        return no_child("the message is about more than one child")

    # 4. Narrow by sex, when the message stated one.
    #
    #    `unknown` is a candidate for BOTH, so a half-filled gender column can never
    #    select a child by virtue of a blank cell — which is the state every child is in
    #    until a registrar uploads it.
    wanted = _GENDERED.get(reference, "")
    candidates = (
        [c for c in roster if c.gender in (wanted, "unknown")] if wanted else list(roster)
    )
    if wanted and not candidates:
        # The parent said "my son" and nobody on file could be one. Their wording is
        # better evidence than the column, so ask rather than declaring them wrong.
        return _ask(roster, f"no child on file could be the {reference}")

    if len(candidates) == 1:
        return _found(candidates[0], "gender", f"the only {reference}")

    # 5. The child already under discussion — but only from among the candidates, so a
    #    sex the parent just stated still overrides a pin that contradicts it. A parent
    #    settled on their daughter who then asks about "my son" is changing subject, not
    #    continuing.
    #
    #    Also required to still be on the roster, so a pin that outlived the parent's
    #    access falls through rather than being used.
    if pin.student_id:
        pinned = next((c for c in candidates if c.student_id == pin.student_id), None)
        if pinned is not None:
            return _found(pinned, "pin", "the child already under discussion")

    return _ask(
        candidates,
        f"{len(candidates)} children and nothing in the message to choose between them",
    )


__all__ = ["ResolvedChild", "no_child", "resolve_child"]
