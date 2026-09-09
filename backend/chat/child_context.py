"""The child a conversation has settled on, remembered across turns.

A parent who has answered "Layla" once must not be asked again on their next question.
That was the intent behind `ChatRequestContext._remembered_child`, but the pin lived on a
per-request object and nothing ever loaded or stored it, so it survived the tool calls of
one turn and died at the turn boundary. This module is where it becomes durable.

State lives in the session metadata beside `pending_hitl`: it survives across requests
**and across processes** without standing up a new store. That matters more here than it looks. Holding conversation state in an in-process
registry keyed by thread id would work on one worker and silently lose the pin on the
next request under any multi-worker deployment — and "the assistant forgot which child"
is not a failure anyone would think to blame on worker affinity.

## The pin is a pointer, never an answer

What is stored is a student id and a label to show. It authorises nothing. Every records
read is re-checked against the school's own guardian link by the service that answers it,
so a stale or wrong pin produces a refusal rather than another family's marks. Keeping
that true is what lets this be cached at all — see `backend/chat/child_roster.py`.

## Why the guardian id is stamped on it

`chat_sessions` is keyed by username (`backend/db/models.py:37`), while the right to read
a child's records is keyed by the guardian handle. Those two can come apart: an
administrator can rebind an account to a different guardian (`identity/routes.py:342`) or
unbind it entirely (`:363`) — the custody-transfer path. Without the stamp, a rebind
leaves a conversation pinned to the previous family's child, and the next vague question
injects that child's name into a prompt. One string comparison closes it.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict

SESSION_CHILD_KEY = "child_context"


class SessionChild(BaseModel):
    """The child this conversation is about, or an empty pin on a fresh one."""

    model_config = ConfigDict(extra="forbid")

    student_id: str = ""
    #: What to call them on screen and in a prompt. Arabic-first, chosen in Python at
    #: the point the roster was read, so no template has to pick between two name
    #: columns and no second place can choose differently.
    label: str = ""
    #: "male" | "female" | "unknown". Carried so a later message that contradicts the
    #: pin ("no, my daughter") can evict it rather than being quietly ignored.
    gender: str = "unknown"
    #: The guardian this pin was resolved under. See the module docstring.
    guardian_id: str = ""

    @property
    def is_set(self) -> bool:
        return bool(self.student_id)

    @classmethod
    def from_metadata(
        cls, metadata: Optional[dict], *, guardian_id: str = ""
    ) -> "SessionChild":
        """Read the pin off session metadata, tolerantly.

        Absent, unreadable, or belonging to a different guardian all produce an empty
        pin rather than an exception. A conversation whose stored state cannot be
        trusted is a conversation that has not settled on a child yet, and that is a
        state this system already handles on every first message — so degrading into it
        costs one question, where raising would cost the turn.
        """
        raw = (metadata or {}).get(SESSION_CHILD_KEY)
        if not isinstance(raw, dict):
            return cls(guardian_id=guardian_id)
        try:
            stored = cls.model_validate(raw)
        except Exception:
            return cls(guardian_id=guardian_id)

        # The rebind check. A pin resolved under a different guardian is not this
        # caller's to inherit, whatever the session id says.
        if guardian_id and stored.guardian_id and stored.guardian_id != guardian_id:
            return cls(guardian_id=guardian_id)
        # A pin written before this field existed, or by a path that had no guardian to
        # stamp, is adopted by the current caller rather than discarded: it is a hint,
        # the read re-checks it, and discarding it would re-ask every parent once on
        # the day this ships for no safety gained.
        if guardian_id and not stored.guardian_id:
            return stored.model_copy(update={"guardian_id": guardian_id})
        return stored

    def to_metadata(self) -> dict:
        return self.model_dump()

    def pin(self, *, student_id: str, label: str = "", gender: str = "") -> None:
        """Settle this conversation on a child.

        Mutates in place because the object is threaded by reference from the turn's
        context back into the metadata that gets saved — the same trick
        `SessionAssetState` uses, and what makes write-back free rather than another
        parameter every call site has to remember to pass.

        A blank `label` or `gender` leaves the stored one alone. A later turn that
        resolves the same child from a thinner source should not erase what an earlier,
        richer one knew.
        """
        if not student_id:
            return
        if student_id != self.student_id:
            # A different child: nothing about the old one carries over.
            self.label = ""
            self.gender = "unknown"
        self.student_id = str(student_id)
        if label:
            self.label = str(label)
        if gender:
            self.gender = str(gender)

    def clear(self) -> None:
        self.student_id = ""
        self.label = ""
        self.gender = "unknown"


def load_child_state(
    metadata: Optional[dict], *, guardian_id: str = ""
) -> SessionChild:
    return SessionChild.from_metadata(metadata, guardian_id=guardian_id)


def save_child_state(save_meta: dict, child: SessionChild) -> dict:
    save_meta[SESSION_CHILD_KEY] = child.to_metadata()
    return save_meta


__all__ = [
    "SESSION_CHILD_KEY",
    "SessionChild",
    "load_child_state",
    "save_child_state",
]
