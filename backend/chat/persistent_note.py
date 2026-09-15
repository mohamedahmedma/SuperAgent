"""The persistent note: which parent it was written for, and whether it may be read.

The note is the assistant's own summary of a conversation, kept in the session metadata
and shown to the model on every later turn. It names the children the conversation was
about — the note prompt asks it to, because a child's name is the first thing a later
question inherits.

`chat_sessions` is keyed by username; the right to read a child's records is keyed by
the guardian handle. An administrator can rebind an account to a different guardian
(`identity/routes.py`, the custody-transfer path), and the child pin already refuses to
survive that (`backend/chat/child_context.py`). The note did not: it was read whatever
guardian had written it, so the previous family's child's name kept steering answers
for the next. The note is stamped with its guardian, and read only for the same one.
"""
from __future__ import annotations

NOTE_KEY = "persistent_note"
#: The guardian the note was written under. Absent on notes written before the stamp
#: existed; those are adopted by the current caller, as an unstamped pin is.
NOTE_GUARDIAN_KEY = "persistent_note_guardian"


def usable_note(metadata: dict | None, guardian_id: str) -> tuple[str, bool]:
    """`(note, discarded)`: the note this caller may read, and whether one was refused.

    A note stamped with a different guardian is another family's memory of the
    conversation. It is returned as empty, and `discarded` tells the turn to clear it so
    it is not consulted — or offered to the note writer as a starting point — again.
    A staff session has no guardian and reads the note as stored.
    """
    metadata = metadata or {}
    note = str(metadata.get(NOTE_KEY) or "")
    stamped = str(metadata.get(NOTE_GUARDIAN_KEY) or "")
    if guardian_id and stamped and stamped != guardian_id:
        return "", bool(note)
    return note, False


def note_patch(note: str, guardian_id: str) -> dict:
    """The metadata a note update writes: the note, and who it was written for."""
    return {NOTE_KEY: note, NOTE_GUARDIAN_KEY: guardian_id or None}


def cleared_note_patch() -> dict:
    """The metadata that drops a note another guardian wrote."""
    return {NOTE_KEY: "", NOTE_GUARDIAN_KEY: None}


__all__ = ["NOTE_GUARDIAN_KEY", "NOTE_KEY", "cleared_note_patch", "note_patch", "usable_note"]
