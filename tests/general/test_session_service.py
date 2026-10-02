"""SessionService: what the session routes did inline, tested without HTTP or a database.

The fakes are as small as the Protocols in application/ports/collaborators.py, which is the
point of having those Protocols: a service's test needs only the methods it calls.
"""

import unittest
from datetime import UTC, datetime

from backend.application.ports import AttachmentRecord
from backend.application.services import SessionService
from backend.domain.errors import NotFound, OperationFailed, operation


def _record(attachment_id: str) -> AttachmentRecord:
    return AttachmentRecord(
        id=attachment_id,
        kind="voice",
        sha256="0" * 64,
        storage_uri=f"local://{attachment_id}",
        content_type="audio/webm",
        byte_size=10,
        duration_ms=1000,
        transcript="hello",
        transcript_status="ok",
        created_at=datetime(2026, 9, 1, tzinfo=UTC),
    )


class _Conversations:
    def __init__(self, infos=None, page=None, deleted=True, fail=None):
        self.infos = infos or []
        self.page = page or {"messages": [], "has_more": False}
        self.deleted = deleted
        self.fail = fail
        self.calls = []

    def list_session_infos(self, user_id):
        self.calls.append(("list", user_id))
        if self.fail:
            raise self.fail
        return list(self.infos)

    def get_session_page(self, user_id, session_id, limit=None, before_id=None):
        self.calls.append(("page", user_id, session_id, limit, before_id))
        if self.fail:
            raise self.fail
        return self.page

    def delete_session(self, user_id, session_id):
        self.calls.append(("delete", user_id, session_id))
        if self.fail:
            raise self.fail
        return self.deleted


class _Attachments:
    def __init__(self, records=()):
        self.records = {record.id: record for record in records}
        self.asked = []

    def get_many(self, username, attachment_ids):
        self.asked.append((username, list(attachment_ids)))
        return {item: self.records[item] for item in attachment_ids if item in self.records}


def _service(conversations=None, attachments=None, restore=None):
    attachments = attachments or _Attachments()
    return SessionService(
        conversations=conversations or _Conversations(),
        attachments=lambda: attachments,
        restore_assets=restore or (lambda messages: messages),
    )


def _no_voice_notes():
    raise AssertionError("only opening a conversation may build the voice notes")


class ListSessionsTests(unittest.TestCase):
    def test_most_recently_active_first(self):
        infos = [
            {"session_id": "a", "updated_at": "2026-09-01T10:00:00"},
            {"session_id": "b", "updated_at": "2026-09-03T10:00:00"},
            {"session_id": "c", "updated_at": "2026-09-02T10:00:00"},
        ]
        result = _service(_Conversations(infos=infos)).list_sessions("parent")
        self.assertEqual(["b", "c", "a"], [item["session_id"] for item in result])

    def test_ties_keep_their_stored_order(self):
        """The route sorted with list.sort(reverse=True), which is stable. So is this."""
        infos = [
            {"session_id": "first", "updated_at": "2026-09-01"},
            {"session_id": "second", "updated_at": "2026-09-01"},
        ]
        result = _service(_Conversations(infos=infos)).list_sessions("parent")
        self.assertEqual(["first", "second"], [item["session_id"] for item in result])

    def test_a_storage_failure_is_reported_with_its_own_message(self):
        """The route answered a 500 with str(e) and nothing else; so does this."""
        conversations = _Conversations(fail=RuntimeError("redis down"))
        with self.assertRaises(OperationFailed) as caught:
            _service(conversations).list_sessions("parent")
        self.assertEqual("redis down", caught.exception.message)


class PageTests(unittest.TestCase):
    def test_passes_the_batch_request_through(self):
        conversations = _Conversations()
        _service(conversations).page("parent", "s1", limit=20, before=55)
        self.assertEqual(("page", "parent", "s1", 20, 55), conversations.calls[0])

    def test_messages_are_restored_and_their_notes_looked_up_once_for_the_batch(self):
        stored = [
            {"id": 1, "type": "human", "content": "q", "timestamp": "t", "attachment_id": "n1"},
            {"id": 2, "type": "ai", "content": "a", "timestamp": "t"},
        ]
        conversations = _Conversations(page={"messages": stored, "has_more": True})
        attachments = _Attachments([_record("n1")])
        restored = []

        def restore(messages):
            restored.append(messages)
            return [dict(message, restored=True) for message in messages]

        page = _service(conversations, attachments, restore).page(
            "parent", "s1", limit=20, before=None
        )

        self.assertEqual([stored], restored)
        self.assertTrue(all(message["restored"] for message in page.messages))
        self.assertEqual([("parent", ["n1", ""])], attachments.asked)
        self.assertEqual({"n1"}, set(page.attachments))
        self.assertTrue(page.has_more)

    def test_notes_are_looked_up_as_the_caller_so_a_strangers_note_resolves_to_nothing(self):
        attachments = _Attachments()
        stored = [
            {"id": 1, "type": "human", "content": "q", "timestamp": "t", "attachment_id": "x"}
        ]
        page = _service(
            _Conversations(page={"messages": stored, "has_more": False}), attachments
        ).page("parent", "s1", limit=20, before=None)
        self.assertEqual("parent", attachments.asked[0][0])
        self.assertEqual({}, page.attachments)


class DeleteTests(unittest.TestCase):
    def test_deletes_the_callers_conversation(self):
        conversations = _Conversations(deleted=True)
        _service(conversations).delete("parent", "s1")
        self.assertEqual(("delete", "parent", "s1"), conversations.calls[0])

    def test_a_conversation_that_is_not_there_is_not_found(self):
        with self.assertRaises(NotFound) as caught:
            _service(_Conversations(deleted=False)).delete("parent", "missing")
        self.assertEqual("Session does not exist", caught.exception.message)

    def test_a_storage_failure_is_not_mistaken_for_not_found(self):
        with self.assertRaises(OperationFailed):
            _service(_Conversations(fail=RuntimeError("db gone"))).delete("parent", "s1")


class WhatEachMethodBuildsTests(unittest.TestCase):
    def test_listing_and_deleting_never_build_the_voice_notes(self):
        """Building them builds the blob store and the speech-to-text model; the old list and
        delete routes depended on the conversation storage alone."""
        service = SessionService(
            conversations=_Conversations(infos=[{"session_id": "a", "updated_at": "1"}]),
            attachments=_no_voice_notes,
            restore_assets=lambda messages: messages,
        )
        self.assertEqual(1, len(service.list_sessions("parent")))
        service.delete("parent", "a")

    def test_voice_notes_that_cannot_be_built_fail_the_page_as_its_operation(self):
        def unbuildable():
            raise RuntimeError("blob backend misconfigured")

        service = SessionService(
            conversations=_Conversations(),
            attachments=unbuildable,
            restore_assets=lambda messages: messages,
        )
        with self.assertRaises(OperationFailed) as caught:
            service.page("parent", "s1", limit=20, before=None)
        self.assertEqual("blob backend misconfigured", caught.exception.message)


class OperationTests(unittest.TestCase):
    def test_a_domain_error_passes_through_untouched(self):
        error = NotFound("nope")
        with self.assertRaises(NotFound) as caught:
            with operation("Failed to do the thing"):
                raise error
        self.assertIs(error, caught.exception)

    def test_anything_else_becomes_the_operations_failure(self):
        with self.assertRaises(OperationFailed) as caught:
            with operation("Failed to read chunks"):
                raise ValueError("milvus down")
        self.assertEqual("Failed to read chunks: milvus down", caught.exception.message)
        self.assertIsInstance(caught.exception.__cause__, ValueError)

    def test_a_block_that_succeeds_is_untouched(self):
        with operation("x"):
            value = 42
        self.assertEqual(42, value)


if __name__ == "__main__":
    unittest.main()
