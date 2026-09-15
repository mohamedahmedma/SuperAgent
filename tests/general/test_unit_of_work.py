"""The unit of work and the repositories behind it, against real Postgres.

The rules here are the ones every service relies on without restating: nothing lands
without `commit()`, a repository cannot be used once its transaction is over, and a
listing of conversations costs one query however many conversations there are.
"""
import unittest
from datetime import UTC, datetime, timedelta

from sqlalchemy import event, select

from backend.application.ports.repositories import NewMessage
from backend.db.models import ChatMessage, ChatSession, User
from tests.general.postgres_support import postgres_schema


class UnitOfWorkTestCase(unittest.TestCase):
    def setUp(self):
        self.schema = postgres_schema(self, User, ChatSession, ChatMessage)
        with self.schema.sessionmaker()() as db:
            db.add_all([User(username="parent", password_hash="x"), User(username="other", password_hash="x")])
            db.commit()

    def count(self, model):
        with self.schema.sessionmaker()() as db:
            return len(db.scalars(select(model)).all())


class TransactionTests(UnitOfWorkTestCase):
    def test_nothing_lands_without_commit(self):
        with self.schema.unit_of_work() as uow:
            uow.conversations.open_session("parent", "s1", {"title": "Fees"})
        self.assertEqual(0, self.count(ChatSession))

    def test_commit_lands_every_staged_write_together(self):
        now = datetime.now(UTC)
        with self.schema.unit_of_work() as uow:
            session = uow.conversations.open_session("parent", "s1", {})
            uow.conversations.add_messages(session, [NewMessage("human", "hi", now, None)])
            uow.commit()
        self.assertEqual((1, 1), (self.count(ChatSession), self.count(ChatMessage)))

    def test_an_exception_rolls_back_and_keeps_travelling(self):
        with self.assertRaises(ValueError):
            with self.schema.unit_of_work() as uow:
                uow.conversations.open_session("parent", "s1", {})
                raise ValueError("halfway")
        self.assertEqual(0, self.count(ChatSession))

    def test_a_repository_outside_its_transaction_says_so(self):
        uow = self.schema.unit_of_work()
        with self.assertRaisesRegex(RuntimeError, "only available inside"):
            uow.conversations
        with uow:
            pass
        with self.assertRaisesRegex(RuntimeError, "only available inside"):
            uow.conversations

    def test_a_unit_of_work_cannot_be_entered_twice_at_once(self):
        uow = self.schema.unit_of_work()
        with uow:
            with self.assertRaisesRegex(RuntimeError, "already open"):
                uow.__enter__()


class ConversationRepositoryTests(UnitOfWorkTestCase):
    def test_an_unknown_user_gets_no_conversation(self):
        with self.schema.unit_of_work() as uow:
            self.assertIsNone(uow.conversations.open_session("nobody", "s1", {}))

    def test_added_messages_get_ids_in_the_order_they_were_given(self):
        now = datetime.now(UTC)
        with self.schema.unit_of_work() as uow:
            session = uow.conversations.open_session("parent", "s1", {})
            ids = uow.conversations.add_messages(
                session, [NewMessage(kind, text, now, None) for kind, text in (("human", "a"), ("ai", "b"), ("human", "c"))]
            )
            uow.commit()
        self.assertEqual(sorted(ids), list(ids))
        with self.schema.unit_of_work() as uow:
            session = uow.conversations.find_session("parent", "s1")
            self.assertEqual(["a", "b", "c"], [m.content for m in uow.conversations.messages(session)])

    def test_summaries_are_newest_first_counted_and_one_query(self):
        start = datetime(2026, 9, 13, 10, 0, tzinfo=UTC)
        with self.schema.unit_of_work() as uow:
            for offset, (name, count) in enumerate((("s1", 3), ("s2", 0), ("s3", 1))):
                session = uow.conversations.open_session("parent", name, {"title": name.upper()})
                uow.conversations.add_messages(
                    session, [NewMessage("human", str(i), start, None) for i in range(count)]
                )
                uow.conversations.patch_session(session, metadata=None, updated_at=start + timedelta(minutes=offset))
            other = uow.conversations.open_session("other", "theirs", {})
            uow.conversations.add_messages(other, [NewMessage("human", "x", start, None)])
            uow.commit()

        statements = []
        listener = lambda *args: statements.append(args[2])  # noqa: E731 — (conn, cursor, statement, ...)
        event.listen(self.schema.engine, "before_cursor_execute", listener)
        try:
            with self.schema.unit_of_work() as uow:
                summaries = uow.conversations.summaries("parent")
        finally:
            event.remove(self.schema.engine, "before_cursor_execute", listener)

        self.assertEqual(
            [("s3", 1), ("s2", 0), ("s1", 3)],
            [(summary.session_id, summary.message_count) for summary in summaries],
        )
        self.assertEqual(1, sum(1 for sql in statements if sql.lstrip().upper().startswith("SELECT")))

    def test_deleting_a_conversation_removes_its_messages_and_nobody_elses(self):
        now = datetime.now(UTC)
        with self.schema.unit_of_work() as uow:
            for username in ("parent", "other"):
                session = uow.conversations.open_session(username, "s1", {})
                uow.conversations.add_messages(session, [NewMessage("human", "hi", now, None)])
            uow.commit()

        with self.schema.unit_of_work() as uow:
            self.assertTrue(uow.conversations.delete_session("parent", "s1"))
            self.assertFalse(uow.conversations.delete_session("parent", "missing"))
            uow.commit()

        self.assertEqual((1, 1), (self.count(ChatSession), self.count(ChatMessage)))


if __name__ == "__main__":
    unittest.main()
