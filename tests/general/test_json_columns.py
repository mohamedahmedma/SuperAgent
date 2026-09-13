"""JSON values lose NUL before they are serialised, because JSONB refuses it — and nothing else.

A NUL reaches the database inside a JSON value as an escape sequence. The old `json`
columns stored it; the `jsonb` columns that revision 0002 created reject the whole write.
These pin the serialiser that closes that gap, and pin that it stops there: JSON never
had the text columns' wider cleaning, and a character the caller wrote other than NUL
comes back exactly as written.
"""
import json
import unittest

from sqlalchemy import select

from backend.db.models import ChatMessage, ChatSession, User
from backend.infra.database import engine, serialize_json
from tests.general.postgres_support import postgres_schema

NUL = "\x00"
# A family emoji: three people held together by zero-width joiners.
FAMILY = "\U0001F468\u200d\U0001F469\u200d\U0001F467"
# "é" written as a letter plus a combining accent, not normalised to one code point.
DECOMPOSED = "e\u0301"


class JsonNulTests(unittest.TestCase):
    def setUp(self):
        self.schema = postgres_schema(self, User, ChatSession, ChatMessage)

    def test_a_nul_inside_a_jsonb_value_is_removed_and_everything_else_kept(self):
        sessions = self.schema.sessionmaker()
        with sessions() as db:
            user = User(username="parent", password_hash="x")
            db.add(user)
            db.flush()
            db.add(ChatSession(
                user_id=user.id,
                session_id="s1",
                metadata_json={"title": f"fees{NUL}policy", "tags": [f"a{NUL}b"], "note": FAMILY + DECOMPOSED},
            ))
            db.commit()

        with sessions() as db:
            stored = db.scalars(select(ChatSession.metadata_json)).one()
        self.assertEqual({"title": "feespolicy", "tags": ["ab"], "note": FAMILY + DECOMPOSED}, stored)

    def test_nul_is_removed_from_nested_values_and_keys(self):
        self.assertEqual('{"kq": ["ab"]}', serialize_json({f"k{NUL}q": [f"a{NUL}b"]}))

    def test_nothing_but_nul_is_removed(self):
        """A zero-width joiner holds an emoji together, and a decomposed character is
        still the character the caller wrote. Text columns strip and normalise those;
        JSON never did, and this keeps it that way."""
        self.assertEqual([FAMILY, DECOMPOSED], json.loads(serialize_json([FAMILY, DECOMPOSED])))

    def test_the_application_engine_uses_it(self):
        """The fix is only real if the engine every request goes through is wired to it."""
        self.assertIs(serialize_json, engine.dialect._json_serializer)


if __name__ == "__main__":
    unittest.main()
