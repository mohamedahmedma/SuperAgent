"""JSON values are cleaned before they are serialised, because JSONB refuses a NUL.

Text columns have always had NUL stripped by the engine's cursor listener. A JSON value
reaches that listener already serialised, with the NUL turned into an escape sequence
the filter does not match. The old `json` columns stored it; the `jsonb` columns that
revision 0002 created reject the whole write. These pin the serialiser that closes the
gap, on the columns and on the engine every request goes through.
"""
import unittest

from sqlalchemy import select

from backend.db.models import ChatMessage, ChatSession, User
from backend.infra.database import engine, serialize_json
from tests.general.postgres_support import postgres_schema

NUL = "\x00"


class JsonNulTests(unittest.TestCase):
    def setUp(self):
        self.schema = postgres_schema(self, User, ChatSession, ChatMessage)

    def test_a_nul_inside_a_jsonb_value_is_stripped_rather_than_rejected(self):
        sessions = self.schema.sessionmaker()
        with sessions() as db:
            user = User(username="parent", password_hash="x")
            db.add(user)
            db.flush()
            db.add(ChatSession(
                user_id=user.id,
                session_id="s1",
                metadata_json={"title": f"fees{NUL}policy", "tags": [f"a{NUL}b"]},
            ))
            db.commit()

        with sessions() as db:
            stored = db.scalars(select(ChatSession.metadata_json)).one()
        self.assertEqual({"title": "feespolicy", "tags": ["ab"]}, stored)

    def test_nested_values_are_cleaned_like_text(self):
        self.assertEqual('{"q": ["ab"]}', serialize_json({"q": [f"a{NUL}b"]}))

    def test_the_application_engine_uses_it(self):
        """The fix is only real if the engine every request goes through is wired to it."""
        self.assertIs(serialize_json, engine.dialect._json_serializer)


if __name__ == "__main__":
    unittest.main()
