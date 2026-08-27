"""Guardian identity, from the HTTP boundary to the tool that uses it.

Three properties are worth protecting here, and they are what these tests are
organised around:

1. Identity is decided once, by verified code, and is immutable thereafter.
2. The bearer token never reaches a log line, a repr, or storage.
3. Absence of identity is a safe, explicit state — not an error, and not a
   half-configured session that reads something it should not.
"""
import asyncio
import unittest

from backend.chat.caller_identity import CallerIdentity
from backend.chat.request_context import ChatRequestContext

TOKEN = "eyJhbGciOiJSUzI1NiJ9.super-secret-bearer-value.signature"


class _Principal:
    """Stands in for whatever the auth layer returns."""

    def __init__(self, username="u", guardian_id="", access_token=""):
        self.username = username
        self.guardian_id = guardian_id
        self.access_token = access_token


class CallerIdentityTests(unittest.TestCase):
    def test_a_parent_needs_both_an_id_and_a_token(self):
        """Either half alone is not a parent session.

        A guardian id with no token cannot be proved to the records facade; a token
        with no id has no subject. Treating either as a parent produces a confusing
        failure deep inside a tool instead of a clear refusal up front.
        """
        self.assertTrue(CallerIdentity("u", "G-1", TOKEN).is_parent)
        self.assertFalse(CallerIdentity("u", "G-1", "").is_parent)
        self.assertFalse(CallerIdentity("u", "", TOKEN).is_parent)
        self.assertFalse(CallerIdentity("u").is_parent)

    def test_for_user_produces_a_non_parent(self):
        identity = CallerIdentity.for_user("someone")
        self.assertEqual("someone", identity.user_id)
        self.assertFalse(identity.is_parent)

    def test_from_principal_reads_a_parent(self):
        identity = CallerIdentity.from_principal(
            _Principal(username="0501234567", guardian_id="G-1", access_token=TOKEN)
        )
        self.assertEqual("0501234567", identity.user_id)
        self.assertEqual("G-1", identity.guardian_id)
        self.assertEqual(TOKEN, identity.guardian_token)
        self.assertTrue(identity.is_parent)

    def test_from_principal_tolerates_a_type_that_predates_guardians(self):
        """Structural, not nominal — so an older or third-party principal still works.

        It simply is not a parent, which is the correct outcome rather than an
        AttributeError at the top of every chat turn.
        """
        class Legacy:
            username = "old"

        identity = CallerIdentity.from_principal(Legacy())
        self.assertEqual("old", identity.user_id)
        self.assertFalse(identity.is_parent)

    def test_from_principal_normalises_none_to_empty(self):
        identity = CallerIdentity.from_principal(
            _Principal(username="u", guardian_id=None, access_token=None)
        )
        self.assertEqual("", identity.guardian_id)
        self.assertEqual("", identity.guardian_token)

    def test_it_is_immutable(self):
        """A tool handed the context must not be able to change whose records it reads."""
        identity = CallerIdentity("u", "G-1", TOKEN)
        with self.assertRaises(Exception):
            identity.guardian_id = "G-2"

    def test_without_credentials_keeps_who_but_drops_the_ability_to_act(self):
        stripped = CallerIdentity("u", "G-1", TOKEN).without_credentials()
        self.assertEqual("G-1", stripped.guardian_id)
        self.assertEqual("", stripped.guardian_token)
        self.assertFalse(stripped.is_parent)

    def test_the_token_never_appears_in_a_repr(self):
        """Reprs reach log lines, tracebacks and captured test output."""
        text = repr(CallerIdentity("u", "G-1", TOKEN))
        self.assertNotIn(TOKEN, text)
        self.assertNotIn("super-secret-bearer-value", text)
        self.assertIn("redacted", text)
        # Still useful for debugging: who it was, and that a token was present.
        self.assertIn("G-1", text)

    def test_a_repr_without_a_token_does_not_claim_one(self):
        self.assertNotIn("redacted", repr(CallerIdentity("u", "G-1", "")))

    def test_the_token_does_not_leak_through_string_formatting(self):
        identity = CallerIdentity("u", "G-1", TOKEN)
        for rendered in (f"{identity!r}", str([identity]), str({"c": identity})):
            self.assertNotIn(TOKEN, rendered)


class ContextWiringTests(unittest.TestCase):
    """What the records tool actually reads off the context."""

    def test_a_context_without_a_caller_is_not_a_parent(self):
        """The default for every existing caller: a job, a test, an anonymous turn."""
        ctx = ChatRequestContext.for_sync(user_id="u", session_id="s")
        self.assertEqual("", ctx.guardian_id)
        self.assertEqual("", ctx.guardian_token)
        self.assertFalse(ctx.is_parent)

    def test_a_parent_caller_reaches_the_context(self):
        ctx = ChatRequestContext.for_sync(
            user_id="u",
            session_id="s",
            caller=CallerIdentity("u", "G-1", TOKEN),
        )
        self.assertEqual("G-1", ctx.guardian_id)
        self.assertEqual(TOKEN, ctx.guardian_token)
        self.assertTrue(ctx.is_parent)

    def test_the_streaming_path_carries_identity_too(self):
        """The two entry points must not diverge.

        A parameter honoured on the sync path and dropped on the streaming one is
        dead in exactly the path real users hit.
        """
        async def build():
            return ChatRequestContext.for_stream(
                user_id="u",
                session_id="s",
                output_queue=asyncio.Queue(),
                caller=CallerIdentity("u", "G-1", TOKEN),
            )

        ctx = asyncio.run(build())
        self.assertEqual("G-1", ctx.guardian_id)
        self.assertTrue(ctx.is_parent)

    def test_a_caller_naming_someone_else_is_refused(self):
        """`user_id` is the storage key.

        A mismatch would write one user's conversation under another's name while
        reading a third party's records — silent, and serious.
        """
        with self.assertRaises(ValueError):
            ChatRequestContext.for_sync(
                user_id="alice", session_id="s", caller=CallerIdentity("bob", "G-1", TOKEN)
            )

    def test_guardian_fields_cannot_be_assigned_on_the_context(self):
        """Read-only views, so a tool cannot promote its own turn."""
        ctx = ChatRequestContext.for_sync(user_id="u", session_id="s")
        for field in ("guardian_id", "guardian_token", "is_parent"):
            with self.subTest(field=field):
                with self.assertRaises(AttributeError):
                    setattr(ctx, field, "G-9")

    def test_closing_the_turn_drops_the_token(self):
        """The context can outlive the request that authorised it.

        A live credential should not outlive the reason it was handed over; who the
        caller was stays readable.
        """
        ctx = ChatRequestContext.for_sync(
            user_id="u", session_id="s", caller=CallerIdentity("u", "G-1", TOKEN)
        )
        ctx.close()
        self.assertEqual("", ctx.guardian_token)
        self.assertEqual("G-1", ctx.guardian_id)
        self.assertFalse(ctx.is_parent)


class ServiceSignatureTests(unittest.TestCase):
    """Both entry points must take identity the same way."""

    def test_both_entry_points_accept_a_keyword_only_caller(self):
        import inspect

        from backend.chat.service import chat_with_agent, chat_with_agent_stream

        for function in (chat_with_agent, chat_with_agent_stream):
            with self.subTest(function=function.__name__):
                parameter = inspect.signature(function).parameters.get("caller")
                self.assertIsNotNone(parameter, "no `caller` parameter")
                self.assertEqual(inspect.Parameter.KEYWORD_ONLY, parameter.kind)
                self.assertIsNone(parameter.default)

    def test_a_caller_is_authoritative_over_the_positional_user_id(self):
        from backend.chat.service import _resolve_caller

        caller, user_id = _resolve_caller(CallerIdentity("real", "G-1", TOKEN), "default_user")
        self.assertEqual("real", user_id)
        self.assertEqual("G-1", caller.guardian_id)

    def test_no_caller_falls_back_to_the_positional_user_id(self):
        from backend.chat.service import _resolve_caller

        caller, user_id = _resolve_caller(None, "someone")
        self.assertEqual("someone", user_id)
        self.assertEqual("someone", caller.user_id)
        self.assertFalse(caller.is_parent)


class RouteWiringTests(unittest.TestCase):
    """The boundary where identity is assembled."""

    def test_the_principal_from_auth_maps_onto_a_caller(self):
        """`AuthenticatedUser` must satisfy `from_principal` structurally.

        These are in different layers and neither imports the other, so nothing but a
        test holds the contract between them.
        """
        from backend.infra.auth import AuthenticatedUser

        principal = AuthenticatedUser(
            username="0501234567", role="parent", guardian_id="G-1", access_token=TOKEN
        )
        identity = CallerIdentity.from_principal(principal)

        self.assertEqual("0501234567", identity.user_id)
        self.assertEqual("G-1", identity.guardian_id)
        self.assertEqual(TOKEN, identity.guardian_token)
        self.assertTrue(identity.is_parent)

    def test_a_staff_principal_produces_a_non_parent(self):
        from backend.infra.auth import AuthenticatedUser

        identity = CallerIdentity.from_principal(
            AuthenticatedUser(username="teacher", role="staff", access_token=TOKEN)
        )
        self.assertFalse(identity.is_parent)

    def test_the_authenticated_user_repr_hides_its_token(self):
        from backend.infra.auth import AuthenticatedUser

        text = repr(
            AuthenticatedUser(username="u", role="parent", guardian_id="G-1", access_token=TOKEN)
        )
        self.assertNotIn(TOKEN, text)

    def test_both_chat_routes_pass_a_caller_through(self):
        """Asserted on the source, because the alternative is booting the whole app.

        The failure this guards against is a route that authenticates correctly and
        then forgets to forward the identity — which looks exactly like a working
        deployment until a parent asks about their child and is told to sign in.
        """
        import inspect

        import backend.api.routes.chat as chat_routes

        source = inspect.getsource(chat_routes)
        self.assertEqual(
            2,
            source.count("caller="),
            "both /chat and /chat/stream must forward a CallerIdentity",
        )
        self.assertIn("CallerIdentity.from_principal", source)


if __name__ == "__main__":
    unittest.main()
