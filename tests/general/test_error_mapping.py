"""The one place a backend error becomes an HTTP response.

The property this refactor rests on is that moving a route from `raise HTTPException(...)` to
a service raising a domain error changes NOTHING a client can see. So the central test here
builds both kinds of route side by side and compares what they answer, byte for byte, rather
than restating the expected body from memory.
"""

import unittest

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from backend.agent.chat.attachments import VoiceNoteRejected
from backend.api.errors import install_error_handlers, provider_failure, status_for
from backend.domain.errors import (
    BackendError,
    FeatureDisabled,
    Forbidden,
    InvalidInput,
    NotFound,
    OperationFailed,
    TurnRefused,
)


def _raising(error):
    """An endpoint that raises `error`, closed over rather than taken as a parameter.

    A default argument would make FastAPI treat it as a query parameter and deep-copy it,
    and an exception whose `__init__` does not match its `args` cannot be copied.
    """

    def endpoint():
        raise error

    return endpoint


def _client(*routes) -> TestClient:
    """A bare app with the handlers installed, one GET route per (path, exception) given."""
    app = FastAPI()
    install_error_handlers(app)
    for path, error in routes:
        app.add_api_route(path, _raising(error), methods=["GET"])
    return TestClient(app)


class WireFormatTests(unittest.TestCase):
    def test_a_domain_error_answers_exactly_what_the_http_exception_it_replaced_did(self):
        cases = [
            (NotFound("Session does not exist"), HTTPException(404, "Session does not exist")),
            (
                InvalidInput("Filename cannot be empty"),
                HTTPException(400, "Filename cannot be empty"),
            ),
            (
                Forbidden("This account cannot send attachments."),
                HTTPException(403, "This account cannot send attachments."),
            ),
            (
                OperationFailed("Failed to read chunks", RuntimeError("milvus down")),
                HTTPException(500, "Failed to read chunks: milvus down"),
            ),
        ]
        for domain_error, http_error in cases:
            with self.subTest(domain_error=type(domain_error).__name__):
                client = _client(("/new", domain_error), ("/old", http_error))
                new, old = client.get("/new"), client.get("/old")
                self.assertEqual(old.status_code, new.status_code)
                self.assertEqual(old.json(), new.json())
                self.assertEqual(old.headers["content-type"], new.headers["content-type"])

    def test_detail_stays_a_string_because_the_web_app_reads_it_as_one(self):
        body = _client(("/x", NotFound("Asset not found"))).get("/x").json()
        self.assertEqual({"detail": "Asset not found"}, body)
        self.assertIsInstance(body["detail"], str)

    def test_a_refused_turn_says_when_to_try_again(self):
        response = _client(("/x", TurnRefused("Slow down", retry_after_seconds=7))).get("/x")
        self.assertEqual(429, response.status_code)
        self.assertEqual("7", response.headers["retry-after"])
        self.assertEqual({"detail": "Slow down"}, response.json())

    def test_a_rejected_voice_note_keeps_its_object_envelope(self):
        """The recorder reads `code` out of this one, so it keeps the shape it always had."""
        cases = [
            (VoiceNoteRejected.TOO_LARGE, 413),
            (VoiceNoteRejected.UNSUPPORTED_TYPE, 415),
            (VoiceNoteRejected.EMPTY, 400),
            (VoiceNoteRejected.TOO_LONG, 400),
        ]
        for reason, expected in cases:
            with self.subTest(reason=reason):
                response = _client(("/x", VoiceNoteRejected(reason, "nope"))).get("/x")
                self.assertEqual(expected, response.status_code)
                self.assertEqual({"detail": {"code": reason, "message": "nope"}}, response.json())


class StatusTests(unittest.TestCase):
    def test_each_error_maps_to_the_status_its_routes_always_used(self):
        self.assertEqual(400, status_for(InvalidInput("x")))
        self.assertEqual(403, status_for(Forbidden("x")))
        self.assertEqual(404, status_for(NotFound("x")))
        self.assertEqual(429, status_for(TurnRefused("x", retry_after_seconds=1)))
        self.assertEqual(500, status_for(OperationFailed("x")))

    def test_status_is_inherited_so_a_new_subclass_needs_no_table_edit(self):
        self.assertEqual(404, status_for(FeatureDisabled("Asset support is disabled.")))

        class NoSuchPair(NotFound):
            code = "no_such_pair"

        self.assertEqual(404, status_for(NoSuchPair("x")))

    def test_an_unmapped_error_is_a_client_error_not_a_crash(self):
        class Unmapped(BackendError):
            pass

        self.assertEqual(400, status_for(Unmapped("x")))

    def test_every_error_carries_a_code(self):
        for klass in (
            BackendError,
            InvalidInput,
            NotFound,
            FeatureDisabled,
            Forbidden,
            OperationFailed,
        ):
            with self.subTest(klass=klass.__name__):
                self.assertTrue(klass.code)
        self.assertNotEqual(NotFound.code, FeatureDisabled.code)


class OperationFailedTests(unittest.TestCase):
    """The message is the one the routes built by hand, in each of the three forms they used."""

    def test_operation_and_cause(self):
        error = OperationFailed("Failed to retrieve document list", ValueError("boom"))
        self.assertEqual("Failed to retrieve document list: boom", error.message)

    def test_operation_alone(self):
        error = OperationFailed("Document processing failed: could not extract content")
        self.assertEqual("Document processing failed: could not extract content", error.message)

    def test_cause_alone(self):
        self.assertEqual("boom", OperationFailed(cause=ValueError("boom")).message)

    def test_the_cause_is_kept_for_the_log(self):
        cause = ValueError("boom")
        self.assertIs(cause, OperationFailed("x", cause).cause)


class ProviderFailureTests(unittest.TestCase):
    """Moved verbatim from the chat route; pinned so the move changed nothing."""

    def test_quota_is_a_429_with_the_explanation(self):
        error = provider_failure(RuntimeError("Error code: 429 - rate limited"))
        self.assertEqual(429, error.status_code)
        self.assertIn("rate limiting/quota limits (429)", error.detail)
        self.assertIn("Original error: Error code: 429 - rate limited", error.detail)

    def test_credentials_keep_their_status_and_message(self):
        for code in (401, 403):
            with self.subTest(code=code):
                error = provider_failure(RuntimeError(f"Error code: {code} - denied"))
                self.assertEqual(code, error.status_code)
                self.assertEqual(f"Error code: {code} - denied", error.detail)

    def test_any_other_named_status_is_passed_through(self):
        error = provider_failure(RuntimeError("Error code: 503 - overloaded"))
        self.assertEqual(503, error.status_code)

    def test_no_status_named_is_a_500(self):
        error = provider_failure(RuntimeError("connection reset"))
        self.assertEqual(500, error.status_code)
        self.assertEqual("connection reset", error.detail)


if __name__ == "__main__":
    unittest.main()
