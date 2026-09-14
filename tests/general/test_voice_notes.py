"""Voice notes: stored once, transcribed, and played back from any device.

A recording used to live only in the browser tab that made it, while the backend received
the literal text "[Voice recording attached]" and answered that. Reopening the chat lost
the player, and the assistant never heard a word. These pin the pieces that replace it:
the service that keeps and transcribes a recording, the transcriber over the provider's
Whisper endpoint, the routes, and the message that carries its note through a reload.
"""
import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.assets.blobs import LocalBlobStore
from backend.chat.attachments import ChatAttachments, VoiceNoteLimits, VoiceNoteRejected
from backend.chat.storage import ConversationStorage, MessageToStore
from backend.chat.transcription import NoTranscriber, Transcript, WhisperTranscriber, build_transcriber
from backend.composition import Services
from backend.infra.auth import AuthenticatedUser, get_current_user
from tests.general.postgres_support import postgres_schema
from tests.general.test_asset_delivery import DictCache

WEBM = b"\x1aE\xdf\xa3" + bytes(range(256)) * 8  # an EBML header followed by noise


class FakeTranscriber:
    def __init__(self, text: str = "", status: str = Transcript.OK) -> None:
        self.text, self.status, self.calls = text, status, []

    def transcribe(self, data: bytes, content_type: str) -> Transcript:
        self.calls.append((len(data), content_type))
        return Transcript(self.text, self.status)


class _VoiceNoteTestCase(unittest.TestCase):
    def setUp(self):
        from backend.db.models import ChatAttachment, ChatMessage, ChatSession, User

        self.schema = postgres_schema(self, User, ChatSession, ChatMessage, ChatAttachment)
        db = self.schema.sessionmaker()()
        db.add_all([User(username="parent", password_hash="x"), User(username="other", password_hash="x")])
        db.commit()
        db.close()

        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.blobs = LocalBlobStore(Path(self._tmp.name))
        self.transcriber = FakeTranscriber("إمتى الباص بييجي؟")
        self.notes = self._service(self.transcriber)

    def _service(self, transcriber, **limits) -> ChatAttachments:
        return ChatAttachments(
            unit_of_work=self.schema.unit_of_work,
            blob_store=self.blobs,
            transcriber=transcriber,
            limits=VoiceNoteLimits(**limits) if limits else None,
        )


class StoringAVoiceNote(_VoiceNoteTestCase):
    def test_a_recording_is_kept_once_transcribed_and_recorded_for_its_owner(self):
        note = self.notes.store_voice_note("parent", WEBM, "audio/webm;codecs=opus", duration_ms=4200)

        self.assertEqual("voice", note.kind)
        self.assertEqual("audio/webm", note.content_type, "the codec parameter is not part of the container")
        self.assertEqual(len(WEBM), note.byte_size)
        self.assertEqual(4200, note.duration_ms)
        self.assertEqual("إمتى الباص بييجي؟", note.transcript)
        self.assertEqual(Transcript.OK, note.transcript_status)
        self.assertTrue(self.blobs.exists(note.sha256, "audio/webm"), "the bytes are in the blob store")
        self.assertEqual(WEBM, self.notes.read_bytes(note))
        self.assertEqual([(len(WEBM), "audio/webm")], self.transcriber.calls)
        self.assertEqual(note, self.notes.get("parent", note.id))

    def test_the_same_recording_sent_twice_is_two_notes_over_one_blob(self):
        first = self.notes.store_voice_note("parent", WEBM, "audio/webm")
        second = self.notes.store_voice_note("parent", WEBM, "audio/webm")

        self.assertNotEqual(first.id, second.id)
        self.assertEqual(first.storage_uri, second.storage_uri, "content-addressed: stored once")

    def test_a_note_is_read_only_by_the_account_that_sent_it(self):
        note = self.notes.store_voice_note("parent", WEBM, "audio/webm")

        self.assertIsNone(self.notes.get("other", note.id))
        self.assertEqual({}, self.notes.get_many("other", [note.id]))
        self.assertEqual({note.id: note}, self.notes.get_many("parent", [note.id, "", "ghost"]))

    def test_an_unknown_user_cannot_store_a_note(self):
        with self.assertRaises(LookupError):
            self.notes.store_voice_note("stranger", WEBM, "audio/webm")

    def test_a_recording_outside_the_limits_is_refused_by_reason(self):
        cases = [
            (b"", "audio/webm", 0, VoiceNoteRejected.EMPTY, {}),
            (WEBM, "audio/webm", 0, VoiceNoteRejected.TOO_LARGE, {"max_bytes": 16}),
            (WEBM, "video/mp4", 0, VoiceNoteRejected.UNSUPPORTED_TYPE, {}),
            (WEBM, "text/plain", 0, VoiceNoteRejected.UNSUPPORTED_TYPE, {}),
            (WEBM, "audio/webm", 6 * 60 * 1000, VoiceNoteRejected.TOO_LONG, {}),
        ]
        for data, content_type, duration, reason, limits in cases:
            with self.subTest(reason=reason):
                with self.assertRaises(VoiceNoteRejected) as caught:
                    self._service(self.transcriber, **limits).store_voice_note("parent", data, content_type, duration)
                self.assertEqual(reason, caught.exception.reason)
        self.assertEqual([], self.transcriber.calls, "nothing refused reaches the transcriber")

    def test_a_recording_is_kept_even_when_nothing_can_transcribe_it(self):
        """No model configured, or the call failed: the parent keeps the recording and
        is told there is no transcript, rather than losing the note."""
        note = self._service(NoTranscriber()).store_voice_note("parent", WEBM, "audio/webm")

        self.assertIsNone(note.transcript)
        self.assertEqual(Transcript.UNAVAILABLE, note.transcript_status)
        self.assertEqual(WEBM, self.notes.read_bytes(note))

    def test_a_message_keeps_its_recording_through_storage(self):
        """The link from the stored question to its note survives the write, the page
        read and the cache."""
        note = self.notes.store_voice_note("parent", WEBM, "audio/webm")
        storage = ConversationStorage(unit_of_work=self.schema.unit_of_work, cache=DictCache())
        storage.append("parent", "s", [MessageToStore("human", note.transcript or "", attachment_id=note.id)])
        storage.append("parent", "s", [MessageToStore("ai", "07:30.")])

        for attempt in ("cold", "warm"):
            with self.subTest(cache=attempt):
                page = storage.get_session_page("parent", "s", limit=10)
                self.assertEqual(note.id, page["messages"][0].get("attachment_id"))
                self.assertNotIn("attachment_id", page["messages"][1], "an answer carries no note")


class WhisperTranscriberTests(unittest.TestCase):
    def _client(self, text=" إمتى الباص؟ ", error=None):
        create = Mock(side_effect=error) if error else Mock(return_value=SimpleNamespace(text=text))
        return SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create))), create

    def test_the_recording_is_posted_with_a_filename_the_endpoint_can_read(self):
        client, create = self._client()
        transcriber = WhisperTranscriber(
            api_key="k", model="openai/whisper-large-v3", client_factory=lambda: client
        )

        transcript = transcriber.transcribe(WEBM, "audio/webm;codecs=opus")

        self.assertEqual(Transcript("إمتى الباص؟", Transcript.OK), transcript)
        kwargs = create.call_args.kwargs
        self.assertEqual("openai/whisper-large-v3", kwargs["model"])
        name, data, content_type = kwargs["file"]
        self.assertTrue(name.endswith(".webm"), "the endpoint infers the container from the name")
        self.assertEqual((WEBM, "audio/webm"), (data, content_type))

    def test_a_failed_call_is_unavailable_not_an_error(self):
        client, _ = self._client(error=RuntimeError("502 from the provider"))
        transcriber = WhisperTranscriber(api_key="k", model="m", client_factory=lambda: client)

        with self.assertLogs("backend.chat.transcription", level="WARNING"):
            transcript = transcriber.transcribe(WEBM, "audio/webm")
        self.assertEqual(Transcript("", Transcript.UNAVAILABLE), transcript)

    def test_silence_is_empty_and_no_bytes_are_not_sent(self):
        client, create = self._client(text="   ")
        transcriber = WhisperTranscriber(api_key="k", model="m", client_factory=lambda: client)

        self.assertEqual(Transcript.EMPTY, transcriber.transcribe(WEBM, "audio/webm").status)
        self.assertEqual(Transcript.EMPTY, transcriber.transcribe(b"", "audio/webm").status)
        self.assertEqual(1, create.call_count)

    def test_the_transcriber_is_built_from_the_resolved_provider_names(self):
        """`TRANSCRIPTION_MODEL` is what the provider block resolves onto — the same way
        `MODEL` and `GRADE_MODEL` reach the chat models."""
        configured = build_transcriber({"TRANSCRIPTION_MODEL": "openai/whisper-large-v3", "ARK_API_KEY": "k", "BASE_URL": "https://api.together.xyz/v1"})
        self.assertIsInstance(configured, WhisperTranscriber)
        self.assertEqual("openai/whisper-large-v3", configured.model)

        self.assertIsInstance(build_transcriber({"ARK_API_KEY": "k"}), NoTranscriber)
        self.assertIsInstance(build_transcriber({"TRANSCRIPTION_MODEL": "m"}), NoTranscriber)

    def test_the_provider_block_can_name_the_transcription_model(self):
        from backend.llm_provider import PROVIDERS, resolve

        resolution = resolve(PROVIDERS["together"], {
            "LLM_PROVIDER": "together", "TOGETHER_API_KEY": "k", "TOGETHER_TRANSCRIPTION_MODEL": "openai/whisper-large-v3",
        })
        self.assertEqual("openai/whisper-large-v3", resolution.values["TRANSCRIPTION_MODEL"])


class VoiceNoteRouteTests(_VoiceNoteTestCase):
    def setUp(self):
        super().setUp()
        from backend.api.routes.attachments import router as attachments_router
        from backend.api.routes.sessions import router as sessions_router

        self.storage = ConversationStorage(unit_of_work=self.schema.unit_of_work, cache=DictCache())
        self.services = Services(attachments=self.notes, conversations=self.storage)
        self.client = self._client_for("parent")

    def _client_for(self, username: str) -> TestClient:
        from backend.api.routes.attachments import router as attachments_router
        from backend.api.routes.sessions import router as sessions_router

        app = FastAPI()
        app.state.services = self.services
        app.include_router(attachments_router)
        app.include_router(sessions_router)
        app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(username=username, role="user")
        return TestClient(app)

    def _upload(self, client=None, data=WEBM, content_type="audio/webm", duration_ms="4200"):
        return (client or self.client).post(
            "/chat/attachments",
            files={"file": ("voice-note.webm", io.BytesIO(data), content_type)},
            data={"duration_ms": duration_ms},
        )

    def test_an_upload_returns_the_note_and_its_transcript(self):
        response = self._upload()

        self.assertEqual(201, response.status_code, response.text)
        body = response.json()
        self.assertEqual("voice", body["kind"])
        self.assertEqual(f"/chat/attachments/{body['id']}", body["url"])
        self.assertEqual("إمتى الباص بييجي؟", body["transcript"])
        self.assertEqual("ok", body["transcript_status"])
        self.assertEqual(4200, body["duration_ms"])

    def test_the_bytes_come_back_for_the_owner_with_a_strong_etag(self):
        url = self._upload().json()["url"]

        first = self.client.get(url)
        self.assertEqual(200, first.status_code)
        self.assertEqual("audio/webm", first.headers["content-type"])
        self.assertEqual(WEBM, first.content)
        self.assertIn("private", first.headers["cache-control"])

        again = self.client.get(url, headers={"If-None-Match": first.headers["etag"]})
        self.assertEqual(304, again.status_code)

    def test_another_account_cannot_play_or_reference_the_note(self):
        url = self._upload().json()["url"]
        self.assertEqual(404, self._client_for("other").get(url).status_code)
        self.assertEqual(404, self.client.get("/chat/attachments/ghost").status_code)

    def test_a_refused_recording_names_why(self):
        self.assertEqual(415, self._upload(content_type="video/mp4").status_code)
        self.assertEqual(400, self._upload(data=b"").status_code)

        self.services = Services(attachments=self._service(self.transcriber, max_bytes=16), conversations=self.storage)
        self.assertEqual(413, self._upload(self._client_for("parent")).status_code)

    def test_a_reopened_conversation_carries_its_voice_notes(self):
        note = self._upload().json()
        self.storage.append("parent", "s", [MessageToStore("human", note["transcript"], attachment_id=note["id"])])
        self.storage.append("parent", "s", [MessageToStore("ai", "07:30.")])

        page = self.client.get("/sessions/s").json()

        self.assertEqual(note["id"], page["messages"][0]["attachment"]["id"])
        self.assertEqual(note["url"], page["messages"][0]["attachment"]["url"])
        self.assertEqual(note["transcript"], page["messages"][0]["content"])
        self.assertIsNone(page["messages"][1]["attachment"])


if __name__ == "__main__":
    unittest.main()
