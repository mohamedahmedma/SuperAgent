"""Speech to text for the voice notes parents send.

One protocol, two implementations. `WhisperTranscriber` posts the recording to the
OpenAI-compatible `/audio/transcriptions` endpoint every provider this backend can be
pointed at serves — Together and Groq both do — with the same key and endpoint the text
models use; only the model id is its own (`TRANSCRIPTION_MODEL`, or the provider block's
`<PREFIX>_TRANSCRIPTION_MODEL`, see backend/llm_provider.py). `NoTranscriber` is what a
deployment that names no model gets: the note is stored and played back, and the client
is told there is no transcript to send.

Never raises into the request. A transcriber that is down costs the parent a transcript,
which the client reports and asks them to type; it must not cost them the recording.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol

logger = logging.getLogger(__name__)

#: The extension the transcription endpoint infers the container from. It reads the
#: filename, not the content type, so a name is built for every upload.
_EXTENSIONS = {
    "audio/webm": ".webm",
    "audio/ogg": ".ogg",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/aac": ".aac",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
}


@dataclass(frozen=True)
class Transcript:
    """What the recording said, and whether that is known.

    `status` is "ok" when `text` is the transcript, "empty" when the model heard nothing
    worth writing (silence, noise), and "unavailable" when no transcript could be made —
    no model configured, or the call failed. The last two look the same to a reader of
    `text` and mean different things to the parent: one asks them to record again, the
    other to type.
    """

    text: str
    status: str

    OK = "ok"
    EMPTY = "empty"
    UNAVAILABLE = "unavailable"


class Transcriber(Protocol):
    def transcribe(self, data: bytes, content_type: str) -> Transcript:
        """The recording's words. Never raises."""
        ...


class NoTranscriber:
    """The deployment has named no speech-to-text model."""

    def transcribe(self, data: bytes, content_type: str) -> Transcript:
        return Transcript("", Transcript.UNAVAILABLE)


class WhisperTranscriber:
    """`/audio/transcriptions` on an OpenAI-compatible endpoint.

    `client_factory` builds the OpenAI client and is injectable so a test can hand over a
    stand-in without a network; production leaves it to build the real one, lazily, so
    constructing the service does not import the client library.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: Optional[str] = None,
        timeout_seconds: float = 60.0,
        client_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url or None
        self._timeout = timeout_seconds
        self._client_factory = client_factory
        self._client: Any = None

    @property
    def model(self) -> str:
        return self._model

    def _get_client(self) -> Any:
        if self._client is None:
            if self._client_factory is not None:
                self._client = self._client_factory()
            else:
                from openai import OpenAI

                self._client = OpenAI(api_key=self._api_key, base_url=self._base_url, timeout=self._timeout)
        return self._client

    def transcribe(self, data: bytes, content_type: str) -> Transcript:
        if not data:
            return Transcript("", Transcript.EMPTY)
        filename = f"voice-note{_EXTENSIONS.get(_bare(content_type), '.webm')}"
        try:
            result = self._get_client().audio.transcriptions.create(
                model=self._model,
                file=(filename, data, _bare(content_type) or "application/octet-stream"),
                response_format="json",
            )
        except Exception:
            # Logged, not raised: the recording is already stored, and the parent is told
            # to type. The model id names what to look at; the audio itself is not logged.
            logger.warning("transcription with %s failed; the voice note keeps no transcript", self._model, exc_info=True)
            return Transcript("", Transcript.UNAVAILABLE)
        text = str(getattr(result, "text", "") or "").strip()
        return Transcript(text, Transcript.OK if text else Transcript.EMPTY)


def _bare(content_type: str) -> str:
    """`audio/webm;codecs=opus` -> `audio/webm`. The recorder reports the codec too."""
    return (content_type or "").split(";", 1)[0].strip().lower()


def build_transcriber(environ: Optional[Mapping[str, str]] = None) -> Transcriber:
    """The transcriber this deployment configured, read from the same names the text
    models use once `backend.llm_provider` has resolved the provider block onto them."""
    source = os.environ if environ is None else environ
    model = (source.get("TRANSCRIPTION_MODEL") or "").strip()
    api_key = (source.get("ARK_API_KEY") or "").strip()
    if not model or not api_key:
        return NoTranscriber()
    return WhisperTranscriber(api_key=api_key, model=model, base_url=(source.get("BASE_URL") or "").strip())


__all__ = ["NoTranscriber", "Transcriber", "Transcript", "WhisperTranscriber", "build_transcriber"]
