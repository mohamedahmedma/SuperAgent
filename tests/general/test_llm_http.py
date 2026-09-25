"""RAG_FIX_PLAN item 48: a model call hands its connection back instead of closing it.

The OpenAI SDK stops reading a stream at `[DONE]` and closes the response with the end of
the chunked body unread, so httpx discards the connection: under load that was a new
connection for two model calls in three, and against a real provider a TLS handshake each
(+208 ms measured to Groq). Two fixes, one per kind of call:

  * the answer, which must stream, goes through the async client's draining transport;
  * every other role does not stream at all (`backend/llm.py`), because LangChain streams
    any call made under the chat turn's streaming callback, and the sync client has no
    such transport.

These tests run the SDK against a real HTTP/1.1 server that counts connections, so they
fail if either the SDK's behaviour or our fix changes.
"""
import asyncio
import json
import socket
import threading
import time
import unittest
from typing import TypedDict

import openai
from pydantic import BaseModel

from backend.llm import ROLES, STREAMED_ROLES, sampling
from backend.llm_http import DRAIN_MAX_SECONDS, ProviderHttpClients


class _ProviderServer:
    """A keep-alive HTTP/1.1 chat-completions server that counts connections.

    A streamed request gets `chunks` answer chunks (or, when it asked for structured
    output, the JSON below in pieces) as chunked SSE, then `[DONE]`, then the chunked
    terminator — separate writes, as a provider sends them. `trickle`
    seconds between chunks lets a test abandon a stream mid-answer. A plain request gets
    one JSON completion whose content is `{"ok": true}`, so it also serves structured
    output. `streamed` counts the requests that asked to stream.
    """

    def __init__(self, chunks=3, trickle=0.0):
        self.chunks, self.trickle, self.connections, self.streamed = chunks, trickle, 0, 0
        self._sock = socket.socket()
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(64)
        self.url = f"http://127.0.0.1:{self._sock.getsockname()[1]}/v1"
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            self.connections += 1
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    @staticmethod
    def _read_request(conn):
        blank = b"\r\n\r\n"
        data = b""
        while blank not in data:
            chunk = conn.recv(65536)
            if not chunk:
                return None
            data += chunk
        head, _, body = data.partition(blank)
        length = next((int(l.split(b":", 1)[1]) for l in head.split(b"\r\n")
                       if l.lower().startswith(b"content-length:")), 0)
        while len(body) < length:
            body += conn.recv(65536)
        return json.loads(body or b"{}")

    def _serve(self, conn):
        def chunk(payload: bytes) -> bytes:
            return f"{len(payload):x}\r\n".encode() + payload + b"\r\n"

        try:
            while (request := self._read_request(conn)) is not None:  # keep-alive
                if not request.get("stream"):
                    completion = json.dumps({
                        "id": "c", "object": "chat.completion", "created": 0, "model": "m",
                        "choices": [{"index": 0, "finish_reason": "stop", "message": {
                            "role": "assistant", "content": '{"ok": true}'}}],
                    }).encode()
                    conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                                 + f"Content-Length: {len(completion)}\r\n\r\n".encode()
                                 + completion)
                    continue
                self.streamed += 1
                conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                             b"Transfer-Encoding: chunked\r\n\r\n")
                # Structured output streams its JSON, so a role that regresses into
                # streaming fails on the counts below rather than on a parse error.
                pieces = (['{"ok"', ": ", "true}"] if request.get("response_format")
                          else [f"w{i} " for i in range(self.chunks)])
                for i, piece in enumerate(pieces):
                    delta = {"role": "assistant", "content": piece} if i == 0 else {"content": piece}
                    event = {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
                             "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
                    conn.sendall(chunk(f"data: {json.dumps(event)}\n\n".encode()))
                    if self.trickle:
                        time.sleep(self.trickle)
                conn.sendall(chunk(b"data: [DONE]\n\n"))
                conn.sendall(b"0\r\n\r\n")
        except OSError:
            pass
        finally:
            conn.close()

    def close(self):
        self._sock.close()


async def _stream_n(client, n, *, abandon=False):
    for _ in range(n):
        stream = await client.chat.completions.create(
            model="m", stream=True, messages=[{"role": "user", "content": "q"}])
        async for _ in stream:
            if abandon:
                break
        await stream.close()


class ConnectionReuseTests(unittest.TestCase):
    def test_the_sdk_on_its_own_discards_a_connection_per_stream(self):
        """The defect, pinned. If this ever passes with 1 the SDK has fixed it and the
        transport in backend/llm_http.py can go."""
        server = _ProviderServer()
        try:
            async def main():
                client = openai.AsyncOpenAI(api_key="x", base_url=server.url, max_retries=0)
                await _stream_n(client, 5)
                await client.close()
            asyncio.run(main())
            self.assertEqual(5, server.connections)
        finally:
            server.close()

    def test_through_the_shared_clients_one_connection_serves_every_stream(self):
        server = _ProviderServer()
        try:
            async def main():
                shared = ProviderHttpClients()
                client = openai.AsyncOpenAI(api_key="x", base_url=server.url, max_retries=0,
                                            http_client=shared.async_client)
                await _stream_n(client, 5)
                await shared.async_client.aclose()
            asyncio.run(main())
            self.assertEqual(1, server.connections)
        finally:
            server.close()

    def test_a_stream_abandoned_mid_answer_is_closed_not_read_to_the_end(self):
        """A parent pressing stop must not cost the full generation: the drain gives up
        within its budget and the connection is discarded, as before."""
        server = _ProviderServer(chunks=200, trickle=0.02)  # ~4 s to finish if drained
        try:
            async def main():
                shared = ProviderHttpClients()
                client = openai.AsyncOpenAI(api_key="x", base_url=server.url, max_retries=0,
                                            http_client=shared.async_client)
                stream = await client.chat.completions.create(
                    model="m", stream=True, messages=[{"role": "user", "content": "q"}])
                async for _ in stream:
                    break
                # Only the close is timed: opening a first connection to a fresh server
                # can take a second on its own, and is not what is under test.
                started = time.perf_counter()
                await stream.close()
                elapsed = time.perf_counter() - started
                await shared.async_client.aclose()
                return elapsed
            elapsed = asyncio.run(main())
            self.assertLess(elapsed, DRAIN_MAX_SECONDS + 0.5)
        finally:
            server.close()

    def test_cancelling_a_streaming_turn_still_cancels_it(self):
        server = _ProviderServer(chunks=200, trickle=0.02)
        try:
            async def main():
                shared = ProviderHttpClients()
                client = openai.AsyncOpenAI(api_key="x", base_url=server.url, max_retries=0,
                                            http_client=shared.async_client)
                task = asyncio.create_task(_stream_n(client, 1))
                await asyncio.sleep(0.3)
                task.cancel()
                try:
                    await task
                    return False
                except asyncio.CancelledError:
                    return True
                finally:
                    await shared.async_client.aclose()
            self.assertTrue(asyncio.run(main()))
        finally:
            server.close()


class _Grade(BaseModel):
    ok: bool


class _State(TypedDict):
    out: str


def _in_a_streaming_turn(call):
    """Run `call` inside a LangGraph node streamed in "messages" mode, as a chat turn is.

    That run attaches LangGraph's streaming callback handler to every model call made
    beneath it, which is what made LangChain stream the grader.
    """
    from langgraph.graph import START, StateGraph

    def node(state):
        call()
        return {"out": "done"}

    graph = StateGraph(_State)
    graph.add_node("node", node)
    graph.add_edge(START, "node")
    list(graph.compile().stream({"out": ""}, stream_mode="messages"))


def _model(server, clients, role):
    from langchain.chat_models import init_chat_model

    return init_chat_model(model="m", model_provider="openai", api_key="x",
                           base_url=server.url, **clients.model_kwargs(), **sampling(role))


class InternalCallsDoNotStreamTests(unittest.TestCase):
    def test_the_sync_sdk_on_its_own_discards_a_connection_per_stream(self):
        """Why no role but the answer may stream: the sync client has no draining
        transport, so every sync stream costs its connection."""
        server = _ProviderServer()
        try:
            client = openai.OpenAI(api_key="x", base_url=server.url, max_retries=0)
            for _ in range(5):
                for _ in client.chat.completions.create(
                        model="m", stream=True, messages=[{"role": "user", "content": "q"}]):
                    pass
            client.close()
            self.assertEqual(5, server.connections)
        finally:
            server.close()

    def test_only_the_answer_role_streams(self):
        self.assertEqual({"answer"}, set(STREAMED_ROLES))
        for role in ROLES:
            with self.subTest(role=role):
                self.assertEqual(role not in STREAMED_ROLES,
                                 sampling(role).get("disable_streaming", False))

    def test_a_grade_inside_a_streaming_turn_is_one_plain_call_on_one_connection(self):
        server = _ProviderServer()
        clients = ProviderHttpClients()
        try:
            grader = _model(server, clients, "grade").with_structured_output(_Grade)
            grades = []
            _in_a_streaming_turn(lambda: grades.extend(grader.invoke("grade") for _ in range(5)))
            self.assertEqual([_Grade(ok=True)] * 5, grades)
            self.assertEqual(0, server.streamed)
            self.assertEqual(1, server.connections)
        finally:
            clients.close()
            server.close()

    def test_the_answer_still_streams_inside_a_streaming_turn(self):
        """The control: the harness above would have caught a role that streams."""
        server = _ProviderServer()
        clients = ProviderHttpClients()
        try:
            answer = _model(server, clients, "answer")
            _in_a_streaming_turn(lambda: answer.invoke("q"))
            self.assertEqual(1, server.streamed)
        finally:
            clients.close()
            server.close()


class WiringTests(unittest.TestCase):
    def test_the_composition_root_hands_every_role_the_same_clients(self):
        from backend.composition import Services

        services = Services()
        kwargs = services.provider_http.model_kwargs()
        self.assertIs(kwargs["http_async_client"], services.provider_http.async_client)
        self.assertIs(services.provider_http, services.provider_http)  # one per container

    def test_the_model_factory_passes_them_to_what_it_builds(self):
        from backend.llm_models import ChatModelFactory

        built = {}

        def build(**kwargs):
            built.update(kwargs)
            return object()

        marker = object()
        factory = ChatModelFactory(environ={"ARK_API_KEY": "k", "GRADE_MODEL": "g"},
                                   build=build, http_kwargs={"http_async_client": marker})
        factory.grader()
        self.assertIs(marker, built["http_async_client"])


if __name__ == "__main__":
    unittest.main()
