"""Which provider the running system actually contacts.

`test_llm_provider.py` proves the resolution is right: given an environment, the correct
values land in `os.environ`. That is necessary, and it is not the question a deployment
asks. The question a deployment asks is "where did the packet go", and in production the
two came apart -- the configuration read `LLM_PROVIDER=together` with a complete
`TOGETHER_*` block, every unit test passed, and Groq quota errors kept arriving.

So these tests assert on the WIRE. Every model client in this backend is an
openai-compatible one built by `init_chat_model`, and every one of them sends through
`httpx` -- so recording the URL of each outbound request captures all of them at once,
including the ones nobody thought to check.

The distinction that matters, and the one the production incident turned on:

    the TEXT models follow LLM_PROVIDER     ->  ARK_API_KEY / BASE_URL
    the VISION model does NOT               ->  VISION_API_KEY / VISION_BASE_URL

The vision trio is deliberately generic so that it survives a provider switch: a
deployment whose text model is on one provider and whose vision model is on another is
the normal case, and `backend/assets/vision.py` documents it. The cost of that decision
is exactly the reported symptom -- moving the text models to Together leaves every figure
read on Groq, still spending a Groq quota, while every unit test agrees the switch worked.
`VisionTests` exists to make that visible rather than surprising.
"""
from __future__ import annotations

import os
import unittest
from urllib.parse import urlsplit

import httpx

from backend.env import load_env

load_env()

from backend.assets.vision import resolve_vision_credentials  # noqa: E402
from backend.llm_provider import active_provider, resolve  # noqa: E402
from tests.general.integration_support import requires_llm  # noqa: E402

#: Hosts that are an LLM provider rather than our own infrastructure. Matched as an exact
#: host or a suffix, so a lookalike domain cannot pass as one of these.
PROVIDER_HOSTS = {
    "api.groq.com": "groq",
    "api.together.ai": "together",
    "api.together.xyz": "together",
    "openrouter.ai": "openrouter",
    "api.openai.com": "openai",
}


def provider_for(url: str) -> str:
    """The provider a URL belongs to, or "" for anything that is not one."""
    host = (urlsplit(url).hostname or "").lower()
    for candidate, name in PROVIDER_HOSTS.items():
        if host == candidate or host.endswith("." + candidate):
            return name
    return ""


class RecordingTransport:
    """Records every outbound httpx request, then lets it proceed.

    Patched onto `httpx.Client.send` and its async twin rather than onto anything in
    langchain: the point is to catch calls this test does not know about, and the only
    layer all of them share is the HTTP client.
    """

    def __init__(self):
        self.urls = []

    def __enter__(self):
        self._sync = httpx.Client.send
        self._async = httpx.AsyncClient.send
        recorder = self

        def sync_send(client, request, **kwargs):
            recorder.urls.append(str(request.url))
            return recorder._sync(client, request, **kwargs)

        async def async_send(client, request, **kwargs):
            recorder.urls.append(str(request.url))
            return await recorder._async(client, request, **kwargs)

        httpx.Client.send = sync_send
        httpx.AsyncClient.send = async_send
        return self

    def __exit__(self, *exc):
        httpx.Client.send = self._sync
        httpx.AsyncClient.send = self._async
        return False

    def providers(self):
        return {p for p in (provider_for(u) for u in self.urls) if p}


class ConfigurationTests(unittest.TestCase):
    """What the process resolved, before any packet moves."""

    def test_a_provider_is_selected_and_its_block_supplies_the_credentials(self):
        """The production failure this catches: LLM_PROVIDER set, block absent, and every
        value quietly inherited from the generic names of whatever was there before."""
        provider = active_provider()
        if provider is None:
            self.skipTest("LLM_PROVIDER is unset; the generic names are used as written")

        sources = resolve(provider).sources
        for setting in ("ARK_API_KEY", "BASE_URL"):
            self.assertTrue(
                sources.get(setting, "").startswith(provider.prefix),
                f"{setting} came from {sources.get(setting)!r}, not from the "
                f"{provider.name} block -- the switch is only half applied",
            )

    def test_the_endpoint_belongs_to_the_selected_provider(self):
        provider = active_provider()
        if provider is None:
            self.skipTest("LLM_PROVIDER is unset")
        self.assertEqual(
            provider.name,
            provider_for(os.environ.get("BASE_URL", "")),
            f"BASE_URL={os.environ.get('BASE_URL')!r} is not a {provider.name} endpoint",
        )

    def test_the_key_format_matches_the_selected_provider(self):
        """A Together endpoint holding a gsk_ key is the shape of a half-applied switch,
        and it surfaces as a 401 from a host nobody named."""
        provider = active_provider()
        if provider is None:
            self.skipTest("LLM_PROVIDER is unset")
        key = os.environ.get("ARK_API_KEY") or ""
        prefixes = {"groq": "gsk_", "together": "tgp_"}
        expected = prefixes.get(provider.name)
        if not expected or not key:
            self.skipTest(f"no known key prefix for {provider.name}")
        self.assertTrue(
            key.startswith(expected),
            f"{provider.name} is selected but ARK_API_KEY starts with "
            f"{key[:4]!r}, not {expected!r}",
        )


class WireTests(unittest.TestCase):
    """Where the packets actually went.

    `backend.chat.runtime` builds its clients at MODULE IMPORT, from the environment as it
    stands at that moment. In production that moment is boot, just after `load_env()`, so
    the clients get the resolved provider. Inside a suite it is whenever some earlier test
    first imported the module -- and several of them (`test_llm_provider`,
    `test_vision_config`, `test_figure_pipeline`) blank `MODEL` and `ARK_API_KEY` inside a
    `patch.dict`. Importing during that window caches a client with no model id, which
    fails later as `_init_chat_model_helper() missing 'model'` in a test that has nothing
    to do with whoever blanked it.

    So the module is reloaded here, with the real environment restored. That is not a
    workaround: it reproduces the production sequence exactly -- resolve the provider,
    then construct the clients -- and it makes these tests independent of which file
    happened to import the runtime first.
    """

    @classmethod
    def setUpClass(cls):
        import importlib

        import backend.chat.runtime as runtime

        if not (os.environ.get("MODEL") or "").strip():
            raise unittest.SkipTest("MODEL is unset; nothing to route")
        cls.runtime = importlib.reload(runtime)

    @requires_llm
    def test_an_answering_call_goes_to_the_selected_provider_and_nowhere_else(self):
        model = self.runtime.model

        with RecordingTransport() as recorded:
            model.invoke("Say OK.")

        self.assertTrue(recorded.urls, "no HTTP request was made at all")
        seen = recorded.providers()
        provider = active_provider()
        expected = provider.name if provider else provider_for(os.environ.get("BASE_URL", ""))
        self.assertEqual(
            {expected},
            seen,
            f"expected every call to reach {expected}; the wire shows "
            f"{seen or 'nothing'} ({recorded.urls})",
        )

    @requires_llm
    def test_no_answering_call_reaches_groq_when_another_provider_is_selected(self):
        """Stated separately from the equality above so that a failure names the symptom
        the deployment actually reported rather than a set difference."""
        provider = active_provider()
        if provider is None or provider.name == "groq":
            self.skipTest("Groq is the selected provider, so reaching it is correct")

        model = self.runtime.model

        with RecordingTransport() as recorded:
            model.invoke("Say OK.")

        groq = [u for u in recorded.urls if provider_for(u) == "groq"]
        self.assertEqual([], groq, f"a text-model call still went to Groq: {groq}")

    @requires_llm
    def test_each_model_role_uses_the_same_provider(self):
        """`answer` and `fast` are separate clients built from the same names. A switch
        that reached one and not the other stays invisible until the planner runs."""
        pairs = (("answer", self.runtime.model), ("fast", self.runtime.fast_model))

        for role, client in pairs:
            with RecordingTransport() as recorded:
                client.invoke("Say OK.")
            with self.subTest(role=role):
                self.assertEqual(
                    1,
                    len(recorded.providers()),
                    f"the {role} model reached {recorded.providers()}",
                )


class VisionTests(unittest.TestCase):
    """The vision model is configured separately and does NOT follow LLM_PROVIDER."""

    def test_the_vision_model_is_reported_beside_the_text_provider(self):
        """Not a failure -- a statement of fact the deployment needs in front of it.

        Vision keeps its own credentials on purpose, so a text-model switch leaves it
        where it was. When that is a different provider from the selected one, this
        prints the pair, because the alternative is discovering it as a quota bill from
        a provider the deployment believes it stopped using.
        """
        credentials = resolve_vision_credentials()
        if not credentials.available:
            self.skipTest("no vision model configured")

        vision = provider_for(credentials.base_url) or "an unrecognised host"
        provider = active_provider()
        text = provider.name if provider else provider_for(os.environ.get("BASE_URL", ""))

        if vision != text:
            prefix = provider.prefix if provider else "<PROVIDER>"
            print(
                f"\n  NOTE: text models -> {text}, but the VISION model -> {vision}"
                f"\n        {credentials.model_id} at {credentials.base_url}"
                f"\n        Figure reads and ingest extraction still spend {vision} quota."
                f"\n        To move it: set {prefix}_VISION_MODEL / _API_KEY / _BASE_URL,"
                f"\n        or blank the generic VISION_* so it falls back to the text model."
            )
        self.assertTrue(credentials.model_id)


if __name__ == "__main__":
    unittest.main()
