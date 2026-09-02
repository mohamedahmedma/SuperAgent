"""LLM provider selection.

The failure this guards is quiet and expensive: a deployment sets `LLM_PROVIDER` and
believes it switched, while a leftover generic `BASE_URL` keeps sending every model call
to the old endpoint with the new provider's key. So the ordering is pinned here — block
over generic, generic over built-in default — along with the two states that must stay
harmless: an unset selector (which has to remain a total no-op for every deployment
predating this) and a blank value inside a block (which has to fall through rather than
resolve to an empty model id).
"""
import logging
import os
import unittest
from unittest.mock import patch

from backend.llm_provider import (
    PROVIDERS,
    SETTINGS,
    UnknownProviderError,
    active_provider,
    apply_provider_env,
    log_provider_status,
    prefixed_name,
    resolve,
)

# Every name the resolver reads, blanked so a real .env cannot leak into a test.
PROVIDER_ENV = {
    name: ""
    for provider in PROVIDERS.values()
    for name in [*SETTINGS, *(prefixed_name(provider, s) for s in SETTINGS)]
}
PROVIDER_ENV["LLM_PROVIDER"] = ""


def env(**overrides):
    return patch.dict(os.environ, {**PROVIDER_ENV, **overrides})


class SelectionTests(unittest.TestCase):
    def test_unset_selector_selects_nothing(self):
        with env(ARK_API_KEY="k", MODEL="m"):
            self.assertIsNone(active_provider())

    def test_the_name_is_case_insensitive(self):
        with env(LLM_PROVIDER="Together"):
            self.assertEqual("together", active_provider().name)

    def test_an_unknown_name_is_a_boot_time_error(self):
        """A typo must not silently fall back to whatever the generic variables say."""
        with env(LLM_PROVIDER="togather"), self.assertRaises(UnknownProviderError) as caught:
            active_provider()
        message = str(caught.exception)
        self.assertIn("togather", message)
        # The error has to name the alternatives; it is the only place they are listed.
        self.assertIn("together", message)
        self.assertIn("groq", message)


class ResolutionTests(unittest.TestCase):
    def test_the_block_beats_the_generic_name(self):
        """The whole point of the selector: with both set, the live block wins.

        If it were the other way round, `.env` — which sets the generic names — would
        make `LLM_PROVIDER` a variable that changes nothing.
        """
        with env(
            LLM_PROVIDER="together",
            TOGETHER_API_KEY="tgp", TOGETHER_BASE_URL="https://together", TOGETHER_MODEL="m-tog",
            ARK_API_KEY="generic", BASE_URL="https://generic", MODEL="m-generic",
        ):
            resolution = apply_provider_env()
            self.assertEqual("tgp", os.environ["ARK_API_KEY"])
            self.assertEqual("https://together", os.environ["BASE_URL"])
            self.assertEqual("m-tog", os.environ["MODEL"])
        self.assertEqual("TOGETHER_API_KEY", resolution.sources["ARK_API_KEY"])

    def test_switching_back_to_groq_reaches_the_groq_block(self):
        """Both blocks stay written; the selector is the only thing that moves."""
        both = dict(
            GROQ_API_KEY="gsk", GROQ_BASE_URL="https://groq", GROQ_MODEL="m-groq",
            TOGETHER_API_KEY="tgp", TOGETHER_BASE_URL="https://together", TOGETHER_MODEL="m-tog",
        )
        with env(LLM_PROVIDER="groq", **both):
            apply_provider_env()
            self.assertEqual("gsk", os.environ["ARK_API_KEY"])
            self.assertEqual("https://groq", os.environ["BASE_URL"])
            self.assertEqual("m-groq", os.environ["MODEL"])

    def test_the_generic_name_fills_what_the_block_omits(self):
        """A model id both providers serve stays written once, not once per block."""
        with env(
            LLM_PROVIDER="together",
            TOGETHER_API_KEY="tgp", TOGETHER_BASE_URL="https://together",
            MODEL="shared", FAST_MODEL="shared-fast", GRADE_MODEL="shared-grade",
        ):
            resolution = apply_provider_env()
            self.assertEqual("shared", os.environ["MODEL"])
        self.assertEqual("MODEL", resolution.sources["MODEL"])
        # Nothing to write for a value already in the environment under its own name.
        self.assertNotIn("MODEL", resolution.values)

    def test_a_blank_block_value_falls_through(self):
        """`TOGETHER_MODEL=` means "not set here", not "the empty model id"."""
        with env(LLM_PROVIDER="together", TOGETHER_API_KEY="tgp", TOGETHER_MODEL="   ",
                 MODEL="shared"):
            apply_provider_env()
            self.assertEqual("shared", os.environ["MODEL"])

    def test_base_url_falls_back_to_the_providers_own_endpoint(self):
        """Only base_url has a built-in default — it is a fact about the provider."""
        with env(LLM_PROVIDER="groq", GROQ_API_KEY="gsk", GROQ_MODEL="m"):
            resolution = apply_provider_env()
            self.assertEqual("https://api.groq.com/openai/v1", os.environ["BASE_URL"])
        self.assertEqual("groq default", resolution.sources["BASE_URL"])

    def test_a_generic_base_url_is_consulted_before_the_built_in_default(self):
        """Why every block in .env writes its own BASE_URL out in full."""
        with env(LLM_PROVIDER="groq", GROQ_API_KEY="gsk", BASE_URL="https://leftover"):
            apply_provider_env()
            self.assertEqual("https://leftover", os.environ["BASE_URL"])

    def test_no_default_is_invented_for_a_key_or_a_model(self):
        with env(LLM_PROVIDER="together"):
            resolution = apply_provider_env()
        self.assertEqual({}, {k: v for k, v in resolution.values.items() if k != "BASE_URL"})


class VisionTests(unittest.TestCase):
    """Vision follows the switch only when a block asks it to."""

    def test_generic_vision_settings_survive_a_provider_switch(self):
        """The shipped arrangement: text on Together, vision deliberately left on Groq."""
        with env(
            LLM_PROVIDER="together",
            TOGETHER_API_KEY="tgp", TOGETHER_BASE_URL="https://together", TOGETHER_MODEL="m-tog",
            VISION_MODEL="vl", VISION_API_KEY="gsk", VISION_BASE_URL="https://groq",
        ):
            apply_provider_env()
            self.assertEqual("vl", os.environ["VISION_MODEL"])
            self.assertEqual("gsk", os.environ["VISION_API_KEY"])
            self.assertEqual("https://groq", os.environ["VISION_BASE_URL"])

    def test_a_block_can_move_vision_onto_the_live_provider(self):
        with env(
            LLM_PROVIDER="together",
            TOGETHER_API_KEY="tgp", TOGETHER_VISION_MODEL="tog-vl",
            VISION_MODEL="groq-vl",
        ):
            apply_provider_env()
            self.assertEqual("tog-vl", os.environ["VISION_MODEL"])


class NoOpTests(unittest.TestCase):
    """An unset selector has to leave a pre-`LLM_PROVIDER` deployment untouched."""

    def test_nothing_is_written_when_the_selector_is_unset(self):
        with env(ARK_API_KEY="legacy", BASE_URL="https://legacy", MODEL="m-legacy",
                 GROQ_API_KEY="gsk", TOGETHER_API_KEY="tgp"):
            before = dict(os.environ)
            self.assertIsNone(apply_provider_env())
            self.assertEqual(before, dict(os.environ))


class DiagnosticTests(unittest.TestCase):
    """The boot line is the only signal that a switch actually took, so it has to be
    both present and free of the key it resolved."""

    def test_it_names_the_provider_the_model_and_the_endpoint(self):
        with env(LLM_PROVIDER="together", TOGETHER_API_KEY="tgp-secret",
                 TOGETHER_BASE_URL="https://together", TOGETHER_MODEL="m-tog"):
            apply_provider_env()
            with self.assertLogs("backend.llm_provider", level=logging.INFO) as captured:
                log_provider_status()
        line = "\n".join(captured.output)
        self.assertIn("together", line)
        self.assertIn("m-tog", line)
        self.assertIn("https://together", line)
        self.assertIn("TOGETHER_API_KEY", line)
        self.assertNotIn("tgp-secret", line)

    def test_an_unset_selector_still_says_so(self):
        with env(MODEL="m", ARK_API_KEY="k"):
            with self.assertLogs("backend.llm_provider", level=logging.INFO) as captured:
                log_provider_status()
        self.assertIn("none selected", "\n".join(captured.output))

    def test_a_typo_is_logged_rather_than_raised_at_boot(self):
        """`log_provider_status` runs inside the lifespan, after `load_env` has already
        raised on a bad name. Reaching it with one means something else selected the
        provider, and a diagnostic must not be what takes the app down."""
        with env(LLM_PROVIDER="nope"):
            with self.assertLogs("backend.llm_provider", level=logging.ERROR) as captured:
                log_provider_status()
        self.assertIn("nope", "\n".join(captured.output))


class RegistryTests(unittest.TestCase):
    def test_every_provider_names_its_own_block_consistently(self):
        for provider in PROVIDERS.values():
            self.assertEqual(provider.name, provider.name.lower())
            # ARK_API_KEY is the one setting whose block spelling is not the generic name
            # with a prefix bolted on.
            self.assertEqual(f"{provider.prefix}_API_KEY", prefixed_name(provider, "ARK_API_KEY"))
            self.assertEqual(f"{provider.prefix}_MODEL", prefixed_name(provider, "MODEL"))
            self.assertTrue(provider.default_base_url.startswith("https://"))

    def test_resolve_reports_a_source_for_everything_it_resolves(self):
        with env(LLM_PROVIDER="groq", GROQ_API_KEY="gsk", MODEL="m"):
            resolution = resolve(PROVIDERS["groq"])
        for setting in resolution.values:
            self.assertIn(setting, resolution.sources)


if __name__ == "__main__":
    unittest.main()
