"""Vision credential resolution.

The failure mode this guards is silent: a profile enables vision, the credentials are
absent, and extraction degrades to the heuristic path forever without anything saying
so. These tests pin the fallback chain, the separate-provider case, and the startup
diagnostic that makes the degraded state visible.
"""
import logging
import os
import unittest
from unittest.mock import patch

from backend.assets.vision import (
    VisionCredentials,
    build_vision_model,
    log_vision_status,
    resolve_vision_credentials,
    vision_status,
)
from backend.profiles.registry import load_profile

# Everything the resolver reads, cleared so a real .env cannot leak into a test.
VISION_ENV = {
    "VISION_MODEL": "", "VISION_API_KEY": "", "VISION_BASE_URL": "",
    "MODEL": "", "ARK_API_KEY": "", "BASE_URL": "",
}


def env(**overrides):
    return patch.dict(os.environ, {**VISION_ENV, **overrides})


class ResolutionTests(unittest.TestCase):
    def test_vision_specific_variables_win(self):
        with env(VISION_MODEL="vl-1", VISION_API_KEY="vk", VISION_BASE_URL="https://vision",
                 MODEL="text-1", ARK_API_KEY="tk", BASE_URL="https://text"):
            creds = resolve_vision_credentials()
        self.assertEqual("vl-1", creds.model_id)
        self.assertEqual("vk", creds.api_key)
        self.assertEqual("https://vision", creds.base_url)

    def test_it_falls_back_to_the_text_model_settings(self):
        """A deployment whose MODEL is already vision-capable needs no new settings."""
        with env(MODEL="gpt-vision", ARK_API_KEY="tk", BASE_URL="https://text"):
            creds = resolve_vision_credentials()
        self.assertEqual("gpt-vision", creds.model_id)
        self.assertEqual("tk", creds.api_key)
        self.assertEqual("https://text", creds.base_url)
        self.assertEqual("MODEL", creds.model_source)

    def test_a_vision_model_on_a_different_provider_is_expressible(self):
        """The case the old per-module resolution could not express: it always sent
        the text provider's key to whatever VISION_MODEL named."""
        with env(MODEL="text-1", ARK_API_KEY="tk", BASE_URL="https://text",
                 VISION_MODEL="claude-vision", VISION_API_KEY="anthropic-key",
                 VISION_BASE_URL="https://anthropic"):
            creds = resolve_vision_credentials()
        self.assertEqual("claude-vision", creds.model_id)
        self.assertEqual("anthropic-key", creds.api_key)
        self.assertEqual("https://anthropic", creds.base_url)

    def test_settings_fall_back_independently(self):
        """Only the model differs; the key and endpoint still come from the text side."""
        with env(MODEL="text-1", ARK_API_KEY="tk", BASE_URL="https://text",
                 VISION_MODEL="vl-1"):
            creds = resolve_vision_credentials()
        self.assertEqual("vl-1", creds.model_id)
        self.assertEqual("tk", creds.api_key)
        self.assertEqual("ARK_API_KEY", creds.key_source)

    def test_blank_values_count_as_unset(self):
        with env(VISION_MODEL="   ", MODEL="text-1", ARK_API_KEY="tk"):
            creds = resolve_vision_credentials()
        self.assertEqual("text-1", creds.model_id)

    def test_availability_needs_a_model_and_a_key(self):
        self.assertTrue(VisionCredentials(model_id="m", api_key="k").available)
        self.assertFalse(VisionCredentials(model_id="m").available)
        self.assertFalse(VisionCredentials(api_key="k").available)
        # base_url is optional — some providers have a default endpoint.
        self.assertTrue(VisionCredentials(model_id="m", api_key="k", base_url="").available)

    def test_missing_pieces_are_named_actionably(self):
        gaps = VisionCredentials().missing()
        self.assertIn("VISION_MODEL (or MODEL)", gaps)
        self.assertIn("VISION_API_KEY (or ARK_API_KEY)", gaps)

    def test_the_description_never_leaks_the_key(self):
        """describe() reaches logs."""
        creds = VisionCredentials(model_id="vl", api_key="sk-super-secret",
                                  base_url="https://x", model_source="VISION_MODEL",
                                  key_source="VISION_API_KEY")
        described = creds.describe()
        self.assertNotIn("sk-super-secret", described)
        self.assertIn("VISION_API_KEY", described)
        self.assertIn("vl", described)

    def test_the_description_reports_unavailability(self):
        self.assertIn("unavailable", VisionCredentials().describe())


class ModelBuildingTests(unittest.TestCase):
    def test_incomplete_credentials_yield_none_rather_than_raising(self):
        """Every caller has a working non-vision fallback; a missing key should degrade
        the deployment, not break it."""
        with env():
            self.assertIsNone(build_vision_model())

    def test_a_configured_model_is_built_with_the_resolved_settings(self):
        with patch("langchain.chat_models.init_chat_model") as init:
            build_vision_model(
                0.3, VisionCredentials(model_id="vl", api_key="k", base_url="https://x")
            )
        kwargs = init.call_args.kwargs
        self.assertEqual("vl", kwargs["model"])
        self.assertEqual("k", kwargs["api_key"])
        self.assertEqual("https://x", kwargs["base_url"])
        self.assertEqual(0.3, kwargs["temperature"])

    def test_an_empty_base_url_is_passed_as_none(self):
        """So the provider SDK uses its own default rather than an empty string."""
        with patch("langchain.chat_models.init_chat_model") as init:
            build_vision_model(0.0, VisionCredentials(model_id="vl", api_key="k"))
        self.assertIsNone(init.call_args.kwargs["base_url"])


class StatusTests(unittest.TestCase):
    def _profile(self, figures=False, entities=False):
        profile = load_profile("base").model_copy(deep=True)
        profile.assets.figures.vision_enabled = figures
        profile.assets.entities.enabled = entities
        profile.assets.entities.vision_enabled = entities
        return profile

    def test_nothing_requested_is_not_degraded(self):
        with env():
            status = vision_status(self._profile())
        self.assertEqual([], status["requested_by"])
        self.assertEqual([], status["degraded"])

    def test_requested_without_credentials_is_reported_as_degraded(self):
        with env():
            status = vision_status(self._profile(figures=True, entities=True))
        self.assertEqual({"figures", "entities"}, set(status["requested_by"]))
        self.assertEqual({"figures", "entities"}, set(status["degraded"]))
        self.assertFalse(status["credentials_available"])

    def test_requested_with_credentials_is_healthy(self):
        with env(VISION_MODEL="vl", VISION_API_KEY="k"):
            status = vision_status(self._profile(figures=True))
        self.assertEqual(["figures"], status["requested_by"])
        self.assertEqual([], status["degraded"])
        self.assertTrue(status["credentials_available"])

    def test_each_pipeline_is_tracked_separately(self):
        with env(VISION_MODEL="vl", VISION_API_KEY="k"):
            status = vision_status(self._profile(figures=True, entities=False))
        self.assertEqual(["figures"], status["requested_by"])

    def test_disabling_assets_stops_figures_being_requested(self):
        profile = self._profile(figures=True)
        profile.assets.figures.enabled = False
        with env():
            self.assertEqual([], vision_status(profile)["requested_by"])

    def test_the_degraded_state_is_logged_as_a_warning(self):
        with env():
            with self.assertLogs("backend.assets.vision", level=logging.WARNING) as captured:
                log_vision_status(self._profile(figures=True))
        message = "".join(captured.output)
        self.assertIn("VISION_MODEL", message)
        self.assertIn("figures", message)

    def test_the_healthy_state_is_logged_at_info(self):
        with env(VISION_MODEL="vl", VISION_API_KEY="k"):
            with self.assertLogs("backend.assets.vision", level=logging.INFO) as captured:
                log_vision_status(self._profile(figures=True))
        self.assertIn("Vision extraction active", "".join(captured.output))

    def test_the_disabled_state_is_logged_without_alarm(self):
        with env():
            with self.assertLogs("backend.assets.vision", level=logging.INFO) as captured:
                log_vision_status(self._profile())
        output = "".join(captured.output)
        self.assertIn("disabled by profile", output)
        self.assertNotIn("WARNING", output)


class SingleSourceOfTruthTests(unittest.TestCase):
    """All three vision call sites must resolve credentials the same way."""

    def test_no_asset_module_reads_the_environment_directly(self):
        import pathlib
        import re

        assets_dir = pathlib.Path(__file__).resolve().parents[1] / "backend" / "assets"
        offenders = []
        for path in assets_dir.glob("*.py"):
            if path.name == "vision.py":
                continue
            text = path.read_text(encoding="utf-8")
            for match in re.finditer(r'os\.getenv\(\s*["\'](VISION_\w+|ARK_API_KEY|MODEL|BASE_URL)', text):
                offenders.append(f"{path.name}: {match.group(1)}")
        self.assertEqual([], offenders)

    def test_every_builder_degrades_without_credentials(self):
        from backend.assets.entity_extractor import HeuristicEntityExtractor, build_entity_extractor
        from backend.assets.extractors import HeuristicExtractor, build_extractor
        from backend.assets.reader import build_figure_reader

        profile = load_profile("ecommerce")
        figures = profile.assets.figures.model_copy(update={"vision_enabled": True})
        entities = profile.assets.entities.model_copy(update={"vision_enabled": True})
        from backend.assets.attributes import build_attribute_schema

        schema = build_attribute_schema(profile.assets.entities)

        with env():
            self.assertIsInstance(build_extractor(figures), HeuristicExtractor)
            self.assertIsInstance(build_entity_extractor(entities, schema), HeuristicEntityExtractor)
            self.assertFalse(build_figure_reader(figures).available)

    def test_every_builder_activates_with_credentials(self):
        from backend.assets.entity_extractor import VisionEntityExtractor, build_entity_extractor
        from backend.assets.extractors import VisionExtractor, build_extractor
        from backend.assets.reader import build_figure_reader
        from backend.assets.attributes import build_attribute_schema

        profile = load_profile("ecommerce")
        figures = profile.assets.figures.model_copy(update={"vision_enabled": True})
        entities = profile.assets.entities.model_copy(update={"vision_enabled": True})
        schema = build_attribute_schema(profile.assets.entities)

        with env(VISION_MODEL="vl", VISION_API_KEY="k", VISION_BASE_URL="https://v"):
            self.assertIsInstance(build_extractor(figures), VisionExtractor)
            self.assertIsInstance(build_entity_extractor(entities, schema), VisionEntityExtractor)
            self.assertTrue(build_figure_reader(figures).available)

    def test_a_disabled_profile_never_activates_vision_even_with_credentials(self):
        from backend.assets.extractors import HeuristicExtractor, build_extractor

        figures = load_profile("base").assets.figures.model_copy(update={"vision_enabled": False})
        with env(VISION_MODEL="vl", VISION_API_KEY="k"):
            self.assertIsInstance(build_extractor(figures), HeuristicExtractor)


if __name__ == "__main__":
    unittest.main()


class RateLimitRetryTests(unittest.TestCase):
    """A rate limit is transient. Treating it as a permanent extraction failure is
    what silently stripped every figure out of a document."""

    def setUp(self):
        from backend.profiles.registry import load_profile as _load

        self.config = _load("base").assets.figures.model_copy(
            update={"vision_retry_attempts": 3, "vision_retry_base_seconds": 0.01,
                    "vision_retry_max_seconds": 0.05}
        )

    def _error(self, message, status=None):
        exc = RuntimeError(message)
        if status is not None:
            exc.status_code = status
        return exc

    def test_groq_413_and_429_are_both_recognised(self):
        from backend.assets.vision import is_rate_limit_error

        # Groq answers 413 for "this single request exceeds your per-minute allowance".
        self.assertTrue(is_rate_limit_error(self._error("Request too large", status=413)))
        self.assertTrue(is_rate_limit_error(self._error("slow down", status=429)))

    def test_the_body_is_the_reliable_discriminator(self):
        from backend.assets.vision import is_rate_limit_error

        self.assertTrue(is_rate_limit_error(self._error(
            "Error code: 413 - rate_limit_exceeded on tokens per minute (TPM): Limit 8000"
        )))
        self.assertTrue(is_rate_limit_error(self._error("Too Many Requests")))

    def test_ordinary_failures_are_not_mistaken_for_quota(self):
        from backend.assets.vision import is_rate_limit_error

        self.assertFalse(is_rate_limit_error(self._error("model does not support json_schema", status=400)))
        self.assertFalse(is_rate_limit_error(self._error("connection reset")))

    def test_the_providers_stated_delay_is_honoured(self):
        """A per-minute window refills on a schedule; guessing shorter burns an attempt."""
        from backend.assets.vision import retry_after_seconds

        self.assertEqual(8.5, retry_after_seconds(self._error("please try again in 8.5s"), 99))
        self.assertEqual(0.25, retry_after_seconds(self._error("try again in 250ms"), 99))
        self.assertEqual(120.0, retry_after_seconds(self._error("try again in 2m"), 99))
        self.assertEqual(7.0, retry_after_seconds(self._error("no hint here"), 7.0))

    def test_a_transient_limit_is_retried_and_then_succeeds(self):
        from backend.assets.vision import call_with_rate_limit_retry

        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise self._error("rate_limit_exceeded: tokens per minute")
            return "extracted"

        self.assertEqual("extracted", call_with_rate_limit_retry(flaky, self.config))
        self.assertEqual(3, calls["n"])

    def test_retries_are_bounded(self):
        from backend.assets.vision import call_with_rate_limit_retry

        calls = {"n": 0}

        def always_limited():
            calls["n"] += 1
            raise self._error("rate_limit_exceeded")

        with self.assertRaises(RuntimeError):
            call_with_rate_limit_retry(always_limited, self.config)
        self.assertEqual(3, calls["n"])

    def test_a_non_quota_error_is_raised_immediately(self):
        """Retrying an unsupported feature just multiplies the delay before fallback."""
        from backend.assets.vision import call_with_rate_limit_retry

        calls = {"n": 0}

        def unsupported():
            calls["n"] += 1
            raise self._error("model does not support response format json_schema", status=400)

        with self.assertRaises(RuntimeError):
            call_with_rate_limit_retry(unsupported, self.config)
        self.assertEqual(1, calls["n"])

    def test_a_quota_error_does_not_fall_through_to_the_next_method(self):
        """The next method sends the same payload against the same allowance, so
        falling through would burn the budget again and misreport the cause."""
        from unittest.mock import Mock

        from backend.assets.extractors import FigureExtraction
        from backend.assets.vision import invoke_structured

        model = Mock()
        bound = Mock()
        bound.invoke.side_effect = self._error("rate_limit_exceeded: tokens per minute")
        model.with_structured_output.return_value = bound

        with self.assertRaises(RuntimeError):
            invoke_structured(model, FigureExtraction, [{"role": "user", "content": "x"}],
                              config=self.config)
        # json_schema only — never re-bound for function_calling.
        self.assertEqual(1, model.with_structured_output.call_count)


class VisionParameterTests(unittest.TestCase):
    def test_extra_params_reach_the_model(self):
        from unittest.mock import patch as _patch

        from backend.assets.vision import VisionCredentials, build_vision_model

        with _patch("langchain.chat_models.init_chat_model") as init:
            build_vision_model(
                0.0, VisionCredentials(model_id="vl", api_key="k"),
                max_tokens=4096, extra_params={"reasoning_effort": "none"},
            )
        kwargs = init.call_args.kwargs
        self.assertEqual(4096, kwargs["max_tokens"])
        self.assertEqual("none", kwargs["reasoning_effort"])

    def test_no_extra_params_adds_nothing(self):
        from unittest.mock import patch as _patch

        from backend.assets.vision import VisionCredentials, build_vision_model

        with _patch("langchain.chat_models.init_chat_model") as init:
            build_vision_model(0.0, VisionCredentials(model_id="vl", api_key="k"))
        self.assertNotIn("reasoning_effort", init.call_args.kwargs)
        self.assertNotIn("max_tokens", init.call_args.kwargs)

    def test_the_output_budget_default_fits_a_small_quota(self):
        """input (~3k for a page image) + this must stay inside an 8k/min window."""
        from backend.profiles.registry import load_profile as _load

        self.assertLessEqual(_load("base").assets.figures.vision_max_output_tokens, 5000)
