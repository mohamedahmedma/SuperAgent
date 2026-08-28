"""Per-role model sampling: what reaches `init_chat_model`, and what deliberately does not.

Two properties matter here and neither is visible from a passing request:

1. **Unset must mean absent.** `reasoning_effort` and `max_tokens` are rejected — not
   ignored — by providers that do not implement them, and a `max_tokens` sent as 0
   would cap every response at nothing. Both have to be dropped from the kwargs rather
   than passed as a falsy default.
2. **The roles must not drift.** Each role reads three `<role>_*` fields off
   ModelConfig by name; adding a role without its fields, or renaming a field, fails at
   the first request rather than at import, so it is asserted directly.
"""
import os
import unittest
from unittest.mock import patch

import backend.profiles.registry as registry
from backend.llm import ROLES, sampling
from backend.profiles.registry import ProfileError, load_profile, set_profile



class SamplingTestCase(unittest.TestCase):
    """The profile is process-global and `sampling()` reads it, so every test restores
    the cache to avoid leaking a tuned profile into the modules other tests import."""

    def tearDown(self):
        set_profile(None)


class SamplingShapeTests(SamplingTestCase):
    def test_temperature_is_always_present(self):
        for role in ROLES:
            with self.subTest(role=role):
                self.assertIn("temperature", sampling(role))

    def test_effort_is_omitted_when_unset(self):
        """Empty is the schema default and must reach the provider as an absent field —
        a model without an effort parameter errors on the key, it does not ignore it."""
        with patch.dict(os.environ, {"GRADE_REASONING_EFFORT": "none"}):
            set_profile(load_profile("base"))
            self.assertNotIn("reasoning_effort", sampling("grade"))

    def test_effort_is_passed_through_when_set(self):
        with patch.dict(os.environ, {"GRADE_REASONING_EFFORT": "high"}):
            set_profile(load_profile("base"))
            self.assertEqual("high", sampling("grade")["reasoning_effort"])

    def test_max_tokens_is_omitted_when_zero(self):
        """0 means 'no ceiling'. Passed through literally it would cap the response at
        zero tokens, which is the difference between a knob being off and being lethal."""
        set_profile(load_profile("base"))
        for role in ROLES:
            with self.subTest(role=role):
                self.assertNotIn("max_tokens", sampling(role))

    def test_max_tokens_is_passed_through_when_positive(self):
        with patch.dict(os.environ, {"PLANNER_MAX_TOKENS": "128"}):
            set_profile(load_profile("base"))
            self.assertEqual(128, sampling("planner")["max_tokens"])

    def test_unknown_role_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            sampling("summariser")
        self.assertIn("summariser", str(ctx.exception))


class RoleDriftTests(SamplingTestCase):
    def test_every_role_has_its_three_config_fields(self):
        """`sampling()` resolves fields by name, so a role added without its fields
        would raise on the first request instead of at startup."""
        models = load_profile("base").models
        for role in ROLES:
            for suffix in ("temperature", "reasoning_effort", "max_tokens"):
                with self.subTest(role=role, field=suffix):
                    self.assertTrue(
                        hasattr(models, f"{role}_{suffix}"),
                        f"ModelConfig is missing {role}_{suffix}",
                    )

    def test_every_role_is_env_overridable(self):
        """Each knob is meant to be tuned per deployment; a role missing from
        ENV_OVERRIDES is silently pinned to whatever the profile says."""
        for role in ROLES:
            for env_suffix in ("TEMPERATURE", "REASONING_EFFORT", "MAX_TOKENS"):
                name = f"{role.upper()}_{env_suffix}"
                with self.subTest(env=name):
                    self.assertIn(name, registry.ENV_OVERRIDES)


class EffortValidationTests(SamplingTestCase):
    def test_invalid_effort_fails_at_startup(self):
        """Fatal on purpose: the alternative is a 400 on every call this role makes."""
        with patch.dict(os.environ, {"GRADE_REASONING_EFFORT": "ultra"}):
            with self.assertRaises(ProfileError):
                load_profile("base")

    def test_off_is_accepted_as_a_synonym_for_none(self):
        with patch.dict(os.environ, {"SCOPE_REASONING_EFFORT": "off"}):
            set_profile(load_profile("base"))
            self.assertNotIn("reasoning_effort", sampling("scope"))

    def test_effort_is_case_insensitive(self):
        with patch.dict(os.environ, {"ANSWER_REASONING_EFFORT": "HIGH"}):
            set_profile(load_profile("base"))
            self.assertEqual("high", sampling("answer")["reasoning_effort"])

    def test_blank_env_falls_through_to_the_profile(self):
        """Blank is 'unset' everywhere in this codebase, which is exactly why `none`
        exists — without it a deployment could not switch off a profile's effort."""
        with patch.dict(os.environ, {"PLANNER_REASONING_EFFORT": ""}):
            set_profile(load_profile("base"))
            self.assertEqual("low", sampling("planner")["reasoning_effort"])


class VisionEffortTests(SamplingTestCase):
    """The vision model is the one per-turn model outside ModelConfig — `view_figure`
    reaches it mid-turn — so it gets the same knob, kept beside the other vision
    settings rather than moved in with the text roles."""

    def test_effort_is_env_overridable_through_a_nested_path(self):
        """assets.figures.* is three levels deep; a resolver that split the path once
        would drop this override silently instead of failing."""
        with patch.dict(os.environ, {"VISION_REASONING_EFFORT": "low"}):
            profile = load_profile("base")
            self.assertEqual("low", profile.assets.figures.vision_reasoning_effort)

    def test_effort_defaults_to_unset(self):
        """Empty means no effort field is sent at all, which is what a vision model
        without the parameter requires."""
        self.assertEqual("", load_profile("base").assets.figures.vision_reasoning_effort)

    def test_invalid_effort_fails_at_startup(self):
        with patch.dict(os.environ, {"VISION_REASONING_EFFORT": "maximum"}):
            with self.assertRaises(ProfileError):
                load_profile("base")

    def test_none_is_normalised_to_unset(self):
        with patch.dict(os.environ, {"VISION_REASONING_EFFORT": "none"}):
            self.assertEqual("", load_profile("base").assets.figures.vision_reasoning_effort)

    def test_builder_omits_the_field_when_unset(self):
        from backend.assets.vision import VisionCredentials, build_vision_model

        creds = VisionCredentials(model_id="v", api_key="k", base_url="https://x.test/v1")
        with patch("langchain.chat_models.init_chat_model") as init:
            build_vision_model(0.0, creds, max_tokens=256, reasoning_effort="")
            self.assertNotIn("reasoning_effort", init.call_args.kwargs)

            init.reset_mock()
            build_vision_model(0.0, creds, max_tokens=256, reasoning_effort="low")
            self.assertEqual("low", init.call_args.kwargs["reasoning_effort"])

    def test_extra_params_still_wins_over_the_typed_field(self):
        """A profile already suppressing reasoning through the passthrough dict must
        keep behaving exactly as it did — the typed field is a better spelling, not a
        silent change of precedence."""
        from backend.assets.vision import VisionCredentials, build_vision_model

        creds = VisionCredentials(model_id="v", api_key="k", base_url="https://x.test/v1")
        with patch("langchain.chat_models.init_chat_model") as init:
            build_vision_model(
                0.0,
                creds,
                reasoning_effort="high",
                extra_params={"reasoning_effort": "none"},
            )
            self.assertEqual("none", init.call_args.kwargs["reasoning_effort"])


class ShippedDefaultTests(SamplingTestCase):
    def test_structured_roles_ship_at_low_effort(self):
        """These four emit a fixed shape — a label, a grade, a rewritten query — so
        deliberation cannot improve the result but is charged for and, because output
        tokens are serial, is most of the turn's wall-clock."""
        models = load_profile("base").models
        for role in ("planner", "grade", "rewrite", "scope"):
            with self.subTest(role=role):
                self.assertEqual("low", getattr(models, f"{role}_reasoning_effort"))

    def test_answer_role_keeps_headroom_above_the_structured_roles(self):
        self.assertEqual("medium", load_profile("base").models.answer_reasoning_effort)

    def test_context_window_is_seven_turns(self):
        """Counted in messages: 14 = 7 user/assistant pairs. Asserted because halving
        it looks like a harmless trim but drops the assistant side of every exchange —
        the half that later follow-ups refer back to."""
        self.assertEqual(14, load_profile("base").agent.context_window_messages)

    def test_context_window_is_env_overridable(self):
        with patch.dict(os.environ, {"CONTEXT_WINDOW_MESSAGES": "6"}):
            self.assertEqual(6, load_profile("base").agent.context_window_messages)


if __name__ == "__main__":
    unittest.main()

