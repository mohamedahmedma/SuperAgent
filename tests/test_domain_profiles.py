"""Domain profile composition, precedence, and no-drift guarantees.

The most important tests here are the drift guards: the `supermew` profile must keep
reproducing the behaviour the system had when these values were hardcoded, because
that profile is what every existing deployment loads by default.
"""
import os
import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import backend.profiles.registry as registry
from backend.profiles.registry import (
    DEFAULT_PROFILE,
    ProfileError,
    available_profiles,
    load_profile,
    reload_profile,
    set_profile,
)
from backend.profiles.schema import DomainProfile


class ProfileTestCase(unittest.TestCase):
    """Profiles are process-global; every test restores the cache and ACTIVE_PROFILE
    so it cannot leak into the modules other test files import."""

    def setUp(self):
        self._saved_active = os.environ.get(registry.PROFILE_ENV_VAR)

    def tearDown(self):
        if self._saved_active is None:
            os.environ.pop(registry.PROFILE_ENV_VAR, None)
        else:
            os.environ[registry.PROFILE_ENV_VAR] = self._saved_active
        set_profile(None)


class ShippedProfilesTests(ProfileTestCase):
    def test_every_shipped_profile_loads_and_validates(self):
        names = available_profiles()
        self.assertIn("base", names)
        self.assertIn(DEFAULT_PROFILE, names)
        for name in names:
            with self.subTest(profile=name):
                profile = load_profile(name)
                self.assertIsInstance(profile, DomainProfile)
                self.assertEqual(name, profile.name)

    def test_default_profile_is_supermew(self):
        set_profile(None)
        os.environ.pop(registry.PROFILE_ENV_VAR, None)
        self.assertEqual("supermew", load_profile().name)

    def test_active_profile_env_var_selects_the_profile(self):
        os.environ[registry.PROFILE_ENV_VAR] = "ecommerce"
        self.assertEqual("ecommerce", reload_profile().name)

    def test_unknown_profile_fails_loudly(self):
        with self.assertRaises(ProfileError) as ctx:
            load_profile("does_not_exist")
        self.assertIn("does_not_exist", str(ctx.exception))


class CompositionTests(ProfileTestCase):
    def test_child_inherits_prompts_from_base(self):
        base = load_profile("base")
        child = load_profile("document_kb")
        # document_kb overrides identity only, so RAG prompts must come through intact.
        self.assertEqual(base.rag.evidence_grade_prompt, child.rag.evidence_grade_prompt)
        self.assertEqual(base.rag.complexity_prompt, child.rag.complexity_prompt)
        self.assertNotEqual(base.identity.api_title, child.identity.api_title)

    def test_lists_replace_rather_than_merge(self):
        """A profile must be able to REMOVE an inherited tool, which append-semantics
        would make impossible."""
        base = load_profile("base")
        child = load_profile("document_kb")
        self.assertIn("get_current_weather", base.agent.tools)
        self.assertNotIn("get_current_weather", child.agent.tools)
        self.assertEqual(["search_knowledge_base", "view_figure"], child.agent.tools)

    def test_ecommerce_declares_its_extra_indexes(self):
        profile = load_profile("ecommerce")
        self.assertEqual(["kb_chunks", "entity_attrs", "entity_images"], profile.ingest.indexes)
        self.assertEqual(["figure_pipeline", "entity_pipeline"], profile.ingest.asset_pipelines)

    def test_circular_inheritance_is_rejected(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "a.yaml").write_text("extends: b\nname: a\n", encoding="utf-8")
            (tmp_path / "b.yaml").write_text("extends: a\nname: b\n", encoding="utf-8")
            with patch.object(registry, "DEFINITIONS_DIR", tmp_path):
                with self.assertRaises(ProfileError) as ctx:
                    load_profile("a")
        self.assertIn("Circular", str(ctx.exception))

    def test_unknown_key_in_a_profile_is_rejected(self):
        """extra='forbid' turns a YAML typo into a startup failure instead of a
        silently ignored setting."""
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "typo.yaml").write_text(
                "name: typo\nretrieval:\n  top_kk: 99\n", encoding="utf-8"
            )
            with patch.object(registry, "DEFINITIONS_DIR", tmp_path):
                with self.assertRaises(ProfileError) as ctx:
                    load_profile("typo")
        self.assertIn("top_kk", str(ctx.exception))


class EnvPrecedenceTests(ProfileTestCase):
    def test_env_overrides_profile_value(self):
        with patch.dict(os.environ, {"RETRIEVAL_TOP_K": "42"}):
            self.assertEqual(42, load_profile("base").retrieval.top_k)

    def test_profile_value_is_used_when_env_is_absent(self):
        with patch.dict(os.environ, {"RETRIEVAL_TOP_K": ""}):
            self.assertEqual(10, load_profile("ecommerce").retrieval.top_k)

    def test_boolean_env_override_is_coerced(self):
        with patch.dict(os.environ, {"AUTO_MERGE_ENABLED": "false"}):
            self.assertFalse(load_profile("base").retrieval.auto_merge_enabled)
        with patch.dict(os.environ, {"AUTO_MERGE_ENABLED": "true"}):
            self.assertTrue(load_profile("base").retrieval.auto_merge_enabled)

    def test_invalid_env_override_keeps_the_profile_value(self):
        with patch.dict(os.environ, {"RETRIEVAL_TOP_K": "not-a-number"}):
            self.assertEqual(8, load_profile("base").retrieval.top_k)

    def test_every_env_override_target_exists_on_the_schema(self):
        """ENV_OVERRIDES is the backward-compatibility contract; a stale entry would
        silently stop honouring a variable that live deployments still set."""
        profile = load_profile("base")
        for env_name, dotted in registry.ENV_OVERRIDES.items():
            with self.subTest(env=env_name):
                *parents, field_name = dotted.split(".")
                section = profile
                for part in parents:
                    section = getattr(section, part, None)
                    self.assertIsNotNone(section, f"{dotted} has no section {part}")
                self.assertTrue(hasattr(section, field_name), f"{dotted} missing field")


class DeepMergeTests(ProfileTestCase):
    """Nested sections must merge key-by-key; only the overridden leaf changes."""

    @contextmanager
    def _definitions(self, **files):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            for name, body in files.items():
                (path / f"{name}.yaml").write_text(body, encoding="utf-8")
            with patch.object(registry, "DEFINITIONS_DIR", path):
                yield path

    def test_nested_section_override_preserves_sibling_keys(self):
        child = "extends: base\nname: child\nassets:\n  triage:\n    min_width: 256\n"
        with self._definitions(base=Path(registry.DEFINITIONS_DIR / "base.yaml").read_text("utf-8"),
                               child=child):
            profile = load_profile("child")
        self.assertEqual(256, profile.assets.triage.min_width)
        # Untouched siblings survive at both nesting levels.
        self.assertEqual(64, profile.assets.triage.min_height)
        self.assertEqual(8000, profile.assets.triage.min_area)
        self.assertTrue(profile.assets.gc_orphan_blobs)

    def test_three_level_inheritance_resolves_most_specific_last(self):
        files = {
            "root": "name: root\nretrieval:\n  top_k: 1\n  candidate_multiplier: 9\n",
            "mid": "extends: root\nname: mid\nretrieval:\n  top_k: 2\n",
            "leaf": "extends: mid\nname: leaf\nretrieval:\n  top_k: 3\n",
        }
        with self._definitions(**files):
            with patch.dict(os.environ, {"RETRIEVAL_TOP_K": "", "RETRIEVAL_CANDIDATE_MULTIPLIER": ""}):
                profile = load_profile("leaf")
        self.assertEqual(3, profile.retrieval.top_k)
        self.assertEqual(9, profile.retrieval.candidate_multiplier)
        self.assertEqual("leaf", profile.name)

    def test_inheritance_depth_is_capped(self):
        files = {f"p{i}": f"extends: p{i + 1}\nname: p{i}\n" for i in range(12)}
        files["p12"] = "name: p12\n"
        with self._definitions(**files):
            with self.assertRaises(ProfileError) as ctx:
                load_profile("p0")
        self.assertIn("deeper than", str(ctx.exception))

    def test_extends_pointing_at_a_missing_profile_fails_loudly(self):
        with self._definitions(child="extends: ghost\nname: child\n"):
            with self.assertRaises(ProfileError) as ctx:
                load_profile("child")
        self.assertIn("ghost", str(ctx.exception))

    def test_malformed_yaml_is_rejected(self):
        with self._definitions(broken="name: [unclosed\n"):
            with self.assertRaises(ProfileError) as ctx:
                load_profile("broken")
        self.assertIn("valid YAML", str(ctx.exception))

    def test_non_mapping_yaml_is_rejected(self):
        with self._definitions(listy="- one\n- two\n"):
            with self.assertRaises(ProfileError) as ctx:
                load_profile("listy")
        self.assertIn("mapping", str(ctx.exception))

    def test_available_profiles_on_a_missing_directory(self):
        with patch.object(registry, "DEFINITIONS_DIR", Path("/definitely/not/here")):
            self.assertEqual([], available_profiles())


class CoercionTests(ProfileTestCase):
    def test_each_target_type_is_coerced_from_its_string(self):
        self.assertIs(True, registry._coerce("yes", False, "X"))
        self.assertIs(False, registry._coerce("off", True, "X"))
        self.assertEqual(12, registry._coerce("12", 0, "X"))
        self.assertEqual(1.5, registry._coerce("1.5", 0.0, "X"))
        self.assertEqual("text", registry._coerce("  text  ", "", "X"))

    def test_lists_are_parsed_as_comma_separated(self):
        self.assertEqual(["a", "b", "c"], registry._coerce("a, b ,c", ["z"], "X"))
        self.assertEqual([], registry._coerce(" , ", ["z"], "X"))

    def test_malformed_values_keep_the_existing_value(self):
        self.assertEqual(7, registry._coerce("seven", 7, "X"))
        self.assertEqual(1.5, registry._coerce("one-point-five", 1.5, "X"))
        self.assertIs(True, registry._coerce("maybe", True, "X"))

    def test_booleans_are_not_mistaken_for_integers(self):
        """bool is a subclass of int; the int branch must not swallow it."""
        self.assertIs(False, registry._coerce("false", True, "X"))

    def test_optional_field_with_no_value_still_coerces(self):
        """candidate_k defaults to None, so the target type comes from the schema."""
        with patch.dict(os.environ, {"RETRIEVAL_CANDIDATE_K": "77"}):
            self.assertEqual(77, load_profile("base").retrieval.candidate_k)

    def test_profile_invalid_after_an_env_override_fails_loudly(self):
        with patch.dict(os.environ, {"CHUNK_STRATEGY": "quantum"}):
            with self.assertRaises(ProfileError) as ctx:
                load_profile("base")
        self.assertIn("environment overrides", str(ctx.exception))


class CacheSemanticsTests(ProfileTestCase):
    def test_get_profile_caches_and_set_profile_replaces(self):
        set_profile(None)
        first = registry.get_profile()
        self.assertIs(first, registry.get_profile())

        replacement = load_profile("ecommerce")
        set_profile(replacement)
        self.assertIs(replacement, registry.get_profile())

    def test_set_profile_none_forces_a_rebuild(self):
        first = registry.get_profile()
        set_profile(None)
        self.assertIsNot(first, registry.get_profile())

    def test_reload_profile_switches_the_active_profile_env_var(self):
        reload_profile("document_kb")
        self.assertEqual("document_kb", os.environ[registry.PROFILE_ENV_VAR])
        self.assertEqual("document_kb", registry.get_profile().name)


class SystemPromptTests(ProfileTestCase):
    def test_persona_is_substituted(self):
        profile = load_profile("supermew")
        rendered = profile.render_system_prompt()
        self.assertTrue(rendered.startswith("You are a helpful knowledge-base assistant."))
        self.assertNotIn("{persona}", rendered)

    def test_literal_braces_in_a_prompt_survive_rendering(self):
        """render_system_prompt uses replace(), not format(): prompts contain literal
        braces (JSON examples), which format() would raise on."""
        profile = load_profile("base").model_copy(deep=True)
        profile.agent.system_prompt = '{persona} Return JSON like {"a": 1} and cite [1].'
        rendered = profile.render_system_prompt()
        self.assertIn('{"a": 1}', rendered)
        self.assertIn("cite [1]", rendered)


class NoDriftTests(ProfileTestCase):
    """Guards against the profile silently changing what the system was doing before
    these values moved out of the code."""

    def test_supermew_reproduces_the_original_identity(self):
        # redis_key_prefix and langsmith_project are env-overridable, and a real .env
        # may well set them, so clear those two to assert the profile's own values.
        with patch.dict(os.environ, {"REDIS_KEY_PREFIX": "", "LANGSMITH_PROJECT": ""}):
            profile = load_profile("supermew")
        self.assertEqual("SuperAgent API", profile.identity.api_title)
        self.assertEqual("superagent", profile.identity.redis_key_prefix)
        self.assertEqual("superagent-rag", profile.identity.langsmith_project)

    def test_original_retrieval_and_chunking_defaults(self):
        # Load `base` with retrieval/chunking env cleared so the profile value shows.
        cleared = {
            key: ""
            for key in registry.ENV_OVERRIDES
            if key.startswith(("RETRIEVAL_", "CHUNK_", "AUTO_MERGE_", "LEAF_", "RERANK_", "SEMANTIC_"))
        }
        with patch.dict(os.environ, cleared):
            profile = load_profile("base")
        self.assertEqual(8, profile.retrieval.top_k)
        self.assertEqual(3, profile.retrieval.candidate_multiplier)
        self.assertEqual(3, profile.retrieval.leaf_retrieve_level)
        self.assertTrue(profile.retrieval.auto_merge_enabled)
        self.assertEqual(2, profile.retrieval.auto_merge_threshold)
        self.assertEqual(800, profile.chunking.chunk_size)
        self.assertEqual(100, profile.chunking.chunk_overlap)
        self.assertEqual(4, profile.chunking.merge_target_divisor)
        self.assertFalse(profile.chunking.semantic_dedup_enabled)

    def test_planner_and_grader_stay_deterministic(self):
        """FAST_MODEL serves both the 0.2 note summariser and the 0.0 planner; if these
        collapse to one value, complexity classification stops being reproducible."""
        models = load_profile("base").models
        self.assertEqual(0.0, models.planner_temperature)
        self.assertEqual(0.0, models.grade_temperature)
        self.assertEqual(0.0, models.rewrite_temperature)
        self.assertEqual(0.2, models.fast_temperature)
        self.assertEqual(0.3, models.answer_temperature)

    def test_prompt_placeholders_are_preserved(self):
        """The prompts moved to backend/prompts/templates/, so the placeholders that
        must survive are the templates' — a template that stopped substituting its
        payload would render a grader prompt with no snippets in it and still look
        perfectly well-formed."""
        from backend.prompts import render

        marker = "PLACEHOLDER_MARKER"
        graded = render("rag/evidence_grade.j2", question=marker, context=marker, constraints=[])
        self.assertEqual(2, graded.count(marker))
        # Carried conditions are a third payload, and they must reach the grader as
        # their own section rather than folded into the question — see AssessmentContext.
        with_conditions = render(
            "rag/evidence_grade.j2", question=marker, context=marker,
            constraints=["CONDITION_MARKER"],
        )
        self.assertIn("CONDITION_MARKER", with_conditions)
        self.assertIn("constraints_discriminate", with_conditions)
        self.assertIn(marker, render("rag/complexity.j2", question=marker))
        self.assertIn(marker, render("rag/rewrite.j2", query=marker))

    def test_the_shipped_profiles_carry_no_prompt_text(self):
        """Prompts live in templates now. A non-empty key here is an override, and an
        accidental one would silently pin that deployment to stale wording."""
        profile = load_profile("base")
        self.assertEqual("", profile.rag.evidence_grade_prompt)
        self.assertEqual("", profile.rag.complexity_prompt)
        self.assertEqual("", profile.rag.rewrite_prompt)
        self.assertEqual("", profile.agent.resume_answer_prompt)
        self.assertEqual("", profile.agent.persistent_note_prompt)
        self.assertEqual("", profile.assets.figures.extraction_prompt)

    def test_arabic_fast_path_markers_survived_the_move_to_yaml(self):
        """The Arabic vocabulary is easy to mangle in transit; these entries are load
        bearing for the Arabic-first corpus."""
        rag = load_profile("base").rag
        for marker in ("ما هو", "متى", "كم", "هل"):
            self.assertIn(marker, rag.simple_query_markers)
        for marker in ("قارن", "لماذا", "كيف", "الفرق"):
            self.assertIn(marker, rag.complex_query_markers)
        # Trailing-space markers must keep their space (they guard against prefix
        # collisions such as "أي" inside a longer word).
        self.assertIn("أي ", rag.simple_query_markers)
        self.assertIn(" و ", rag.complex_query_markers)


class ToolRegistryTests(ProfileTestCase):
    def test_unknown_tool_name_fails_loudly(self):
        from backend.chat.request_context import ChatRequestContext
        from backend.tools import UnknownToolError, build_tools

        ctx = ChatRequestContext.for_sync(user_id="u", session_id="s")
        with self.assertRaises(UnknownToolError) as err:
            build_tools(["search_knowledge_base", "not_a_tool"], ctx)
        self.assertIn("not_a_tool", str(err.exception))

    def test_tools_are_built_in_declaration_order(self):
        from backend.chat.request_context import ChatRequestContext
        from backend.tools import build_tools

        ctx = ChatRequestContext.for_sync(user_id="u", session_id="s")
        tools = build_tools(["search_knowledge_base", "get_current_weather"], ctx)
        self.assertEqual(
            ["search_knowledge_base", "get_current_weather"],
            [tool.name for tool in tools],
        )


if __name__ == "__main__":
    unittest.main()
