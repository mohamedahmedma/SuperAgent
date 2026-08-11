"""The `school` profile — the only one that can reach a student's records.

Two opposite failures are guarded here, and they matter about equally:

The tool is bound where it should be, and actually builds. A profile naming a tool
that is not registered raises at process start, so a typo here is a failed deployment
rather than a failing test — unless a test catches it first.

The tool is bound NOWHERE else. Every bound tool ships its schema to the model on
every call, so a stray binding costs unrelated deployments real tokens per turn and
offers the model a capability it will occasionally try to use.
"""
import os
import unittest

import backend.profiles.registry as registry
from backend.chat.caller_identity import CallerIdentity
from backend.chat.request_context import ChatRequestContext
from backend.profiles.registry import (
    DEFAULT_PROFILE,
    available_profiles,
    load_profile,
    set_profile,
)
from backend.tools import TOOL_BUILDERS, build_tools

RECORDS_TOOL = "get_student_records"


class ProfileTestCase(unittest.TestCase):
    """Profiles are process-global; restore the cache and ACTIVE_PROFILE either way."""

    def setUp(self):
        self._saved_active = os.environ.get(registry.PROFILE_ENV_VAR)

    def tearDown(self):
        if self._saved_active is None:
            os.environ.pop(registry.PROFILE_ENV_VAR, None)
        else:
            os.environ[registry.PROFILE_ENV_VAR] = self._saved_active
        set_profile(None)


class SchoolProfileTests(ProfileTestCase):
    def test_it_is_shipped_and_loads(self):
        self.assertIn("school", available_profiles())
        self.assertEqual("school", load_profile("school").name)

    def test_it_binds_the_records_tool(self):
        self.assertIn(RECORDS_TOOL, load_profile("school").agent.tools)

    def test_every_tool_it_names_is_registered(self):
        """A typo here is a failed deployment, not a failing request.

        `build_tools` raises `UnknownToolError` at process start rather than skipping
        the name, so this is the cheapest possible place to find out.
        """
        unknown = [
            name for name in load_profile("school").agent.tools if name not in TOOL_BUILDERS
        ]
        self.assertEqual([], unknown)

    def test_its_tools_actually_build(self):
        profile = load_profile("school")
        ctx = ChatRequestContext.for_sync(user_id="u", session_id="s")

        tools = build_tools(profile.agent.tools, ctx)

        self.assertEqual(len(profile.agent.tools), len(tools))
        self.assertIn(RECORDS_TOOL, {tool.name for tool in tools})

    def test_it_drops_the_tools_a_school_has_no_use_for(self):
        """Lists replace rather than merge, so inheriting base must not drag these in."""
        tools = load_profile("school").agent.tools
        self.assertNotIn("get_current_weather", tools)
        self.assertNotIn("search_products", tools)

    def test_it_still_inherits_from_base(self):
        """It overrides identity, tools and retrieval only — RAG config must survive."""
        school = load_profile("school")
        base = load_profile("base")
        self.assertEqual(base.rag.max_rewrites, school.rag.max_rewrites)
        self.assertEqual(base.agent.recursion_limit, school.agent.recursion_limit)

    def test_its_arabic_display_name_survives_the_yaml_round_trip(self):
        """A mojibake school name is the first thing a parent would see."""
        self.assertEqual("مساعد المدرسة", load_profile("school").identity.display_name)

    def test_its_persona_is_not_the_inherited_cat(self):
        persona = load_profile("school").identity.persona
        self.assertNotIn("cat", persona.lower())
        self.assertNotEqual(load_profile("base").identity.persona, persona)

    def test_it_confirms_out_of_scope_questions_before_refusing(self):
        """Parents ask plenty a school corpus is not about; without this those get
        answered from general knowledge in the school's voice."""
        self.assertTrue(load_profile("school").agent.request_envelope_enabled)


class NoAccidentalExposureTests(ProfileTestCase):
    def test_no_other_shipped_profile_binds_the_records_tool(self):
        for name in available_profiles():
            if name == "school":
                continue
            with self.subTest(profile=name):
                self.assertNotIn(RECORDS_TOOL, load_profile(name).agent.tools)

    def test_adding_this_profile_did_not_change_the_default(self):
        """A new definition file must be inert until something selects it."""
        os.environ.pop(registry.PROFILE_ENV_VAR, None)
        self.assertEqual(DEFAULT_PROFILE, load_profile().name)
        self.assertNotIn(RECORDS_TOOL, load_profile().agent.tools)

    def test_selecting_it_by_environment_variable_works(self):
        os.environ[registry.PROFILE_ENV_VAR] = "school"
        self.assertEqual("school", load_profile().name)
        self.assertIn(RECORDS_TOOL, load_profile().agent.tools)


class BindingGrantsNothingTests(ProfileTestCase):
    """Binding the tool is not authorisation. It only makes the tool callable.

    Whether anything comes back is decided by the identity service (does this account
    carry a guardian claim) and the records facade (is that guardian linked to that
    student). Neither can be reached from a profile.
    """

    def _records_tool(self, ctx):
        tools = build_tools(load_profile("school").agent.tools, ctx)
        return next(tool for tool in tools if tool.name == RECORDS_TOOL)

    def test_a_non_parent_session_is_refused_even_with_the_tool_bound(self):
        ctx = ChatRequestContext.for_sync(user_id="u", session_id="s")
        result = self._records_tool(ctx).invoke({"record_type": "grades"})

        self.assertIn("NOT_A_PARENT_SESSION", result)

    def test_a_guardian_id_without_a_token_is_still_refused(self):
        """Half an identity must not read half a record."""
        ctx = ChatRequestContext.for_sync(
            user_id="u",
            session_id="s",
            caller=CallerIdentity(user_id="u", guardian_id="G-1", guardian_token=""),
        )
        result = self._records_tool(ctx).invoke({"record_type": "grades"})

        self.assertIn("NOT_A_PARENT_SESSION", result)


if __name__ == "__main__":
    unittest.main()
