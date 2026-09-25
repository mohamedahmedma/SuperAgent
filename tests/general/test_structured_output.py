"""RAG_FIX_PLAN item 47: model-output schemas are built once, not on every call.

Under load the serving process spent 10.4% of its CPU regenerating the same JSON schemas
and 4.5% re-creating the same pydantic classes. What these tests pin is that the cure
changes nothing a model or a caller can see — only what it costs.
"""
import unittest
from typing import List, Literal

from pydantic import BaseModel, Field

from backend.structured_output import StructuredOutput


class Plain(BaseModel):
    scope: Literal["in_domain", "out_of_domain"] = Field(description="x")
    tags: List[str] = Field(default_factory=list)


class Cached(StructuredOutput):
    scope: Literal["in_domain", "out_of_domain"] = Field(description="x")
    tags: List[str] = Field(default_factory=list)


class Other(StructuredOutput):
    flag: bool = False


def _untitled(schema):
    return {k: v for k, v in schema.items() if k != "title"}


class CachedSchemaTests(unittest.TestCase):
    def test_the_schema_is_the_one_pydantic_would_have_built(self):
        self.assertEqual(_untitled(Plain.model_json_schema()), _untitled(Cached.model_json_schema()))

    def test_a_caller_editing_its_copy_cannot_change_the_next_callers(self):
        """The OpenAI SDK's strict-schema pass edits the dict it is handed in place."""
        first = Cached.model_json_schema()
        first["properties"]["scope"]["description"] = "tampered"
        first["additionalProperties"] = False
        second = Cached.model_json_schema()
        self.assertEqual("x", second["properties"]["scope"]["description"])
        self.assertNotIn("additionalProperties", second)

    def test_classes_do_not_share_a_cache_entry(self):
        self.assertIn("scope", Cached.model_json_schema()["properties"])
        self.assertIn("flag", Other.model_json_schema()["properties"])
        self.assertNotIn("flag", Cached.model_json_schema()["properties"])

    def test_arguments_are_part_of_the_key(self):
        validation = Cached.model_json_schema(mode="validation")
        serialization = Cached.model_json_schema(mode="serialization")
        self.assertEqual(validation, Cached.model_json_schema(mode="validation"))
        self.assertEqual(serialization, Cached.model_json_schema(mode="serialization"))

    def test_the_openai_sdks_strict_schema_is_unchanged(self):
        from openai.lib._pydantic import to_strict_json_schema

        self.assertEqual(_untitled(to_strict_json_schema(Plain)), _untitled(to_strict_json_schema(Cached)))
        self.assertEqual(to_strict_json_schema(Cached), to_strict_json_schema(Cached))


class EverySchemaAModelAnswersInTests(unittest.TestCase):
    """Built once, at import — never again inside the function that calls the model."""

    def test_they_all_use_the_cached_base(self):
        from backend.chat.resolution import ResolvedQuery
        from backend.chat.signals import RequestEnvelope
        from backend.rag.pipeline import ComplexityResult, EvidenceGrade
        from backend.rag.scope_detector import ScopeVerdict
        from backend.rag.utils import RewritePlan

        for schema in (RequestEnvelope, ResolvedQuery, ScopeVerdict, EvidenceGrade,
                       ComplexityResult, RewritePlan):
            self.assertTrue(issubclass(schema, StructuredOutput), schema.__name__)

    def test_the_hoisted_classes_still_read_what_a_model_returns(self):
        from backend.chat.resolution import ResolvedQuery
        from backend.chat.signals import RequestEnvelope
        from backend.rag.scope_detector import ScopeVerdict

        envelope = RequestEnvelope.model_validate({"scope": "in_domain", "needed_tools": ["x"]})
        self.assertEqual(("in_domain", ["x"], "none"),
                         (envelope.scope, envelope.needed_tools, envelope.child_reference))
        resolved = ResolvedQuery.model_validate({"question": "fees?"})
        self.assertEqual(("fees?", "followup", ""), (resolved.question, resolved.intent, resolved.search_text))
        self.assertEqual("out_of_domain", ScopeVerdict.model_validate({"scope": "out_of_domain"}).scope)


class KnowledgeToolTests(unittest.TestCase):
    def test_the_model_facing_definition_is_unchanged(self):
        from langchain_core.utils.function_calling import convert_to_openai_tool

        from backend.chat.request_context import ChatRequestContext
        from backend.tools.knowledge import KnowledgeQuery, make_search_knowledge_base

        ctx = ChatRequestContext.for_sync(user_id="u", session_id="s")
        try:
            tool = make_search_knowledge_base(ctx)
            spec = convert_to_openai_tool(tool)["function"]
        finally:
            ctx.close()
        self.assertIs(KnowledgeQuery, tool.args_schema)
        self.assertEqual("search_knowledge_base", spec["name"])
        self.assertEqual({"properties": {"query": {"type": "string"}}, "required": ["query"],
                          "type": "object"}, spec["parameters"])


if __name__ == "__main__":
    unittest.main()
