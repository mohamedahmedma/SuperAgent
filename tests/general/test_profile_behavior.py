"""Behavioural tests: the active profile actually changes what the system does.

test_domain_profiles.py verifies the profile OBJECT is composed correctly. These tests
verify the wiring — that swapping a profile changes retrieval vocabulary, chunk sizes,
accepted uploads, agent tools, and user-facing copy at the point of use.

Two consumer shapes need different treatment:

* **call-time** consumers invoke `get_profile()` per call and are tested by swapping
  the cached profile;
* **import-time** consumers snapshot the profile into module constants, so they are
  re-executed from source under the profile being tested.
"""
import importlib.util
import os
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import backend.profiles.registry as registry
from backend.profiles.registry import load_profile, set_profile

REPO_ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def active_profile(name_or_profile):
    """Install a shipped profile as the process-wide active one for a test."""
    profile = load_profile(name_or_profile) if isinstance(name_or_profile, str) else name_or_profile
    set_profile(profile)
    try:
        yield profile
    finally:
        set_profile(None)


@contextmanager
def temp_profile(yaml_body: str, env: dict | None = None, name: str = "under_test"):
    """Build, activate, and tear down a throwaway profile.

    Deliberately has no `extends`, so unspecified values fall back to schema defaults
    and each test controls exactly the fields it cares about.

    `env` is patched BEFORE the profile loads, because environment overrides are baked
    in at load time. Patching afterwards leaves the real .env's values composed into
    the profile and silently invalidates the test — the trap that made an earlier
    version of these tests read 4 where the profile said 12.
    """
    with patch.dict(os.environ, env or {}):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / f"{name}.yaml").write_text(yaml_body, encoding="utf-8")
            with patch.object(registry, "DEFINITIONS_DIR", path):
                profile = load_profile(name)
            set_profile(profile)
            try:
                yield profile
            finally:
                set_profile(None)


def reexec_module(relative_path: str, fake_modules: dict | None = None):
    """Execute a backend module from source so its import-time profile snapshot is
    taken under whatever profile is currently active."""
    module_name = f"under_test_{relative_path.replace('/', '_').replace('.', '_')}_{id(fake_modules)}"
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, fake_modules or {}):
        spec.loader.exec_module(module)
    return module


def _fake_rag_utils():
    """Stand-in for backend.rag.utils so pipeline.py can be re-executed without
    pulling in embeddings, Milvus, or the rerank client."""
    fake_rag = types.ModuleType("backend.rag")
    fake_rag.__path__ = []
    fake_utils = types.ModuleType("backend.rag.utils")
    fake_utils.RETRIEVAL_TOP_K = 5
    fake_utils.retrieve_documents = lambda *a, **k: {"docs": [], "meta": {}}
    fake_utils.rewrite_query_once = lambda query: {}
    fake_utils.dedupe_documents = lambda docs: docs
    fake_utils.retrieval_trace_fields = lambda meta: dict(meta)
    return {"backend.rag": fake_rag, "backend.rag.utils": fake_utils}


def _fake_indexing():
    """Stand-in for the indexing package so rag/utils.py can be re-executed without
    loading the embedding model or connecting to Milvus."""
    fake_indexing = types.ModuleType("backend.indexing")
    fake_indexing.__path__ = []
    fake_milvus = types.ModuleType("backend.indexing.milvus_client")
    fake_milvus.get_milvus_store = lambda: object()
    fake_embedding = types.ModuleType("backend.indexing.embedding")
    fake_embedding.embedding_service = object()
    # Both the domain gate and retrieval ask for the query vector; the real module
    # memoizes so only one forward pass happens. The stub delegates so tests that
    # assert on what was embedded still see the call.
    fake_embedding.embed_query = lambda text: [0.1, 0.2, 0.3]
    fake_embedding.reset_query_vector_cache = lambda: None
    fake_parent = types.ModuleType("backend.indexing.parent_chunk_store")
    fake_parent.ParentChunkStore = type(
        "ParentChunkStore", (), {"get_documents_by_ids": lambda self, ids: []}
    )
    return {
        "backend.indexing": fake_indexing,
        "backend.indexing.milvus_client": fake_milvus,
        "backend.indexing.embedding": fake_embedding,
        "backend.indexing.parent_chunk_store": fake_parent,
    }


class ProfileTestCase(unittest.TestCase):
    def setUp(self):
        self._saved_active = os.environ.get(registry.PROFILE_ENV_VAR)

    def tearDown(self):
        if self._saved_active is None:
            os.environ.pop(registry.PROFILE_ENV_VAR, None)
        else:
            os.environ[registry.PROFILE_ENV_VAR] = self._saved_active
        set_profile(None)


# ---------------------------------------------------------------------------
# Call-time consumers
# ---------------------------------------------------------------------------

class UploadPolicyTests(ProfileTestCase):
    def test_accepted_extensions_follow_the_active_profile(self):
        from backend.api.resources import is_supported_document

        with active_profile("base"):
            self.assertTrue(is_supported_document("legacy.doc"))
            self.assertTrue(is_supported_document("legacy.xls"))

        # ecommerce deliberately omits the legacy binary Office formats.
        with active_profile("ecommerce"):
            self.assertFalse(is_supported_document("legacy.doc"))
            self.assertFalse(is_supported_document("legacy.xls"))
            self.assertTrue(is_supported_document("catalogue.pdf"))

    def test_extension_matching_is_case_insensitive(self):
        from backend.api.resources import is_supported_document

        with active_profile("base"):
            self.assertTrue(is_supported_document("REPORT.PDF"))
            self.assertTrue(is_supported_document("Sheet.XlSx"))

    def test_a_profile_with_no_extensions_accepts_nothing(self):
        from backend.api.resources import is_supported_document

        with temp_profile("name: locked\ningest:\n  supported_extensions: []\n"):
            self.assertFalse(is_supported_document("anything.pdf"))

    def test_rejection_message_comes_from_profile_copy(self):
        from backend.profiles import get_profile

        body = 'name: shouty\nuser_copy:\n  unsupported_file_type: "Nope, PDFs only."\n'
        with temp_profile(body):
            self.assertEqual("Nope, PDFs only.", get_profile().user_copy.unsupported_file_type)


class AgentBudgetTests(ProfileTestCase):
    def test_knowledge_tool_budget_follows_the_profile(self):
        from backend.chat.request_context import ChatRequestContext

        with temp_profile("name: chatty\nagent:\n  max_knowledge_calls_per_turn: 3\n"):
            ctx = ChatRequestContext.for_sync(user_id="u", session_id="s")
            self.assertEqual(
                [True, True, True, False],
                [ctx.acquire_knowledge_tool_slot() for _ in range(4)],
            )

    def test_default_budget_is_a_single_call(self):
        from backend.chat.request_context import ChatRequestContext

        with active_profile("base"):
            ctx = ChatRequestContext.for_sync(user_id="u", session_id="s")
            self.assertTrue(ctx.acquire_knowledge_tool_slot())
            self.assertFalse(ctx.acquire_knowledge_tool_slot())

    def test_budget_reset_respects_the_profile_limit(self):
        from backend.chat.request_context import ChatRequestContext

        with temp_profile("name: two\nagent:\n  max_knowledge_calls_per_turn: 2\n"):
            ctx = ChatRequestContext.for_sync(user_id="u", session_id="s")
            ctx.acquire_knowledge_tool_slot()
            ctx.acquire_knowledge_tool_slot()
            self.assertFalse(ctx.acquire_knowledge_tool_slot())
            ctx.reset_knowledge_tool_budget()
            self.assertTrue(ctx.acquire_knowledge_tool_slot())

    def test_zero_budget_blocks_the_knowledge_tool_entirely(self):
        from backend.chat.request_context import ChatRequestContext

        with temp_profile("name: none\nagent:\n  max_knowledge_calls_per_turn: 0\n"):
            ctx = ChatRequestContext.for_sync(user_id="u", session_id="s")
            self.assertFalse(ctx.acquire_knowledge_tool_slot())


class AgentAssemblyTests(ProfileTestCase):
    def _built_kwargs(self, profile_ctx):
        import backend.chat.runtime as runtime
        from backend.chat.request_context import ChatRequestContext

        ctx = ChatRequestContext.for_sync(user_id="u", session_id="s")
        with profile_ctx:
            with patch.object(runtime, "create_agent") as create_agent:
                runtime.create_agent_for_request(ctx)
        return create_agent.call_args.kwargs

    def test_agent_is_built_with_exactly_the_profile_tools_and_prompt(self):
        kwargs = self._built_kwargs(active_profile("document_kb"))
        self.assertEqual(
            ["search_knowledge_base", "view_figure"], [tool.name for tool in kwargs["tools"]]
        )
        self.assertTrue(kwargs["system_prompt"].startswith("You are a precise document assistant."))

    def test_supermew_profile_keeps_the_full_tool_set(self):
        """supermew is the full-feature test bed, so it carries every registered tool."""
        kwargs = self._built_kwargs(active_profile("supermew"))
        self.assertEqual(
            ["get_current_weather", "search_knowledge_base", "view_figure", "search_products"],
            [tool.name for tool in kwargs["tools"]],
        )
        self.assertTrue(kwargs["system_prompt"].startswith("You are a helpful knowledge-base assistant"))

    def test_tool_order_follows_the_profile_declaration(self):
        body = 'name: reversed\nagent:\n  tools: ["search_knowledge_base", "get_current_weather"]\n'
        kwargs = self._built_kwargs(temp_profile(body))
        self.assertEqual(
            ["search_knowledge_base", "get_current_weather"],
            [tool.name for tool in kwargs["tools"]],
        )

    def test_unregistered_tool_in_a_profile_fails_agent_construction(self):
        import backend.chat.runtime as runtime
        from backend.chat.request_context import ChatRequestContext
        from backend.tools import UnknownToolError

        ctx = ChatRequestContext.for_sync(user_id="u", session_id="s")
        with temp_profile('name: future\nagent:\n  tools: ["compare_products"]\n'):
            with self.assertRaises(UnknownToolError):
                runtime.create_agent_for_request(ctx)

    def test_a_toolless_profile_builds_an_agent_with_no_tools(self):
        kwargs = self._built_kwargs(temp_profile("name: bare\nagent:\n  tools: []\n"))
        self.assertEqual([], kwargs["tools"])


class CacheNamespaceTests(ProfileTestCase):
    def test_redis_prefix_isolates_profiles_sharing_one_instance(self):
        from backend.infra.cache import RedisCache

        with patch.dict(os.environ, {"REDIS_KEY_PREFIX": ""}):
            with active_profile("ecommerce"):
                self.assertEqual("shop", RedisCache().key_prefix)
            with active_profile("document_kb"):
                self.assertEqual("dockb", RedisCache().key_prefix)

    def test_env_still_overrides_the_profile_prefix(self):
        from backend.infra.cache import RedisCache

        with patch.dict(os.environ, {"REDIS_KEY_PREFIX": "pinned"}):
            with active_profile(load_profile("ecommerce")):
                self.assertEqual("pinned", RedisCache().key_prefix)


# ---------------------------------------------------------------------------
# Import-time consumers (module re-execution)
# ---------------------------------------------------------------------------

class PipelineBehaviourTests(ProfileTestCase):
    @staticmethod
    def _pipeline():
        return reexec_module("backend/rag/pipeline.py", _fake_rag_utils())

    def test_fast_path_vocabulary_comes_from_the_active_profile(self):
        with active_profile("ecommerce"):
            pipeline = self._pipeline()

        # An ecommerce-specific marker classifies as simple...
        self.assertIsNotNone(pipeline._simple_question_fast_path_reason("price of the blue shoe"))
        # ...and an ecommerce comparison marker blocks the fast path.
        self.assertIsNone(pipeline._simple_question_fast_path_reason("recommend a running shoe"))

    def test_base_profile_does_not_know_ecommerce_vocabulary(self):
        with active_profile("base"):
            pipeline = self._pipeline()
        self.assertIsNone(pipeline._simple_question_fast_path_reason("price of the blue shoe today"))

    def test_fast_path_length_limit_is_profile_driven(self):
        question = "what is the refund window for online orders placed abroad"  # 58 chars
        with active_profile("base"):  # limit 48
            base_pipeline = self._pipeline()
        with active_profile("ecommerce"):  # limit 64
            shop_pipeline = self._pipeline()

        self.assertIsNone(base_pipeline._simple_question_fast_path_reason(question))
        self.assertIsNotNone(shop_pipeline._simple_question_fast_path_reason(question))

    def test_empty_marker_lists_disable_the_fast_path_markers(self):
        """Only the vocabulary is profile data. The wh-pattern rule is independent of
        it, so this uses a phrasing that pattern cannot match — otherwise the test
        would be asserting that a rule it never configured had been turned off."""
        body = (
            "name: nomarkers\nrag:\n  simple_query_markers: []\n"
            "  simple_override_markers: []\n"
            "  complex_query_markers: []\n  fast_path_short_intent_chars: 0\n"
        )
        with temp_profile(body):
            pipeline = self._pipeline()
        self.assertIsNone(pipeline._simple_question_fast_path_reason("the fee for grade 5"))
        # With the vocabulary restored, the same question is recognised.
        with temp_profile(
            'name: withmarkers\nrag:\n  simple_query_markers: ["the fee"]\n'
            "  fast_path_short_intent_chars: 0\n"
        ):
            restored = self._pipeline()
        self.assertIsNotNone(restored._simple_question_fast_path_reason("the fee for grade 5"))

    @staticmethod
    def _rewrite_report():
        """A HIGH-certainty report asking to rewrite — what the grader produces when the
        evidence is only partially on topic."""
        from backend.rag.evidence import Certainty, ChunkAssessment, EvidenceReport

        return EvidenceReport(
            chunks=[ChunkAssessment(index=1)],
            certainty=Certainty.HIGH,
            relevance="weak",
            sufficiency="partial",
            preferred_route="rewrite",
        )

    @staticmethod
    def _rewrite_report_with_named_gap():
        """As above, but with something specific to ask the user for. Without a named
        slot the policy answers from partial evidence rather than asking — see
        HumanInTheLoopTests."""
        from backend.rag.evidence import Certainty, ChunkAssessment, EvidenceReport

        return EvidenceReport(
            chunks=[ChunkAssessment(index=1)],
            certainty=Certainty.HIGH,
            relevance="weak",
            sufficiency="partial",
            preferred_route="rewrite",
            ambiguity="missing_slot",
            missing_slots=["grade"],
        )

    def test_rewrite_budget_is_profile_driven(self):
        from backend.rag.policy import decide_route

        with temp_profile("name: patient\nrag:\n  max_rewrites: 3\n"):
            pipeline = self._pipeline()

        report = self._rewrite_report()
        kwargs = {"has_docs": True, "is_sub_agent": False, "config": pipeline._RAG}
        self.assertEqual("rewrite", decide_route(report, rewrite_count=2, **kwargs)[0])
        # Budget exhausted at 3: with a named gap the user is asked, without one the
        # partial evidence is answered from.
        named = self._rewrite_report_with_named_gap()
        self.assertEqual("clarify", decide_route(named, rewrite_count=3, **kwargs)[0])
        self.assertEqual("answer", decide_route(report, rewrite_count=3, **kwargs)[0])

    def test_zero_rewrite_budget_never_rewrites(self):
        from backend.rag.policy import decide_route

        with temp_profile("name: strict\nrag:\n  max_rewrites: 0\n"):
            pipeline = self._pipeline()

        route, _ = decide_route(
            self._rewrite_report_with_named_gap(),
            has_docs=True,
            rewrite_count=0,
            is_sub_agent=False,
            config=pipeline._RAG,
        )
        self.assertEqual("clarify", route)

    def test_hitl_copy_is_profile_driven(self):
        with active_profile("ecommerce"):
            pipeline = self._pipeline()

        grade = pipeline.EvidenceGrade(
            relevance="strong",
            answerability="partial",
            ambiguity="multiple_candidates",
            route="scope_select",
        )
        self.assertEqual(
            "I found a few product lines that could match. Which one did you mean?",
            pipeline._default_hitl_prompt("scope_select", grade),
        )

    def test_missing_slot_copy_is_profile_driven(self):
        body = 'name: slots\nuser_copy:\n  hitl_clarify_missing_slots: "Still need: "\n'
        with temp_profile(body):
            pipeline = self._pipeline()

        grade = pipeline.EvidenceGrade(
            relevance="strong",
            answerability="partial",
            ambiguity="missing_slot",
            route="clarify",
            missing_slots=["grade", "term"],
        )
        self.assertEqual("Still need: grade, term", pipeline._default_hitl_prompt("clarify", grade))

    def test_sub_question_cap_is_profile_driven(self):
        with temp_profile("name: wide\nrag:\n  max_sub_questions: 2\n"):
            pipeline = self._pipeline()
        self.assertEqual(2, pipeline._RAG.max_sub_questions)
        with self.assertRaises(Exception):
            pipeline.ComplexityResult(complexity="complex", sub_questions=["a", "b", "c"])

    def test_prompts_are_taken_from_the_profile(self):
        body = (
            'name: terse\nrag:\n  evidence_grade_prompt: "Grade {question} against {context}"\n'
            '  complexity_prompt: "How hard is {question}?"\n'
        )
        with temp_profile(body):
            pipeline = self._pipeline()
        self.assertEqual("Grade {question} against {context}", pipeline.EVIDENCE_GRADE_PROMPT)
        self.assertEqual("How hard is {question}?", pipeline.COMPLEXITY_PROMPT)


class ChunkingBehaviourTests(ProfileTestCase):
    @staticmethod
    def _loader_module():
        return reexec_module("backend/indexing/document_loader.py")

    def test_chunk_sizes_come_from_the_profile(self):
        body = "name: chunky\nchunking:\n  chunk_size: 400\n  chunk_overlap: 40\n"
        cleared = {"CHUNK_SIZE": "", "CHUNK_OVERLAP": "", "CHUNK_L1_SIZE": "", "CHUNK_L2_SIZE": ""}
        with temp_profile(body, env=cleared):
            loader = self._loader_module().DocumentLoader()

        self.assertEqual(600, loader._level_3_size)   # max(600, 400) floor still applies
        self.assertEqual(2000, loader._level_1_size)  # max(2000, 400*3)
        self.assertEqual(1000, loader._level_2_size)  # max(1000, 400*2)

    def test_explicit_l1_l2_in_the_profile_win_over_derivation(self):
        body = "name: explicit\nchunking:\n  chunk_size: 800\n  l1_size: 5000\n  l2_size: 2500\n"
        cleared = {"CHUNK_SIZE": "", "CHUNK_L1_SIZE": "", "CHUNK_L2_SIZE": ""}
        with temp_profile(body, env=cleared):
            loader = self._loader_module().DocumentLoader()

        self.assertEqual(5000, loader._level_1_size)
        self.assertEqual(2500, loader._level_2_size)

    def test_env_overrides_the_profile_chunk_size(self):
        """Precedence at the point of use: constructor arg > env > profile."""
        body = "name: chunky\nchunking:\n  chunk_size: 400\n"
        with temp_profile(body, env={"CHUNK_SIZE": "1200", "CHUNK_L1_SIZE": ""}):
            loader = self._loader_module().DocumentLoader()
        self.assertEqual(3600, loader._level_1_size)  # derived from the env value, not 400

    def test_constructor_argument_beats_both(self):
        body = "name: chunky\nchunking:\n  chunk_size: 400\n"
        with temp_profile(body, env={"CHUNK_SIZE": "1200", "CHUNK_L1_SIZE": ""}):
            loader = self._loader_module().DocumentLoader(chunk_size=900)
        self.assertEqual(2700, loader._level_1_size)

    def test_strategy_comes_from_the_profile(self):
        body = "name: sentences\nchunking:\n  strategy: sentence\n"
        with temp_profile(body, env={"CHUNK_STRATEGY": ""}):
            loader = self._loader_module().DocumentLoader()
        self.assertEqual("sentence", loader._strategy)

    def test_invalid_profile_strategy_is_rejected_at_load(self):
        from backend.profiles.registry import ProfileError

        with self.assertRaises(ProfileError):
            with temp_profile("name: bogus\nchunking:\n  strategy: quantum\n"):
                pass

    def test_merge_target_divisor_is_profile_driven(self):
        with temp_profile("name: coarse\nchunking:\n  merge_target_divisor: 1\n"):
            module = self._loader_module()
        self.assertEqual(1, module.MERGE_TARGET_DIVISOR)

    def test_layout_parser_can_be_disabled_by_the_profile(self):
        body = "name: flat\nchunking:\n  layout_parser_enabled: false\n"
        cleared = {"LAYOUT_PARSER_ENABLED": "", "PDF_LAYOUT_PARSER_ENABLED": ""}
        with temp_profile(body, env=cleared):
            module = self._loader_module()
        self.assertFalse(module.LAYOUT_PARSER_ENABLED)
        # The PDF-specific override defaults to the master switch.
        self.assertFalse(module.PDF_LAYOUT_PARSER_ENABLED)

    def test_pdf_override_can_re_enable_layout_parsing(self):
        body = "name: flat\nchunking:\n  layout_parser_enabled: false\n"
        env = {"LAYOUT_PARSER_ENABLED": "", "PDF_LAYOUT_PARSER_ENABLED": "true"}
        with temp_profile(body, env=env):
            module = self._loader_module()
        self.assertFalse(module.LAYOUT_PARSER_ENABLED)
        self.assertTrue(module.PDF_LAYOUT_PARSER_ENABLED)


class WriterBehaviourTests(ProfileTestCase):
    def test_semantic_dedup_is_profile_driven(self):
        from backend.indexing.milvus_writer import MilvusWriter

        body = "name: dedup\nchunking:\n  semantic_dedup_enabled: true\n  semantic_dedup_threshold: 0.5\n"
        cleared = {"SEMANTIC_DEDUP_ENABLED": "", "SEMANTIC_DEDUP_THRESHOLD": ""}
        with temp_profile(body, env=cleared):
            writer = MilvusWriter(embedding_service=object(), milvus_manager=object())

        self.assertTrue(writer.semantic_dedup_enabled)
        self.assertEqual(0.5, writer.semantic_dedup_threshold)

    def test_env_overrides_profile_dedup_setting(self):
        from backend.indexing.milvus_writer import MilvusWriter

        body = "name: dedup\nchunking:\n  semantic_dedup_enabled: true\n"
        with temp_profile(body, env={"SEMANTIC_DEDUP_ENABLED": "false"}):
            writer = MilvusWriter(embedding_service=object(), milvus_manager=object())
        self.assertFalse(writer.semantic_dedup_enabled)


class RetrievalBehaviourTests(ProfileTestCase):
    @staticmethod
    def _utils_module():
        return reexec_module("backend/rag/utils.py", _fake_indexing())

    def test_retrieval_tuning_comes_from_the_profile(self):
        body = (
            "name: wide\nretrieval:\n  top_k: 12\n  candidate_k: 96\n"
            "  leaf_retrieve_level: 2\n  auto_merge_threshold: 5\n"
        )
        cleared = {
            "RETRIEVAL_TOP_K": "",
            "RETRIEVAL_CANDIDATE_K": "",
            "LEAF_RETRIEVE_LEVEL": "",
            "AUTO_MERGE_THRESHOLD": "",
        }
        with temp_profile(body, env=cleared):
            utils = self._utils_module()

        self.assertEqual(12, utils.RETRIEVAL_TOP_K)
        self.assertEqual(2, utils.LEAF_RETRIEVE_LEVEL)
        self.assertEqual(5, utils.AUTO_MERGE_THRESHOLD)
        # The trace distinguishes a pool size set by the profile from one pinned by env.
        self.assertEqual(
            (96, {"candidate_k_source": "profile", "retrieval_candidate_multiplier": 3}),
            utils.resolve_candidate_k(12),
        )

    def test_candidate_k_source_reports_env_when_env_pins_it(self):
        body = "name: wide\nretrieval:\n  top_k: 12\n  candidate_k: 96\n"
        with temp_profile(body, env={"RETRIEVAL_CANDIDATE_K": "40"}):
            utils = self._utils_module()
        candidate_k, meta = utils.resolve_candidate_k(12)
        self.assertEqual(40, candidate_k)
        self.assertEqual("env", meta["candidate_k_source"])

    def test_null_candidate_k_falls_back_to_the_multiplier(self):
        body = "name: derived\nretrieval:\n  top_k: 5\n  candidate_multiplier: 4\n"
        cleared = {
            "RETRIEVAL_TOP_K": "",
            "RETRIEVAL_CANDIDATE_K": "",
            "RETRIEVAL_CANDIDATE_MULTIPLIER": "",
        }
        with temp_profile(body, env=cleared):
            utils = self._utils_module()

        candidate_k, meta = utils.resolve_candidate_k(5)
        self.assertEqual(20, candidate_k)
        self.assertEqual("multiplier", meta["candidate_k_source"])

    def test_auto_merge_can_be_disabled_by_the_profile(self):
        body = "name: nomerge\nretrieval:\n  auto_merge_enabled: false\n"
        with temp_profile(body, env={"AUTO_MERGE_ENABLED": ""}):
            utils = self._utils_module()
        self.assertFalse(utils.AUTO_MERGE_ENABLED)

    def test_rewrite_prompt_comes_from_the_profile(self):
        body = 'name: terse\nrag:\n  rewrite_prompt: "Rewrite this: {query}"\n'
        with temp_profile(body):
            utils = self._utils_module()
        self.assertEqual("Rewrite this: {query}", utils.REWRITE_PROMPT)

    def test_rerank_limits_come_from_the_profile(self):
        body = "name: picky\nretrieval:\n  rerank_min_score: 0.4\n  rerank_doc_char_limit: 800\n"
        cleared = {"RERANK_MIN_SCORE": "", "RERANK_DOC_CHAR_LIMIT": ""}
        with temp_profile(body, env=cleared):
            utils = self._utils_module()
        self.assertEqual(0.4, utils.RERANK_MIN_SCORE)
        self.assertEqual(800, utils.RERANK_DOC_CHAR_LIMIT)


# ---------------------------------------------------------------------------
# Environment reader semantics
# ---------------------------------------------------------------------------

class EnvReaderTests(unittest.TestCase):
    """A variable set to an empty string must count as unset, everywhere.

    `.env` files routinely contain `FOO=` for something a person meant to disable.
    Under the bare `os.getenv(name, default)` form that returns "", which either
    crashes (`int("")`) or picks the wrong branch — and it made module import fail
    outright before these readers existed.
    """

    def test_blank_is_treated_as_unset(self):
        from backend.env import env_bool, env_float, env_int, env_value

        with patch.dict(os.environ, {"X_TEST": ""}):
            self.assertIsNone(env_value("X_TEST"))
            self.assertEqual(7, env_int("X_TEST", 7))
            self.assertEqual(1.5, env_float("X_TEST", 1.5))
            self.assertTrue(env_bool("X_TEST", True))

    def test_whitespace_only_is_treated_as_unset(self):
        from backend.env import env_int, env_value

        with patch.dict(os.environ, {"X_TEST": "   "}):
            self.assertIsNone(env_value("X_TEST"))
            self.assertEqual(7, env_int("X_TEST", 7))

    def test_absent_variable_uses_the_default(self):
        from backend.env import env_bool, env_float, env_int, env_value

        os.environ.pop("X_ABSENT", None)
        self.assertIsNone(env_value("X_ABSENT"))
        self.assertEqual(3, env_int("X_ABSENT", 3))
        self.assertEqual(2.5, env_float("X_ABSENT", 2.5))
        self.assertFalse(env_bool("X_ABSENT", False))

    def test_values_are_parsed_and_trimmed(self):
        from backend.env import env_bool, env_float, env_int, env_value

        with patch.dict(os.environ, {"X_TEST": "  42  "}):
            self.assertEqual("42", env_value("X_TEST"))
            self.assertEqual(42, env_int("X_TEST", 0))
        with patch.dict(os.environ, {"X_TEST": "0.25"}):
            self.assertEqual(0.25, env_float("X_TEST", 0.0))
        for raw, expected in [("true", True), ("1", True), ("YES", True), ("on", True),
                              ("false", False), ("0", False), ("No", False), ("off", False)]:
            with self.subTest(raw=raw), patch.dict(os.environ, {"X_TEST": raw}):
                self.assertEqual(expected, env_bool("X_TEST", not expected))

    def test_malformed_values_fall_back_to_the_default(self):
        """A typo such as RERANK_DOC_CHAR_LIMIT=#2000 must not crash startup."""
        from backend.env import env_bool, env_float, env_int

        with patch.dict(os.environ, {"X_TEST": "#2000"}):
            self.assertEqual(2000, env_int("X_TEST", 2000))
        with patch.dict(os.environ, {"X_TEST": "#3"}):
            self.assertEqual(5.0, env_float("X_TEST", 5.0))
        with patch.dict(os.environ, {"X_TEST": "maybe"}):
            self.assertTrue(env_bool("X_TEST", True))

    def test_minimum_is_enforced_on_both_value_and_default(self):
        from backend.env import env_float, env_int

        with patch.dict(os.environ, {"X_TEST": "1"}):
            self.assertEqual(200, env_int("X_TEST", 2000, minimum=200))
        with patch.dict(os.environ, {"X_TEST": ""}):
            self.assertEqual(200, env_int("X_TEST", 10, minimum=200))
            self.assertEqual(0.1, env_float("X_TEST", 0.01, minimum=0.1))


if __name__ == "__main__":
    unittest.main()
