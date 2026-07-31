"""DomainProfile schema: every value the system's behaviour, prompts, and identity
depend on, in one declarative place.

Design rules that the rest of the codebase relies on:

1. **Every field has a default equal to the value it replaced.** A profile that omits
   a key behaves exactly like the pre-profile hardcoded system. This is what makes it
   safe to introduce profiles without a behavioural migration.
2. **Nothing here reads the environment.** Env overrides are applied by the registry
   AFTER composition, so precedence is a single documented rule (env > profile >
   default) rather than scattered `os.getenv` calls with their own fallbacks.
3. **Sections are stable, fields are additive.** Downstream code imports a section
   (`profile.rag`), never the whole profile, so adding a field to one section cannot
   ripple into unrelated modules.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from backend.assets.attributes import AttributeSpec


class _Section(BaseModel):
    # Reject unknown keys: a typo in a YAML profile is a silent behaviour change
    # otherwise, and profiles are exactly the kind of file people hand-edit.
    model_config = ConfigDict(extra="forbid")


class IdentityConfig(_Section):
    """Who the deployment says it is. Replaces every hardcoded product string."""

    name: str = "SuperMew"
    display_name: str = "SuperMew"
    api_title: str = "SuperMew API"
    description: str = "Knowledge-base assistant"
    # Injected into the agent system prompt as {persona}; the one place a
    # deployment's voice is defined.
    persona: str = "You are a cute cat bot that loves to help users."
    redis_key_prefix: str = "supermew"
    langsmith_project: str = "supermew-rag"


class ModelConfig(_Section):
    """Sampling settings per model role. Model IDs stay in env — they are deployment
    credentials, not domain behaviour, and belong with the API key that reaches them."""

    answer_temperature: float = 0.3
    # FAST_MODEL is used for two different jobs at two different temperatures:
    # conversational note-keeping (fast_temperature) and deterministic structured
    # planning (planner_temperature). Collapsing them would silently make complexity
    # classification and query rewriting non-deterministic.
    fast_temperature: float = 0.2
    planner_temperature: float = 0.0
    grade_temperature: float = 0.0
    rewrite_temperature: float = 0.0


class AgentConfig(_Section):
    """The tool-calling agent's contract."""

    # Empty means "use the shipped template", which is the normal case — the agent
    # prompt is composed from backend/prompts/templates/agent/system.j2 so that a turn
    # pays only for the capabilities it actually bound.
    #
    # Setting it is an escape hatch: the string is used verbatim (with {persona}
    # substituted) and NOTHING is composed, so a profile that sets it also opts out of
    # per-turn narrowing. Worth it for a deployment that needs total control of the
    # wording; a domain that only wants to change part of the prompt should add a
    # template pack and override one block instead.
    system_prompt: str = ""
    # System prompt for the direct-answer path taken when a HITL clarification is
    # resumed — that path bypasses the agent, so it needs its own instructions.
    resume_answer_prompt: str = ""
    # Prompt for the rolling conversation summary ("persistent note").
    persistent_note_prompt: str = ""

    tools: List[str] = Field(default_factory=lambda: ["get_current_weather", "search_knowledge_base"])
    recursion_limit: int = 8
    max_knowledge_calls_per_turn: int = 1
    # view_figure gets its own budget, carved explicitly out of the single-knowledge-
    # call rule. Sharing that budget would make looking at a figure cost the turn its
    # only retrieval, which is the opposite of what the tool is for.
    max_figure_calls_per_turn: int = 3
    context_window_messages: int = 6
    persistent_note_max_chars: int = 500

    # Messages that are ENTIRELY one of these are answered without searching anything.
    # Matched whole-message after normalisation, never as a prefix or keyword: "thanks"
    # is social, "thanks, and what are the fees?" is a question, and answering the
    # second with a pleasantry is the failure this list must not cause. Empty disables
    # the check. Domain-specific and language-specific, hence profile data.
    social_phrases: List[str] = Field(
        default_factory=lambda: [
            "hi", "hello", "hey", "good morning", "good evening",
            "thanks", "thank you", "thanks a lot", "thank you very much",
            "ok", "okay", "sure", "great", "cool", "nice", "bye", "goodbye",
            "مرحبا", "اهلا", "أهلا", "السلام عليكم", "صباح الخير", "مساء الخير",
            "شكرا", "شكرا لك", "شكرا جزيلا", "تمام", "حسنا", "طيب", "مع السلامة",
        ]
    )
    # How a social turn is answered.
    #   model  — one call with every tool unbound: cheap, and replies stay varied
    #   static — no call at all, using user_copy.social
    # `model` by default because a greeting is the first thing a user sees, and an
    # identical canned string every time reads as a script.
    social_reply_mode: Literal["model", "static"] = "model"

    # The escalation rung of the request classifier: one small call that settles scope
    # and reads disclosed profile fields in a single envelope. Off until wired.
    # Without it the corpus-similarity rung can only ever ADMIT a question — it is
    # never permitted to end a turn on its own.
    request_envelope_enabled: bool = False


class RagConfig(_Section):
    """Prompts and routing policy for the retrieval graph."""

    evidence_grade_prompt: str = ""
    complexity_prompt: str = ""
    rewrite_prompt: str = ""

    max_rewrites: int = 1
    max_sub_questions: int = 4

    # Complexity planning: one FAST_MODEL call that classifies the question and, when it
    # is genuinely multi-part, decomposes it into sub-questions retrieved in parallel.
    #
    # It is a whole subsystem behind one switch, and its cost is not the planner call —
    # that is the cheap part. Each sub-question pays its own retrieval AND its own grader
    # call, so a decomposed turn costs a multiple of a plain one.
    #
    # When False the node is not registered at all: the graph starts at retrieve_initial,
    # and prepare_sub_questions / rag_sub_agent / synthesis become unreachable rather than
    # merely unused. Nothing downstream fabricates a classification to compensate —
    # `complexity` stays unset, because nothing classified anything.
    complexity_planning_enabled: bool = True

    # When to spend a GRADE_MODEL call on the retrieved evidence.
    #
    #   always         — every retrieval is graded (the original behaviour)
    #   uncertain_only — grade unless retrieval is CLEARLY on target
    #   never          — no grading; docs answer, no docs mean no_knowledge.
    #                    This also disables rewrite and HITL, which both derive
    #                    from the grade.
    #
    # The semantic rung of the ladder: a cross-encoder scoring the question against each
    # retrieved chunk. Off by default because it loads a model — enable per deployment
    # once the latency is measured on the target hardware.
    rerank_cross_encoder_enabled: bool = False
    # bge-reranker-v2-m3 is the same family as the bge-m3 embedder and is strong on
    # Arabic. `-base` is roughly half the parameters if CPU latency is the constraint.
    rerank_cross_encoder_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_cross_encoder_device: str = "cpu"
    # Only the opening span of a chunk carries the relevance signal, and pairs are the
    # cost unit, so truncating is nearly free accuracy-wise.
    rerank_doc_char_limit: int = 2000
    # Above this a chunk is treated as carrying the evidence; below `irrelevant` the whole
    # pool is dismissed. Between the two the assessor abstains and the grader decides —
    # that middle band is where an LLM is actually worth its cost.
    rerank_sufficient_score: float = 0.6
    rerank_irrelevant_score: float = 0.05
    rerank_min_supporting_chunks: int = 1

    # ---- Scope: is an incoming question this corpus's subject at all? --------------
    # Judged against a catalogue of the questions the corpus can answer, built once per
    # section at ingest. Scoring questions against questions is the most homogeneous
    # comparison available, which is what lets one cut point behave consistently;
    # scoring against chunk text does not separate (measured: 0.58 in-corpus vs 0.46
    # for a poem).
    scope_index_enabled: bool = False
    # Which level of the chunk hierarchy is a "section". 1 is the top. Document level
    # is deliberately not used: a single-document corpus would yield one vector.
    scope_section_level: int = 1
    # Empty means FAST_MODEL. Summarisation is an ingest cost, paid once per changed
    # section, so a small model is usually right.
    scope_summary_model: str = ""
    scope_min_questions: int = 4
    scope_max_questions: int = 10
    # Languages the anticipated questions are written in. A question only matches a
    # query in the same language, so an Arabic-first corpus must catalogue both.
    scope_question_languages: str = "English and Arabic"
    # Frozen so the corpus catalogue cannot drift between re-indexes. The summariser
    # chooses from this list and invented labels are discarded.
    scope_topic_vocabulary: List[str] = Field(default_factory=list)
    # Percentile of in-scope self-similarity used as the escalate-below floor. The cut
    # point is DERIVED from the corpus at index time, not chosen — a hand-tuned number
    # is picked against one corpus and silently rots on the next re-index. Low on
    # purpose: escalating unnecessarily costs one small call, admitting silently costs
    # a search that would have happened anyway.
    scope_floor_percentile: float = 10.0
    # How many matched questions the scope model is shown. It judges evidence rather
    # than guessing from a topic list, which is the best defence against the one
    # unrecoverable failure here — refusing a valid question.
    scope_match_count: int = 3

    # Rejecting a question the corpus cannot answer before searching for it. Off by
    # default: the threshold has to be calibrated against a corpus before it can be
    # trusted, and an uncalibrated gate refuses valid questions. Turn it on per profile
    # once `evals/` shows where the boundary sits.
    domain_gate_enabled: bool = False
    # Cosine against the best-matching corpus section. Bias this LOW — a false rejection
    # is a silent refusal on a valid question, while a false acceptance costs one search
    # the grader was going to catch anyway.
    domain_gate_min_similarity: float = 0.35
    # How many sections may be carried forward as topic hints for the retrieval filter.
    # More than one on purpose: a single guessed topic that is wrong destroys recall.
    domain_gate_topic_count: int = 3
    # A bare follow-up ("what about grade 6?") carries its topic in the previous turn, so
    # scoring its own words measures nothing.
    domain_gate_skip_with_history: bool = True
    domain_gate_min_question_chars: int = 12

    # How far up the assessor ladder a retrieval must climb before its report may be
    # acted on. Rungs run cheapest-first and stop here, so this one value decides how
    # much a turn spends on judging evidence:
    #   low    — structural and lexical proxies only (fast, semantically blind)
    #   medium — a calibrated cross-encoder relevance score
    #   high   — a language model read the chunks and the question together
    # Held at `high` while the ladder's only semantic rung is the LLM grader; drop to
    # `medium` once the reranker is supplying scores and the grader becomes the
    # exception rather than the rule.
    evidence_required_certainty: Literal["low", "medium", "high"] = "high"
    # Lexical overlap is kept as a cheap negative signal and a trace breadcrumb. It can
    # never conclude on its own, so disabling it changes reporting, not routing.
    evidence_lexical_enabled: bool = True

    # The grader is one synchronous call per retrieval — and one per sub-agent on a
    # decomposed question — so it is usually the dominant latency in a turn. Much of
    # the time it only confirms what retrieval already made obvious.
    grading_mode: Literal["always", "uncertain_only", "never"] = "uncertain_only"

    # Thresholds for "clearly on target". Every one must hold before grading is
    # skipped: skipping wrongly means answering from weak evidence, which is the
    # failure this pipeline exists to prevent, while grading unnecessarily only costs
    # latency. The bias is therefore firmly toward calling the grader.
    skip_grading_min_chunks: int = 3
    # Fraction of the question's own content words that must appear in the top chunks.
    skip_grading_term_coverage: float = 0.75
    skip_grading_top_chunks: int = 3
    # Only applied when a reranker actually ran — its scores are calibrated, whereas
    # raw RRF fusion scores are not comparable across queries.
    skip_grading_min_rerank_score: float = 0.5

    # How much of the retrieved set actually reaches the answer prompt. Retrieval
    # returns a fixed top_k; answering rarely needs all of it, and the tool result is
    # re-sent on every subsequent model call in the turn, so the saving compounds.
    #   adaptive | off
    context_selection_mode: Literal["adaptive", "off"] = "adaptive"
    # Never go below this many chunks. 1 permits a single-chunk answer when that chunk
    # fully covers the question and is long enough to plausibly contain it; 2 is the
    # conservative setting for corpora with heavy cross-referencing between passages.
    context_min_chunks: int = 1
    # Stop once this fraction of the question's content words is covered. 1.0 means
    # every content word must appear before anything is dropped.
    context_target_coverage: float = 1.0
    # A chunk shorter than this is treated as a heading: it may be kept as context but
    # can never be the chunk that ends selection, since headings match a question's
    # vocabulary perfectly while containing no answer.
    context_min_chunk_chars: int = 120

    # Local fast-path classification rules. These are language- and domain-specific:
    # an e-commerce profile wants product-attribute markers where the school corpus
    # wants admissions vocabulary, which is precisely why they are profile data and
    # not module constants.
    # Checked BEFORE complex_query_markers. Some single-fact phrasings contain a
    # substring that also opens a genuinely complex question — "how many students"
    # contains "how ". Without this override the complex marker wins, the fast path is
    # skipped, and a plain lookup is sent to the planner (and possibly decomposed into
    # several sub-questions, each costing its own retrieval and grader call).
    simple_override_markers: List[str] = Field(default_factory=list)
    simple_query_markers: List[str] = Field(default_factory=list)
    complex_query_markers: List[str] = Field(default_factory=list)
    query_dimension_markers: List[str] = Field(default_factory=list)
    fast_path_max_chars: int = 48
    fast_path_short_intent_chars: int = 18


class RetrievalConfig(_Section):
    top_k: int = 8
    candidate_k: Optional[int] = None
    candidate_multiplier: int = 3
    leaf_retrieve_level: int = 3

    auto_merge_enabled: bool = True
    auto_merge_threshold: int = 2

    # Threshold applied instead when a parent's children include a FIGURE chunk.
    #
    # Merging exists to give a text hit its surrounding context, which is worth doing.
    # For figures it costs something specific: two figure leaves collapse into one
    # parent, so an answer citing `[1]` points at two images and the reader has to
    # work out which one was meant. Leaving figure groups unmerged keeps each image
    # individually citable, and the parent's text context is rarely what a figure hit
    # needed anyway — its transcription already travels with it.
    #
    # None means "never merge a group containing a figure". An integer applies that
    # threshold, so a section with many figures can still be collapsed if a domain
    # prefers it. Text-only groups are unaffected and keep auto_merge_threshold.
    auto_merge_figure_threshold: Optional[int] = None

    rerank_min_score: float = 0.0
    rerank_doc_char_limit: int = 2000
    rerank_timeout_seconds: float = 5.0


class ChunkingConfig(_Section):
    """Ingest-side behaviour. Phase 2+ asset pipelines hang off this section."""

    strategy: Literal["recursive", "sentence", "token"] = "recursive"
    chunk_size: int = 800
    chunk_overlap: int = 100
    l1_size: Optional[int] = None
    l2_size: Optional[int] = None
    sentence_overlap: int = 1
    merge_target_divisor: int = 4
    layout_parser_enabled: bool = True
    semantic_dedup_enabled: bool = False
    semantic_dedup_threshold: float = 0.97


class LocalizedText(_Section):
    """A user-facing string in each language the deployment answers in.

    Only strings the system emits *without* a model need this. A model handed an
    Arabic question replies in Arabic on its own; a canned string cannot, and an
    English refusal to an Arabic question is a visible failure in an Arabic-first
    deployment. Both fields default empty so a profile that sets neither degrades to
    the model rather than to a blank reply.
    """

    en: str = ""
    ar: str = ""


class CopyConfig(_Section):
    """User-facing strings. Separated from prompts because these are shown verbatim
    to end users and are the first thing a white-label deployment rewrites."""

    # Emitted with no model call when a classifier that READ the question judged it
    # outside this corpus. Never on a similarity score alone — see chat/turn_policy.
    out_of_domain: LocalizedText = Field(
        default_factory=lambda: LocalizedText(
            en=(
                "That's outside what I can help with here — I answer questions about "
                "this organisation's documents. Ask me about those and I'll do my best."
            ),
            ar=(
                "هذا خارج نطاق ما أستطيع المساعدة به هنا، فأنا أجيب عن الأسئلة المتعلقة "
                "بمستندات هذه الجهة. اسألني عنها وسأبذل قصارى جهدي."
            ),
        )
    )
    # Used only when agent.social_reply_mode is "static".
    social: LocalizedText = Field(
        default_factory=lambda: LocalizedText(
            en="Happy to help — what would you like to know?",
            ar="سعيد بمساعدتك، ما الذي تودّ معرفته؟",
        )
    )

    no_knowledge: str = (
        "The knowledge base does not contain reliable relevant information, "
        "so this question can't be answered from it right now."
    )
    retrieval_error: str = (
        "A temporary technical issue prevented searching the knowledge base. "
        "Please try again in a moment."
    )
    hitl_clarify_default: str = (
        "I found relevant content, but the evidence isn't enough to determine an answer. "
        "Please provide more detail about what you're asking."
    )
    hitl_clarify_missing_slots: str = "I found relevant content, but key information is still missing: "
    hitl_scope_default: str = (
        "I found several possibly relevant directions in the knowledge base. "
        "Which one are you asking about?"
    )
    hitl_clarify_agent: str = (
        "I found relevant knowledge, but I'm still missing one key piece of information. "
        "Please provide it so I can continue."
    )
    hitl_scope_agent: str = (
        "I found several knowledge base scopes that might be relevant. "
        "Please choose which one you'd like to continue querying."
    )
    new_session_title: str = "New session"
    unsupported_file_type: str = "Only PDF, Word, Excel, and HTML documents are supported"


class TriageConfig(_Section):
    """Deterministic pre-filter that runs before any model sees an image.

    This is the largest single cost lever in image ingest: most images in a document
    corpus are logos, rules, and spacers, and every one of them rejected here is a
    VLM call never made. Thresholds are profile data because "too small to matter"
    is domain-specific — a product thumbnail and a diagram detail differ.
    """

    min_width: int = 64
    min_height: int = 64
    min_area: int = 8000
    # Extreme aspect ratios are almost always rules, borders, or banner strips.
    max_aspect_ratio: float = 12.0
    min_byte_size: int = 1024
    # An image whose digest repeats on at least this fraction of a document's pages is
    # page furniture (letterhead, watermark) regardless of how interesting it looks.
    repeat_page_fraction: float = 0.6
    # At or above this pixel area an image is assumed to carry structure worth
    # transcribing (chart, diagram, table screenshot) and is tiered COMPLEX.
    complex_min_area: int = 250000


class FigurePipelineConfig(_Section):
    """Figure extraction: turning document images into a retrievable text surface."""

    enabled: bool = True
    # Vision extraction is opt-in. With it off, figures still become retrievable via
    # alt text and nearby captions — worse recall, but zero model cost and no
    # dependency on a vision-capable endpoint.
    vision_enabled: bool = False
    vision_temperature: float = 0.0
    # Longest edge the image is downscaled to before being sent. Image tokens scale
    # with area, so this is the main per-call cost control.
    vision_max_edge: int = 1024
    # Output-token ceiling for a vision call.
    #
    # This is NOT free headroom. Providers reserve it against a tokens-per-minute
    # quota before generating anything, so `input + this` must fit inside the quota or
    # the request is rejected outright — and a reasoning model spends much of it on a
    # scratchpad, so the value has to clear the truncation floor AND the quota ceiling
    # at once. On a small quota, disabling reasoning (vision_extra_params) buys far
    # more room than raising this does.
    vision_max_output_tokens: int = 4096

    # Rate limits are transient, but without a retry a single 429 permanently demotes
    # an image to the heuristic path and the document silently loses its figures.
    vision_retry_attempts: int = 4
    vision_retry_base_seconds: float = 5.0
    vision_retry_max_seconds: float = 90.0

    # Provider-specific parameters passed straight through to the model, for knobs
    # this schema should not try to enumerate. The one that matters on a small quota
    # is turning the scratchpad off — {"reasoning_effort": "none"} on Groq — which
    # reclaims the output budget it would otherwise consume.
    vision_extra_params: Dict[str, Any] = Field(default_factory=dict)

    # DPI used when rendering a PDF page region to an image. 150 is legible for chart
    # labels without producing megabyte crops.
    render_dpi: int = 150
    # Prose around the image that is passed as extraction context. A caption is worth
    # far more than the pixels for naming what a figure IS.
    context_chars: int = 600
    max_tags: int = 8
    max_answerable_questions: int = 4
    # Below this self-reported confidence a SIMPLE-tier asset is retried at COMPLEX.
    escalate_below_confidence: float = 0.45
    extraction_prompt: str = ""


class EntityPipelineConfig(_Section):
    """Entity (product) extraction: images that ARE the record, not evidence inside one.

    `attributes` is the domain's whole vocabulary. The extraction schema, the
    `search_products` tool signature, filter validation, and the attribute index are
    all generated from it, so a new domain is a YAML block rather than a code change.
    """

    enabled: bool = False
    vision_enabled: bool = False

    # How an image's role is decided during ingest:
    #   figure  — everything is document evidence (the default, and what Phase 3 does)
    #   entity  — everything is a catalogue record (a product-image import)
    #   auto    — the extractor classifies each image (needs vision)
    role_strategy: Literal["figure", "entity", "auto"] = "figure"

    attributes: List["AttributeSpec"] = Field(default_factory=list)

    # Recall width before attribute filtering. Filters narrow hard, so the semantic
    # pass must hand over enough candidates for them to narrow from.
    candidate_pool: int = 200
    max_results: int = 20
    extraction_prompt: str = ""


class AssetDeliveryConfig(_Section):
    """How retrieved assets reach a client.

    Deliberately free of anything UI-specific. The backend publishes structured
    references; each consumer declares its own capabilities per request and the
    presenter adapts. Swapping the frontend changes nothing here.
    """

    # Mount path for the asset byte endpoint.
    #
    # Deliberately NOT "/assets": that is Vite's default build output directory, and an
    # API mounted there shadows the frontend's own bundle — the SPA 404s on its own
    # JavaScript. Any value here must also appear in the dev server's proxy list.
    base_path: str = "/media"

    # Attach asset references to chat responses automatically when retrieved chunks
    # reference them. Off for deployments that resolve assets in a second call.
    attach_to_response: bool = True
    max_assets_per_response: int = 8

    # Attach only the images belonging to chunks the answer actually cited.
    #
    # Retrieval surfaces every figure near the topic — a uniform question hits all the
    # uniform images — but the answer usually rests on one. The `[n]` markers the agent
    # already emits say exactly which, so this costs nothing to honour. An answer that
    # cites nothing still gets everything surfaced.
    attach_only_cited: bool = True

    # Ceiling for base64 inlining, applied on top of whatever a client requests — a
    # client asking to inline a 40 MB scan must not be able to do it.
    max_inline_bytes: int = 512_000

    # Bytes are content-addressed and therefore immutable, so they can be cached hard.
    cache_max_age_seconds: int = 31_536_000

    # How many times one asset may be re-read at the pixel level within a session.
    # After the first read its observations live in the conversation, so further
    # questions are answered from text at no cost.
    max_pixel_reads_per_asset: int = 2
    # Observations retained per asset before the oldest are dropped.
    max_observations_per_asset: int = 6


class AssetsConfig(_Section):
    """Image/asset handling. Consumed from Phase 2 (storage) onward."""

    enabled: bool = True

    # Blob backend. Credentials never live in a profile — they come from the
    # environment (ASSET_S3_* / MINIO_*), because profiles are committed to the repo.
    blob_backend: Literal["local", "s3"] = "local"
    blob_root: str = "data/assets"
    blob_bucket: str = "kb-assets"

    # Orphaned blobs are removed when the last document referencing them is deleted.
    # Extraction rows are always kept: they are derived text keyed by an irreversible
    # digest, they are the expensive artifact, and re-upload is the common case.
    gc_orphan_blobs: bool = True

    max_images_per_document: int = 500
    triage: TriageConfig = Field(default_factory=TriageConfig)
    figures: FigurePipelineConfig = Field(default_factory=FigurePipelineConfig)
    entities: EntityPipelineConfig = Field(default_factory=EntityPipelineConfig)
    delivery: AssetDeliveryConfig = Field(default_factory=AssetDeliveryConfig)


class IngestConfig(_Section):
    """Which ingest capabilities and stores this profile activates.

    Phase 1 only carries `supported_extensions`; `asset_pipelines` and `indexes` are
    declared now so Phase 2 (AssetDossier) extends this section rather than
    introducing a competing config surface.
    """

    supported_extensions: List[str] = Field(
        default_factory=lambda: [".pdf", ".docx", ".doc", ".xlsx", ".xls", ".html", ".htm"]
    )
    asset_pipelines: List[str] = Field(default_factory=list)
    indexes: List[str] = Field(default_factory=lambda: ["kb_chunks"])


class DomainProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Bumped when a section's MEANING changes in a way stored data depends on.
    # Phase 2 stamps this onto every AssetDossier so a schema migration knows
    # which assets need re-extraction.
    profile_version: int = 1
    name: str = "default"
    extends: Optional[str] = None

    identity: IdentityConfig = Field(default_factory=IdentityConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    rag: RagConfig = Field(default_factory=RagConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    assets: AssetsConfig = Field(default_factory=AssetsConfig)
    # Named `user_copy` rather than `copy` so it does not shadow BaseModel.copy.
    user_copy: CopyConfig = Field(default_factory=CopyConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)

    def render_system_prompt(
        self,
        tools: Optional[Sequence[str]] = None,
        language: Optional[str] = None,
    ) -> str:
        """Agent system prompt for a turn binding `tools`, answered in `language`.

        `tools=None` means "everything the profile allows", which is both the default
        and what any uncertainty produces: narrowing is an optimisation, so it must
        never be the reason a capability goes unmentioned while its tool is bound.

        The tool list is passed rather than read from the profile because the caller
        decides it per turn (see backend/chat/runtime.py). Prompt and binding must come
        from ONE decision — a prompt describing an unbound tool invites a call for a
        tool that does not exist, and a bound tool whose instructions were dropped gets
        used wrongly or not at all.

        A profile that sets `agent.system_prompt` takes that verbatim. Uses str.replace
        rather than str.format because prompts routinely contain literal braces (JSON
        examples, citation markers) that format() would raise on.
        """
        if self.agent.system_prompt:
            return self.agent.system_prompt.replace("{persona}", self.identity.persona).strip()

        # Imported here, not at module scope: backend.tools pulls in the request context
        # and every tool module, and the profile package is imported by several of them.
        from backend.chat.language import language_name
        from backend.prompts import render
        from backend.tools import GROUNDED_TOOLS

        bound = list(self.agent.tools if tools is None else tools)
        return render(
            "agent/system.j2",
            persona=self.identity.persona,
            grounded=any(name in GROUNDED_TOOLS for name in bound),
            # "" for anything the detector cannot positively name, which keeps the
            # generic instruction. See LANGUAGE_NAMES for why that matters.
            language_name=language_name(language) if language else "",
        )

    def as_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="python")
