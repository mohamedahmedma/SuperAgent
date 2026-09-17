"""The retrieval graph's steps, each a class handed what it reaches for.

Every step used to be a function reaching this package's module globals for its retriever,
its rewriter, its models and its configuration — which is why the tests that drive the
graph re-execute `pipeline.py` with its imports stubbed out. A step here is given those
collaborators through `RagDependencies` instead, and imports none of them.

`pipeline.py` supplies the dependencies as a live view over its own globals, resolved at
the moment a step asks. So a stub installed when the module loads, an attribute assigned
after it, and a `patch.object` on the pipeline module all still reach the step that uses
them, although the graph was compiled before any of them happened.

Behaviour is unchanged: every progress label, trace key and state update is the one the
function it replaces produced. The test suite asserts few of the graph's progress labels;
a golden run over every branch, compared before and after, checked the rest.
"""
import operator
import re
from typing import Annotated, List, Literal, Optional, Protocol, TypedDict

from langgraph.types import Send

from backend.chat.child_names import strip_child_names
from backend.chat.request_context import ChatRequestContext
from backend.prompts import resolve as resolve_prompt
from backend.rag.evidence import (
    AssessmentContext,
    Certainty,
    ChunkAssessment,
    EvidenceReport,
    build_ladder,
)
from backend.rag.grading_view import format_docs
from backend.rag.policy import decide_route, offerable_directions, select_context_indices
from backend.schemas.chat import normalize_rag_sub_trace


class RAGState(TypedDict):
    question: str
    query: str
    context: str
    docs: List[dict]
    route: Optional[str]
    retrieval_status: Optional[str]
    retrieval_failed: Optional[bool]
    evidence_relevance: Optional[str]
    evidence_answerability: Optional[str]
    evidence_ambiguity: Optional[str]
    evidence_confidence: Optional[float]
    missing_slots: Optional[List[str]]
    hitl_prompt: Optional[str]
    hitl_options: Optional[List[str]]
    rewrite_count: int
    # HITL turns already spent on this question, surviving the resume boundary.
    hitl_rounds: int
    rewrite_method: Optional[str]
    rewritten_query: Optional[str]
    step_back_question: Optional[str]
    hyde_document: Optional[str]
    rag_trace: Optional[dict]
    # Fields added for complexity routing
    complexity: Optional[str]
    complexity_reason: Optional[str]
    sub_questions: Optional[List[str]]
    is_sub_agent: bool
    sub_results: Annotated[List[dict], operator.add]
    request_context: ChatRequestContext
    rag_step_group: Optional[str]
    rag_step_group_label: Optional[str]
    # Corpus sections the turn planner matched, forwarded as a retrieval hint. None
    # means no planner ran or it abstained. Never a filter — see chat/turn_policy.
    retrieval_sections: Optional[List[str]]
    # Catalogued questions the planner matched, one per section. Routing offers these
    # as a scope_select rather than guessing with a rewrite — see policy.decide_route.
    scope_options: Optional[List[str]]
    # Conditions carried over from earlier turns ("grades up to Year 6"). Appended to
    # the retrieval query and stated to the answering model, because narrowing the
    # search is only half of it — the right fee tables still yield an answer covering
    # every year group unless something tells the model which years were asked about.
    carried_constraints: Optional[List[str]]
    # Whether this turn inherited its subject from the conversation. Routing reads it to
    # decide whether a choice of corpus directions could possibly narrow anything.
    is_followup: Optional[bool]
    # The turn's language, for document-pair routing: where the same document was
    # uploaded in both Arabic and English, retrieval answers from the half that matches
    # the question rather than letting both compete. Empty searches everything, which is
    # also what an unpaired corpus does — see rag/utils.language_filter_clause.
    #
    # Declared here or dropped: LangGraph keeps only the keys the state schema names, so
    # for as long as this line was missing `_initial_state` wrote the language and every
    # node read None — both halves of a paired document competed on every bilingual turn.
    language: Optional[str]
    # The year group the school's records put this turn's child in. Beside the question,
    # never appended to it: a year group is absent from every passage the corpus wrote
    # once for everybody, so stapling it to the query dilutes recall exactly as carried
    # conditions did. It reaches the grader and the answer prompt instead.
    child_year: Optional[str]


class RagDependencies(Protocol):
    """What the graph's steps reach outside themselves for."""

    @property
    def config(self): ...

    @property
    def copy(self): ...

    @property
    def top_k(self) -> int: ...

    @property
    def complexity_prompt(self) -> str: ...

    @property
    def complexity_schema(self) -> type: ...

    def retrieve_documents(self, query: str, *, top_k: int, language: str) -> dict: ...

    def rewrite_query_once(self, question: str) -> dict: ...

    def dedupe_documents(self, docs: List[dict]) -> List[dict]: ...

    def retrieval_trace_fields(self, meta: dict) -> dict: ...

    def model_assessors(self) -> list:
        """The model-backed rungs of the evidence ladder, built for this assessment."""

    def complexity_model(self): ...

    def initial_state(self, question: str, ctx: ChatRequestContext, **kwargs) -> dict: ...


def emit(state: RAGState, icon: str, label: str, detail: str = "") -> None:
    ctx = state["request_context"]
    ctx.emit_rag_step(
        icon,
        label,
        detail,
        group=state.get("rag_step_group"),
        group_label=state.get("rag_step_group_label"),
    )


def search_query(state: RAGState) -> str:
    """The text this state searches for: the question, and nothing appended to it.

    An earlier revision appended the turn's carried conditions here, on the reasoning
    that the agent's tool query might omit a condition the user set two turns ago.
    Measured over a twenty-turn conversation (tests/test_conversation_sequence.py) that
    cost three of twenty turns outright: "what time does the school day start (the child
    is 5 years old)" ranks the term-dates passage below anything sharing the words
    "child" or "years", because those terms appear nowhere in a passage about opening
    hours. Every query term absent from the target passage is dilution, and a condition
    is absent from every passage the corpus wrote once for everybody.

    The condition still reaches retrieval when it belongs there — the RESOLVED question
    states it in natural language ("the fees for the years up to Year 6"), which is
    vocabulary the fee table actually contains. What it must not do is arrive twice, or
    arrive as bare keywords stapled to a question about something else.

    Constraints now travel to the two stages that can act on them without costing
    recall: the grader, which reports whether the material varies by them, and the
    answer prompt, which narrows or does not on that verdict.

    The child's NAME goes the other way: out. It is the same argument as the paragraph
    above, only stronger, because a name is not merely absent from the target passage —
    it is absent from the whole corpus, which is the school's own material and is written
    once for every family. On the sparse half that is worse than dilution: a name is a
    rare term, so its IDF is high, and any chunk that happens to carry it outranks the
    one that answers the question.

    Cut HERE rather than from `state["question"]` itself, and that distinction is the
    feature. This function is what every retrieval reads; the question is what the
    grader, the HITL prompts and the resumed answer read, and that last one hands it to a
    model that has to write the parent a sentence naming their child. Which one of a
    parent's children a turn is about still reaches records, the prompt and the answer —
    it just stops reaching the search box.
    """
    question = state["question"]
    stripped, _cuts = strip_child_names(question, child_names(state))
    return stripped


def child_names(state: RAGState) -> list:
    """The name spellings this turn's message used, as the planner settled them.

    Off the request context, like every other hint the graph reads, and defaulted to
    nothing — a sub-agent state, a direct `run_rag_graph` in a test, or a profile with no
    roster behind it has no child and needs no special case.
    """
    ctx = state.get("request_context")
    return list(getattr(ctx, "child_names", None) or [])


class RetrieveInitial:
    """The first search for a question."""

    def __init__(self, dependencies: RagDependencies) -> None:
        self._deps = dependencies

    def __call__(self, state: RAGState) -> RAGState:
        query = search_query(state)
        emit(state, "🔍", "Searching the knowledge base...", "Initial retrieval")
        if query != state["question"]:
            # Said once, here, rather than inside `search_query` — which also runs for the
            # rewriter and would report the same cut twice on one turn. The DETAIL names no
            # child: this trace is persisted per message and streamed to a browser, and the
            # rule `turn_policy.as_trace` states holds here too — report that a decision was
            # made, never what it was.
            emit(
                state, "🙈", "Searching without the child's name",
                "the corpus is the school's own material and names no pupil",
            )
        if state.get("carried_constraints"):
            emit(
                state, "🧷", "Conditions carried from earlier turns",
                "Applied when the answer is written, not to the search — "
                + "; ".join(state["carried_constraints"]),
            )
        retrieved = self._deps.retrieve_documents(
            query, top_k=self._deps.top_k, language=str(state.get("language") or "")
        )
        results = retrieved.get("docs", [])
        retrieve_meta = retrieved.get("meta", {})
        retrieval_failed = retrieve_meta.get("retrieval_mode") == "failed"
        context = format_docs(results)
        if retrieval_failed:
            emit(
                state,
                "🚧",
                "Knowledge base temporarily unreachable",
                retrieve_meta.get("retrieval_error") or "retrieval backend error",
            )
        else:
            emit(
                state,
                "🧱",
                "Three-tier chunk retrieval",
                (
                    f"Leaf level L{retrieve_meta.get('leaf_retrieve_level', 3)} recall, "
                    f"candidates {retrieve_meta.get('candidate_k', 0)}"
                ),
            )
            emit(
                state,
                "🧩",
                "Auto-merging",
                (
                    f"Enabled: {bool(retrieve_meta.get('auto_merge_enabled'))}, "
                    f"Applied: {bool(retrieve_meta.get('auto_merge_applied'))}, "
                    f"Replaced chunks: {retrieve_meta.get('auto_merge_replaced_chunks', 0)}"
                ),
            )
            emit(state, "✅", f"Retrieval complete, found {len(results)} snippets", f"Mode: {retrieve_meta.get('retrieval_mode', 'hybrid')}")
            if not results:
                emit(state, "⚠️", "No snippets available, proceeding to the evidence-grading short-circuit check")
        rag_trace = {
            "tool_used": True,
            "tool_name": "search_knowledge_base",
            "query": query,
            "retrieved_chunks": results,
            "initial_retrieved_chunks": results,
            "retrieval_stage": "initial",
            "child_name_removed": query != state["question"],
            "complexity": state.get("complexity"),
            "complexity_reason": state.get("complexity_reason"),
            **self._deps.retrieval_trace_fields(retrieve_meta),
        }
        return {
            "query": query,
            "docs": results,
            "context": context,
            "retrieval_failed": retrieval_failed,
            "rag_trace": rag_trace,
        }


def route_after_grade(state: RAGState) -> Literal["rewrite_question", "end"]:
    if state.get("route") == "rewrite":
        return "rewrite_question"
    return "end"


def route_after_rewrite(state: RAGState) -> Literal["retrieve_rewritten", "end"]:
    """Only re-retrieve when a rewrite was actually planned.

    This edge used to be unconditional, so a node that gave up on planning one fell
    into `retrieve_rewritten`, which requires `rewrite_method` and raises `ValueError`
    without it. A planner that returned nothing — an unconfigured FAST_MODEL, a
    provider error, a response that failed schema validation — therefore failed the
    whole turn, and did it on the path meant to degrade gracefully.
    """
    return "retrieve_rewritten" if state.get("rewrite_method") else "end"


def route_after_complexity(state: RAGState):
    """Simple questions go straight to retrieval; complex questions retrieve the planned sub-questions in parallel."""
    if state.get("complexity") == "complex":
        return "prepare_sub_questions"
    return "retrieve_initial"


class GradeDocuments:
    """Assess the retrieved evidence once, then apply the policies that read it.

    This step orchestrates and nothing more: it holds no thresholds, computes no
    signals, and does not know which assessors exist. Assessment produces one report;
    routing and context sizing are pure functions over it.
    """

    def __init__(self, dependencies: RagDependencies) -> None:
        self._deps = dependencies

    def conditions(self, state) -> List[str]:
        """Every condition the grader should judge the material against.

        The child's year joins the conditions the user set rather than getting a channel of
        its own, because the question being asked of the grader is the same for both: does
        this material give different answers depending on them? A fee table does; a document
        list written once for everybody does not, and that verdict is what stops a general
        rule being withheld from a parent whose child is in Year 1.

        Its wording says where it came from. The grader is not told the user set it, because
        the user did not — the roster did, and a condition that misreports its own source is
        the fabricated-provenance pattern `backend/rag/evidence.py` exists to prevent.
        """
        conditions = list(state.get("carried_constraints") or [])
        year = str(state.get("child_year") or "").strip()
        if year:
            conditions.append(self._deps.config.child_year_condition.format(year=year))
        return conditions

    def assess(self, state: RAGState) -> EvidenceReport:
        """Climb the ladder for this retrieval, once.

        Rungs are ordered by the certainty each can establish, which is also their cost, so
        the grader is reached only when the cheaper rungs cannot conclude.
        """
        docs = state.get("docs") or []
        ladder = build_ladder(self._deps.config, extra=self._deps.model_assessors())
        return ladder.run(
            AssessmentContext(
                # The QUESTION, not the constrained query. Folding the conditions in was a
                # mistake with one visible failure mode: a grader asked whether a general
                # admissions document list answers "what documents are required (grades up
                # to Year 6)" sees no year mentioned anywhere, honestly returns
                # `relevance: none`, and `decide_route` denies — which is the one outcome
                # that policy exists to make impossible while material is in hand.
                #
                # The conditions travel beside the question instead, and the grader reports
                # on them in `constraints_discriminate` rather than scoring against them.
                question=state["question"],
                docs=docs,
                retrieval_meta=state.get("rag_trace") or {},
                config=self._deps.config,
                constraints=self.conditions(state),
            )
        )

    @staticmethod
    def retrieval_status_for_route(route: str, report: EvidenceReport) -> str:
        if route == "answer":
            # Only `sufficient` earns "answerable". Everything else routed to answer rests
            # on less than the question asked for — including `none`, which reaches this
            # route now that on-subject evidence is never denied. That distinction is what
            # the knowledge tool reads to tell the model to answer from what it has and name
            # the gap, so collapsing it into "answerable" would silently drop the guidance
            # on exactly the turns that need it.
            return "answerable" if report.sufficiency == "sufficient" else "partial"
        if route == "rewrite":
            return "needs_rewrite"
        if route == "clarify":
            return "needs_clarification"
        if route == "scope_select":
            return "needs_scope_selection"
        if route == "retrieval_error":
            return "retrieval_error"
        return "no_knowledge"

    def default_hitl_prompt(self, route: str, report: EvidenceReport) -> str:
        if report.hitl_prompt:
            return report.hitl_prompt
        copy = self._deps.copy
        if route == "scope_select":
            return copy.hitl_scope_default
        if report.missing_slots:
            return copy.hitl_clarify_missing_slots + ", ".join(report.missing_slots)
        return copy.hitl_clarify_default

    def report_update(
        self,
        report: EvidenceReport,
        route: str,
        scope_options: List[str] | None = None,
        *,
        is_followup: bool = False,
    ) -> dict:
        """Flatten a report plus its route into the trace fields the rest of the system
        already reads. Names are unchanged so the frontend and any integrating client keep
        working; `evidence_certainty` and `evidence_assessed_by` are additive."""
        hitl_prompt = self.default_hitl_prompt(route, report) if route in ("clarify", "scope_select") else ""
        # The grader's own list wins when it wrote one. The catalogued directions are the
        # fallback, and on a scope_select routed BY those directions they are the only list
        # there is — a scope_select with no options is never asked, so without this the
        # route decided in policy would be silently dropped here.
        #
        # `offerable_directions` is applied rather than the raw list, so this fallback obeys
        # the same rule routing does. Without it a follow-up could still be handed the
        # catalogue neighbours that routing had just refused to ask about, arriving through
        # a grader-initiated scope_select instead.
        options = list(report.hitl_options) or offerable_directions(
            scope_options or [], is_followup=is_followup
        )
        return {
            **report.as_trace(),
            "retrieval_status": self.retrieval_status_for_route(route, report),
            "hitl_prompt": hitl_prompt,
            "hitl_options": options if route == "scope_select" else list(report.hitl_options),
            "route": route,
        }

    def __call__(self, state: RAGState) -> RAGState:
        if state.get("retrieval_failed"):
            # A backend outage says nothing about what the KB contains, so this must not
            # reach an assessor or surface as no_knowledge. Static short-circuit only.
            rag_trace = state.get("rag_trace", {}) or {}
            rag_trace.update({
                "retrieval_status": "retrieval_error",
                "route": "retrieval_error",
                "evidence_reason": "knowledge_base_unreachable",
            })
            emit(state, "\U0001f6a7", "Retrieval failed, returning a static retry notice", "Not treated as missing knowledge")
            return {
                "route": "retrieval_error",
                "retrieval_status": "retrieval_error",
                "docs": [],
                "context": "",
                "rag_trace": rag_trace,
            }

        docs = state.get("docs") or []
        if docs:
            emit(state, "\U0001f4ca", "Evaluating evidence quality...")

        config = self._deps.config
        report = self.assess(state)
        scope_options = list(state.get("scope_options") or [])
        route, route_reason = decide_route(
            report,
            has_docs=bool(docs),
            rewrite_count=int(state.get("rewrite_count") or 0),
            is_sub_agent=bool(state.get("is_sub_agent")),
            config=config,
            hitl_rounds=int(state.get("hitl_rounds") or 0),
            scope_options=scope_options,
            is_followup=bool(state.get("is_followup")),
        )

        report_update = self.report_update(
            report, route, scope_options, is_followup=bool(state.get("is_followup"))
        )
        rag_trace = state.get("rag_trace", {}) or {}
        rag_trace.update(report_update)
        rag_trace["route_reason"] = route_reason

        if route == "answer":
            if report.sufficiency == "partial":
                emit(state, "\U0001f7e1", "Keeping partially relevant evidence", f"Confidence: {report.confidence:.2f}")
            else:
                emit(state, "✅", "Evidence sufficient, returning retrieved snippets", f"Confidence: {report.confidence:.2f}")
        elif route == "rewrite":
            emit(state, "⚠️", "Evidence insufficient, will rewrite the query once", f"Confidence: {report.confidence:.2f}")
        elif route in ("clarify", "scope_select"):
            emit(state, "❓", "Needs more information from the user", report_update["hitl_prompt"])
        elif route == "retrieval_error":
            # Retrieval worked but assessment could not reach the required standard, so
            # neither answering nor denying would be honest. Say "try again" instead.
            emit(state, "\U0001f6a7", "Evidence could not be assessed", route_reason)
        else:
            emit(state, "⛔", "No usable evidence found in the knowledge base", route_reason)

        update = {
            "route": route,
            "retrieval_status": report_update["retrieval_status"],
            "evidence_relevance": report.relevance,
            "evidence_answerability": report.sufficiency,
            "evidence_ambiguity": report.ambiguity,
            "evidence_confidence": report.confidence,
            "missing_slots": list(report.missing_slots),
            "hitl_prompt": report_update["hitl_prompt"],
            "hitl_options": list(report_update["hitl_options"]),
            "rag_trace": rag_trace,
        }

        if route in ("no_knowledge", "clarify", "scope_select", "retrieval_error"):
            # Dropping the chunks from the TRACE too, not just from the state. Asset
            # attachment falls back to `rag_trace.retrieved_chunks` whenever the knowledge
            # tool pinned nothing (backend/chat/assets_bridge.py), and on these routes it
            # pins nothing by design — so leaving them here attached the figures from
            # chunks this turn just decided not to answer from. "The knowledge base has no
            # reliable information on this" arriving with the picture that answers the
            # question is the most confusing thing the system can do. `initial_retrieved_chunks`
            # still holds the full set for the trace panel.
            if docs:
                rag_trace["retrieved_chunks"] = []
            update.update({"docs": [], "context": ""})
            return update

        if route == "answer" and docs:
            keep, reason = select_context_indices(
                report, docs, config, answer_ceiling=self._deps.top_k,
            )
            rag_trace["context_selection_reason"] = reason
            rag_trace["context_chunks_available"] = len(docs)
            if keep:
                kept = [docs[i - 1] for i in keep]
                emit(
                    state, "✂️", f"Sending {len(kept)} of {len(docs)} chunks to the model", reason,
                )
                update.update({"docs": kept, "context": format_docs(kept)})
                # retrieved_chunks has to match what the answer was built from: citation
                # markers and asset attribution both index into it. The full set stays in
                # initial_retrieved_chunks for the trace panel.
                rag_trace["retrieved_chunks"] = kept
                rag_trace["context_trimmed"] = True
                rag_trace["context_chunks_kept"] = len(kept)
            else:
                rag_trace["context_trimmed"] = False
                rag_trace["context_chunks_kept"] = len(docs)

        return update


class RewriteQuestion:
    """Plan one step-back or HyDE rewrite of a question the evidence did not settle."""

    def __init__(self, dependencies: RagDependencies) -> None:
        self._deps = dependencies

    @staticmethod
    def stop(state: RAGState, reason: str, label: str, detail: str) -> RAGState:
        """End the rewrite branch without re-retrieving, keeping the first pass.

        The rewrite is a bonus second attempt, so failing to plan one must cost the turn
        nothing it already had. Both callers used to return `no_knowledge` with `docs`
        cleared — which denied knowledge the first pass had retrieved and graded on-subject,
        the exact outcome `decide_route` exists to prevent — while their own step message
        said "answering from the first pass only". The first pass is now what the turn
        answers from, and only a genuinely empty one still denies.
        """
        docs = state.get("docs") or []
        status = "partial" if docs else "no_knowledge"
        rag_trace = state.get("rag_trace", {}) or {}
        rag_trace.update({
            "retrieval_status": status,
            "route": "answer" if docs else "no_knowledge",
            "evidence_reason": reason,
        })
        if not docs:
            rag_trace["retrieved_chunks"] = []
        emit(state, "⚠️" if docs else "⛔", label, detail)
        return {
            "route": "answer" if docs else "no_knowledge",
            "retrieval_status": status,
            "docs": docs,
            "context": state.get("context") or "",
            "rag_trace": rag_trace,
        }

    def __call__(self, state: RAGState) -> RAGState:
        question = search_query(state)
        emit(state, "✏️", "Rewriting the query...")

        rewrite_count = int(state.get("rewrite_count") or 0)
        if rewrite_count >= self._deps.config.max_rewrites:
            return self.stop(
                state,
                "rewrite_budget_exhausted",
                "Rewrite budget exhausted, stopping retrieval",
                "Answering from what the earlier passes found",
            )

        emit(state, "🧠", "Choosing between Step-back / HyDE rewrite")
        rewrite = self._deps.rewrite_query_once(question)
        if not rewrite:
            return self.stop(
                state,
                "rewrite_unavailable",
                "Could not plan a rewrite, stopping retrieval",
                "Answering from the first pass only",
            )

        rewrite_method = (rewrite.get("rewrite_method") or "").strip()
        step_back_question = (rewrite.get("step_back_question") or "").strip()
        hyde_document = (rewrite.get("hyde_document") or "").strip()
        rewritten_query = (rewrite.get("rewritten_query") or "").strip()

        method_label = "Step-back" if rewrite_method == "step_back" else "HyDE"
        emit(state, "✅", f"Selected {method_label} rewrite", "Only this rewrite method will run this round")

        rag_trace = state.get("rag_trace", {}) or {}
        rag_trace.update({
            "rewrite_method": rewrite_method,
            "rewritten_query": rewritten_query,
            "rewrite_count": rewrite_count + 1,
        })
        if step_back_question:
            rag_trace["step_back_question"] = step_back_question
        if hyde_document:
            rag_trace["hyde_document"] = hyde_document

        return {
            "rewrite_method": rewrite_method,
            "rewritten_query": rewritten_query,
            "step_back_question": step_back_question,
            "hyde_document": hyde_document,
            "rewrite_count": rewrite_count + 1,
            "rag_trace": rag_trace,
        }


class RetrieveRewritten:
    """The second search, with the rewritten query, added to what the first pass found."""

    def __init__(self, dependencies: RagDependencies) -> None:
        self._deps = dependencies

    def __call__(self, state: RAGState) -> RAGState:
        rewrite_method = (state.get("rewrite_method") or "").strip()
        if rewrite_method not in ("step_back", "hyde"):
            raise ValueError("rewrite_method is required for rewritten retrieval")
        rewritten_query = (state.get("rewritten_query") or "").strip()
        if not rewritten_query:
            raise ValueError("rewritten_query is required for rewritten retrieval")
        # The rewriter was handed the stripped question, so this is normally a no-op. It is
        # here because the rewrite is written by a MODEL, and a step-back question composed
        # from a turn about one child is exactly the place one would reappear.
        rewritten_query, _cuts = strip_child_names(rewritten_query, child_names(state))
        method_label = "Step-back" if rewrite_method == "step_back" else "HyDE"
        emit(state, "🔄", f"Re-retrieving with the {method_label} query...")
        retrieved = self._deps.retrieve_documents(
            rewritten_query, top_k=self._deps.top_k, language=str(state.get("language") or "")
        )
        results = retrieved.get("docs", [])
        retrieve_meta = retrieved.get("meta", {})
        retrieval_failed = retrieve_meta.get("retrieval_mode") == "failed"

        # The rewrite ADDS to the first pass, it does not replace it. A step-back or HyDE
        # query is a different question by construction, so its results can easily miss a
        # chunk the literal question found — and that chunk was already graded on-subject,
        # or this branch would not be running. Replacing the set meant a rewrite could
        # leave the turn with less evidence than it started with, and the second grading
        # pass would then deny knowledge the first pass had in hand.
        #
        # Rewritten results lead because they are the targeted retry; the first pass keeps
        # whatever they did not rediscover. The graded set can reach 2 x top_k on the
        # minority of turns that rewrite at all, and adaptive context selection trims it
        # back to the chunks the grader cites before any of it reaches the answer prompt.
        first_pass = [] if retrieval_failed else list(state.get("docs") or [])
        merged = self._deps.dedupe_documents(results + first_pass)
        for index, item in enumerate(merged, 1):
            item["rrf_rank"] = index
        context = format_docs(merged)
        if retrieval_failed:
            emit(
                state,
                "🚧",
                "Knowledge base temporarily unreachable during rewritten retrieval",
                retrieve_meta.get("retrieval_error") or "retrieval backend error",
            )
        else:
            emit(
                state,
                "🧱",
                f"{method_label} three-tier retrieval",
                (
                    f"L{retrieve_meta.get('leaf_retrieve_level', 3)} recall, "
                    f"candidates {retrieve_meta.get('candidate_k', 0)}, "
                    f"merge-replaced {retrieve_meta.get('auto_merge_replaced_chunks', 0)}"
                ),
            )
            emit(
                state,
                "✅",
                f"Rewritten retrieval complete, {len(merged)} snippets total",
                f"{len(results)} from the {method_label} query, "
                f"{len(merged) - len(results)} kept from the first pass",
            )
        rag_trace = state.get("rag_trace", {}) or {}
        rag_trace.update({
            "rewrite_method": rewrite_method,
            "rewritten_query": rewritten_query,
            "retrieved_chunks": merged,
            # The rewritten pass ALONE, so a trace still shows what the rewrite itself
            # found rather than the union it was folded into.
            "rewrite_retrieved_chunks": results,
            "retrieval_stage": "rewritten",
            **self._deps.retrieval_trace_fields(retrieve_meta),
        })
        if state.get("step_back_question"):
            rag_trace["step_back_question"] = state["step_back_question"]
        if state.get("hyde_document"):
            rag_trace["hyde_document"] = state["hyde_document"]
        return {"docs": merged, "context": context, "retrieval_failed": retrieval_failed, "rag_trace": rag_trace}


class ClassifyComplexity:
    """Decide whether a question is one lookup or several, and plan the several."""

    def __init__(self, dependencies: RagDependencies) -> None:
        self._deps = dependencies

    def fast_path_reason(self, question: str) -> Optional[str]:
        """Return a reason only when a local rule can confidently classify a simple query.

        The marker vocabulary is language- AND domain-specific, so it is profile data: a
        school corpus needs admissions vocabulary that another corpus would not.
        """
        config = self._deps.config
        simple_override_markers = tuple(config.simple_override_markers)
        simple_query_markers = tuple(config.simple_query_markers)
        complex_query_markers = tuple(config.complex_query_markers)
        query_dimension_markers = tuple(config.query_dimension_markers)

        normalized = re.sub(r"\s+", " ", (question or "").strip()).lower()
        if not normalized or len(normalized) > config.fast_path_max_chars:
            return None
        # Overrides run FIRST. "how many students" is a single-fact lookup, but it contains
        # "how ", which also opens genuinely analytical questions. Without this the complex
        # marker wins and a plain lookup is sent to the planner — which may decompose it
        # into several sub-questions, each paying its own retrieval and grader call.
        if any(marker in normalized for marker in simple_override_markers):
            return "obvious_simple_fast_path:single_fact_override"
        if any(marker in normalized for marker in complex_query_markers):
            return None
        if "、" in normalized:
            return None
        if re.search(r"[一-鿿]", normalized) and normalized.count(" ") >= 2:
            return None
        if sum(marker in normalized for marker in query_dimension_markers) >= 2:
            return None
        if sum(normalized.count(mark) for mark in ("?", "？", ";", "；")) > 1:
            return None
        if any(marker in normalized for marker in simple_query_markers):
            return "obvious_simple_fast_path:single_fact_marker"
        # A wh-question opening directly onto a copula ("what are the partners") or with an
        # attribute in between ("what element is X"). Both are single-fact lookups that the
        # literal markers miss, and the first form is common enough that missing it sent
        # ordinary plural questions down the decomposition path.
        if re.match(
            r"^(what|which|who|where|when)\s+(?:\w+\s+)?(is|are|was|were|does|do)\b",
            normalized,
        ):
            return "obvious_simple_fast_path:wh_attribute_question"
        # Terminators include Arabic ؟ and ۔ — otherwise an Arabic question keeps its
        # mark and measures one character longer against the length rule below.
        if len(normalized.rstrip("?？。.!！؟۔،")) <= config.fast_path_short_intent_chars:
            return "obvious_simple_fast_path:short_single_intent"
        return None

    def __call__(self, state: RAGState) -> RAGState:
        """Uses FAST_MODEL to determine question complexity."""
        question = state["question"]
        emit(state, "🧭", "Analyzing question complexity...")

        fast_path_reason = self.fast_path_reason(question)
        if fast_path_reason:
            emit(state, "⚡", "Fast-classified as a simple question → using the standard RAG flow")
            return {"complexity": "simple", "complexity_reason": fast_path_reason}

        model = self._deps.complexity_model()
        if not model:
            raise RuntimeError("FAST_MODEL is required for complexity planning")

        prompt = resolve_prompt(self._deps.complexity_prompt, "rag/complexity.j2", question=question)
        result = model.with_structured_output(self._deps.complexity_schema).invoke(
            [{"role": "user", "content": prompt}]
        )
        complexity = (result.complexity or "simple").strip().lower()
        reason = (result.reason or "").strip()
        sub_questions = [
            item.strip()
            for item in (result.sub_questions or [])
            if item and item.strip()
        ][: self._deps.config.max_sub_questions]
        if complexity not in ("simple", "complex"):
            raise ValueError(f"Unsupported complexity result: {complexity}")
        if complexity == "complex" and not sub_questions:
            raise ValueError("Complexity planner returned no sub-questions")

        if complexity == "simple":
            emit(state, "✅", "Simple question → using the standard RAG flow", f"Reason: {reason[:60]}")
        else:
            emit(state, "🔀", "Complex question → decomposing into sub-questions for parallel retrieval", f"Reason: {reason[:60]}")

        return {
            "complexity": complexity,
            "complexity_reason": reason,
            "sub_questions": sub_questions if complexity == "complex" else [],
        }


class PrepareSubQuestions:
    """Announce the sub-questions produced by the complexity planner."""

    def __call__(self, state: RAGState) -> RAGState:
        planned_sub_questions = [
            item.strip()
            for item in (state.get("sub_questions") or [])
            if item and item.strip()
        ]
        for i, sq in enumerate(planned_sub_questions, 1):
            emit(state, "📌", f"Sub-question {i}", f"{sq[:80]} added to parallel retrieval")
        return {"sub_questions": planned_sub_questions}


class FanOutSubQuestions:
    """Dispatch the planned sub-questions to the sub-agent in parallel, via the Send API."""

    def __init__(self, dependencies: RagDependencies) -> None:
        self._deps = dependencies

    def __call__(self, state: RAGState):
        sub_qs = state.get("sub_questions") or []
        ctx = state["request_context"]
        return [
            Send(
                "rag_sub_agent",
                self._deps.initial_state(
                    sq,
                    ctx,
                    is_sub_agent=True,
                    rag_step_group=f"Sub-question {i}",
                    rag_step_group_label=sq,
                ),
            )
            for i, sq in enumerate(sub_qs, 1)
        ]


class RagSubAgent:
    """Run the only reachable sub-agent path directly: retrieve, then grade."""

    def __init__(self, retrieve: RetrieveInitial, grade: GradeDocuments) -> None:
        self._retrieve = retrieve
        self._grade = grade

    def __call__(self, state: RAGState) -> RAGState:
        question = state.get("question", "")
        result = dict(state)
        result.update(self._retrieve(result))
        result.update(self._grade(result))
        trace = result.get("rag_trace") or {}
        return {
            "sub_results": [{
                "question": question,
                "docs": result.get("docs", []),
                "retrieval_status": result.get("retrieval_status") or trace.get("retrieval_status"),
                "route": result.get("route") or trace.get("route"),
                "rag_trace": trace,
            }],
        }


class Synthesis:
    """Merge the sub-agents' graded documents into one context and one verdict."""

    def __init__(self, dependencies: RagDependencies) -> None:
        self._deps = dependencies

    @staticmethod
    def report(docs: List[dict], retrieval_status: str, sub_results: List[dict]) -> EvidenceReport:
        """The merged evidence report for a decomposed question.

        Synthesis does not assess anything itself — every document here was already graded by
        the sub-agent that retrieved it. So the report inherits HIGH certainty and records
        that provenance, rather than restating conclusions in its own words. Before this, this
        function hand-wrote `relevance="strong" if has_docs else "none"`, which is the same
        fabricated-grade pattern the ladder exists to remove.
        """
        assessed_by = ["synthesis"]
        for result in sub_results:
            for name in (result.get("rag_trace") or {}).get("evidence_assessed_by") or []:
                if name not in assessed_by:
                    assessed_by.append(name)

        if not docs:
            return EvidenceReport(
                certainty=Certainty.HIGH,
                relevance="none",
                sufficiency="none",
                assessed_by=assessed_by,
                reasons=[f"no sub-question produced usable evidence ({retrieval_status})"],
            )

        return EvidenceReport(
            chunks=[ChunkAssessment(index=i) for i, _ in enumerate(docs, 1)],
            certainty=Certainty.HIGH,
            relevance="strong",
            sufficiency="partial" if retrieval_status == "partial" else "sufficient",
            preferred_route="answer",
            assessed_by=assessed_by,
            reasons=[f"merged {len(docs)} graded chunk(s) from {len(sub_results)} sub-question(s)"],
        )

    def __call__(self, state: RAGState) -> RAGState:
        """Merges all documents retrieved by the sub-agents, dedupes and ranks them, and outputs the final context."""
        sub_results = state.get("sub_results", [])
        emit(state, "🔬", f"Synthesizing retrieval results from {len(sub_results)} sub-questions...")

        all_docs: List[dict] = []
        for result in sub_results:
            status = result.get("retrieval_status")
            if status not in ("answerable", "partial"):
                continue
            docs = result.get("docs", [])
            all_docs.extend(docs)

        deduped = self._deps.dedupe_documents(all_docs)
        for idx, item in enumerate(deduped, 1):
            item["rrf_rank"] = idx

        # An outage on any sub-question with zero merged docs means the empty result reflects
        # backend availability, not corpus coverage — surface retry, not no_knowledge.
        retrieval_outage = (not deduped) and any(
            result.get("retrieval_status") == "retrieval_error" for result in sub_results
        )

        context = format_docs(deduped)
        if deduped:
            emit(state, "✅", f"Synthesis complete, {len(deduped)} deduplicated snippets total")
        elif retrieval_outage:
            emit(state, "🚧", "Knowledge base temporarily unreachable for the sub-questions", "Returning a static retry notice")
        else:
            emit(state, "⛔", "None of the sub-questions had usable evidence")

        # Merge rag_trace from all sub-agents
        sub_traces = []
        for result in sub_results:
            trace = result.get("rag_trace")
            if trace:
                normalized_trace = normalize_rag_sub_trace(trace)
                if normalized_trace:
                    sub_traces.append(normalized_trace)

        original_trace = state.get("rag_trace") or {}
        has_docs = bool(deduped)
        retrieval_status = "answerable" if has_docs else "no_knowledge"
        if has_docs and any(result.get("retrieval_status") == "partial" for result in sub_results):
            retrieval_status = "partial"
        if retrieval_outage:
            retrieval_status = "retrieval_error"
        hitl_traces = [
            trace for trace in sub_traces
            if trace.get("retrieval_status") in ("needs_clarification", "needs_scope_selection")
        ]
        hitl_route = None
        hitl_prompt = ""
        hitl_options: List[str] = []
        # Outage outranks HITL: asking the user to clarify can't fix an unreachable backend.
        if not has_docs and not retrieval_outage and hitl_traces:
            scope_trace = next(
                (trace for trace in hitl_traces if trace.get("retrieval_status") == "needs_scope_selection"),
                None,
            )
            chosen_trace = scope_trace or hitl_traces[0]
            retrieval_status = chosen_trace.get("retrieval_status") or "needs_clarification"
            hitl_route = "scope_select" if retrieval_status == "needs_scope_selection" else "clarify"
            prompts = [
                trace.get("hitl_prompt")
                for trace in hitl_traces
                if trace.get("hitl_prompt")
            ]
            hitl_prompt = "; ".join(dict.fromkeys(prompts))
            for trace in hitl_traces:
                for option in trace.get("hitl_options") or []:
                    if option not in hitl_options:
                        hitl_options.append(option)

        fallback_route = "retrieval_error" if retrieval_outage else "no_knowledge"
        route = "answer" if has_docs else (hitl_route or fallback_route)
        rag_trace = {
            **original_trace,
            "tool_used": True,
            "tool_name": "search_knowledge_base",
            "query": state["question"],
            "retrieved_chunks": deduped,
            "retrieval_stage": "synthesis",
            "complexity": "complex",
            "complexity_reason": state.get("complexity_reason", ""),
            "sub_questions": state.get("sub_questions", []),
            "sub_agent_count": len(sub_results),
            "synthesis_merged_count": len(all_docs),
            "sub_traces": sub_traces,
            "retrieval_status": retrieval_status,
            "route": route,
            "hitl_prompt": hitl_prompt,
            "hitl_options": hitl_options,
            **self.report(deduped, retrieval_status, sub_results).as_trace(),
        }

        return {
            "docs": deduped,
            "context": context,
            "route": route,
            "retrieval_status": retrieval_status,
            "hitl_prompt": hitl_prompt,
            "hitl_options": hitl_options,
            "rag_trace": rag_trace,
        }


class ResumeRetrieval:
    """The targeted search a resumed clarification runs, graded like any other."""

    def __init__(self, dependencies: RagDependencies, grade: GradeDocuments) -> None:
        self._deps = dependencies
        self._grade = grade

    def __call__(self, state: dict) -> dict:
        emit(state, "🔎", "Running targeted retrieval using the HITL follow-up", "Skipping complexity classification and sub-question decomposition")
        query = search_query(state)
        retrieved = self._deps.retrieve_documents(
            query, top_k=self._deps.top_k, language=str(state.get("language") or "")
        )
        results = retrieved.get("docs", [])
        retrieve_meta = retrieved.get("meta", {})
        retrieval_failed = retrieve_meta.get("retrieval_mode") == "failed"
        context = format_docs(results)
        if retrieval_failed:
            emit(
                state,
                "🚧",
                "Knowledge base temporarily unreachable during HITL retrieval",
                retrieve_meta.get("retrieval_error") or "retrieval backend error",
            )
        else:
            emit(
                state,
                "🧱",
                "HITL three-tier chunk retrieval",
                (
                    f"Leaf level L{retrieve_meta.get('leaf_retrieve_level', 3)} recall, "
                    f"candidates {retrieve_meta.get('candidate_k', 0)}"
                ),
            )
            emit(
                state,
                "🧩",
                "Auto-merging",
                (
                    f"Enabled: {bool(retrieve_meta.get('auto_merge_enabled'))}, "
                    f"Applied: {bool(retrieve_meta.get('auto_merge_applied'))}, "
                    f"Replaced chunks: {retrieve_meta.get('auto_merge_replaced_chunks', 0)}"
                ),
            )
            emit(state, "✅", f"HITL targeted retrieval complete, found {len(results)} snippets", f"Mode: {retrieve_meta.get('retrieval_mode', 'hybrid')}")
        rag_trace = state.get("rag_trace") or {}
        rag_trace.update({
            "tool_used": True,
            "tool_name": "search_knowledge_base",
            "query": query,
            "retrieved_chunks": results,
            "hitl_targeted_retrieved_chunks": results,
            "hitl_resumed": True,
            "hitl_resume_strategy": "targeted_retrieval",
            "retrieval_stage": "hitl_targeted_retrieval",
            **self._deps.retrieval_trace_fields(retrieve_meta),
        })
        state.update({
            "query": query,
            "docs": results,
            "context": context,
            "retrieval_failed": retrieval_failed,
            "rag_trace": rag_trace,
        })
        state.update(self._grade(state))
        return state


__all__ = [
    "ClassifyComplexity",
    "FanOutSubQuestions",
    "GradeDocuments",
    "PrepareSubQuestions",
    "RAGState",
    "RagDependencies",
    "RagSubAgent",
    "ResumeRetrieval",
    "RetrieveInitial",
    "RetrieveRewritten",
    "RewriteQuestion",
    "Synthesis",
    "child_names",
    "emit",
    "route_after_complexity",
    "route_after_grade",
    "route_after_rewrite",
    "search_query",
]
