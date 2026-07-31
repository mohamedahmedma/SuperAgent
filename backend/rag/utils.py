from collections import defaultdict
from typing import List, Tuple, Dict, Any, Literal, Optional
import logging
import os
import json
import requests
from langsmith import traceable

from backend.indexing.milvus_client import get_milvus_store
from backend.indexing.embedding import embed_query, embedding_service as _embedding_service
from backend.env import env_bool, env_float, env_int, env_value
from backend.indexing.parent_chunk_store import ParentChunkStore
from backend.profiles import get_profile
from backend.prompts import resolve as resolve_prompt
from backend.text_normalization import normalize_query
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def _optional_env(name: str) -> Optional[str]:
    value = (os.getenv(name) or "").strip()
    if not value:
        return None
    normalized = value.lower()
    if (
        normalized.startswith(("your_", "your-", "replace-with"))
        or "your-rerank" in normalized
        or "your_rerank" in normalized
    ):
        return None
    return value


ARK_API_KEY = os.getenv("ARK_API_KEY")
FAST_MODEL = os.getenv("FAST_MODEL")
BASE_URL = os.getenv("BASE_URL")

# Retrieval tuning defaults come from the active domain profile; the environment
# readers below still take precedence, so the effective order is
# env > profile > schema default. The profile object itself is env-overlaid by
# backend/profiles/registry.py, so both paths agree on the final value.
_PROFILE = get_profile()
_RETRIEVAL = _PROFILE.retrieval

RERANK_MODEL = _optional_env("RERANK_MODEL")
RERANK_BINDING_HOST = _optional_env("RERANK_BINDING_HOST")
RERANK_API_KEY = _optional_env("RERANK_API_KEY")
RERANK_ENABLED = bool(RERANK_MODEL and RERANK_API_KEY and RERANK_BINDING_HOST)
RERANK_TIMEOUT_SECONDS = env_float(
    "RERANK_TIMEOUT_SECONDS", _RETRIEVAL.rerank_timeout_seconds, minimum=0.1
)
RERANK_DOC_CHAR_LIMIT = env_int(
    "RERANK_DOC_CHAR_LIMIT", _RETRIEVAL.rerank_doc_char_limit, minimum=200
)
AUTO_MERGE_ENABLED = env_bool("AUTO_MERGE_ENABLED", _RETRIEVAL.auto_merge_enabled)
AUTO_MERGE_THRESHOLD = env_int("AUTO_MERGE_THRESHOLD", _RETRIEVAL.auto_merge_threshold)
# None keeps figure-bearing groups unmerged so each image stays individually citable.
AUTO_MERGE_FIGURE_THRESHOLD = _RETRIEVAL.auto_merge_figure_threshold
LEAF_RETRIEVE_LEVEL = env_int("LEAF_RETRIEVE_LEVEL", _RETRIEVAL.leaf_retrieve_level)


RETRIEVAL_CANDIDATE_MULTIPLIER = env_int(
    "RETRIEVAL_CANDIDATE_MULTIPLIER", _RETRIEVAL.candidate_multiplier, minimum=1
)
RETRIEVAL_TOP_K = env_int("RETRIEVAL_TOP_K", _RETRIEVAL.top_k, minimum=1)
RERANK_MIN_SCORE = env_float("RERANK_MIN_SCORE", _RETRIEVAL.rerank_min_score)

# An explicit candidate pool size can come from either layer, and the retrieval trace
# reports WHICH — "env" and "profile" are different operational stories when someone
# is working out why a deployment recalls more than its profile says it should.
_CANDIDATE_K_FROM_ENV = env_value("RETRIEVAL_CANDIDATE_K")
if _CANDIDATE_K_FROM_ENV is not None:
    _RETRIEVAL_CANDIDATE_K_RAW = _CANDIDATE_K_FROM_ENV
    _CANDIDATE_K_EXPLICIT_SOURCE = "env"
elif _RETRIEVAL.candidate_k is not None:
    _RETRIEVAL_CANDIDATE_K_RAW = str(_RETRIEVAL.candidate_k)
    _CANDIDATE_K_EXPLICIT_SOURCE = "profile"
else:
    _RETRIEVAL_CANDIDATE_K_RAW = ""
    _CANDIDATE_K_EXPLICIT_SOURCE = "multiplier"

RETRIEVAL_TRACE_FIELDS = (
    "retrieval_pipeline",
    "retrieval_mode",
    "retrieval_error",
    "candidate_k",
    "candidate_k_source",
    "candidate_k_config_error",
    "retrieval_candidate_multiplier",
    "retrieval_top_k",
    "leaf_retrieve_level",
    "recall_count",
    "post_merge_candidate_count",
    "candidate_count",
    "auto_merge_enabled",
    "auto_merge_applied",
    "auto_merge_threshold",
    "auto_merge_figure_threshold",
    "auto_merge_replaced_chunks",
    "auto_merge_steps",
    "rerank_enabled",
    "rerank_applied",
    "rerank_model",
    "rerank_endpoint",
    "rerank_error",
    "rerank_timeout_seconds",
    "rerank_min_score",
    "post_rerank_count",
    "post_threshold_count",
    "retrieval_empty",
)

# Initialize retrieval dependencies globally (shares embedding_service with the API to keep BM25 state consistent)
_milvus_manager = get_milvus_store()
_parent_chunk_store = ParentChunkStore()

_rewrite_model = None


def resolve_candidate_k(top_k: int) -> Tuple[int, Dict[str, Any]]:
    """Resolve the Milvus candidate pool size; RETRIEVAL_CANDIDATE_K takes priority, otherwise top_k × multiplier."""
    if _RETRIEVAL_CANDIDATE_K_RAW:
        try:
            candidate_k = max(int(_RETRIEVAL_CANDIDATE_K_RAW), top_k)
        except ValueError:
            candidate_k = max(top_k * RETRIEVAL_CANDIDATE_MULTIPLIER, top_k)
            return candidate_k, {
                "candidate_k_source": "multiplier",
                "retrieval_candidate_multiplier": RETRIEVAL_CANDIDATE_MULTIPLIER,
                "candidate_k_config_error": "invalid RETRIEVAL_CANDIDATE_K",
            }
        return candidate_k, {
            "candidate_k_source": _CANDIDATE_K_EXPLICIT_SOURCE,
            "retrieval_candidate_multiplier": RETRIEVAL_CANDIDATE_MULTIPLIER,
        }
    candidate_k = max(top_k * RETRIEVAL_CANDIDATE_MULTIPLIER, top_k)
    return candidate_k, {
        "candidate_k_source": "multiplier",
        "retrieval_candidate_multiplier": RETRIEVAL_CANDIDATE_MULTIPLIER,
    }


def retrieval_trace_fields(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the retrieval fields that should be written into rag_trace from the retrieve meta."""
    return {key: meta[key] for key in RETRIEVAL_TRACE_FIELDS if key in meta and meta[key] is not None}


def _get_rerank_endpoint() -> str:
    if not RERANK_BINDING_HOST:
        return ""
    host = RERANK_BINDING_HOST.strip().rstrip("/")
    return host if host.endswith("/v1/rerank") else f"{host}/v1/rerank"


def _effective_score(doc: dict) -> Optional[float]:
    """Prefer the rerank score, otherwise use the recall score; used for merge aggregation and post-merge reranking."""
    rerank_score = doc.get("rerank_score")
    if rerank_score is not None:
        return float(rerank_score)
    score = doc.get("score")
    if score is not None:
        return float(score)
    return None


def _meets_rerank_min_score(doc: dict) -> bool:
    score = _effective_score(doc)
    if score is None:
        return RERANK_MIN_SCORE <= 0
    return score >= RERANK_MIN_SCORE


def _merge_rank_score_into(target: dict, source: dict) -> None:
    incoming = _effective_score(source)
    if incoming is None:
        return
    uses_rerank = source.get("rerank_score") is not None or target.get("rerank_score") is not None
    if uses_rerank:
        existing = target.get("rerank_score")
        if existing is None:
            target["rerank_score"] = incoming
        else:
            target["rerank_score"] = max(float(existing), incoming)
        return
    existing = target.get("score")
    if existing is None:
        target["score"] = incoming
    else:
        target["score"] = max(float(existing), incoming)


def _is_figure_chunk(doc: dict) -> bool:
    """Whether a chunk carries an image. `asset_ids` is checked as well as `modality`
    so a chunk written before the modality field existed is still recognised."""
    return doc.get("modality") == "figure" or bool(doc.get("asset_ids"))


def _merge_to_parent_level(
    docs: List[dict],
    threshold: int = 2,
    figure_threshold: Optional[int] = None,
) -> Tuple[List[dict], int]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    for doc in docs:
        parent_id = (doc.get("parent_chunk_id") or "").strip()
        if parent_id:
            groups[parent_id].append(doc)

    merge_parent_ids: List[str] = []
    for parent_id, children in groups.items():
        if any(_is_figure_chunk(child) for child in children):
            # Merging two figure leaves into one parent makes a single citation point
            # at several images, which is exactly the ambiguity the citation was
            # supposed to remove. Text groups keep the normal threshold.
            if figure_threshold is None or len(children) < figure_threshold:
                continue
        elif len(children) < threshold:
            continue
        merge_parent_ids.append(parent_id)
    if not merge_parent_ids:
        return docs, 0

    parent_docs = _parent_chunk_store.get_documents_by_ids(merge_parent_ids)
    parent_map = {item.get("chunk_id", ""): item for item in parent_docs if item.get("chunk_id")}

    merged_docs: List[dict] = []
    parent_slot: Dict[str, int] = {}
    merged_count = 0
    for doc in docs:
        parent_id = (doc.get("parent_chunk_id") or "").strip()
        if not parent_id or parent_id not in parent_map:
            merged_docs.append(doc)
            continue

        if parent_id in parent_slot:
            existing = merged_docs[parent_slot[parent_id]]
            _merge_rank_score_into(existing, doc)
            merged_count += 1
            continue

        parent_doc = dict(parent_map[parent_id])
        _merge_rank_score_into(parent_doc, doc)
        parent_doc["merged_from_children"] = True
        parent_doc["merged_child_count"] = len(groups[parent_id])
        parent_slot[parent_id] = len(merged_docs)
        merged_docs.append(parent_doc)
        merged_count += 1

    return merged_docs, merged_count


def _empty_merge_meta() -> Dict[str, Any]:
    return {
        "auto_merge_enabled": AUTO_MERGE_ENABLED,
        "auto_merge_applied": False,
        "auto_merge_threshold": AUTO_MERGE_THRESHOLD,
        "auto_merge_figure_threshold": AUTO_MERGE_FIGURE_THRESHOLD,
        "auto_merge_replaced_chunks": 0,
        "auto_merge_steps": 0,
        "post_merge_candidate_count": 0,
    }


@traceable(name="auto_merge_candidates", run_type="tool")
def _auto_merge_candidates(docs: List[dict]) -> Tuple[List[dict], Dict[str, Any]]:
    """Perform L3→L2→L1 merging over the full set of recall candidates; order is unchanged, reranking is handled by a later step."""
    meta = _empty_merge_meta()
    meta["post_merge_candidate_count"] = len(docs)
    if not AUTO_MERGE_ENABLED or not docs:
        return docs, meta

    merged_docs, merged_count_l3_l2 = _merge_to_parent_level(
        docs, threshold=AUTO_MERGE_THRESHOLD, figure_threshold=AUTO_MERGE_FIGURE_THRESHOLD
    )
    merged_docs, merged_count_l2_l1 = _merge_to_parent_level(
        merged_docs, threshold=AUTO_MERGE_THRESHOLD, figure_threshold=AUTO_MERGE_FIGURE_THRESHOLD
    )

    replaced_count = merged_count_l3_l2 + merged_count_l2_l1
    meta.update({
        "auto_merge_applied": replaced_count > 0,
        "auto_merge_replaced_chunks": replaced_count,
        "auto_merge_steps": int(merged_count_l3_l2 > 0) + int(merged_count_l2_l1 > 0),
        "post_merge_candidate_count": len(merged_docs),
    })
    return merged_docs, meta


def _sort_by_rank_score(docs: List[dict]) -> List[dict]:
    return sorted(docs, key=lambda item: _effective_score(item) or 0.0, reverse=True)


def dedupe_documents(docs: List[dict]) -> List[dict]:
    """Dedupe by chunk_id; duplicates keep the higher rank score (rerank_score takes priority)."""
    by_key: Dict[str, dict] = {}
    order: List[str] = []
    for item in docs:
        chunk_id = (item.get("chunk_id") or "").strip()
        key = chunk_id or f"{item.get('filename')}|{item.get('page_number')}|{item.get('text')}"
        if key not in by_key:
            by_key[key] = item
            order.append(key)
            continue
        _merge_rank_score_into(by_key[key], item)
    return [by_key[key] for key in order]


@traceable(name="rerank_documents", run_type="tool")
def _rerank_documents(query: str, docs: List[dict], top_k: int) -> Tuple[List[dict], Dict[str, Any]]:
    docs_with_rank = [{**doc, "rrf_rank": i} for i, doc in enumerate(docs, 1)]
    meta: Dict[str, Any] = {
        "rerank_enabled": RERANK_ENABLED,
        "rerank_applied": False,
        "rerank_model": RERANK_MODEL,
        "rerank_endpoint": _get_rerank_endpoint(),
        "rerank_error": None,
        "rerank_timeout_seconds": RERANK_TIMEOUT_SECONDS,
        "candidate_count": len(docs_with_rank),
    }
    if not docs_with_rank or not meta["rerank_enabled"]:
        return _sort_by_rank_score(docs_with_rank)[:top_k], meta

    payload = {
        "model": RERANK_MODEL,
        "query": query,
        # Truncated: rerank providers bill per ~500-token document unit, and the
        # relevance signal sits in the opening span of a chunk anyway.
        "documents": [doc.get("text", "")[:RERANK_DOC_CHAR_LIMIT] for doc in docs_with_rank],
        "top_n": min(top_k, len(docs_with_rank)),
        "return_documents": False,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RERANK_API_KEY}",
    }
    try:
        response = requests.post(
            meta["rerank_endpoint"],
            headers=headers,
            json=payload,
            timeout=RERANK_TIMEOUT_SECONDS,
        )
        if response.status_code >= 400:
            meta["rerank_error"] = f"HTTP {response.status_code}: {response.text}"
            return _sort_by_rank_score(docs_with_rank)[:top_k], meta

        items = response.json().get("results", [])
        reranked = []
        for item in items:
            idx = item.get("index")
            if isinstance(idx, int) and 0 <= idx < len(docs_with_rank):
                doc = dict(docs_with_rank[idx])
                score = item.get("relevance_score")
                if score is not None:
                    doc["rerank_score"] = score
                reranked.append(doc)

        if reranked:
            # Set here, not before the call: "applied" has to mean these documents
            # carry reranker scores. Setting it on entry made it mean "attempted",
            # which stayed true through every failure path below and misreported a
            # silent RRF fallback as a successful rerank.
            meta["rerank_applied"] = True
            return reranked[:top_k], meta

        meta["rerank_error"] = "empty_rerank_results"
        return _sort_by_rank_score(docs_with_rank)[:top_k], meta
    except (requests.RequestException, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        meta["rerank_error"] = str(e)
        return _sort_by_rank_score(docs_with_rank)[:top_k], meta


class RewritePlan(BaseModel):
    method: Literal["step_back", "hyde"] = Field(
        description="The single query-rewrite method used this round"
    )
    step_back_question: str = Field(
        default="",
        max_length=300,
        description="The abstracted step-back question, filled in only when method=step_back",
    )
    hyde_document: str = Field(
        default="",
        max_length=1200,
        description="The hypothetical answer document, filled in only when method=hyde",
    )


# Prompt text lives in the active profile (backend/profiles/definitions/*.yaml) so a
# domain can retune retrieval wording without a code change.
REWRITE_PROMPT = _PROFILE.rag.rewrite_prompt


def _get_rewrite_model():
    global _rewrite_model
    if not ARK_API_KEY or not FAST_MODEL:
        return None
    if _rewrite_model is None:
        _rewrite_model = init_chat_model(
            model=FAST_MODEL,
            model_provider="openai",
            api_key=ARK_API_KEY,
            base_url=BASE_URL,
            temperature=_PROFILE.models.rewrite_temperature,
            stream_usage=True,
        )
    return _rewrite_model


def rewrite_query_once(query: str) -> Optional[dict]:
    """Plan one rewrite, or return None when the plan cannot be produced.

    None rather than an exception. A rewrite is an OPTIONAL second attempt at
    retrieval that only runs when the first pass graded weak; if the planner fails,
    the correct outcome is "no rewrite happened", not a failed turn for the user.

    That distinction is not theoretical: providers enforcing OpenAI-style *strict*
    structured output (Groq among them) require every declared property to be present
    in the response. A planner told to leave the unused field "empty" tends to omit it
    instead, which fails schema validation. The prompt now asks for an empty string
    explicitly, and the field is inferred from whichever one came back populated — but
    the outer guard is what makes the path safe regardless of provider behaviour.
    """
    model = _get_rewrite_model()
    if not model:
        logger.warning("FAST_MODEL is not configured; skipping query rewrite")
        return None

    try:
        result = model.with_structured_output(RewritePlan).invoke(
            [{"role": "user", "content": resolve_prompt(REWRITE_PROMPT, "rag/rewrite.j2", query=query)}]
        )
    except Exception:
        logger.exception("Query-rewrite planning failed; continuing without a rewrite")
        return None

    method = (getattr(result, "method", "") or "").strip()
    step_back_question = (getattr(result, "step_back_question", "") or "").strip()
    hyde_document = (getattr(result, "hyde_document", "") or "").strip()

    # Trust the populated field over the declared method: a planner that fills in
    # hyde_document while labelling itself step_back has still expressed a usable
    # intent, and discarding it would waste the call.
    if method not in ("step_back", "hyde"):
        method = "step_back" if step_back_question else ("hyde" if hyde_document else "")
    if method == "step_back" and not step_back_question and hyde_document:
        method = "hyde"
    if method == "hyde" and not hyde_document and step_back_question:
        method = "step_back"

    if method == "step_back" and step_back_question:
        return {
            "rewrite_method": "step_back",
            "rewritten_query": f"{query}\n\nStep-back question: {step_back_question}",
            "step_back_question": step_back_question,
            "hyde_document": "",
        }
    if method == "hyde" and hyde_document:
        return {
            "rewrite_method": "hyde",
            "rewritten_query": f"{query}\n\nHypothetical answer document: {hyde_document}",
            "step_back_question": "",
            "hyde_document": hyde_document,
        }

    logger.warning("Query-rewrite plan was empty (method=%r); continuing without a rewrite", method)
    return None


def _finalize_retrieval(
    query: str,
    retrieved: List[dict],
    top_k: int,
    retrieval_mode: str,
    candidate_k: int,
    candidate_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Production pipeline: recall candidates → Auto-merge → Rerank (top_k) → threshold filtering."""
    candidates, merge_meta = _auto_merge_candidates(retrieved)
    reranked_docs, rerank_meta = _rerank_documents(query=query, docs=candidates, top_k=top_k)
    post_rerank_count = len(reranked_docs)
    final_docs = [d for d in reranked_docs if _meets_rerank_min_score(d)]
    meta = {
        **rerank_meta,
        **merge_meta,
        **candidate_config,
        "retrieval_mode": retrieval_mode,
        "retrieval_pipeline": "recall_merge_rerank",
        "candidate_k": candidate_k,
        "retrieval_top_k": top_k,
        "leaf_retrieve_level": LEAF_RETRIEVE_LEVEL,
        "recall_count": len(retrieved),
        "rerank_min_score": RERANK_MIN_SCORE,
        "post_rerank_count": post_rerank_count,
        "post_threshold_count": len(final_docs),
        "retrieval_empty": len(final_docs) == 0,
    }
    return {"docs": final_docs, "meta": meta}


@traceable(name="retrieve_documents", run_type="retriever")
def retrieve_documents(query: str, top_k: int = RETRIEVAL_TOP_K) -> Dict[str, Any]:
    # Normalize the query with the SAME rules applied to indexed text. Arabic
    # pasted from a PDF arrives as presentation forms / tatweel / zero-width
    # marks, which is a different character sequence from the indexed chunk —
    # without this the BM25 side cannot match and the dense side degrades.
    query = normalize_query(query) or query
    candidate_k, candidate_config = resolve_candidate_k(top_k)
    filter_expr = f"chunk_level == {LEAF_RETRIEVE_LEVEL}"
    try:
        # Memoized: the domain gate has usually already embedded this exact normalized
        # text, so this is a dictionary hit rather than a second forward pass.
        dense_embedding = embed_query(query)
    except Exception:
        return {
            "docs": [],
            "meta": {
                "rerank_enabled": RERANK_ENABLED,
                "rerank_applied": False,
                "rerank_model": RERANK_MODEL,
                "rerank_endpoint": _get_rerank_endpoint(),
                "rerank_error": "embedding_failed",
                "rerank_timeout_seconds": RERANK_TIMEOUT_SECONDS,
                "retrieval_error": "embedding_failed",
                "retrieval_mode": "failed",
                "retrieval_pipeline": "recall_merge_rerank",
                "candidate_k": candidate_k,
                **candidate_config,
                "retrieval_top_k": top_k,
                "leaf_retrieve_level": LEAF_RETRIEVE_LEVEL,
                "recall_count": 0,
                **_empty_merge_meta(),
                "candidate_count": 0,
                "rerank_min_score": RERANK_MIN_SCORE,
                "post_rerank_count": 0,
                "post_threshold_count": 0,
                "retrieval_empty": True,
            },
        }

    try:
        retrieved = _milvus_manager.hybrid_retrieve(
            dense_embedding=dense_embedding,
            query=query,
            top_k=candidate_k,
            filter_expr=filter_expr,
        )
        return _finalize_retrieval(
            query=query,
            retrieved=retrieved,
            top_k=top_k,
            retrieval_mode="hybrid",
            candidate_k=candidate_k,
            candidate_config=candidate_config,
        )
    except Exception:
        try:
            retrieved = _milvus_manager.dense_retrieve(
                dense_embedding=dense_embedding,
                top_k=candidate_k,
                filter_expr=filter_expr,
            )
            return _finalize_retrieval(
                query=query,
                retrieved=retrieved,
                top_k=top_k,
                retrieval_mode="dense_fallback",
                candidate_k=candidate_k,
                candidate_config=candidate_config,
            )
        except Exception:
            return {
                "docs": [],
                "meta": {
                    "rerank_enabled": RERANK_ENABLED,
                    "rerank_applied": False,
                    "rerank_model": RERANK_MODEL,
                    "rerank_endpoint": _get_rerank_endpoint(),
                    "rerank_error": "retrieve_failed",
                    "rerank_timeout_seconds": RERANK_TIMEOUT_SECONDS,
                    "retrieval_error": "retrieve_failed",
                    "retrieval_mode": "failed",
                    "retrieval_pipeline": "recall_merge_rerank",
                    "candidate_k": candidate_k,
                    **candidate_config,
                    "retrieval_top_k": top_k,
                    "leaf_retrieve_level": LEAF_RETRIEVE_LEVEL,
                    "recall_count": 0,
                    **_empty_merge_meta(),
                    "candidate_count": 0,
                    "rerank_min_score": RERANK_MIN_SCORE,
                    "post_rerank_count": 0,
                    "post_threshold_count": 0,
                    "retrieval_empty": True,
                },
            }
