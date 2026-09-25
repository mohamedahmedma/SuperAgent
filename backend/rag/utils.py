from collections import defaultdict
import math
from typing import List, Tuple, Dict, Any, Literal, Optional
import logging
import os
import json
import requests
from langsmith import traceable

from backend.indexing.embedding import embed_query
from backend.rag.rerank_assessor import CrossEncoderProvider
from backend.env import env_bool, env_float, env_int, env_value
from backend.profiles import get_profile
from backend.prompts import resolve as resolve_prompt
from backend.text_matching import search_key
from backend.text_normalization import normalize_query
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
# The largest text one retrieved chunk may carry — see RetrievalConfig. Leaves are
# bounded by the chunking budgets; this is what bounds the merge path, and together they
# are what make the size of a prompt a property of `top_k` rather than of the corpus.
EVIDENCE_WINDOW_CHARS = env_int(
    "RETRIEVAL_EVIDENCE_WINDOW_CHARS", _RETRIEVAL.evidence_window_chars, minimum=200
)
# How many of the final slots one image may occupy — see `_limit_per_asset`.
MAX_CHUNKS_PER_ASSET = env_int(
    "RETRIEVAL_MAX_CHUNKS_PER_ASSET", _RETRIEVAL.max_chunks_per_asset, minimum=1
)
# A local cross-encoder over the candidate pool — see `_local_rerank`.
RERANK_LOCAL_ENABLED = env_bool("RERANK_LOCAL_ENABLED", _RETRIEVAL.rerank_local_enabled)
RERANK_LOCAL_MODEL = (os.getenv("RERANK_LOCAL_MODEL") or _RETRIEVAL.rerank_local_model).strip()
RERANK_LOCAL_DEVICE = (os.getenv("RERANK_LOCAL_DEVICE") or _RETRIEVAL.rerank_local_device).strip()
RERANK_LOCAL_BATCH_SIZE = env_int(
    "RERANK_LOCAL_BATCH_SIZE", _RETRIEVAL.rerank_local_batch_size, minimum=1
)
# How many fused candidates the cross-encoder scores — the whole cost of reranking, and
# the ceiling on what it can rescue. See RetrievalConfig.
RERANK_LOCAL_TOP_N = env_int(
    "RERANK_LOCAL_TOP_N", _RETRIEVAL.rerank_local_top_n, minimum=0
)

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
    "evidence_window_chars",
    # The largest chunk this retrieval actually returned. Reported so the invariant is
    # VISIBLE rather than merely intended: a value above `evidence_window_chars` means
    # something reached the model that the bound says cannot, and on a deployment the
    # likeliest cause is an index built before figures were split — which a reindex, not
    # a code change, is what fixes.
    "max_chunk_chars",
    "rerank_enabled",
    "rerank_applied",
    "rerank_model",
    "rerank_endpoint",
    "rerank_error",
    "rerank_timeout_seconds",
    "rerank_min_score",
    "post_rerank_count",
    "max_chunks_per_asset",
    "chunks_crowded_out",
    "post_threshold_count",
    "retrieval_empty",
)

def _milvus():
    """The vector store, from the process container.

    It used to be opened at import, so a process that only wanted
    `language_filter_clause` connected to Milvus to get it. The container builds one on
    first real use and shares it with ingest, which is what keeps the BM25 state and the
    embedder consistent between the two.
    """
    from backend.composition import default_services

    return default_services().milvus


def _parent_chunks():
    """The parent-chunk store, from the process container.

    Resolved per call rather than held here: a module-level instance is built at import,
    which is what step 4 removed. The container caches it, so this costs a dictionary
    lookup.
    """
    from backend.composition import default_services

    return default_services().parent_chunks



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


#: How a figure's text announces itself inside a chunk. Imported rather than spelled
#: again: it is written by `AssetDossier.surrogate_parts` and read here, and two
#: spellings would fail silently rather than loudly.
from backend.assets.dossier import FIGURE_MARKER as _FIGURE_MARKER  # noqa: E402
from backend.structured_output import StructuredOutput


def _block_candidates(child_text: str) -> List[str]:
    """The child as one block, and the child without the section prefix its level added.

    A leaf is not reliably a substring of its own parent, and the reason is
    `_apply_section_prefix`: it prepends a SYNTHESISED path ("Fees > Payment") that the
    parent normally does not contain, because the parent holds those headings as
    separate block lines. It is applied independently at each level and skipped when the
    section title already appears near the top, so whether a given child is a substring
    varies child by child.

    Measured over the whole school corpus, 211 child/parent pairs: the text locates
    51.7% of them exactly and dropping its first line a further 47.9%, leaving 0.5% to
    the line strategy below and nothing unlocated. The second is not a fallback nobody
    expects to run — it carries nearly half the corpus.
    """
    text = (child_text or "").strip()
    if not text:
        return []
    lines = text.splitlines()
    candidates = [text]
    if len(lines) > 1:
        candidates.append("\n".join(lines[1:]).strip())
    return candidates


def _parent_line_offsets(parent_text: str) -> Dict[str, List[int]]:
    """Where each of the parent's lines begins, by the line's own text."""
    offsets: Dict[str, List[int]] = defaultdict(list)
    cursor = 0
    for line in parent_text.split("\n"):
        offsets[line.strip()].append(cursor)
        cursor += len(line) + 1
    return offsets


def _match_spans(parent_text: str, child_text: str, budget: int) -> List[Tuple[int, int]]:
    """Where a matched child sits inside its parent. Empty when it cannot be placed.

    Two strategies, and the second one is why this is not a single `find`.

    As one block, the child either appears once or it does not, and that settles it.

    Line by line, it does not settle anything, because the lines this pipeline repeats
    are the ones a child shares with its parent's other children:
    `_split_table_row_groups` re-emits the header row in every group and
    `_figure_passages` repeats the caption on every passage. Both are usually LONGER
    than the data rows beneath them, so "anchor on the longest line" anchors on the one
    line that says nothing about where the child is. A child holding rows 35 to 40 of a
    fee table anchored at offset 0 and the window came back holding rows 1 to 34 — the
    parent's own text, none of the child's, and a fee for the wrong year group. That is
    item 26's failure arriving as a mechanism rather than a model error, and it is
    silent: the chunk still reports the merge it did and a size inside the bound.

    So the lines are used together rather than one being trusted. Each line that occurs
    exactly once in the parent is an anchor; the anchors are then taken around their
    median and any that sit further than a window away are dropped. A repeated header is
    not an anchor at all, and a unique header pulling towards the top of the table is
    outvoted by the rows that came with it.

    Whole lines, not substrings: "figure line 2" is a substring of "figure line 20", so
    substring uniqueness calls an ordinary line ambiguous and refuses a merge that is
    perfectly locatable.
    """
    text = (child_text or "").strip()
    if not text or not parent_text:
        return []

    for candidate in _block_candidates(text):
        start = parent_text.find(candidate)
        if start >= 0 and parent_text.find(candidate, start + 1) < 0:
            return [(start, start + len(candidate))]

    offsets = _parent_line_offsets(parent_text)
    spans: List[Tuple[int, int]] = []
    for line in (raw.strip() for raw in text.split("\n")):
        found = offsets.get(line) or []
        if line and len(found) == 1:
            spans.append((found[0], found[0] + len(line)))
    if not spans:
        return []

    starts = sorted(start for start, _ in spans)
    median = starts[len(starts) // 2]
    return [span for span in spans if abs(span[0] - median) <= budget]


def _parent_window(parent_text: str, children: List[dict], budget: int) -> Optional[str]:
    """The parent text around what actually matched, within `budget`. None if unlocatable.

    Auto-merging replaces matched children with their whole parent, and it does that
    BEFORE ranking, so the passage that earned the hit can end up anywhere inside a much
    larger chunk. That was survivable while the grader was shown a prefix of each chunk
    and merely expensive; it is the thing that decides the size of a prompt once the
    grader reads chunks whole.

    So the merge still gives a hit its surrounding context — which is what it is for —
    but the context is chosen by where the match IS. Whole lines, grown outward from the
    match in both directions until the budget is reached, because a line is a table row
    here and half a row is worse than no row.

    A parent already inside the budget comes back unchanged, which is the common case
    and byte-identical to the old behaviour.

    Returning None rather than the whole parent is deliberate: a merge that cannot find
    its own match cannot claim to be sending the text around it, and the children it
    would have replaced are the passages that actually matched and are leaf-bounded
    already. Keeping them is both the safer evidence and the safer size.
    """
    text = parent_text or ""
    if len(text) <= budget:
        return text

    spans = [
        span
        for child in children
        for span in _match_spans(text, child.get("text", ""), budget)
    ]
    if not spans:
        return None

    lines = text.split("\n")
    bounds: List[Tuple[int, int]] = []
    cursor = 0
    for line in lines:
        bounds.append((cursor, cursor + len(line)))
        cursor += len(line) + 1

    core = [
        index for index, (start, end) in enumerate(bounds)
        if any(start < span_end and end > span_start for span_start, span_end in spans)
    ]
    if not core:
        return None

    first, last = core[0], core[-1]
    size = sum(len(lines[index]) + 1 for index in range(first, last + 1)) - 1
    if size > budget:
        # The match itself is wider than the budget — a single enormous row is the case.
        # Cut from the EARLIEST match rather than from whichever child came first in the
        # candidate list, so a second match that sits before it is not cut away. A window
        # of whole lines is not a bound on its own, because one line can be longer than
        # the whole window.
        return text[min(start for start, _ in spans):][:budget]

    before, after = first - 1, last + 1
    while True:
        grew = False
        if before >= 0 and size + len(lines[before]) + 1 <= budget:
            size += len(lines[before]) + 1
            before -= 1
            grew = True
        if after < len(lines) and size + len(lines[after]) + 1 <= budget:
            size += len(lines[after]) + 1
            after += 1
            grew = True
        if not grew:
            break
    return "\n".join(lines[before + 1:after])


def _merge_to_parent_level(
    docs: List[dict],
    threshold: int = 2,
    figure_threshold: Optional[int] = None,
    window_chars: Optional[int] = None,
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

    parent_docs = _parent_chunks().get_documents_by_ids(merge_parent_ids)
    parent_map = {item.get("chunk_id", ""): item for item in parent_docs if item.get("chunk_id")}

    budget = int(window_chars if window_chars is not None else EVIDENCE_WINDOW_CHARS)
    windows: Dict[str, str] = {}
    for parent_id, parent in parent_map.items():
        window = _parent_window(parent.get("text", ""), groups.get(parent_id, []), budget)
        if window is None:
            logger.info(
                "not merging %s: its matched children could not be located in it; "
                "keeping the children, which matched and are already leaf-sized",
                parent_id,
            )
            continue
        windows[parent_id] = window

    merged_docs: List[dict] = []
    parent_slot: Dict[str, int] = {}
    merged_count = 0
    for doc in docs:
        parent_id = (doc.get("parent_chunk_id") or "").strip()
        if not parent_id or parent_id not in windows:
            merged_docs.append(doc)
            continue

        if parent_id in parent_slot:
            existing = merged_docs[parent_slot[parent_id]]
            _merge_rank_score_into(existing, doc)
            merged_count += 1
            continue

        parent_doc = dict(parent_map[parent_id])
        window = windows[parent_id]
        if window != parent_doc.get("text"):
            parent_doc["text"] = window
            parent_doc["merged_window_applied"] = True
            # The picture may have been windowed out. A chunk that no longer contains a
            # figure must not still claim one: the model is shown `[FIGURE n]` per asset
            # the chunk carries, so keeping the id here attaches a picture to an answer
            # written from two paragraphs that do not mention it.
            #
            # Presence of the marker, not which asset — a windowed parent holding one of
            # its two figures still reports both, and that over-reports rather than
            # inventing. Splitting images makes this path common rather than rare, since
            # one image now spans several parents.
            if _FIGURE_MARKER not in window:
                parent_doc["asset_ids"] = []
                if parent_doc.get("modality") == "figure":
                    parent_doc["modality"] = "text"
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
        "evidence_window_chars": EVIDENCE_WINDOW_CHARS,
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


def _limit_per_asset(docs: List[dict], limit: int, top_k: int) -> Tuple[List[dict], int]:
    """The best `top_k` chunks, with no single image taking more than `limit` of them.

    A new risk, and one that only exists because images are now split: while an image was
    one chunk it could occupy one slot, and now a transcribed calendar is ten passages
    that all match a question about the calendar. Measured on the school corpus after
    splitting, one image already took 5 of 8 slots on one question and 4 on another —
    the rest of the corpus was crowded out of a set that is meant to be the whole
    evidence for an answer.

    Applied to the RANKED order and before the cut, so a capped chunk gives its slot to
    the next best thing rather than shrinking the set. Per IMAGE, not per document: this
    corpus is one document, so a per-document cap would be either a no-op or a gag, and
    the thing that multiplies is passages of one picture.

    A question genuinely answered by an image still gets `limit` of its passages, which
    is the discovery passage plus the transcription around the match.
    """
    if limit <= 0:
        return docs[:top_k], 0
    kept: List[dict] = []
    per_asset: Dict[str, int] = defaultdict(int)
    dropped = 0
    for doc in docs:
        assets = [asset for asset in (doc.get("asset_ids") or []) if asset]
        if assets and any(per_asset[asset] >= limit for asset in assets):
            dropped += 1
            continue
        for asset in assets:
            per_asset[asset] += 1
        kept.append(doc)
        if len(kept) >= top_k:
            break
    return kept, dropped


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value)) if -30.0 < value < 30.0 else (0.0 if value <= 0 else 1.0)


#: The local cross-encoder, loaded once per process including its failure. Its own
#: instance rather than the assessor's, so the two cannot silently share a model name.
_local_reranker = CrossEncoderProvider()


def _local_rerank(query: str, docs: List[dict]) -> Optional[List[dict]]:
    """The candidates ordered by a cross-encoder, or None if it is not available here.

    A cross-encoder reads the question and the chunk TOGETHER, which is the thing neither
    half of hybrid retrieval does: dense compares two independent embeddings and BM25
    compares terms, and on a corpus written in one language and questioned in another the
    sparse half contributes almost nothing. RRF then fuses two rank lists without anything
    ever having judged a pair.

    Scores are squashed to (0, 1). The raw output of a one-label cross-encoder is a logit
    and can be negative, and `_meets_rerank_min_score` compares it against
    `RERANK_MIN_SCORE`, which is 0.0 — so passing logits through would silently drop every
    chunk the model was unsure about, and at a threshold that was written for a different
    scale entirely. Squashing keeps the ORDER identical (sigmoid is monotonic) and makes
    the number mean the same kind of thing the threshold does.
    """
    model = _local_reranker.model(RERANK_LOCAL_MODEL, RERANK_LOCAL_DEVICE)
    if model is None:
        return None

    # Only the head of the fused order is scored, because a forward pass per pair is the
    # whole cost of this and it is paid on every turn. The tail keeps its fused order and
    # stays BELOW everything scored — a chunk RRF placed 13th cannot be promoted, which is
    # the ceiling `rerank_local_top_n` documents, but it is not discarded either.
    depth = RERANK_LOCAL_TOP_N if RERANK_LOCAL_TOP_N > 0 else len(docs)
    head, tail = docs[:depth], docs[depth:]
    pairs = [(query, str(doc.get("text") or "")[:RERANK_DOC_CHAR_LIMIT]) for doc in head]
    try:
        scores = model.predict(pairs, batch_size=RERANK_LOCAL_BATCH_SIZE, show_progress_bar=False)
    except Exception:
        logger.exception("local rerank failed; falling back to the fused retrieval order")
        return None
    scored = []
    for doc, score in zip(head, scores):
        value = float(score)
        scored.append({**doc, "rerank_score": value if 0.0 <= value <= 1.0 else _sigmoid(value)})
    scored.sort(key=lambda item: item["rerank_score"], reverse=True)
    return scored + list(tail)


@traceable(name="rerank_documents", run_type="tool")
def _rerank_documents(query: str, docs: List[dict], top_k: int) -> Tuple[List[dict], Dict[str, Any]]:
    """The candidates in relevance order. ORDERED, not cut.

    The cut moved to `_finalize_retrieval`, because the per-image cap has to be applied
    to the ranking before the final slots are handed out — a cap applied after the cut
    can only shrink the set, never let the next best chunk take the slot.
    """
    docs_with_rank = [{**doc, "rrf_rank": i} for i, doc in enumerate(docs, 1)]
    meta: Dict[str, Any] = {
        "rerank_enabled": RERANK_ENABLED or RERANK_LOCAL_ENABLED,
        "rerank_applied": False,
        "rerank_model": RERANK_MODEL,
        "rerank_endpoint": _get_rerank_endpoint(),
        "rerank_error": None,
        "rerank_timeout_seconds": RERANK_TIMEOUT_SECONDS,
        "candidate_count": len(docs_with_rank),
    }
    if not docs_with_rank:
        return docs_with_rank, meta

    # The LOCAL cross-encoder first when it is configured. Each branch is gated on its own
    # flag, and deliberately: `rerank_enabled` in the meta means "something reranked this",
    # which is what a trace wants to say, and reading it back as "the REMOTE one is on" is
    # a bug this function already had — with the local reranker enabled and no endpoint
    # configured it fell straight through to an HTTP POST at an empty URL, failed, and
    # reported a silent RRF fallback in 118 ms.
    if RERANK_LOCAL_ENABLED and not RERANK_ENABLED:
        scored = _local_rerank(query, docs_with_rank)
        if scored is not None:
            meta.update({
                "rerank_applied": True,
                "rerank_model": RERANK_LOCAL_MODEL,
                "rerank_endpoint": f"local:{RERANK_LOCAL_DEVICE}",
            })
            return scored, meta
        meta["rerank_error"] = "local_cross_encoder_unavailable"
        return _sort_by_rank_score(docs_with_rank), meta

    if not RERANK_ENABLED:
        return _sort_by_rank_score(docs_with_rank), meta

    payload = {
        "model": RERANK_MODEL,
        "query": query,
        # Truncated: rerank providers bill per ~500-token document unit, and the
        # relevance signal sits in the opening span of a chunk anyway.
        "documents": [doc.get("text", "")[:RERANK_DOC_CHAR_LIMIT] for doc in docs_with_rank],
        # Every candidate, because the caller cuts. Scores and indices only come back
        # (`return_documents` is false), so asking for the full ordering costs nothing
        # and is what lets the per-image cap promote the next best chunk into a slot.
        "top_n": len(docs_with_rank),
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
            return _sort_by_rank_score(docs_with_rank), meta

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
            return reranked, meta

        meta["rerank_error"] = "empty_rerank_results"
        return _sort_by_rank_score(docs_with_rank), meta
    except (requests.RequestException, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        meta["rerank_error"] = str(e)
        return _sort_by_rank_score(docs_with_rank), meta


class RewritePlan(StructuredOutput):
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
    """The query-planning model, or None when this deployment has not configured one.

    A function rather than a direct container call at the one call site: it is the seam
    the rewrite tests substitute.
    """
    from backend.composition import default_services

    return default_services().models.rewriter()


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
    ranked, rerank_meta = _rerank_documents(query=query, docs=candidates, top_k=top_k)
    selected, crowded_out = _limit_per_asset(ranked, MAX_CHUNKS_PER_ASSET, top_k)
    post_rerank_count = len(selected)
    final_docs = [d for d in selected if _meets_rerank_min_score(d)]
    oversized = [d for d in final_docs if len(d.get("text") or "") > EVIDENCE_WINDOW_CHARS]
    if oversized:
        # Not trimmed here, and deliberately not: trimming at the edge is the grading
        # view again, one stage later, and it would cost the answer the same evidence.
        # The bound belongs where the size is made — chunking and the merge window — so
        # this says so instead of hiding it.
        logger.warning(
            "%d retrieved chunk(s) exceed evidence_window_chars=%d (largest %d): the "
            "index predates figure splitting, so reindex the affected documents",
            len(oversized), EVIDENCE_WINDOW_CHARS,
            max(len(d.get("text") or "") for d in oversized),
        )
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
        "max_chunks_per_asset": MAX_CHUNKS_PER_ASSET,
        "chunks_crowded_out": crowded_out,
        "post_threshold_count": len(final_docs),
        "retrieval_empty": len(final_docs) == 0,
        "max_chunk_chars": max((len(d.get("text") or "") for d in final_docs), default=0),
    }
    return {"docs": final_docs, "meta": meta}


def language_filter_clause(language: str) -> str:
    """The `and ...` that keeps a bilingual corpus from answering twice, or "".

    Excludes the redundant half of a PAIRED document — the one whose twin is in the
    language being asked in. Everything else stays eligible, so a document that exists
    in one language only still answers questions asked in the other. See
    `DocumentPairService.superseded_filenames` for why this is an exclusion and not a
    filter down to the asked language.

    Returns "" whenever there is nothing to exclude, which is the common case and also
    every case before an admin has paired anything — so this costs an empty list lookup
    and changes no behaviour until the feature is actually used.
    """
    if not language:
        return ""
    try:
        # Resolved per call, not at module scope: this module is re-executed with
        # `backend.indexing` stubbed by the retrieval symmetry tests, and a caller that
        # never routes should not pay for the database layer at import.
        from backend.composition import default_services

        superseded = default_services().document_pairs.superseded_filenames(language)
    except Exception:
        # A pairing lookup is an optimisation, not a gate. If the table cannot be read
        # the right outcome is to search the whole corpus and possibly answer from the
        # other language, never to fail the turn.
        logger.exception("could not read document pairs; retrieving without language routing")
        return ""
    if not superseded:
        return ""
    # json.dumps rather than manual quoting: filenames are admin-supplied and a stray
    # quote or backslash would otherwise produce an expression that is either invalid
    # or, worse, valid and wrong.
    return f" and filename not in {json.dumps(superseded, ensure_ascii=False)}"


@traceable(name="retrieve_documents", run_type="retriever")
def retrieve_documents(
    query: str, top_k: int = RETRIEVAL_TOP_K, language: str = ""
) -> Dict[str, Any]:
    # Normalize the query with the SAME rules applied to indexed text. Arabic
    # pasted from a PDF arrives as presentation forms / tatweel / zero-width
    # marks, which is a different character sequence from the indexed chunk —
    # without this the BM25 side cannot match and the dense side degrades.
    query = normalize_query(query) or query
    candidate_k, candidate_config = resolve_candidate_k(top_k)
    # Defaulted to "" so every existing caller — a test, a sub-agent, the entity
    # retriever — keeps searching the whole corpus. Language routing is opt-in per
    # call, and a caller that does not know the turn's language must not silently get
    # a narrowed corpus.
    filter_expr = f"chunk_level == {LEAF_RETRIEVE_LEVEL}" + language_filter_clause(language)
    try:
        # Memoized: the domain gate has usually already embedded this exact normalized
        # text, so this is a dictionary hit rather than a second forward pass.
        dense_embedding = embed_query(query)
    except Exception:
        logger.exception("could not embed the query %r", query[:120])
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
        retrieved = _milvus().hybrid_retrieve(
            dense_embedding=dense_embedding,
            # The SPARSE half only. `bm25_text` was folded and light-stemmed by
            # search_key on the way into the index, so the query has to be put through
            # the same function or the two are different strings and Arabic stops
            # matching altogether — this is the symmetry backend/text_matching.py exists
            # to hold. The dense half keeps the natural query above: bge-m3 reads Arabic
            # morphology, and folding before embedding throws away signal it would use.
            query=search_key(query),
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
            retrieved = _milvus().dense_retrieve(
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
            # Logged, not just counted. Both retrieval paths degrade to the same
            # "try again" notice, which is right for the user and useless for whoever
            # has to fix it — the cause has been Milvus not running, an auth failure and
            # a schema mismatch, and the trace said "failed" for all three.
            logger.exception("retrieval failed for query %r after dense fallback", query[:120])
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
