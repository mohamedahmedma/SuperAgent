"""Summarising corpus sections so scope can be judged without searching.

The question "is this our subject at all?" is expensive to answer with retrieval,
because answering it that way means running the search you were trying to avoid. It is
cheap to answer against a small index built once — provided the index describes what
the corpus is ABOUT rather than what it literally says.

Three decisions shape this module, and each is a departure from the obvious version:

**Sections, not documents.** A document-level summary index has one vector per file,
which is useless on a single-document corpus — and a school handbook, a policy manual
or a product catalogue is usually one file. Summarising the top level of the existing
chunk hierarchy gives real granularity without inventing a second one.

**Questions, not descriptions.** The embedded field is `answers` — the questions a user
would actually ask that this section can answer — not the prose summary. Queries arrive
as questions, so question-to-question is the most homogeneous comparison available, and
homogeneous comparisons are what let a single cut point behave consistently. Each
question is embedded separately and scored by max: averaging six questions into one
vector puts the centroid near none of them.

**Cached by content hash.** A section whose text has not changed reuses its summary
verbatim. Re-indexing an unchanged corpus costs nothing, and identical input provably
yields identical output — so the scope boundary cannot drift between deployments or
between runs. Same reasoning as AssetExtraction, and the same shape.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence

from backend.prompts import render

logger = logging.getLogger(__name__)


@dataclass
class SectionRecord:
    """One section's catalogue entry, before or after it reaches the database."""

    chunk_id: str
    content_sha256: str = ""
    filename: str = ""
    chunk_level: int = 0
    summary: str = ""
    answers: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    # Parallel to `answers`. Empty until the build step embeds them; the index treats
    # a record without vectors as needing re-embedding rather than as having none.
    question_vectors: List[Sequence[float]] = field(default_factory=list)
    embedding_model: str = ""
    model_used: str = ""

    @property
    def usable(self) -> bool:
        """A record with no questions contributes nothing to the index.

        Kept as a property rather than filtered at construction so a failed section is
        still visible in the store — an empty `answers` list is the signal that a
        section needs re-summarising, and silently dropping it would hide that.
        """
        return bool(self.answers)


def content_hash(text: str) -> str:
    """Identity of a section's text, for cache reuse.

    Whitespace-normalised: re-chunking can reflow a section without changing a word,
    and re-summarising identical prose would spend a model call to produce the same
    answer with different wording.
    """
    normalized = " ".join((text or "").split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class SummarySchema:
    """The structured output the summariser asks for.

    Built dynamically because the topic vocabulary is profile data: constraining
    `topics` to an enum in the schema is what stops a model inventing labels, which is
    what would let the corpus catalogue drift between re-indexes.
    """

    @staticmethod
    def build(vocabulary: Sequence[str], min_questions: int, max_questions: int):
        from typing import List as _List

        from pydantic import BaseModel, Field

        vocab = [str(item) for item in vocabulary if str(item).strip()]

        class SectionCatalogue(BaseModel):
            summary: str = Field(description="One or two sentences on what this section is about")
            answers: _List[str] = Field(
                description="Questions a user would ask that this section answers",
                min_length=1,
                max_length=max(max_questions, 1),
            )
            topics: _List[str] = Field(default_factory=list, description="Labels from the supplied list")

        # The vocabulary is enforced after generation rather than as a schema enum:
        # several providers reject enums inside arrays, and a rejected schema costs the
        # whole record where a filtered label costs one label.
        SectionCatalogue.model_config = {"json_schema_extra": {"vocabulary": vocab}}
        return SectionCatalogue


def summarise_section(
    text: str,
    *,
    invoke,
    vocabulary: Sequence[str],
    persona: str = "",
    languages: str = "",
    min_questions: int = 4,
    max_questions: int = 10,
) -> Optional[Dict[str, Any]]:
    """One section in, one catalogue entry out. `invoke(prompt, schema)` does the call.

    Returns None on failure. A section that cannot be summarised is skipped rather than
    guessed at: an invented question would teach the gate to accept a subject the
    corpus cannot actually help with, which is worse than the section being absent.
    """
    prompt = render(
        "rag/section_summary.j2",
        section_text=text,
        vocabulary=list(vocabulary),
        persona=persona,
        languages=languages,
        min_questions=min_questions,
        max_questions=max_questions,
    )
    schema = SummarySchema.build(vocabulary, min_questions, max_questions)
    try:
        result = invoke(prompt, schema)
    except Exception:
        logger.warning("section summarisation failed", exc_info=True)
        return None

    payload = result if isinstance(result, dict) else getattr(result, "model_dump", lambda: {})()
    answers = _clean_questions(payload.get("answers"), max_questions)
    if not answers:
        return None

    return {
        "summary": str(payload.get("summary") or "").strip(),
        "answers": answers,
        "topics": _clean_topics(payload.get("topics"), vocabulary),
    }


def _clean_questions(raw, limit: int) -> List[str]:
    """Deduplicate and bound. Near-identical questions cost a vector each and add no
    coverage, so they are collapsed case-insensitively."""
    seen = set()
    questions: List[str] = []
    for item in raw or []:
        text = " ".join(str(item).split())
        if len(text) < 3:
            continue
        key = text.lower().rstrip("?")
        if key in seen:
            continue
        seen.add(key)
        questions.append(text)
        if len(questions) >= limit:
            break
    return questions


def _clean_topics(raw, vocabulary: Sequence[str]) -> List[str]:
    """Keep only labels the profile actually declares.

    Enforced here rather than trusted from the model: an invented label would enter the
    catalogue, reach the scope prompt, and differ between two runs over the same corpus
    — which is exactly the drift the frozen vocabulary exists to prevent.

    An EMPTY vocabulary is the bootstrap case, and means the opposite: nothing has been
    frozen yet, so proposed labels are kept as-is for a human to review and freeze.
    Discarding them would make the first build produce no catalogue at all, and there
    would be nothing to derive the vocabulary FROM.
    """
    proposed = [" ".join(str(item).split()).strip().lower() for item in (raw or [])]
    proposed = [item for item in proposed if item]

    if not vocabulary:
        deduped: List[str] = []
        for item in proposed:
            if item not in deduped:
                deduped.append(item)
        return deduped[:3]

    allowed = {str(item).strip().lower(): str(item).strip() for item in vocabulary}
    topics: List[str] = []
    for item in proposed:
        canonical = allowed.get(item)
        if canonical and canonical not in topics:
            topics.append(canonical)
    return topics


def plan_sections(sections: Iterable[dict], existing: Dict[str, str]) -> Dict[str, List[dict]]:
    """Split sections into those needing a model call and those already catalogued.

    `existing` maps chunk_id to the content hash already stored. Anything whose text is
    unchanged is reused, which is what makes a re-index of an edited corpus cost only
    the edited sections.
    """
    todo: List[dict] = []
    reuse: List[dict] = []
    for section in sections:
        digest = content_hash(section.get("text", ""))
        entry = {**section, "content_sha256": digest}
        (reuse if existing.get(section.get("chunk_id")) == digest else todo).append(entry)
    return {"summarise": todo, "reuse": reuse}


def corpus_catalogue(records: Sequence[SectionRecord], limit: int = 24) -> str:
    """A one-line description of what the corpus covers, for the scope prompt.

    Topics rather than section headings: headings are internal structure and often
    meaningless out of context ("3. ACTIVITIES"), while topics come from a vocabulary
    written to be read.

    Superseded by `build_corpus_digest` for the scope prompt, and kept for the build
    log and as the fallback when no digest has been generated yet — a comma list is a
    poor description but a working one, and the gate must not lose its corpus
    description because a paragraph could not be written.
    """
    seen: List[str] = []
    for record in records:
        for topic in record.topics:
            if topic not in seen:
                seen.append(topic)
    return ", ".join(seen[:limit])


# How much section-summary prose to put in one digest call. Comfortably inside any
# modern context window; the point of the bound is that the reduce below stays
# predictable, not that the model could not take more.
DIGEST_INPUT_BUDGET = 12_000


def sections_fingerprint(records: Sequence[SectionRecord]) -> str:
    """Identity of the section SET, so a stale digest is detectable rather than assumed.

    Over (chunk_id, content_sha256) sorted: adding, removing or editing any section
    changes it, while re-running an unchanged corpus does not. Reordering does not
    either — the digest describes a corpus, not a sequence.
    """
    parts = sorted(
        f"{record.chunk_id}:{record.content_sha256}"
        for record in records
        if getattr(record, "chunk_id", "")
    )
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _digest_batches(
    records: Sequence[SectionRecord],
    budget: int,
    min_items: int = 1,
) -> List[List[SectionRecord]]:
    """Group sections so each batch's prose fits one call.

    `min_items` exists for the reduce rounds. A batch holding one piece summarises that
    piece into another piece, which is not a reduction — with a budget too small to fit
    two partials, the loop below would produce the same number of pieces every round
    and only ever terminate by hitting its round cap. Requiring two per batch during a
    reduce makes every round at least halve the count, so it converges by construction.
    Overshooting the budget on a batch of two is the lesser problem: the budget is a
    guideline about prompt size, while non-convergence spends a model call per section
    per round.
    """
    batches: List[List[SectionRecord]] = []
    current: List[SectionRecord] = []
    size = 0
    for record in records:
        cost = len(record.summary or "") + 2
        if len(current) >= min_items and size + cost > budget:
            batches.append(current)
            current, size = [], 0
        current.append(record)
        size += cost
    if current:
        batches.append(current)
    return batches


def build_corpus_digest(
    records: Sequence[SectionRecord],
    *,
    invoke,
    persona: str = "",
    languages: str = "",
    budget: int = DIGEST_INPUT_BUDGET,
) -> str:
    """A paragraph describing what the whole corpus covers. `invoke(prompt)` returns text.

    **Every section, not a sample.** The description the scope model reads decides
    whether a user gets an answer at all, so a corpus whose later half went undescribed
    would refuse questions about it while looking perfectly healthy. When the summaries
    do not fit one call they are reduced in batches and the partial paragraphs are
    reduced again — as many rounds as the corpus needs. Nothing is truncated away.

    Returns "" on failure, which the caller treats as "keep the previous digest": a
    corpus description that is one ingest out of date is a far smaller problem than no
    description at all.
    """
    usable = [record for record in records if (record.summary or "").strip()]
    if not usable:
        return ""

    pieces = [record.summary.strip() for record in usable]
    round_number = 0
    while True:
        batches = _digest_batches(
            [SectionRecord(chunk_id=str(i), summary=piece) for i, piece in enumerate(pieces)],
            budget,
            # The first pass groups whatever fits; a lone oversized section summary
            # still has to be sent. Every later pass is a reduce and must shrink.
            min_items=1 if round_number == 0 else 2,
        )
        outputs: List[str] = []
        for batch in batches:
            prompt = render(
                "rag/corpus_digest.j2",
                summaries=[record.summary for record in batch],
                persona=persona,
                languages=languages,
                # A partial round is still describing part of a corpus, and saying so
                # stops the model writing "this document covers..." about a fragment.
                partial=len(batches) > 1,
            )
            try:
                text = str(invoke(prompt) or "").strip()
            except Exception:
                logger.warning("corpus digest call failed", exc_info=True)
                text = ""
            if text:
                outputs.append(text)

        if not outputs:
            return ""
        if len(outputs) == 1:
            return " ".join(outputs[0].split())

        # More than one partial: reduce again. Every reduce round batches at least two
        # pieces, so the count at least halves and this terminates in log2(sections)
        # rounds. The cap is a backstop against a pathological input, not the mechanism.
        round_number += 1
        if round_number > 10:
            logger.warning("corpus digest did not converge; joining %d partials", len(outputs))
            return " ".join(" ".join(outputs).split())
        pieces = outputs
