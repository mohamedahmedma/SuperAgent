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
    """
    seen: List[str] = []
    for record in records:
        for topic in record.topics:
            if topic not in seen:
                seen.append(topic)
    return ", ".join(seen[:limit])
