"""Scope detection: the two rungs that decide whether a question is this corpus's subject.

Rung 1 scores the query against the catalogue of questions the corpus can answer. It
may ADMIT — nothing else in the system can say "yes, proceed" as cheaply — but it may
never refuse, because a low score has too many innocent explanations: a phrasing the
catalogue did not anticipate, a language it was not written in, vocabulary the corpus
spells differently.

Rung 2 is a model, and it runs only on the questions rung 1 could not admit. It is
shown the closest catalogued questions and their scores, so it judges evidence rather
than guessing from a topic list. That matters because it is the only component allowed
to end a turn, and its false negative is the one unrecoverable failure in this design:
a valid question refused outright.

Two rules are enforced in code rather than in the prompt, because a prompt is a request
and this needs to be a guarantee:

  * only rung 2 may produce a HIGH-certainty rejection
  * disclosing a personal detail overrides a rejection — someone answering "he is 9" is
    continuing a conversation, not changing the subject
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from backend.chat.signals import RequestSignals, Scope
from backend.llm import sampling
from backend.rag.evidence import Certainty
from backend.rag.scope_index import ScopeIndex, ScopeMatch, build_index
from backend.text_normalization import normalize_query

logger = logging.getLogger(__name__)


class ScopeIndexStore:
    """Builds the scope index once per process and holds it.

    A failed build is cached as failed: the catalogue is missing or the database is
    down, and retrying on every request would turn one outage into a per-turn cost for
    a feature that is meant to be free.
    """

    def __init__(self, builder=None):
        self._builder = builder
        self._index: Optional[ScopeIndex] = None
        self._lock = threading.Lock()
        self._failed = False

    def get(self) -> ScopeIndex:
        if self._index is not None:
            return self._index
        with self._lock:
            if self._index is not None:
                return self._index
            if self._failed or self._builder is None:
                return ScopeIndex()
            try:
                self._index = self._builder() or ScopeIndex()
            except Exception:
                logger.warning("scope index build failed; scope checking disabled", exc_info=True)
                self._failed = True
                return ScopeIndex()
        return self._index

    def invalidate(self) -> None:
        """After a re-index. The corpus changed, so the catalogue and its derived floor
        both have to be rebuilt — the floor is a statistic of the corpus, not a
        constant."""
        with self._lock:
            self._index = None
            self._failed = False


def _default_builder() -> ScopeIndex:
    import os

    from backend.indexing.embedding import embedding_service
    from backend.indexing.section_summary import corpus_catalogue
    from backend.indexing.summary_store import load_digest, load_records
    from backend.profiles import get_profile

    profile = get_profile()
    records = [record for record in load_records(profile.name) if record.usable]
    digest = load_digest(profile.name)

    # The paragraph when ingest wrote one, the topic list when it did not. The fallback
    # is deliberately not an error: a deployment that has not re-run the catalogue build
    # since this shipped still needs a working gate, and a comma list is a worse
    # description rather than no description.
    catalogue = digest.paragraph or corpus_catalogue(records)
    if not digest.paragraph:
        logger.info("no corpus digest stored; describing the corpus by topic list")

    return build_index(
        records,
        embed=embedding_service.get_embeddings,
        floor_percentile=float(getattr(profile.rag, "scope_floor_percentile", 10.0)),
        catalogue=catalogue,
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
        cached_floor=(digest.floor_sha256, digest.floor) if digest.floor_sha256 else None,
    )


index_store = ScopeIndexStore(builder=_default_builder)


class CatalogueScopeDetector:
    """Rung 1. Free, and permitted only to admit.

    MEDIUM on admission because a strong question-to-question match is real evidence.
    LOW on a miss, which by the ladder's rules cannot end the climb — the next rung has
    to look before anything is refused.
    """

    name = "scope_catalogue"
    certainty = Certainty.MEDIUM

    def __init__(self, store: Optional[ScopeIndexStore] = None):
        self._store = store or index_store

    def detect(self, ctx, signals: RequestSignals) -> Optional[RequestSignals]:
        if not getattr(ctx.config, "scope_index_enabled", False):
            return None
        index = self._store.get()
        if not index.ready:
            return None

        # `text_to_score` is the resolved question when a resolver produced one. That is
        # the whole difference between measuring this turn and measuring an average of
        # it and the one before: "and what is the fees for this years", concatenated
        # onto a question about uniforms, landed nearest the catalogue's UNIFORM
        # entries, and those were then offered to the user as the directions to choose
        # between for a question about fees.
        vector = _embed(ctx.text_to_score)
        if vector is None:
            return None

        limit = max(1, int(getattr(ctx.config, "scope_match_count", 3)))
        matches = index.best_matches(vector, limit)
        if not matches:
            return None

        signals.scope_matches = matches
        signals.candidate_sections = _sections_of(matches)
        signals.scope_options = distinct_directions(
            matches,
            index.floor,
            duplicate_similarity=float(
                getattr(ctx.config, "scope_option_duplicate_similarity", 0.92)
            ),
            max_score_gap=float(getattr(ctx.config, "scope_option_max_score_gap", 0.06)),
        )
        top = matches[0]

        if top.score >= index.floor:
            signals.scope = Scope.IN_DOMAIN
            signals.scope_certainty = Certainty.MEDIUM
            signals.reasons.append(
                f"matches {top.question!r} at {top.score:.2f} (floor {index.floor:.2f})"
            )
        else:
            signals.scope = Scope.OUT_OF_DOMAIN
            signals.scope_certainty = Certainty.LOW
            signals.reasons.append(
                f"best catalogue match {top.score:.2f} below floor {index.floor:.2f}"
            )
        return signals


class ScopeModelDetector:
    """Rung 2. The only component that may refuse a turn.

    Shown rung 1's matches and their scores. A model asked "is this in scope?" with
    nothing but a topic list has to guess; the same model shown the four nearest
    questions the corpus can answer is checking, and checking is what this decision
    needs given that its mistakes are unrecoverable.
    """

    name = "scope_model"
    certainty = Certainty.HIGH

    def __init__(self, invoke=None):
        self._invoke = invoke

    def detect(self, ctx, signals: RequestSignals) -> Optional[RequestSignals]:
        if not getattr(ctx.config, "request_envelope_enabled", False):
            return None
        invoke = self._invoke or _default_scope_invoke
        result = invoke(ctx, signals)
        if not isinstance(result, dict):
            return None

        signals.personal_data = [str(item) for item in (result.get("personal_data") or [])]
        verdict = str(result.get("scope") or "").strip().lower()
        reason = str(result.get("reason") or "").strip()

        if verdict == Scope.OUT_OF_DOMAIN.value:
            signals.scope = Scope.OUT_OF_DOMAIN
            signals.scope_certainty = Certainty.HIGH
            signals.reasons.append(reason or "scope model: different subject")
        elif verdict == Scope.IN_DOMAIN.value:
            signals.scope = Scope.IN_DOMAIN
            signals.scope_certainty = Certainty.HIGH
            signals.reasons.append(reason or "scope model: this corpus's subject")
        else:
            signals.reasons.append("scope model returned no usable verdict")
            return signals

        # Enforced here, not asked for in the prompt: a prompt is a request, and this
        # has to hold every time. Supplying a detail the assistant asked for is a
        # continuation, and refusing it would strand the user mid-conversation.
        if signals.personal_data and signals.scope is Scope.OUT_OF_DOMAIN:
            signals.scope = Scope.IN_DOMAIN
            signals.reasons.append("overridden: message discloses personal details")
        return signals


def _default_scope_invoke(ctx, signals: RequestSignals) -> Optional[Dict[str, Any]]:
    """One structured call, prompted with rung 1's evidence."""
    import os

    from langchain.chat_models import init_chat_model
    from pydantic import BaseModel, Field
    from typing import List as _List, Literal as _Literal

    from backend.assets.vision import call_with_rate_limit_retry, invoke_structured
    from backend.prompts import render
    from backend.profiles import get_profile

    profile = get_profile()
    personal_fields = list(getattr(ctx.config, "personal_data_fields", None) or [])

    class ScopeVerdict(BaseModel):
        scope: _Literal["in_domain", "out_of_domain"] = Field(
            description="Whether the message is this knowledge base's subject"
        )
        reason: str = Field(default="", description="One short sentence")
        personal_data: _List[str] = Field(default_factory=list)

    index = index_store.get()
    prompt = render(
        "rag/scope_check.j2",
        # The resolved question, so the model judges the same words rung 1 scored.
        # Showing it the raw "and what about those?" while telling it those scores
        # describe something else is how a rung that reads meaning gets misled by the
        # rung below it.
        question=ctx.text_to_score,
        matches=[
            {"question": m.question, "score": m.score, "above_floor": m.score >= index.floor}
            for m in (signals.scope_matches or [])
        ],
        catalogue=index.catalogue,
        persona=profile.identity.persona,
        history=_history_text(ctx),
        personal_fields=personal_fields,
        # A similarity score means nothing without the scale it lives on. Handing over
        # a bare 0.31 invites the model to read it as "no match" when this corpus may
        # never score higher; the floor is the corpus's own calibration and is the only
        # thing that makes the number interpretable.
        floor=index.floor,
        # Distinguishes "the search found nothing like this" from "the search did not
        # run". They look identical in an empty match list and mean opposite things.
        index_ready=index.ready,
    )
    model = init_chat_model(
        model=getattr(ctx.config, "scope_summary_model", "") or os.getenv("FAST_MODEL"),
        model_provider="openai",
        api_key=os.getenv("ARK_API_KEY"),
        base_url=os.getenv("BASE_URL"),
        **sampling("scope"),
    )
    # Same quota as everything else in the turn, so the same treatment: a 429 here
    # would make this rung abstain, and abstaining leaves the tentative rejection from
    # the rung below — which cannot end a turn, so the question proceeds. Safe, but it
    # spends the search this rung existed to avoid.
    class _Retry:
        vision_retry_attempts = int(getattr(ctx.config, "model_retry_attempts", 3))
        vision_retry_base_seconds = float(getattr(ctx.config, "model_retry_base_seconds", 5.0))
        vision_retry_max_seconds = float(getattr(ctx.config, "model_retry_max_seconds", 60.0))

    result = call_with_rate_limit_retry(
        lambda: invoke_structured(model, ScopeVerdict, [{"role": "user", "content": prompt}]),
        config=_Retry(),
        description="scope check",
    )
    return result if isinstance(result, dict) else result.model_dump()


def _embed(text: str) -> Optional[Sequence[float]]:
    from backend.indexing.embedding import embed_query

    normalized = normalize_query(text) or text
    try:
        return embed_query(normalized)
    except Exception:
        logger.warning("scope: could not embed the question, abstaining", exc_info=True)
        return None


def _history_text(ctx, limit: int = 4) -> str:
    """Recent dialogue, both sides.

    It used to be the previous USER line alone, which cannot resolve a follow-up
    pointing at something only the assistant said — "the fees you mentioned" refers to
    an assistant turn, and a history without one leaves the model guessing.
    """
    from backend.chat.resolution import conversation_text

    return conversation_text(getattr(ctx, "history", ()) or (), limit=limit)


def _sections_of(matches: Sequence[ScopeMatch]) -> List[str]:
    sections: List[str] = []
    for match in matches:
        if match.chunk_id and match.chunk_id not in sections:
            sections.append(match.chunk_id)
    return sections


def distinct_directions(
    matches: Sequence[ScopeMatch],
    floor: float,
    *,
    duplicate_similarity: float = 0.92,
    max_score_gap: float = 0.06,
    max_directions: int = 2,
) -> List[str]:
    """The catalogued questions worth offering the user as a choice, best first.

    Matching is recall, and recall should be generous. Offering is a question put to a
    person, and every option that survives to here makes the interruption more likely —
    the once-per-question ask is unlocked by having two or more directions, so a list
    that is long for the wrong reason is not a cosmetic problem. Four conditions:

    **At or above the floor.** The corpus's own derived statistic for "this resembles
    something we can answer", so a direction is only offered when the corpus vouches for
    it, and it re-derives on every re-index like everything else that reads it.

    **Not a paraphrase of a direction already offered.** Compared vector-to-vector
    between the matches themselves, not against the query. This is the condition the
    shipped system was missing, and it was visible to users: "What subjects are taught
    IN Years 3 to 6?" and "What subjects are taught FOR Years 3 to 6?" were presented
    as two things to choose between, and being two was itself what authorised the ask.
    The catalogue summariser does collapse near-duplicates WITHIN a section, which is
    why this was believed to be unnecessary — but nothing collapses them ACROSS
    sections, and two sections covering the same ground is the normal state of a real
    corpus.

    **Not far behind the leader.** A clear winner is not an ambiguity. If the best match
    is more than `max_score_gap` ahead of a direction, the query did pick between them,
    and asking which was meant is a question whose answer is already on the screen.

    **A crowded field is noise, not ambiguity.** Real ambiguity is a small, tight cluster:
    the query landed between two things the corpus treats separately. When many entries
    survive instead, the similarity did not discriminate at all, and a question built from
    it asks the user to pick from a menu the system does not itself understand. Measured
    across fifteen realistic queries against the shipped catalogue, every field of two was
    a coherent choice — "documents for all students" vs "for international students",
    "partner organisations" vs "partnership contact" — and every field of three or more
    was noise. "what time does school end" produced "What are the school hours", "When
    does the school open" and "What are the school timings", which is one question three
    times; "is there a school bus" produced "Does the school have a library" and "Where is
    the school located". Both of those are questions the system should simply answer, and
    both would have interrupted the user to ask which of several near-identical or
    unrelated things they meant. So above `max_directions` the ask is declined and the
    turn proceeds, which is the conservative direction: a missed ask costs a rewrite, a
    bad ask costs the user's confidence.

    Sections are NOT the discriminator, though the first version of this used them —
    one option per section, on the theory that two questions from one section are two
    phrasings of one direction. Measured against the shipped catalogue, that was wrong
    in the case it was written for. "what is partner" matches "Which partner
    organisations are listed?" at 0.586 and "Who is the contact for partnership
    enquiries?" at 0.585, and both live in the same chunk. They are plainly different
    questions; a section is a unit of the document, not a unit of meaning, and a corpus
    chunked coarsely enough keeps every reading of a topic in one of them. Similarity
    between the questions themselves is the discriminator the section proxy was reaching
    for, and it says the right thing in both of these cases.
    """
    ranked = sorted(
        (match for match in matches if (match.question or "").strip() and match.score >= floor),
        key=lambda match: match.score,
        reverse=True,
    )
    if not ranked:
        return []

    best = ranked[0].score
    options: List[str] = []
    # Unit vectors of the options already accepted, so the duplicate test is one dot
    # product per pair. Normalising here rather than inside a cosine helper is what
    # keeps this linear in matches rather than quadratic in normalisations.
    kept: List[Any] = []
    seen: set = set()
    for match in ranked:
        question = match.question.strip()
        if question in seen:
            continue
        if max_score_gap > 0 and (best - match.score) > max_score_gap:
            continue
        unit = _unit_vector(match.vector)
        if unit is not None and any(
            float(unit @ other) >= duplicate_similarity for other in kept
        ):
            continue
        seen.add(question)
        if unit is not None:
            kept.append(unit)
        options.append(question)
        if len(options) > max_directions:
            # Counted AFTER the filters above, so this measures how many genuinely
            # distinct directions survived rather than how many matches were requested.
            # Returning nothing rather than a truncated list: the overflow is evidence
            # that the scores did not separate, which disqualifies the whole field, not
            # just the entries past the limit.
            return []
    return options


def _unit_vector(vector):
    """A catalogued question's vector, normalised, or None when it cannot be used.

    None means "cannot be compared", and every caller reads that as "not a duplicate".
    That is the safe direction: an index built before vectors were carried on a match
    offers the list it always did rather than silently collapsing options it never
    actually compared.
    """
    if vector is None:
        return None
    try:
        array = np.asarray(vector, dtype=np.float32).ravel()
        if not array.size:
            return None
        norm = float(np.linalg.norm(array))
        return array / norm if norm > 0.0 else None
    except Exception:  # pragma: no cover - a comparison must never break a turn
        logger.debug("could not normalise a scope option vector", exc_info=True)
        return None
