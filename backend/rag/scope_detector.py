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

from backend.chat.signals import RequestSignals, Scope
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
    from backend.indexing.embedding import embedding_service
    from backend.indexing.section_summary import corpus_catalogue
    from backend.indexing.summary_store import load_records
    from backend.profiles import get_profile

    profile = get_profile()
    records = [record for record in load_records(profile.name) if record.usable]
    return build_index(
        records,
        embed=embedding_service.get_embeddings,
        floor_percentile=float(getattr(profile.rag, "scope_floor_percentile", 10.0)),
        catalogue=corpus_catalogue(records),
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

        vector = _embed(_text_to_score(ctx))
        if vector is None:
            return None

        limit = max(1, int(getattr(ctx.config, "scope_match_count", 3)))
        matches = index.best_matches(vector, limit)
        if not matches:
            return None

        signals.scope_matches = matches
        signals.candidate_sections = _sections_of(matches)
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

    from backend.assets.vision import invoke_structured
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

    prompt = render(
        "rag/scope_check.j2",
        question=ctx.question,
        matches=[{"question": m.question, "score": m.score} for m in (signals.scope_matches or [])],
        catalogue=index_store.get().catalogue,
        persona=profile.identity.persona,
        history=_history_text(ctx),
        personal_fields=personal_fields,
    )
    model = init_chat_model(
        model=getattr(ctx.config, "scope_summary_model", "") or os.getenv("FAST_MODEL"),
        model_provider="openai",
        api_key=os.getenv("ARK_API_KEY"),
        base_url=os.getenv("BASE_URL"),
        temperature=0.0,
    )
    result = invoke_structured(model, ScopeVerdict, [{"role": "user", "content": prompt}])
    return result if isinstance(result, dict) else result.model_dump()


def _embed(text: str) -> Optional[Sequence[float]]:
    from backend.indexing.embedding import embed_query

    normalized = normalize_query(text) or text
    try:
        return embed_query(normalized)
    except Exception:
        logger.warning("scope: could not embed the question, abstaining", exc_info=True)
        return None


def _text_to_score(ctx) -> str:
    """The message, prefixed by the previous user turn when there is one.

    A bare follow-up carries its subject in the conversation, not in its own words, and
    measuring nothing looks identical to being off-topic.
    """
    from backend.chat.signals import _last_user_text

    if not getattr(ctx, "has_history", False):
        return ctx.question
    previous = _last_user_text(ctx.history)
    return f"{previous}\n{ctx.question}" if previous else ctx.question


def _history_text(ctx, limit: int = 4) -> str:
    from backend.chat.signals import _last_user_text

    previous = _last_user_text(getattr(ctx, "history", ()) or ())
    return f"User previously said: {previous}" if previous else ""


def _sections_of(matches: Sequence[ScopeMatch]) -> List[str]:
    sections: List[str] = []
    for match in matches:
        if match.chunk_id and match.chunk_id not in sections:
            sections.append(match.chunk_id)
    return sections
