"""The question in the corpus's language, for searching only.

Hybrid retrieval fuses a dense list and a sparse one. The sparse half is BM25, which
matches TERMS — so when a parent writes Arabic and the corpus is English, it contributes
almost nothing: the only tokens the two sides share are the ones the parent happened to
type in Latin script ("Year 3", "FS1", "CAT4"). RRF then fuses one useful ranking with
one that is close to noise, and the dense half carries the turn alone.

Measured on the school set, 349 questions, same index and same code, changing only the
language the question is asked in:

    stage          Arabic question    English question
    recalled       337                343
    ranked, k=8    320                338
    ranked, k=4    273                314

The `recalled -> ranked` loss falls from 17 questions to 5. That is the sparse half
waking up, and it is a larger effect than anything reranking reached on this corpus —
a cross-encoder over the same pool LOST 25 questions and cost five seconds a turn.

## What this does NOT change

Only the text handed to the retriever. The question the grader judges, the question the
answer is written from, and the language the user is answered in are all untouched — the
same distinction `search_query` already draws for the child's name, and for the same
reason: retrieval wants the words the CORPUS uses, and every other stage wants the words
the USER used.

## Gated, and free when it cannot help

A question already in the corpus's language costs nothing: `detect_language` is a script
count, not a model. So an English-speaking deployment, or an English question in an
Arabic-first one, never pays for this at all.

## Failure is abstention

Every failure path returns the query unchanged. A translator that is misconfigured, rate
limited or returning nonsense costs the improvement, never the turn — the same rule
`rewrite_query_once` follows.
"""
from __future__ import annotations

import logging
import re
from threading import Lock
from typing import Dict, Optional, Tuple

from backend.chat.language import detect_language
from backend.profiles import get_profile
from backend.prompts import resolve as resolve_prompt
from backend.text_normalization import normalize_query

logger = logging.getLogger(__name__)

_PROFILE = get_profile()
_RAG = _PROFILE.rag
_RETRIEVAL = _PROFILE.retrieval

#: Typography a model returns that a term index will not match. The translator is asked
#: for plain ASCII and mostly complies; this is what makes it a guarantee rather than a
#: hope. It is not cosmetic — an early run came back with "Pre-K" spelled using a
#: non-breaking hyphen (U+2011), which BM25 scores as a different token entirely, so the
#: measurement would have been of the translator rather than of the retrieval.
_TYPOGRAPHY = {
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
    "−": "-", "‘": "'", "’": "'", "“": '"', "”": '"',
    " ": " ", " ": " ",
}

#: Arabic-Indic and Eastern Arabic-Indic digits. A translator that "localises" 105,000
#: into ١٠٥٬٠٠٠ breaks the term match and the citation at once, so a reply carrying them
#: is rejected rather than repaired: it has rewritten the one thing it was told to keep.
_FOREIGN_DIGITS = re.compile(r"[٠-٩۰-۹]")

_WHITESPACE = re.compile(r"\s+")

#: Things a translation must carry through UNCHANGED, checked in the output rather than
#: asked for in the prompt.
#:
#: The prompt asks; this verifies. A fee row is found by "105,000 EGP" and by nothing
#: else, so a translator that drops, rounds or localises it has removed the only term
#: that identifies the answer — and it fails silently, as a question that suddenly
#: retrieves the wrong year group. A second model asked to check the first would only be
#: another opinion; this is arithmetic.
_URL = re.compile(r"https?://\S+|\b[\w.-]+\.(?:example|com|org|net|edu|eg)\b(?:/\S*)?", re.I)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
#: Trailing punctuation a question mark or comma leaves on a token, in either script.
_EDGE = "؟?.,،:;!\"')("


def protected_tokens(text: str) -> set:
    """Every fragment of `text` a translation is not allowed to alter.

    Anything carrying a digit (amounts, year groups, dates, times, percentages), plus
    emails and URLs whole. Deliberately not "proper nouns": those are a judgement, and a
    rule that cannot be checked mechanically does not belong here.
    """
    found = {match.group(0) for match in _EMAIL.finditer(text or "")}
    found |= {match.group(0) for match in _URL.finditer(text or "")}
    for word in (text or "").split():
        token = word.strip(_EDGE)
        if token and any(char.isdigit() for char in token):
            found.add(token)
    return found

#: Process-local, bounded, and keyed on the normalized query — the same shape as the
#: query-embedding memo. A turn that rewrites searches twice from the same base question,
#: and a parent who asks the same thing twice should not pay twice.
_CACHE_LIMIT = 256
_cache: Dict[str, str] = {}
_cache_lock = Lock()


def _fast_model():
    """The small model that does this. A function so tests can substitute it."""
    from backend.composition import default_services

    return default_services().models.planner()


def _clean(text: str) -> str:
    for bad, good in _TYPOGRAPHY.items():
        text = text.replace(bad, good)
    return _WHITESPACE.sub(" ", text).strip().strip('"').strip()


def _usable(original: str, candidate: str, target: str) -> bool:
    """Whether a translation is safe to search with.

    Rejection is cheap and a bad translation is not: the query is the only thing standing
    between the user and the corpus, so anything suspicious falls back to the original
    rather than searching for something the user did not ask.
    """
    if not candidate:
        return False
    if _FOREIGN_DIGITS.search(candidate):
        logger.warning("translated query localised its digits; searching with the original")
        return False
    lost = [token for token in protected_tokens(original) if token not in candidate]
    if lost:
        logger.warning(
            "translated query dropped %s; searching with the original", sorted(lost)[:4]
        )
        return False
    if detect_language(candidate) != target:
        logger.warning("translated query is not in %s; searching with the original", target)
        return False
    # A translation is a rephrasing, not an essay. Anything several times longer has
    # answered the question, explained it, or hallucinated context onto it, and every
    # extra term it invented is dilution on both halves of the retrieval.
    if len(candidate) > max(120, len(original) * 3):
        logger.warning("translated query is %d chars for a %d-char question; searching with "
                       "the original", len(candidate), len(original))
        return False
    return True


def _cached(key: str) -> Optional[str]:
    with _cache_lock:
        return _cache.get(key)


def _remember(key: str, value: str) -> None:
    with _cache_lock:
        if len(_cache) >= _CACHE_LIMIT:
            _cache.clear()
        _cache[key] = value


def reset_cache() -> None:
    """Forget every memoized translation. For tests."""
    with _cache_lock:
        _cache.clear()


def needs_translation(question: str) -> Tuple[bool, str]:
    """Whether searching for this question needs it translated first.

    Free: a script count, not a model. Exported because the RESOLVER uses the same
    judgement — when a turn needs resolving and translating, one call does both, and when
    it needs neither there is no call at all.
    """
    target = (getattr(_RETRIEVAL, "query_translation_language", "") or "en").strip()
    if not getattr(_RETRIEVAL, "query_translation_enabled", False):
        return False, "disabled"
    text = (question or "").strip()
    if not text:
        return False, "empty question"
    source = detect_language(text)
    if source == target:
        return False, f"already in {target}"
    return True, f"{source} -> {target}"


def verify(original: str, candidate: str) -> bool:
    """Whether a translation of `original` is safe to search with.

    Exported so the merged resolver call validates its `search_text` field by exactly the
    same rules as a standalone translation — a field produced by a bigger prompt is not a
    more trustworthy field.
    """
    target = (getattr(_RETRIEVAL, "query_translation_language", "") or "en").strip()
    return _usable(original, _clean(candidate), target)


def remember_translation(question: str, search_text: str) -> bool:
    """Record a translation produced elsewhere, after checking it.

    The resolver calls this with the `search_text` its own call returned. Retrieval then
    finds it in the memo instead of asking a model again, which is what makes "resolve and
    translate" one call rather than two — without retrieval having to know that the
    resolver exists, or trust it.
    """
    text = (question or "").strip()
    candidate = _clean(search_text or "")
    if not text or not verify(text, candidate):
        return False
    _remember(normalize_query(text) or text, candidate)
    return True


def translate_for_search(query: str, *, invoke=None) -> Tuple[str, dict]:
    """`(text to search with, trace fields)`. The query unchanged whenever in doubt."""
    target = (getattr(_RETRIEVAL, "query_translation_language", "") or "en").strip()
    text = (query or "").strip()
    if not getattr(_RETRIEVAL, "query_translation_enabled", False):
        return query, {"query_translated": False, "query_translation_reason": "disabled"}
    if not text:
        return query, {"query_translated": False, "query_translation_reason": "empty question"}

    wanted, reason = needs_translation(text)
    if not wanted:
        return query, {"query_translated": False, "query_translation_reason": reason}
    source = detect_language(text)

    key = normalize_query(text) or text
    hit = _cached(key)
    if hit is not None:
        return hit, {
            "query_translated": True,
            "query_translation_reason": "memoized",
            "query_search_text": hit,
        }

    try:
        prompt = resolve_prompt(
            getattr(_RAG, "query_translation_prompt", ""),
            "rag/translate_query.j2",
            question=text,
            language=target,
        )
        caller = invoke or (lambda messages: _fast_model().invoke(messages))
        reply = caller([{"role": "user", "content": prompt}])
    except Exception:
        logger.exception("query translation failed; searching with the original")
        return query, {"query_translated": False, "query_translation_reason": "translator failed"}

    candidate = _clean(str(getattr(reply, "content", reply) or ""))
    if not _usable(text, candidate, target):
        return query, {"query_translated": False, "query_translation_reason": "rejected"}

    _remember(key, candidate)
    return candidate, {
        "query_translated": True,
        "query_translation_reason": f"{source} -> {target}",
        "query_search_text": candidate,
    }
