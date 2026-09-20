"""The same school questions, asked in English instead of Egyptian Arabic.

The Arabic set (`school_dataset`) measures the cross-language path end to end: an Arabic
question against an English corpus. This one holds everything else fixed — same cases,
same ids, same gold spans, same splits — and changes only the language the question is
asked in.

The pair exists to answer one question that the Arabic set alone cannot: **is the sparse
half of hybrid retrieval contributing anything cross-language?** BM25 matches terms, and
an Arabic question shares almost no terms with an English chunk beyond what the parent
happens to type in English ("Year 3", "FS1", "CAT4"). If that is what is costing recall,
then asking the same question in the corpus's own language should recover it, and the
size of the difference is the value of either translating the query at resolution time or
indexing an Arabic twin of the corpus.

It is a MEASUREMENT artifact and not a product surface. Nothing in the product asks
questions in English; parents write Arabic.

The English is a translation of the Arabic by the deployment's own fast model, made once
and committed (`data/questions_en.json`) so a score is reproducible rather than depending
on a model call at eval time. Numbers, year-group codes, dates, emails and URLs were held
fixed in the translation, and the result is normalised to ASCII punctuation — an early
pass came back with "Pre-K" spelled using a non-breaking hyphen, which BM25 would have
scored as a different token and would have measured the translator rather than the
retrieval.

Gold spans are NOT translated. They are English either way, because they are quotations
from an English corpus, so both datasets are scored against exactly the same evidence.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import List

from tests.evals.school_dataset import (  # noqa: F401 — re-exported so the two read alike
    CORPUS_FILENAME,
    Case,
    missing,
    satisfied,
)
from tests.evals.school_dataset import CASES as _ARABIC_CASES
from tests.evals.school_dataset import DATASET_VERSION as _ARABIC_VERSION

#: Tied to the Arabic set's version: these are the same cases in another language, so a
#: score is only comparable against an English run of the same underlying dataset.
DATASET_VERSION = _ARABIC_VERSION

#: Beside this module rather than under a `data/` directory: `data/` is gitignored
#: for runtime uploads and assets, and this is a committed fixture — a score has to
#: be reproducible from a clean checkout.
_TRANSLATIONS = Path(__file__).with_name("school_questions_en.json")


def _english_questions() -> dict:
    if not _TRANSLATIONS.exists():
        raise FileNotFoundError(
            f"{_TRANSLATIONS} is missing. It is generated once from the Arabic set and "
            f"committed; regenerate it rather than translating at eval time, or a score "
            f"stops being reproducible."
        )
    return json.loads(_TRANSLATIONS.read_text(encoding="utf-8"))


def _build() -> List[Case]:
    english = _english_questions()
    untranslated = [case.id for case in _ARABIC_CASES if not english.get(case.id, "").strip()]
    if untranslated:
        raise ValueError(
            f"{len(untranslated)} case(s) have no English question: {untranslated[:5]}. "
            f"A case silently keeping its Arabic would be scored as an English run and "
            f"quietly flatter whichever side it favours."
        )
    return [
        replace(case, question=english[case.id], context=english.get(f"{case.id}::context", case.context))
        for case in _ARABIC_CASES
    ]


CASES: List[Case] = _build()


def cases(split: str = "dev") -> List[Case]:
    """`dev` (default), `holdout`, or `all` — the same split as the Arabic set."""
    if split == "all":
        return list(CASES)
    if split == "holdout":
        return [c for c in CASES if c.holdout]
    if split == "dev":
        return [c for c in CASES if not c.holdout]
    raise ValueError(f"unknown split {split!r}; use dev, holdout, or all")
