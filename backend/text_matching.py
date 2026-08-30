"""Text prepared for MATCHING — never for display, and never for a model.

The sibling of `backend/text_normalization.py`, and the distinction between them is the
whole point of this module existing separately.

`text_normalization.sanitize_text` is LOSSLESS in the ways a reader would notice: it
repairs PDF damage (presentation forms, tatweel, invisible bidi marks) but deliberately
preserves diacritics, hamza forms and ZWNJ, because its output is what a user reads and
what the model composes an answer from. `tests/general/test_arabic_and_adversarial.py`
asserts that preservation.

This module is LOSSY on purpose. It folds away exactly the distinctions that make two
spellings of the same word compare unequal — and it may only ever be applied to text
that is used as a lookup key. Two rules follow, and breaking either one is a defect:

  1. **Never feed this to the model, and never store it as `text`.** Folded Arabic is
     out-of-distribution for an LLM: it has lost the orthography the model was trained
     on, and asking it to compose fluent Arabic from mangled input maximises the chance
     it invents the surface form of a quote, an amount or a date.
  2. **Never feed this to the dense embedder.** bge-m3 handles Arabic morphology
     natively; folding and stemming before embedding throws away signal it would have
     used. The dense lane gets `normalize_query`; only the sparse lane gets this.

## Why two keys and not one

`name_key` folds. `search_key` folds AND light-stems. They are different because a
proper noun must not be stemmed:

    ليلى  -> fold -> ليلي  -> stem -> ليل      (a different name)
    أميرة -> fold -> اميره -> stem -> امير     (collides with the male name أمير)

Stemming is symmetric, so a stemmed roster would still match a stemmed query — but it
would ALSO match a sibling, and selecting the wrong one of a parent's own children is
the failure `child_resolution` is written to avoid. Names get folding only.

## Why English is not stemmed here

It already is, one layer down. `milvus_client.build_analyzer_params` runs lowercase +
`_english_` stop words + a Porter stemmer over `bm25_text` and over the query, server
side, and that chain is tuned (see `_QUESTION_STOP_WORDS`). Stemming English again here
would be redundant work on a path that already behaves. So `search_key` rewrites Arabic
tokens and leaves Latin ones exactly as they are, for the analyzer to handle.

## Symmetry is not optional

Whatever this module does to `bm25_text` at index time it must do to the sparse query at
retrieval time. Fold one side only and Arabic retrieval does not degrade — it stops
working, because the indexed term and the query term become different strings. That is
the same argument `text_normalization`'s docstring makes, and it is why both callers
import from one place rather than each doing "roughly the same thing".

## Adopting this needs a REBUILD, not a redeploy

Two things changed underneath an existing collection, and neither reaches one that is
already there:

  - `bm25_text` now stores folded, stemmed text. Chunks written before this still hold
    raw text, and the query is folded, so those chunks stop matching on the sparse half.
  - the analyzer's stop list gained the Arabic entries, and analyzer params are fixed at
    collection creation. `MilvusStore.ensure_collection` returns early when the
    collection exists, so it will NOT pick them up.

So: `MilvusStore.drop_collection()` and re-ingest. Until that happens the dense half
carries retrieval on its own, which is what it was already doing for Arabic — nothing
gets worse in the meantime, it just does not get better.
"""
from __future__ import annotations

import re
from functools import lru_cache

import snowballstemmer
from camel_tools.utils.dediac import dediac_ar
from camel_tools.utils.normalize import (
    normalize_alef_ar,
    normalize_alef_maksura_ar,
    normalize_teh_marbuta_ar,
    normalize_unicode,
)

from backend.text_normalization import sanitize_text

# Arabic, Arabic Supplement and Arabic Extended-A. Presentation forms are absent on
# purpose: `sanitize_text` has already mapped them back to standard letters, so a
# character still in that range here would be a bug upstream, not a token to stem.
_ARABIC_RANGES = "؀-ۿݐ-ݿࢠ-ࣿ"
_HAS_ARABIC = re.compile(f"[{_ARABIC_RANGES}]")

# Word tokens. Rewriting via `sub` rather than splitting and re-joining leaves the
# separators the caller built between tokens (the " > " in `_apply_bm25_section_prefix`
# is structured text, not a bag of words) instead of flattening them away.
_TOKEN = re.compile(r"\w+", re.UNICODE)

# The definite article, alone or behind a conjunction or preposition. Stripped HERE
# rather than left to Snowball, and that is a correctness fix rather than a tidy-up.
#
# Snowball's own prefix handling is length-sensitive, so whether it strips a suffix
# depends on whether the article was there when it started:
#
#     الدرجات  -> درج      but   درجات  -> درجا
#     الاجازات -> اجاز     but   اجازات -> اجازا
#
# Those are the SAME word, and a corpus writing "الدرجات" would not match a parent
# typing "درجات". Removing the article first means the stemmer always sees the same
# base, so the term a word produces no longer depends on how it was introduced.
#
# Only the article forms are stripped, never a bare و/ب/ل — "ولد" must not become "لد".
# Snowball still handles a bare conjunction itself.
_DEFINITE_ARTICLE = re.compile("^(?:وال|فال|بال|كال|لل|ال)")

# Below this many letters, removing the article would leave nothing to match on, so the
# word is stemmed as it stands. Two rather than three because "الزي" (the uniform) is a
# real question word and leaves exactly two.
_MIN_LETTERS_AFTER_ARTICLE = 2

# The sound feminine plural. Removed here for the same reason as the article: Snowball
# takes the ت off but leaves the ا, so the singular and plural of one noun end up as two
# terms that never meet —
#
#     اجازة -> اجاز    but    اجازات -> اجازا
#     درجة  -> درج     but    درجات  -> درجا
#
# A parent asks about "الاجازة" and the calendar is written about "الاجازات"; on the
# sparse half those were simply different words. Stripping the suffix before stemming
# puts them back together.
_FEMININE_PLURAL = re.compile("ات$")

# Kept at three so that stripping cannot reduce a short word to a fragment: بنات (girls)
# would otherwise become بن, which matches far too much.
_MIN_LETTERS_AFTER_PLURAL = 3

# Arabic function and interrogative words, for the BM25 stop list. The direct
# counterpart of `_QUESTION_STOP_WORDS` in milvus_client, and it exists for the same
# measured reason: in a Q&A corpus these open nearly every question while saying nothing
# about which chunk answers it, and on a corpus this small their IDF stays high enough
# to drag unrelated chunks into the candidate set.
#
# Curated rather than taken from a general-purpose stop list (NLTK ships ~750 Arabic
# words) for two reasons. A prose stop list is far too aggressive for questions — it
# removes negation and quantity words that change what was asked — and NLTK's requires a
# runtime `nltk.download`, which is a network call at import time in production.
#
# Written in NATURAL spelling. `arabic_stop_words_for_analyzer()` puts them through
# `search_key` before the analyzer ever sees them, because the analyzer is looking at
# text this module has already folded and stemmed — a list written in surface form would
# simply never match, silently.
#
# One exclusion is deliberate and load-bearing: على is absent. Folding ى->ي makes the
# preposition على and the given name علي the same string, and this deployment's SIS
# carries a child called علي عثمان. Stopping it would delete that child's name from the
# index entirely. The general rule for adding to this list: never add a token that is
# also a plausible Egyptian given name (على, هنا, أمل, نور).
_ARABIC_STOP_WORDS_NATURAL = [
    # interrogatives, MSA and Egyptian
    "ما", "ماذا", "مين", "من", "هل", "كيف", "ازاي", "ايه", "متى", "امتى",
    "اين", "فين", "كم", "كام", "لماذا", "ليه",
    # pronouns, prepositions, particles
    "انا", "انت", "هو", "هي", "احنا", "نحن", "هم", "في", "الى", "عن", "مع",
    "عند", "بين", "بعد", "قبل", "لكن", "او", "ثم", "قد", "لقد",
    "هذا", "هذه", "ذلك", "التي", "الذي", "كل", "بعض",
    # conversational filler that opens a question without narrowing it
    "عايز", "عاوز", "ممكن", "لو", "سمحت", "اريد", "احتاج",
    "يا", "بس", "طيب", "تمام", "شكرا", "رجاء", "فضلك",
]


@lru_cache(maxsize=1)
def arabic_stop_words_for_analyzer() -> list:
    """The stop list in the form the Milvus analyzer actually sees.

    Derived rather than written out, so the two can never drift: the analyzer runs over
    text `search_key` produced, so a stop word must be what `search_key` would produce
    for it. Deduplicated because stemming collapses several entries onto one form.
    """
    return sorted({key for key in (search_key(word) for word in _ARABIC_STOP_WORDS_NATURAL) if key})


# Words whose ة-form is a DIFFERENT NOUN, not a feminine variant of the word beside it.
#
# Snowball strips the feminine ending, which is usually right and usually helps: طالب
# and طالبة are one word about one kind of person, and a corpus that writes one while a
# parent types the other should still match. معلم/معلمة, مدير/مديرة and ناظر/ناظرة are
# the same story.
#
# These are not. مدرسة is a PLACE and مدرس is a PERSON; مكتبة is a library and مكتب an
# office. Letting them share a BM25 term means "مين مدرس الرياضيات؟" scores against
# every chunk that merely says "the school" — which, in a school corpus, is nearly all
# of them. That is the one place in this domain where the feminine ending carries the
# whole meaning, so it is the one place the stemmer is overruled.
#
# Written as the base form AFTER folding and after the article is removed, which is the
# point the check happens. Kept deliberately short: every entry is a word the stemmer
# will now under-merge, so it earns its place by being a distinct noun in THIS corpus,
# not by being a noun that could be one.
_DISTINCT_FEMININE_NOUNS = frozenset({"مدرسه", "مكتبه", "حاسبه"})


@lru_cache(maxsize=1)
def _arabic_stemmer():
    """Snowball's Arabic stemmer, built once.

    Snowball rather than NLTK's ISRI, and that is the load-bearing choice in this
    module. ISRI reduces a word to its triliteral root, and Arabic roots are heavily
    polysemous: it maps طالب (a student) and طلب (an application) onto the same term,
    along with مدرسة/مدرس/تدريس/دراسة. On a school corpus that is not a small loss of
    precision, it is two different questions retrieving each other's answers.

    Snowball's Arabic stemmer is a LIGHT stemmer — it strips clitics and inflectional
    affixes and stops there, so الرسوم/بالرسوم/للرسوم converge on رسوم while طالب and
    طلب stay apart. That is the behaviour `test_text_matching.py` pins.
    """
    return snowballstemmer.stemmer("arabic")


def _map_tokens(text: str, transform) -> str:
    return _TOKEN.sub(lambda match: transform(match.group(0)), text)


def has_arabic(text: str) -> bool:
    """Whether `text` contains any Arabic-script letter."""
    return bool(_HAS_ARABIC.search(text or ""))


def fold(text: str) -> str:
    """Orthographic folding: the spelling differences that are not word differences.

    Runs `sanitize_text` first so that PDF-extracted text is repaired before it is
    folded — the two are a chain, not alternatives.

    The four camel-tools normalizers are the standard Arabic IR set:
      dediac              strip the diacritics that are optional in ordinary writing
      alef                أ إ آ ٱ -> ا
      alef_maksura        ى -> ي
      teh_marbuta         ة -> ه

    Deliberately NOT normalizing ؤ/ئ onto و/ي, which pyarabic's `normalize_hamza` would
    do. It buys مسئول/مسؤول at the cost of collapsing رئيس onto رييس, and camel-tools
    leaving it out reflects the usual practice. The variants actually present in this
    deployment's SIS data are alef, teh marbuta and alef maksura, all covered above.

    Casefolds too, so the one key works for Latin text as well — which is what lets a
    parent writing "Ali" match a roster row carrying an English name.
    """
    text = sanitize_text(text or "")
    if not text:
        return ""
    text = normalize_unicode(text)
    text = dediac_ar(text)
    text = normalize_alef_ar(text)
    text = normalize_alef_maksura_ar(text)
    text = normalize_teh_marbuta_ar(text)
    return " ".join(text.casefold().split())


def name_key(text: str) -> str:
    """A lookup key for a PROPER NOUN — folded, never stemmed.

    Used for children's names and subject names, where the risk is not failing to match
    but matching the wrong one. See the module docstring for why stemming a name is
    unsafe even though it is symmetric.
    """
    return fold(text)


def _stem_token(token: str) -> str:
    if not _HAS_ARABIC.search(token):
        # Latin, digits, mixed identifiers: left for the Milvus analyzer's English
        # chain, which already lowercases, stops and Porter-stems them.
        return token
    bare = _DEFINITE_ARTICLE.sub("", token)
    if len(bare) >= _MIN_LETTERS_AFTER_ARTICLE:
        token = bare
    # Checked after the article is gone, so every spelling of "the school" reaches the
    # protection rather than only the bare one. Before the plural strip too: مدرسات
    # (female teachers) is not this word and must keep going to the stemmer.
    if token in _DISTINCT_FEMININE_NOUNS:
        return token
    singular = _FEMININE_PLURAL.sub("", token)
    if len(singular) >= _MIN_LETTERS_AFTER_PLURAL:
        token = singular
    return _arabic_stemmer().stemWord(token)


def search_key(text: str) -> str:
    """A lookup key for FREE TEXT — folded and light-stemmed. The BM25 surface.

    Applied to `bm25_text` on the way into Milvus and to the sparse half of the query on
    the way out, and it must be both or neither.

    Stop words are NOT removed here. They are removed by the analyzer, which applies one
    list to the indexed text and the query alike; doing it in two places would make the
    two sides drift the first time somebody edited one list.
    """
    if not text:
        return ""
    return _map_tokens(fold(text), _stem_token)


__all__ = [
    "arabic_stop_words_for_analyzer",
    "fold",
    "has_arabic",
    "name_key",
    "search_key",
]
