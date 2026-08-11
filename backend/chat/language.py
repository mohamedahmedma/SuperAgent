"""Which language to answer a turn in.

Only needed where the system speaks for itself — static replies that no model
composes. A model given an Arabic question answers in Arabic without being told; a
canned string cannot, and an English refusal to an Arabic question is a visible
failure in an Arabic-first deployment.

Script counting rather than language identification. It is not trying to tell Arabic
from Persian, and it does not need to: the question is only ever "which of the copies
we actually have should this turn get". A dedicated langid model would be more
accurate at a question nobody is asking, and would cost a load and a forward pass to
answer it.
"""
from __future__ import annotations

from typing import Iterable

# Arabic, Arabic Supplement, Arabic Extended-A, and the presentation-form blocks.
# Presentation forms matter because text pasted out of a PDF arrives in them.
_ARABIC_RANGES = (
    ("؀", "ۿ"),
    ("ݐ", "ݿ"),
    ("ࢠ", "ࣿ"),
    ("ﭐ", "﷿"),
    ("ﹰ", "﻿"),
)

ARABIC = "ar"
ENGLISH = "en"

# Languages this module can name POSITIVELY, for a prompt that wants to say "reply in
# X" rather than "reply in the user's language".
#
# Arabic only, and English's absence is the whole point. `detect_language` is a
# two-way script test: Arabic script or not. `en` is what it returns when it did not
# find Arabic, which is a fallback, not a detection — Chinese and French both come
# back `en`, and the base profile ships Chinese fast-path markers, so that is a real
# case rather than a hypothetical. Telling one of those users to "reply in English"
# would be worse than saying nothing, so anything not in this map keeps the generic
# instruction. Adding a script here means teaching the detector to establish it first.
LANGUAGE_NAMES = {ARABIC: "Arabic"}


def language_name(code: str) -> str:
    """The English name of a positively-detected language, or "" when there is none."""
    return LANGUAGE_NAMES.get(code or "", "")


def _in_ranges(char: str, ranges: Iterable[tuple]) -> bool:
    return any(low <= char <= high for low, high in ranges)


def arabic_ratio(text: str) -> float:
    """Share of the letters in `text` written in Arabic script.

    Digits, punctuation and whitespace are ignored: a phone number or a date says
    nothing about which language someone is writing in, and counting them would drag
    a short Arabic question toward the English side.
    """
    letters = [char for char in (text or "") if char.isalpha()]
    if not letters:
        return 0.0
    arabic = sum(1 for char in letters if _in_ranges(char, _ARABIC_RANGES))
    return arabic / len(letters)


def detect_language(text: str, *, default: str = ENGLISH, threshold: float = 0.2) -> str:
    """`ar` or `en` for a user message.

    The threshold is deliberately low. Arabic questions routinely carry Latin-script
    proper nouns, product names and numbers — "ما هو الـ IB Diploma؟" is an Arabic
    question with a third of its letters in Latin script. Requiring a majority would
    answer it in English. The reverse mistake is far rarer: English sentences do not
    accumulate Arabic letters by accident.
    """
    if not (text or "").strip():
        return default
    return ARABIC if arabic_ratio(text) >= threshold else ENGLISH
