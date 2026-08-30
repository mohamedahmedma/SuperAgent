"""Does this document actually hold the language the upload form said it did?

The bilingual upload form has two columns, and the column an admin drops a file into is
what everything downstream believes. That belief is load-bearing: pairing an Arabic
document with an English one is what lets an Arabic question be answered from the Arabic
half, so a file in the wrong column does not fail — it silently routes every Arabic
question to English text and every English question to Arabic text, and the only symptom
is answers that read oddly to whoever notices first.

A swap is the single most likely mistake with a two-column form, it is invisible once
made, and it is cheap to catch: the text has already been parsed by the time this runs,
so checking it costs a character scan and no model call.

## Why a ratio and not a language identifier

The same reason `backend/chat/language.py` gives: the question is never "which of the
world's languages is this", it is "is this the Arabic side or the English side". Script
counting answers that exactly, and a langid model would be more accurate at a question
nobody is asking.

The thresholds are deliberately wide apart, and the gap between them abstains. Real
documents are not monolingual — an Arabic school policy carries English course codes and
Latin-script proper nouns, and an English one carries Arabic names — so a check that
demanded purity would reject the corpus it exists to protect. Only a document that looks
plainly, overwhelmingly like the OTHER language is rejected.
"""
from __future__ import annotations

from dataclasses import dataclass

from backend.chat.language import ARABIC, ENGLISH, arabic_ratio

#: At or above this share of Arabic letters, the text is Arabic beyond argument.
ARABIC_FLOOR = 0.50
#: At or below this, it holds essentially no Arabic and cannot be the Arabic half.
LATIN_CEILING = 0.10
#: Letters needed before a ratio means anything at all.
#:
#: Without this the check inverts on empty text: `arabic_ratio("")` is 0.0, which is
#: below LATIN_CEILING, so a document that parsed to nothing would be REJECTED as
#: "not Arabic" — the one case this module is supposed to stay out of. A scanned page
#: that yielded no text has a different problem and the upload job reports that one.
MIN_LETTERS = 30


@dataclass(frozen=True)
class LanguageVerdict:
    """Whether a parsed document matches the column it was uploaded into."""

    declared: str
    ratio: float
    #: False only when the text plainly contradicts the column. An inconclusive
    #: document — mixed script, a scanned page that parsed to almost no letters —
    #: is accepted, because the admin said so and nothing here knows better.
    agrees: bool
    reason: str = ""

    @property
    def looks_like(self) -> str:
        """The language the CONTENT suggests, for an error message that helps."""
        if self.ratio >= ARABIC_FLOOR:
            return ARABIC
        if self.ratio <= LATIN_CEILING:
            return ENGLISH
        return ""


def verify(text: str, declared: str) -> LanguageVerdict:
    """Check parsed `text` against the column it arrived in.

    Abstains (agrees=True) on anything inconclusive, including empty text: a document
    that parsed to nothing has a different problem, and the upload job reports that one
    on its own.
    """
    declared = (declared or "").strip()
    ratio = arabic_ratio(text or "")

    if sum(1 for char in (text or "") if char.isalpha()) < MIN_LETTERS:
        return LanguageVerdict(declared, ratio, True, "too little text to judge")

    if declared == ARABIC and ratio <= LATIN_CEILING:
        return LanguageVerdict(
            declared, ratio, False,
            f"uploaded as Arabic but only {ratio:.0%} of its letters are Arabic script",
        )
    if declared == ENGLISH and ratio >= ARABIC_FLOOR:
        return LanguageVerdict(
            declared, ratio, False,
            f"uploaded as English but {ratio:.0%} of its letters are Arabic script",
        )
    return LanguageVerdict(declared, ratio, True)


def describe_mismatch(filename: str, verdict: LanguageVerdict) -> str:
    """The message an admin sees. Names the file, says what was found, and says what to
    do about it — a rejection that only says "language mismatch" invites a retry of the
    same upload."""
    suggestion = ""
    if verdict.looks_like == ARABIC:
        suggestion = " It looks like the Arabic version — try the Arabic column."
    elif verdict.looks_like == ENGLISH:
        suggestion = " It looks like the English version — try the English column."
    return f"{filename}: {verdict.reason}.{suggestion}"
