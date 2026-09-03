"""Folding two spellings of one Arabic name onto one key, for matching only.

A registrar types `فاطمه` looking for `فاطمة`, and `احمد` looking for `أحمد`. Both are
the same name. Arabic writes several of its letters more than one way, and which way a
name reaches this service is decided by whoever typed the spreadsheet — not by the
child. A search that compares the raw strings answers "no such child" to a name that is
plainly on the register, which is the failure this module exists to remove.

**This output is a lookup key and never a display value.** It is deliberately lossy: it
throws away exactly the distinctions that make two spellings of one name compare
unequal, and a screen that showed it would be showing a misspelling. Nothing here is
stored, and nothing here is written back to a name column.

**Both sides must be folded with the same table.** Fold the query alone and Arabic
search does not degrade — it stops working, because the stored term and the typed term
become different strings. `SEARCH_FOLDING` is therefore the source of both halves: this
module folds the query, and `people_repository` builds the SQL that folds the column
from the same tuple, so the two cannot drift apart.

**Names are folded, never stemmed.** A stemmer is symmetric and would still match, but
it would also match a sibling — `أميرة` stems onto `أمير`, a different person in the
same family — and selecting the wrong one of a parent's own children is worse than not
finding her. Orthography is folded; morphology is left alone.

The Latin half of a name is untouched: none of these characters appears in it, so
folding an English name is a no-op, and case is handled by `ILIKE` as it always was.

Nothing here reads a clock, a database or the environment.
"""
from typing import Final

# Every pair is (as it may be written, as it is matched). Written with the codepoint in
# the comment because a literal Arabic mark is a literal no reviewer can see — several
# of these are zero-width on screen, and one silently deleted by an editor that strips
# "stray" bytes would break matching with nothing visible in the diff.
#
# The letters first. Each of these is one letter an Egyptian keyboard offers in several
# forms, and a school roster carries all of them:
SEARCH_FOLDING: Final[tuple[tuple[str, str], ...]] = (
    ("أ", "ا"),  # أ  alef with hamza above  -> ا   أحمد / احمد
    ("إ", "ا"),  # إ  alef with hamza below  -> ا   إيمان / ايمان
    ("آ", "ا"),  # آ  alef with madda        -> ا   آية / اية
    ("ٱ", "ا"),  # ٱ  alef wasla             -> ا
    ("ة", "ه"),  # ة  teh marbuta            -> ه   فاطمة / فاطمه
    ("ى", "ي"),  # ى  alef maksura           -> ي   ليلى / ليلي
    ("ؤ", "و"),  # ؤ  waw with hamza         -> و   رؤوف / روءوف
    ("ئ", "ي"),  # ئ  yeh with hamza         -> ي   فائز / فايز
    # Then the marks. None of them is part of a name's identity; all of them survive a
    # copy-paste out of a document that was typeset rather than typed.
    ("ـ", ""),  # ـ  tatweel: decorative elongation, no meaning
    ("ً", ""),  # ً  fathatan
    ("ٌ", ""),  # ٌ  dammatan
    ("ٍ", ""),  # ٍ  kasratan
    ("َ", ""),  # َ  fatha
    ("ُ", ""),  # ُ  damma
    ("ِ", ""),  # ِ  kasra
    ("ّ", ""),  # ّ  shadda
    ("ْ", ""),  # ْ  sukun
    ("ٰ", ""),  # ٰ  superscript alef
)


def fold_for_search(text: str) -> str:
    """One spelling of a name, as the key every spelling of it folds onto.

    For matching only — see the module docstring. Latin text passes through unchanged.
    """
    for written, matched in SEARCH_FOLDING:
        text = text.replace(written, matched)
    return text
