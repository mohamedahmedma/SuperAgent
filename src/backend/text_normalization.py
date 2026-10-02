"""Shared text normalization for indexing AND retrieval.

This lives outside `backend.indexing` on purpose: documents and queries MUST be
normalized identically, otherwise a query can never match the text that was
indexed. Keeping the rules in one place, imported by both layers, is what makes
that symmetry structural rather than a convention someone has to remember.

Arabic notes (this corpus is Arabic-first):
- PDF producers frequently emit Arabic as presentation-form glyphs
  (U+FB50-U+FEFF) instead of standard letters. NFC leaves them alone, so such
  text indexes as a different character sequence than the same words typed
  normally. Targeted NFKC maps them back without NFKC's unrelated effects on
  Latin ligatures/full-width characters elsewhere.
- Tatweel / kashida (U+0640) is a purely decorative letter-elongation with no
  semantic content; justified Arabic PDFs are full of it. Left in place it
  splits "الرســوم" and "الرسوم" into different BM25 tokens.
- ZWNJ/ZWJ are ORTHOGRAPHIC in Arabic-script languages and are preserved.
"""
from __future__ import annotations

import re
import unicodedata

# Non-printable C0/C1 control characters (keeps common formatting: \t, \n, \r).
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Zero-width and invisible bidi formatting characters. U+200C ZWNJ and U+200D ZWJ
# are deliberately EXCLUDED: stripping ZWNJ merges Persian/Arabic words into
# different words (می‌رود -> میرود) and it also carries meaning in emoji sequences.
_INVISIBLE_CHAR_RE = re.compile(r"[​﻿‏‪-‮]")

# Arabic presentation forms (display glyphs) — normalized back to standard letters.
_ARABIC_PRESENTATION_RE = re.compile(r"[ﭐ-﷿ﹰ-﻿]+")

# Tatweel / kashida: decorative elongation, no semantic value.
_TATWEEL = "ـ"


def _normalize_arabic_presentation_forms(text: str) -> str:
    return _ARABIC_PRESENTATION_RE.sub(
        lambda match: unicodedata.normalize("NFKC", match.group(0)), text
    )


def sanitize_text(text: str) -> str:
    """Enterprise-grade text sanitizer, applied to BOTH documents and queries.

    1. Normalization: converts uniformly to standard NFC form, merging decomposed
       diacritics and accents so character representation is consistent across sources.
    2. Removes invalid and invisible bytes: NUL, zero-width space, BOM markers, and
       invisible bidi override marks (ZWNJ/ZWJ preserved — orthographic in Arabic).
    3. Arabic: maps presentation-form glyphs back to standard letters and strips
       tatweel, so PDF-extracted Arabic matches Arabic typed by a user.
    4. Cleans non-printable characters and PUA placeholder glyphs.
    5. Encoding convergence safeguard: utf-8 ignore to strip orphaned surrogates.
    """
    if not text:
        return ""

    # 1. Normalize to Unicode NFC form
    text = unicodedata.normalize("NFC", text)

    # 2. Remove invisible zero-width characters, BOM, and bidi format controls
    text = _INVISIBLE_CHAR_RE.sub("", text)

    # 3. Arabic-specific convergence
    text = _normalize_arabic_presentation_forms(text)
    text = text.replace(_TATWEEL, "")

    # 4. Clean non-printable control characters and PUA placeholder glyphs
    text = _CONTROL_CHAR_RE.sub("", text)
    text = re.sub(r"[-]", "", text)

    # 5. Fully strip orphaned surrogate pairs, converging to compliant UTF-8
    try:
        cleaned = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
    except Exception:
        chars = []
        for char in text:
            if 0xD800 <= ord(char) <= 0xDFFF:
                continue
            chars.append(char)
        cleaned = "".join(chars)

    return cleaned


def normalize_query(query: str) -> str:
    """Normalize a user query exactly as indexed text was normalized.

    Without this, an Arabic query pasted from a PDF (presentation forms, tatweel,
    zero-width marks) is a different character sequence from the indexed chunk and
    matches neither the BM25 side nor the dense side reliably.
    """
    return sanitize_text(query).strip()


def fit_utf8_bytes(text: str, max_bytes: int) -> str:
    """Truncate to at most max_bytes of UTF-8 WITHOUT splitting a character.

    Milvus VARCHAR limits are byte limits, but chunk budgets are counted in
    characters: Arabic costs 2 bytes/char and CJK 3, so a character-based cap
    lets a non-Latin chunk exceed the column limit and fail the whole insert
    batch after all embedding work is already paid for.
    """
    if not text:
        return text
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")
