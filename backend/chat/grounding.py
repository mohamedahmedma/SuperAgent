"""Does the answer's arithmetic exist in the evidence? Checked, not asked.

The failure this module answers to: a parent asked what their son's fees were, the
corpus returned nothing, and the assistant replied «مصاريف الصف الرابع 45 ألف جنيه على
تلات دفعات» with a citation marker attached. The figure was invented, the grade was
invented, and the citation pointed at a chunk that had never existed.

Two things were already true and neither helped. The system prompt asks the model to
cite what it uses and to answer only from retrieved material — a prompt is a request,
and this model granted it on most turns and not on that one. And the runtime had a guard
for exactly this case, which duplicate tool calls had quietly disengaged
(`backend/chat/runtime.py`). Restoring that guard stops the corpus-said-nothing route.
This module closes the other one: retrieval returned real chunks, and the model still
stated a number that is in none of them.

## Why numbers

Because they are the one class of claim that can be checked with no model, no embedding
and no judgement, and because in this deployment they are the whole harm surface. A fee,
an instalment count, a deadline, a percentage — a parent acts on those. A verifier that
tried to check prose would need a model, and a model is the thing whose output is in
question. So the rule is narrow and total: **every figure in the answer must appear in
the evidence the turn actually retrieved, or be derivable from figures that do.**

Nothing here is heuristic about what the model meant. It extracts numbers, normalises
them, and does set membership. The same input gives the same verdict every time, which
is the property the whole design is for.

## What it deliberately does not flag

  * Numbers below `floor` (default 100). "3 instalments" against evidence that spells
    «ثلاث دفعات» in words is a formatting difference, not a fabrication, and chasing it
    would produce false positives on every turn for no safety gained.
  * Arithmetic the evidence supports. Evidence saying «45,000 على ثلاث دفعات» justifies
    an answer that says 15,000 per instalment. Sums, differences, products and
    quotients of grounded figures are grounded — see `_derivable`.
  * A percentage applied to a grounded figure. «الرسوم 30,000» plus «خصم 10% للأخ
    التاني» supports 27,000, and a sibling discount is one of the most ordinary
    questions this deployment gets — an earlier revision scored it as a fabrication
    because 27,000 is not a sum or quotient of {30000, 10}.
  * Citation markers themselves. `[1]` is stripped before extraction; it is provenance,
    not a claim.

Digits arrive in three scripts (ASCII, Arabic-Indic ٠-٩, Eastern Arabic ۰-۹) and figures
carry multipliers in two languages («45 ألف» is 45000, and so is "45k"). All of it is
normalised to one number before comparison, because a check that could be defeated by
writing the same figure a different way would not be a check.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set, Tuple

#: Arabic-Indic and Eastern Arabic digits onto ASCII. A figure written «٤٥٠٠٠» and one
#: written "45000" are the same claim and must not compare as different ones.
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

#: Words that scale the number in front of them, in both languages the corpus uses.
_MULTIPLIERS = {
    "ألف": 1000, "الف": 1000, "ألفا": 1000, "ألفاً": 1000, "الفا": 1000,
    "آلاف": 1000, "الاف": 1000, "ألاف": 1000,
    "مليون": 1_000_000, "ملايين": 1_000_000, "مليونا": 1_000_000,
    "thousand": 1000, "thousands": 1000, "k": 1000,
    "million": 1_000_000, "millions": 1_000_000, "m": 1_000_000, "mn": 1_000_000,
}

#: The longest multiplier plus room for a currency word between it and the number.
_MULTIPLIER_WINDOW = 14

#: A run of digits, with thousands separators (ASCII and Arabic ٬) and a decimal part
#: (ASCII and Arabic ٫) allowed inside it.
_NUMBER = re.compile(r"\d[\d,٬]*(?:[.٫]\d+)?")

#: Citation markers, removed before anything counts digits.
_CITATION = re.compile(r"\[\s*(\d{1,3})\s*\]")

#: A word, in either script, for reading the token that follows a number.
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)

#: A percentage in the evidence. Read separately from the plain figures because a
#: percentage is an OPERATOR on them, not another amount: 10 next to 30,000 means 3,000
#: or 27,000, and treating it as a quantity is what made a sibling discount look invented.
_PERCENT = re.compile(r"(\d+(?:[.٫]\d+)?)\s*(?:%|٪|في المئة|في المية|بالمئة|percent)")

#: A clock time. Read as ONE typed value and then removed, never as the loose digits it
#: is made of, because those digits are not claims about anything.
#:
#: This is the typed half of the rule the module docstring states. «10:00» is a time; it
#: is not a 10 and a 0, and it is certainly not a 10 that the next word may scale. Both
#: readings did real damage:
#:
#:   * A physics lesson after «10:00» made 10,000 — the lookahead read a truncated
#:     «الفيزياء» as «الف» — and a correct timetable was discarded as ungrounded.
#:   * «45 حصة» passed as grounded because 45 appears inside «07:45». A wrong figure
#:     colliding with the digits of an unrelated time is a false NEGATIVE in a safety
#:     check, which is the worse of the two failures and the quieter one.
#:
#: Masked in the answer and in the evidence by the same function, so the two can never
#: disagree about what a time is.
_TIME = re.compile(r"(?<!\d)(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?(?!\d)")

#: Below this, a figure is a count rather than a claim. See the module docstring.
DEFAULT_FLOOR = 100

#: Ceiling on how many evidence figures take part in the derivation search, which is
#: quadratic. Corpora with more than this are answering from a table, and a table's
#: figures are in the set already.
_DERIVATION_LIMIT = 120

#: The largest number of parts a grounded total may be split into. Instalments, terms
#: and school months all sit well inside this; beyond it the "derivation" would admit
#: almost any figure and the check would stop meaning anything.
_MAX_PARTS = 12


def normalize_digits(text: str) -> str:
    """Every digit script folded onto ASCII."""
    return (text or "").translate(_DIGITS)


def strip_citations(text: str) -> str:
    """Remove `[n]` markers so their indices are never read as figures."""
    return _CITATION.sub(" ", text or "")


def strip_times(text: str) -> str:
    """Remove clock times, so a time is never read as a quantity.

    A space, not an empty string: the digits must not be allowed to close up against
    whatever followed them, or a masked time would leave the next word adjacent to the
    number before it and the multiplier lookahead would read the two as one claim.
    """
    return _TIME.sub(" ", text or "")


def citation_indices(text: str) -> List[int]:
    """The chunk numbers an answer claims to be citing, in the order written."""
    return [int(match.group(1)) for match in _CITATION.finditer(normalize_digits(text or ""))]


def numeric_claims(text: str, *, floor: int = 0) -> Set[float]:
    """Every figure stated in `text`, normalised and scaled by any multiplier word.

    Citation markers are removed first. A number is scaled when a multiplier follows it
    within a short window, so «45 ألف جنيه» and "45 thousand pounds" both read as 45000
    while "45 students" stays 45.
    """
    cleaned = strip_times(normalize_digits(strip_citations(text)))
    found: Set[float] = set()
    for match in _NUMBER.finditer(cleaned):
        raw = match.group(0).replace(",", "").replace("٬", "").replace("٫", ".")
        try:
            value = float(raw)
        except ValueError:
            continue
        value *= _multiplier_after(cleaned, match.end())
        if abs(value) >= floor:
            found.add(value)
    return found


def percentages_in(text: str) -> Set[float]:
    """Every percentage stated in `text`, as a number out of 100.

    Separate from `numeric_claims` because the two answer different questions: that one
    asks "what amounts does this state", this one asks "what proportions does it apply".
    A discount is only derivable if the check knows which of the two a 10 is.
    """
    cleaned = normalize_digits(text or "")
    found: Set[float] = set()
    for match in _PERCENT.finditer(cleaned):
        try:
            found.add(float(match.group(1).replace("٫", ".")))
        except ValueError:
            continue
    return found


def _multiplier_after(text: str, position: int) -> int:
    """The scale word following a number, or 1.

    Only the FIRST word after the number is considered, and only when it STARTS within a
    short window. Looking further would let a «مليون» three sentences away scale a figure
    it has nothing to do with.

    The word is read WHOLE, out of the full text; the window bounds only where it may
    BEGIN. Slicing the text first and matching inside the slice truncates whatever word
    straddles the boundary, and a truncated word can collide with a real multiplier —
    «الفيزياء» cut at 14 characters is «الف», which is the Arabic for "thousand".

    That is not hypothetical. A parent asked for their daughter's timetable, the model
    wrote «10:00» and then, on the next line, a physics lesson; the truncation read
    10,000, no such figure was in the evidence, and the entire correct timetable was
    replaced with "I couldn't verify those figures". Every Arabic word beginning الف
    collided the same way — الفصل, الفسحة, الفلسفة, الفنون, الفرنسية — as did مليونير
    against مليون, and «الفصل الدراسي» after a number is about as common as Arabic gets.
    """
    match = _WORD.search(text, position)
    if match is None or match.start() >= position + _MULTIPLIER_WINDOW:
        return 1
    return _MULTIPLIERS.get(match.group(0).strip().lower(), 1)


def _close(left: float, right: float) -> bool:
    """Whether two figures are the same claim, allowing for rounding in the answer."""
    return abs(left - right) <= max(0.51, abs(right) * 1e-9)


def _derivable(value: float, grounded: Sequence[float], percentages: Sequence[float] = ()) -> bool:
    """Whether `value` follows arithmetically from figures that ARE in evidence.

    Evidence saying «45,000 على ثلاث دفعات» supports an answer of 15,000 per instalment.
    That is the model doing its job, not inventing, and a verifier that called it a
    fabrication would be training its operators to ignore it.

    The split allowance is separate from the pairwise one because the divisor is usually
    not a figure at all: this corpus writes instalment counts as words — «ثلاث دفعات» —
    so there is no 3 in evidence to divide by, and pairwise arithmetic alone rejected a
    correct per-instalment answer. A grounded total divided into a small whole number of
    parts (or a grounded part multiplied up to a total) is therefore grounded too.

    It keeps its teeth: against evidence of 45,000 the allowance admits 22,500 and
    15,000 and 90,000, and still rejects 30,000 — which is the class of number the check
    exists to catch.
    """
    for left in grounded:
        for right in grounded:
            if _close(left + right, value) or _close(left - right, value):
                return True
            if _close(left * right, value):
                return True
            if right and _close(left / right, value):
                return True
    for total in grounded:
        for parts in range(2, _MAX_PARTS + 1):
            if _close(value * parts, total) or _close(value, total * parts):
                return True

    # A percentage applied to a grounded figure. Without this the check rejected the
    # CORRECT answer to one of the most ordinary questions this deployment gets: a
    # corpus stating «الرسوم 30,000» and «خصم 10% للأخ التاني» supports 27,000, and
    # 27,000 is not a sum, product, quotient or n-way split of {30000, 10} — so a
    # sibling discount, a late-payment surcharge and an early-payment discount were all
    # scored as fabrications. Percentages have to be read AS percentages.
    for percent in percentages or ():
        share = percent / 100.0
        for base in grounded:
            if (
                _close(base * share, value)          # the discount or surcharge itself
                or _close(base * (1 - share), value)  # the price after a discount
                or _close(base * (1 + share), value)  # the price after a surcharge
            ):
                return True
    return False


@dataclass(frozen=True)
class GroundingReport:
    """What the check found. `ok` is the only field a caller has to read."""

    ok: bool
    evidence_count: int
    checked: int
    ungrounded: Tuple[float, ...] = ()
    invalid_citations: Tuple[int, ...] = ()
    cited_without_evidence: bool = False
    #: Whether this turn read a tool at all. A FACT about the turn, never a verdict: the
    #: citation rule below is unchanged by it, and `ok` does not consult it. It exists so
    #: a caller can tell "cited a source that never existed" from "cited on a turn whose
    #: evidence simply carries no `[n]`" — two failures with the same signature and very
    #: different remedies. What to DO about the second is policy, and policy lives in
    #: `backend/chat/service.py` beside `answer_grounding_mode`, not here.
    tool_evidence: bool = False

    @property
    def reason(self) -> str:
        """One line naming what failed, for the trace and the log."""
        parts = []
        if self.cited_without_evidence:
            parts.append("cited evidence on a turn that retrieved none")
        if self.invalid_citations:
            parts.append(
                f"citation(s) {', '.join(str(i) for i in self.invalid_citations)} "
                f"outside the {self.evidence_count} chunk(s) retrieved"
            )
        if self.ungrounded:
            shown = ", ".join(_render(value) for value in self.ungrounded)
            parts.append(f"figure(s) not in the evidence: {shown}")
        return "; ".join(parts) or "grounded"

    def as_trace(self) -> dict:
        return {
            "grounding_ok": self.ok,
            "grounding_evidence_chunks": self.evidence_count,
            "grounding_numbers_checked": self.checked,
            "grounding_ungrounded_numbers": [_render(v) for v in self.ungrounded],
            "grounding_invalid_citations": list(self.invalid_citations),
            "grounding_tool_evidence": self.tool_evidence,
            "grounding_reason": self.reason,
        }


def _render(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def verify(
    answer: str,
    evidence: Iterable[str],
    *,
    floor: int = DEFAULT_FLOOR,
    check_citations: bool = True,
    extra_evidence: Iterable[str] = (),
) -> GroundingReport:
    """Check an answer against the evidence the turn retrieved.

    `evidence` is the text of the retrieved chunks. An empty answer, or one stating no
    figure and citing nothing, passes trivially — most turns do, and the check has to be
    free on those.

    `extra_evidence` is material the turn genuinely read that is NOT a numbered chunk —
    today, what `get_student_records` returned. Its figures ground the answer exactly as
    a chunk's would, and it is deliberately kept out of `evidence_count`, which is the
    denominator the citation check counts against: a child's marks carry no `[n]` and
    admitting them there would make `[3]` valid on a turn that retrieved two chunks.
    Numbers and provenance are two different questions, and this argument is the reason
    the parameter exists rather than the caller simply concatenating the two lists.
    """
    answer = answer or ""
    evidence_texts = [text for text in evidence if text]
    evidence_count = len(evidence_texts)
    tool_texts = [text for text in extra_evidence if text]
    has_tool_evidence = bool(tool_texts)
    evidence_blob = "\n".join(evidence_texts + tool_texts)

    grounded_all = numeric_claims(evidence_blob, floor=0)
    grounded = sorted(grounded_all)[:_DERIVATION_LIMIT]
    percentages = sorted(percentages_in(evidence_blob))[:_DERIVATION_LIMIT]

    stated = numeric_claims(answer, floor=floor)
    ungrounded = tuple(
        sorted(
            value
            for value in stated
            if not any(_close(value, known) for known in grounded_all)
            and not _derivable(value, grounded, percentages)
        )
    )

    invalid: Tuple[int, ...] = ()
    cited_without_evidence = False
    if check_citations:
        indices = citation_indices(answer)
        if indices and evidence_count == 0:
            # Unchanged, and deliberately so: tool text is NOT a citable chunk, and
            # counting it as one would make `[1]` valid on a turn that retrieved nothing
            # to cite. `tool_evidence` reports the distinction without softening the rule.
            cited_without_evidence = True
        else:
            invalid = tuple(
                sorted({i for i in indices if i < 1 or i > evidence_count})
            )

    return GroundingReport(
        ok=not (ungrounded or invalid or cited_without_evidence),
        evidence_count=evidence_count,
        checked=len(stated),
        ungrounded=ungrounded,
        invalid_citations=invalid,
        cited_without_evidence=cited_without_evidence,
        tool_evidence=has_tool_evidence,
    )


__all__ = [
    "DEFAULT_FLOOR",
    "GroundingReport",
    "citation_indices",
    "normalize_digits",
    "numeric_claims",
    "percentages_in",
    "strip_citations",
    "verify",
]
