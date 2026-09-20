"""The last checks on an answer, and the copy served instead of one.

Each check asks one question a prompt cannot be trusted to have settled — did the answer
deny a record the turn actually read, did the tool the turn required actually run, was
anything left once the transcript was stripped — and returns replacement copy or "". The
canned replies for outcomes no model needs to compose live here too, because they are what
those checks put in an answer's place.

Moved out of `service.py` with its behaviour unchanged. Copy is read from the active
profile per call rather than captured at import.
"""
import logging
import re

from backend.chat.finalize import Finalizer
from backend.profiles import get_profile
from backend.text_matching import name_key

logger = logging.getLogger(__name__)


def terminal_reply(status: str, language: str) -> str:
    """The profile's own wording for an outcome the model does not need to compose."""
    from backend.chat.turn_policy import localized

    copy = get_profile().user_copy.retrieval_error if status == "retrieval_error" else get_profile().user_copy.no_knowledge
    return localized(copy, language)


def _no_knowledge_response() -> str:
    return get_profile().user_copy.no_knowledge


def _retrieval_error_response() -> str:
    return get_profile().user_copy.retrieval_error


def nothing_usable_reply(finalizer: Finalizer, turn_plan) -> str:
    """Copy for a turn whose model output contained no answer at all.

    Measured on the live provider: on one turn in three the model emitted a transcript —
    reasoning plus a fabricated tool call — and never opened a final channel. The
    finalizer correctly withholds all of it, and the turn then had nothing to say, so
    the parent got an empty bubble. Suppressing a non-answer is right; showing nothing
    in its place is not.

    `retrieval_error` is the honest copy for it. The knowledge base was fine — the
    model's reply was unusable — and what that copy tells a parent is exactly what this
    situation warrants: a brief technical problem, try again in a moment. Inventing a
    cheerier message would be claiming to know something about a turn that produced no
    information at all.

    Returns "" when the turn legitimately had nothing to say — a social reply that was
    short-circuited, or a plan that answered without the model — so this never
    manufactures an error out of a quiet success.
    """
    if turn_plan is not None and getattr(turn_plan, "short_circuit", False):
        return ""
    # Only when the finalizer actually withheld something. A model that returned an
    # empty string on its own is a different fault and not one this copy describes.
    trace = finalizer.as_trace()
    withheld = (
        trace.get("finalize_dropped_tool_call_messages")
        or trace.get("finalize_harmony_messages")
        or trace.get("finalize_dropped_chars")
    )
    if not withheld:
        return ""
    logger.warning(
        "the model produced no answer channel this turn; serving the retry copy"
    )
    return get_profile().user_copy.retrieval_error


#: Outcomes where `get_student_records` actually returned a child's record. Anything
#: else — no_records, unavailable, not_authorized, which_student — is a turn that
#: legitimately has nothing to report, and an answer saying so is the CORRECT answer.
#: Every outcome name that means a record actually came back. It must grow with each new
#: record tool: an outcome missing from here cannot trip `_denies_the_records` at all, so
#: a turn that read a child's record and then told the parent nothing was found passes
#: unnoticed. `timetable` was missing for exactly that reason until the classroom tools
#: were added and the gap was noticed.
RECORDS_RETRIEVED = frozenset(
    {
        "grades",
        "subject",
        "attendance",
        "timetable",
        # One day of a week is a record that came back, exactly as the whole week is.
        # Missing from here, an answer narrowed to tomorrow could tell a parent no
        # timetable was found on a turn that read one — which is the failure this set
        # exists to catch, and it would have been invisible on the commonest question
        # this deployment gets.
        "timetable_day",
        "class",
        "subjects",
        "teachers",
        "subject_teacher",
    }
)


def _denies_the_records(ctx, answer: str, *, phrases) -> bool:
    """Whether the answer tells the parent nothing was found, on a turn that found it.

    The failure, verbatim from the deployment: `get_student_records` returned 87.5% and
    91.0% for a named child, and the assistant replied that it could not find any
    records. Nothing in the system contradicted it — the graph knew the tool had been
    called and not what it returned, and the numeric check cannot see a claim that
    states no number.

    Both halves are required and they come from opposite ends. What the tool returned is
    fact, reported by the tool itself (`note_tool_outcome`). What the answer claims is a
    phrase list, which is a guess — so the phrases are the deployment's own copy, and the
    mode this drives starts at `observe` for exactly that reason.

    `phrases` is that wording, handed in rather than read here, so this stays a question
    about three values — what ran, what was said, what counts as denying it — and a test
    states the wording it means instead of reaching into the profile.
    """
    phrases = list(phrases or [])
    if not phrases:
        return False
    outcomes = getattr(ctx, "tool_outcomes", None) or []
    if not any(outcome in RECORDS_RETRIEVED for name, outcome in outcomes):
        return False
    folded = name_key(answer or "")
    if not folded:
        return False
    # Folded first, then tested for emptiness — not `if phrase`, which was the bug.
    # A phrase of "   " is truthy and folds to "", and "" is a substring of every answer,
    # so one stray blank line in a deployment's `records_denial_phrases` made EVERY
    # records answer read as a denial. Under `records_denial_mode: enforce` that would
    # have replaced every correct answer about a child's marks with the could-not-verify
    # copy — a config typo turning into a total outage of the feature it guards.
    needles = [key for key in (name_key(phrase) for phrase in phrases) if key]
    return any(needle in folded for needle in needles)


def enforce_records_agreement(finalizer: Finalizer, ctx, turn_plan) -> str:
    """Replacement copy for an answer that denies a record the turn actually read.

    The numeric grounding check that used to sit beside this is gone: it read the
    model's prose about a table the model no longer writes. This one asks a different
    question and survives it — what the tool RETURNED against what the answer CLAIMED,
    which is not a figure comparison at all. Contract unchanged: "" when there is
    nothing to do.
    """
    mode = getattr(get_profile().agent, "records_denial_mode", "off")
    if mode == "off" or turn_plan is None or getattr(turn_plan, "short_circuit", False):
        return ""
    phrases = getattr(get_profile().agent, "records_denial_phrases", None)
    if not _denies_the_records(ctx, finalizer.answer or "", phrases=phrases):
        return ""
    logger.warning(
        "the answer denies a record this turn retrieved; mode=%s", mode
    )
    return get_profile().user_copy.unverified_answer if mode == "enforce" else ""


def enforce_forced_tool_ran(finalizer: Finalizer, ctx, turn_plan) -> str:
    """Replacement copy for an answer whose required tool never actually ran.

    `_ForcePlannedTool` asks the provider to require a tool; it cannot make it. When the
    endpoint returns an ordinary assistant message instead, that middleware relaxes the
    requirement and gives the model one more pass — and an unforced model may answer the
    question from memory, which is the single outcome forcing exists to prevent.

    So the requirement is checked here, where the turn's actual tool traffic is known,
    rather than trusted at the point it was requested. Two failures are closed at once,
    and it is the same replacement that closes both:

      * **The invented record.** Nothing else checks it. The figures a parent reads
        now come from a rendered block, so an answer with no tool behind it has no
        block either — and its prose is the one thing left that could be invented.
      * **The doubled answer.** The retry happens after the first attempt's prose has
        already streamed to the browser, so the reader would otherwise see the rejected
        answer followed by the second one. A replacement is an assignment on the client,
        not an append, so it clears both.

    Same shape and same contract as its two siblings above — "" when there is nothing to
    do — and it asks only the question those cannot: was a tool this turn REQUIRED to
    call among the tools it called?
    """
    forced = (getattr(ctx, "forced_tool", "") or "").strip()
    if not forced or turn_plan is None or getattr(turn_plan, "short_circuit", False):
        return ""
    # The planner's own dispatch satisfies the requirement as surely as a model call
    # does: a seeded result is the tool having run. `tool_outcomes` records both.
    if any(name == forced for name, _ in (getattr(ctx, "tool_outcomes", None) or [])):
        return ""
    logger.warning(
        "the turn required %s and no such tool ran; replacing the answer", forced
    )
    return get_profile().user_copy.unverified_answer


def resumed_static_reply(rag_result: dict | None) -> str | None:
    """The copy for a resumed clarification with nothing to answer from, or None.

    One rule for both entry points, and it used to be two. The retrieval-error branch
    was added to the sync answer and never carried to the streamed one, which looked only
    at whether any documents came back — so a parent resuming a clarification while the
    knowledge base was unreachable was told the school had no information on it. That is
    precisely the reading B1 forbids: an outage says nothing about what the corpus holds.

    None means the retrieved documents stand and the model should answer from them. A
    string, even an empty one, is the reply — the distinction matters because a profile
    may configure empty copy, and that must still skip the model.
    """
    if not isinstance(rag_result, dict):
        return _no_knowledge_response()
    trace = rag_result.get("rag_trace") or {}
    status = rag_result.get("retrieval_status") or trace.get("retrieval_status")
    route = rag_result.get("route") or trace.get("route")
    if status == "retrieval_error" or route == "retrieval_error":
        return _retrieval_error_response()
    if status == "no_knowledge" or route == "no_knowledge" or not (rag_result.get("docs") or []):
        return _no_knowledge_response()
    return None


#: The fewest digits a figure must have for this check to look at it.
#:
#: Three, and the number is the whole design. The grounding check this replaces was
#: retired (commit f40f8c2) for two extractor faults, and both of them live below three
#: digits: "45 حصة" passed as verified because 45 appears inside "07:45", and a correct
#: timetable was withdrawn because "10:00" read as 10,000. Ignoring one- and two-digit
#: figures removes times, dates, ordinals, small counts and percentages from the check in
#: one stroke — and leaves exactly the class that caused the incident this exists for: an
#: amount. "88,000" is five digits, and it was the wrong year group's fee.
#:
#: So it is deliberately narrow rather than thorough. A check that fires on a correct
#: answer is worse than one that stays quiet on a wrong one, because the first teaches a
#: deployment to switch it off.
_FIGURE_MIN_DIGITS = 3

#: Digits in the scripts this deployment sees, folded to Western before comparison: a
#: model may write ١٠٥٬٠٠٠ where the corpus wrote 105,000, and they are the same figure.
_DIGIT_FOLD = {ord(c): str(i % 10) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹")}

#: Separators inside a figure. Stripped so "105,000", "105 000" and "105٬000" compare
#: equal, and NOT interpreted — the retired check parsed the Arabic multiplier "ألف" as
#: a thousand and turned a clock into a price. Nothing here reads a figure's meaning; it
#: compares the digits as written.
_FIGURE = re.compile(r"\d[\d,\u066c\u00a0\u202f ]*\d|\d")


def _figures(text: str) -> set:
    """Every figure in `text` with at least `_FIGURE_MIN_DIGITS` digits, as bare digits.

    Bare digits, so "105,000 EGP" and "105000" are one figure, and so a figure is
    compared as a WHOLE token — the substring match is what let 45 be satisfied by 07:45.
    """
    folded = (text or "").translate(_DIGIT_FOLD)
    found = set()
    for match in _FIGURE.finditer(folded):
        digits = "".join(ch for ch in match.group(0) if ch.isdigit())
        if len(digits) >= _FIGURE_MIN_DIGITS:
            found.add(digits)
    return found


def ungrounded_figures(answer: str, evidence: str) -> list:
    """Figures the answer states that the evidence it was given does not contain.

    The failure, verbatim from the deployment: asked for Year 3 fees, both grader modes
    answered 88,000 EGP — the FS1-FS2 row. Nothing caught it. The grader had approved the
    evidence, the evidence was right, and the answer read the wrong line of it.

    This asks the one question no model call can be trusted with and no prompt can
    enforce: is every amount in this sentence actually in the material it was written
    from? It costs no tokens and it cannot hallucinate, because it only ever compares
    digits that are already on the page.

    What it does NOT do is judge. A figure absent from the evidence is reported, not
    corrected — the answer may have summed two rows or converted a percentage, both of
    which are legitimate and neither of which this can tell apart from an invention. That
    is why it ships in `observe`: the trace says what it found, and a deployment reads its
    own false-positive rate before it lets this replace an answer.
    """
    stated = _figures(answer)
    if not stated:
        return []
    return sorted(stated - _figures(evidence))


def enforce_answer_figures(finalizer: Finalizer, turn_plan, rag_trace) -> str:
    """Replacement copy for an answer stating an amount its evidence does not contain.

    Scoped to the knowledge-base path, and that scope is the other half of why this can
    exist again. Records no longer carry a model-written figure at all — the tool renders
    the grid and the model writes only the sentence beside it (commit 79d2810) — so the
    prose worth checking is what a turn wrote FROM RETRIEVED CHUNKS, which is where the
    fee incident happened and where nothing else looks.

    Same contract as its siblings: "" when there is nothing to do.
    """
    mode = getattr(get_profile().agent, "answer_figures_mode", "off")
    if mode == "off" or turn_plan is None or getattr(turn_plan, "short_circuit", False):
        return ""
    chunks = (rag_trace or {}).get("retrieved_chunks") or []
    if not chunks:
        return ""
    evidence = "\n".join(str(chunk.get("text", "")) for chunk in chunks)
    missing = ungrounded_figures(finalizer.answer or "", evidence)
    if not missing:
        return ""
    logger.warning(
        "the answer states %s, which its evidence does not contain; mode=%s",
        missing[:4], mode,
    )
    return get_profile().user_copy.unverified_answer if mode == "enforce" else ""
