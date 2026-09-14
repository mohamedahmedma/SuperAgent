"""What a tool rendered, placed under the answer that refers to it.

A record tool returns a table, and a table is the one thing a model should not be asked to
retype. The tool renders it once; this module narrows it to what the answer talks about,
puts it under the prose, and marks it so it stays out of the model's own history. Figure
markers resolve here for the same reason: the model points at a picture, and the picture is
placed where it pointed.

Moved out of `service.py` with its behaviour unchanged.
"""
import logging
import re

from backend.profiles import get_profile
from backend.schemas.chat import normalize_answer_blocks
from backend.text_matching import name_key

logger = logging.getLogger(__name__)


#: How every branch of `tools/records_result.j2` and `tools/knowledge_result.j2` opens.
#: These headers address the MODEL — they name an outcome and carry instructions — and a
#: parent must never see one.
#:
#: Uppercase ASCII on an Arabic-first deployment, so there is no wording a reply could
#: legitimately contain that collides with them.
_EVIDENCE_MARKERS = re.compile(
    r"^\s*(?:TIMETABLE|STUDENT_GRADES|SUBJECT_DETAIL|SUBJECTS|TEACHERS|SUBJECT_TEACHER"
    r"|CLASS|ATTENDANCE|NO_RECORDS|NO_STUDENTS_LINKED|NO_CLASS_THIS_TERM|NOT_AUTHORIZED"
    r"|NOT_A_PARENT_SESSION|RECORDS_UNAVAILABLE|TOOL_CALL_LIMIT_REACHED"
    r"|NEEDS_STUDENT_CHOICE|NEEDS_SUBJECT_CHOICE|TIMETABLE_NOT_PUBLISHED"
    r"|SUBJECTS_NOT_PUBLISHED|TEACHERS_NOT_ASSIGNED"
    # The one-day timetable's three. `\b` does not split on an underscore, so
    # `TIMETABLE` above never covered `TIMETABLE_FOR_ONE_DAY` — each header is its own
    # alternative here, exactly as `TIMETABLE_NOT_PUBLISHED` already was.
    r"|TIMETABLE_FOR_ONE_DAY|NOT_A_SCHOOL_DAY|NOTHING_TIMETABLED_THAT_DAY"
    # The knowledge tool's own headers. Same kind of string and the same rule — they
    # name an outcome to the model — and the records set was only ever listed first
    # because a records header is what was caught reaching a parent. A model that
    # pastes one of these pastes the other.
    r"|NEEDS_CLARIFICATION|NEEDS_SCOPE_SELECTION|NO_KNOWLEDGE|PARTIAL_EVIDENCE"
    r"|RETRIEVAL_ERROR)\b",
    re.MULTILINE,
)


def _drop_leaked_evidence(answer: str) -> str:
    """The answer up to the point where it starts relaying the tool's own text.

    Asking the model not to retype the grid moved the problem rather than solving it: it
    stopped reformatting the table and began pasting the MODEL-FACING render instead —
    outcome header, raw `07:45:00` timestamps, English day keys and all. Measured on the
    live model, first try.

    A prompt cannot be relied on for this, which is the lesson of every other guard in
    this file. The headers are a closed set this repo owns, so the cut is exact: the
    prose before the first one is the framing sentence that was asked for, and everything
    from it onward is evidence the reader was never meant to see. The properly rendered
    block is appended afterwards regardless, so nothing is lost by cutting.
    """
    text = answer or ""
    found = _EVIDENCE_MARKERS.search(text)
    if not found:
        return text
    logger.warning("the answer relayed tool evidence; cut at %r", found.group(0).strip())
    return text[: found.start()].rstrip()


def settle_answer_blocks(answer: str, ctx) -> tuple[str, list]:
    """The answer with each tool-rendered block underneath it, and those blocks as data.

    The data a record tool returns is a table, and a table is the one thing a model
    should not be asked to retype. Every time it did, it was one paraphrase away from a
    figure that verification then had to catch — and catching it meant discarding the
    whole answer, so a formatting habit cost a parent their timetable.

    Rendered once by the tool, appended here. Nothing the parent reads as data passes
    through the model at all, which is a stronger guarantee than any check applied
    afterwards could be.

    Narrowed to what was asked, and marked so it stays out of the model's history — see
    `_narrow_block` and `BLOCK_MARKER`.

    Each block leaves here twice, from one narrowing decision. As TEXT, under the prose
    after its marker: what is stored, what the model's history strips, and what any client
    that knows nothing else shows. And as DATA, for a client that draws the table itself
    (`AnswerBlock` in backend/schemas/chat.py), carrying the `index` of the marker it
    draws. A block with no data, or data that fails the contract, simply leaves its
    marker to be shown as text — so a drawn table is only ever an improvement on the text,
    never a condition for seeing the record at all.

    Returns `(answer, blocks)`, the blocks already validated.
    """
    # Cut FIRST, above the block check rather than below it. This used to sit after the
    # early return, so the guard ran only on a turn that produced a table — and
    # `_PRESENTED` holds three outcomes out of the twenty-three the template can render.
    # Every other record (subjects, the classroom, teachers, and every refusal) skipped
    # the cut entirely, which is how «SUBJECTS for فاطمه محمد ابوالحسن — …» reached a
    # parent verbatim, the model's own citation marker still on the end of it.
    prose = _drop_leaked_evidence(answer).rstrip()
    blocks = [b for b in (getattr(ctx, "answer_blocks", None) or []) if b]
    if not blocks:
        # An answer that was NOTHING but evidence leaves nothing to show, and an empty
        # bubble is the one outcome worse than the leak. There is no table to fall back
        # on here — that is what makes this case different from the one below — so the
        # turn is reported as unverified, the same copy its sibling guards use.
        #
        # Only when the cut is what emptied it. A turn that legitimately said nothing
        # (a short-circuit, a quiet success) must stay silent rather than be handed a
        # failure message it did not earn.
        if not prose and (answer or "").strip():
            logger.warning("the whole answer was tool evidence; nothing left to show")
            return get_profile().user_copy.unverified_answer, []
        return prose, []
    language = getattr(ctx, "language", "")
    language = language if isinstance(language, str) else ""
    rendered: list[str] = []
    structured: list[dict] = []
    for block in blocks:
        if isinstance(block, dict):
            kind, text, data = block.get("kind", ""), block.get("text", ""), block.get("data")
        else:
            kind, text, data = "", str(block), None
        shown = _narrow_block(kind, text, prose)
        if not shown:
            continue
        if isinstance(data, dict) and data:
            # Checked BEFORE it is narrowed, so the narrowing only ever walks a shape the
            # contract vouches for — this runs inside the turn, and a tool's malformed
            # data must cost the drawing, never the answer. Narrowing only removes rows,
            # so what it returns still satisfies the contract it was checked against.
            checked = normalize_answer_blocks(
                [{"kind": kind, "index": len(rendered), "language": language, "data": data}]
            )
            if checked:
                block_out = checked[0]
                block_out["data"] = _narrow_block_data(kind, block_out["data"], prose)
                structured.append(block_out)
        rendered.append(f"{BLOCK_MARKER}\n{shown}")
    settled = "\n\n".join(([prose] if prose else []) + rendered)
    return settled, structured


def _append_answer_blocks(answer: str, ctx) -> str:
    """The answer with each tool-rendered block underneath it, or unchanged.

    The text half of `settle_answer_blocks`, for callers that store or show text only.
    """
    return settle_answer_blocks(answer, ctx)[0]


def attach_answer_blocks(rag_trace: dict | None, blocks: list) -> dict | None:
    """Record the turn's blocks on its trace, which is what persists them.

    The reasoning is `attach_assets_to_trace`'s: the trace is the only place a stored
    message keeps anything beside its text, so a turn with no trace yet gets one — rather
    than drawing the table live and printing its markdown after the next reload.
    """
    if not blocks:
        return rag_trace
    return {**(rag_trace or {}), "answer_blocks": blocks}


#: Put on the line before every rendered block. The frontend's markdown renderer drops
#: raw HTML outright (`renderer.html = () => ''`), so a reader never sees this — and the
#: backend can therefore find where a block starts in a stored message.
#:
#: It exists because the block belongs to the READER and not to the model's context. See
#: `strip_answer_blocks`.
BLOCK_MARKER = "<!--record-block-->"


def strip_answer_blocks(text: str) -> str:
    """A stored answer with its rendered blocks removed, for the model to read back.

    The block is 95% of the message it is attached to — a week's timetable is about 1,240
    characters against 52 of prose. History reaches the resolver and the classifier
    through `conversation_text`, which clips each message to 600 characters, so once a
    block was stored the next turn's context was a wall of lesson rows and almost none of
    the sentence that said what the turn was about.

    Measured: «ومين بيديها في الفصل» — who teaches her — was resolved against that wall,
    classified as a timetable question, and answered with the timetable again. The model
    was not wrong; it was handed the wrong record because the previous record had crowded
    the question out.

    So the block stays in the stored message, where the reader and a re-rendered history
    still get the table, and is dropped from what the model reads. The prose survives, and
    the prose is what a follow-up actually needs: "her timetable for the second term" is
    the subject; the forty-five rows are not.
    """
    body = text or ""
    cut = body.find(BLOCK_MARKER)
    prose = body[:cut].rstrip() if cut != -1 else body
    # Figure anchors come out for the same reason the block does, and a sharper one: the
    # anchor carries an asset_id, and an id in the model's history is an id in its next
    # answer — shown one, a small model writes it back as an image link that cannot load.
    # The reader keeps the picture; the model reads the sentence that surrounded it.
    return _FIGURE_ANCHOR_RE.sub("", prose)


#: What a resolved figure marker becomes in the stored answer.
#:
#: An HTML comment for the same reason `BLOCK_MARKER` is one: the frontend's markdown
#: renderer drops raw HTML outright, so a reader never sees it. That also makes the
#: feature degrade instead of breaking — a frontend that predates it renders clean prose
#: and still shows the pictures in the trailing block, rather than printing an anchor.
_FIGURE_ANCHOR = "<!--figure:%s-->"


_FIGURE_ANCHOR_RE = re.compile(r"<!--figure:.+?-->")


#: `[FIGURE 2]`, `[figure 2]`, `[الشكل ٢]`. The Arabic forms and the Arabic-Indic digits
#: are here because this corpus is Arabic: a model writing Arabic prose writes «الشكل ٢»
#: as readily as the English marker it was shown, and a parser that only knew ASCII would
#: have silently dropped most real markers and shown no picture.
_FIGURE_MARKER_RE = re.compile(
    r"\[\s*(?:FIGURE|الشكل|شكل)\s*([0-9٠-٩۰-۹]+)\s*\]",
    re.IGNORECASE,
)


#: Arabic-Indic and Extended Arabic-Indic digits to ASCII, so «٢» and "2" name the same
#: figure.
_FIGURE_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def resolve_figure_markers(answer: str, ctx) -> str:
    """The answer with each figure marker replaced by an anchor for that picture.

    The model is shown `[FIGURE 1]` on a chunk header and asked to write the same marker
    where the picture belongs. This turns the ones it wrote into anchors the frontend
    renders the image at, so the figure lands inside the sentence that describes it
    instead of as a card underneath the whole answer.

    An unknown number is DELETED, and nothing else happens — no refusal, no correction,
    no annotation. That rule is the entire lesson of the grounding layer this replaced:
    it withheld answers whose citations it could not verify, and a correct answer
    withdrawn over a marker is a far worse outcome than a missing picture. A model that
    invents `[FIGURE 9]` costs the reader nothing.

    Deleting is also what keeps the marker from ever being *seen*: the raw text streams
    to the client in `content` deltas before this runs, so an unresolvable marker would
    otherwise sit in the bubble. Both problems, one rule.
    """
    body = answer or ""
    if "[" not in body:
        return body
    numbers = dict(getattr(ctx, "figure_numbers", None) or {})
    dropped = False

    def _anchor(match) -> str:
        nonlocal dropped
        try:
            number = int(match.group(1).translate(_FIGURE_DIGITS))
        except ValueError:  # pragma: no cover - the pattern only matches digits
            dropped = True
            return ""
        asset_id = numbers.get(number)
        if not asset_id:
            logger.info(
                "the answer named figure %s; this turn retrieved %s",
                number, sorted(numbers) or "none",
            )
            dropped = True
            return ""
        # `-->` would close the comment early and leak the rest of the id as text. It
        # cannot occur in a `build_asset_id` output, which is why this is a guard and
        # not an encoding scheme.
        return _FIGURE_ANCHOR % str(asset_id).replace("-->", "")

    resolved = _FIGURE_MARKER_RE.sub(_anchor, body)
    if dropped:
        # A deletion mid-sentence leaves two spaces where the marker was. Runs of spaces
        # and tabs only — collapsing newlines would join paragraphs.
        resolved = re.sub(r"[ \t]{2,}", " ", resolved)
    return resolved


def _narrow_block(kind: str, block: str, answer: str) -> str:
    """The block, cut down to the rows the answer actually talks about.

    A parent asking «هي جابت كام في العربي» was given the Arabic mark in the sentence and
    then every other subject's mark underneath it, which answers a question nobody asked
    and buries the one they did.

    The model's own sentence is the filter, and that is the whole idea: the tool decides
    what is TRUE and the model decides what is RELEVANT, which is the division of labour
    it is actually good at. Nothing is rewritten — a row either survives or it does not,
    so a figure the parent reads is still the tool's own.

    Falls back to the whole block whenever the answer names nothing, because "show me her
    grades" should still show all of them.
    """
    lines = [line for line in (block or "").split("\n") if line.strip()]
    if not lines or not (answer or "").strip():
        return block
    folded = name_key(answer)

    if kind == "grades":
        # `subject: 84.0% (B)` — the label is what precedes the colon.
        kept = [ln for ln in lines if name_key(ln.split(":")[0]) and name_key(ln.split(":")[0]) in folded]
        return "\n".join(kept) if kept else block

    if kind == "timetable":
        # Day headings own the rows beneath them, so a day is kept or dropped whole.
        days, current, keeping = [], [], False
        for line in lines:
            if line.startswith("**"):
                keeping = name_key(line.strip("*")) in folded
                current = [line] if keeping else []
                if keeping:
                    days.append(current)
                continue
            if keeping and current is not None:
                current.append(line)
        kept = ["\n".join(day) for day in days if len(day) > 1]
        return "\n".join(kept) if kept else block

    return block


def _narrow_block_data(kind: str, data: dict, answer: str) -> dict:
    """`_narrow_block` for a block's DATA: the same rows kept, by the same rule.

    Narrowed separately because one copy is lines of text and the other a structure, but
    the decision has to be the same one — a phone drawing Sunday alone while the stored
    text keeps the whole week would be two answers to one question. So each branch reads
    the label its text branch reads (a day's heading, a mark's subject), folds it the same
    way, and falls back to the whole record in the same cases. test_answer_blocks.py holds
    the two in step.
    """
    if not (answer or "").strip():
        return data
    folded = name_key(answer)

    if kind == "timetable":
        # The text heading is `**{shows_as or name}**`, which is exactly what `label` holds.
        kept = [
            day
            for day in data.get("days") or []
            if day.get("slots") and name_key(day.get("label") or day.get("day") or "") in folded
        ]
        return {**data, "days": kept} if kept else data

    if kind == "grades":
        # The text row is `subject: 84.0% (B)` and its rule reads what precedes the colon.
        kept = []
        for course in data.get("courses") or []:
            label = name_key(str(course.get("subject") or "").split(":")[0])
            if label and label in folded:
                kept.append(course)
        return {**data, "courses": kept} if kept else data

    return data
