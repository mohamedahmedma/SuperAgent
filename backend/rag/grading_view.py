"""Retrieved chunks as the answer model and the grader each need to see them.

The answer model gets every chunk whole: a fee table read out of an image is only useful
entire. The grader is deciding whether the snippets are on the subject and whether they
settle it, and neither question is answered by row 200 of a table — so it sees each chunk
topped and tailed, and a figure summarised down to what it is about.

Moved out of `pipeline.py` with its behaviour unchanged. Pure: nothing here reads the
profile, a model or the retriever.
"""
from typing import List


def format_docs(docs: List[dict]) -> str:
    if not docs:
        return ""
    chunks = []
    for i, doc in enumerate(docs, 1):
        source = doc.get("filename", "Unknown")
        page = doc.get("page_number", "N/A")
        text = doc.get("text", "")
        chunks.append(f"[{i}] {source} (Page {page}):\n{text}")
    return "\n\n---\n\n".join(chunks)


#: How much of one chunk the GRADER sees. The answer model still gets all of it.
#:
#: A grade is a judgement about subject and sufficiency, and the top of a chunk settles
#: both: `AssetDossier.render_surrogate` writes caption first, then description, then the
#: transcription — so a cap taken from the top leaves exactly what the question is being
#: matched against and drops the literal contents, which is evidence for ANSWERING and
#: noise for grading.
#:
#: It is also the only bound on the grading prompt as a whole. `format_docs` passes
#: every retrieved chunk in full, so at `top_k: 8` the prompt was whatever the corpus
#: happened to hold — and a figure whose transcription is a school calendar rendered as
#: a year of table rows pushed one deployment's grader past its completion budget, which
#: comes back as `finish_reason: length` and no JSON at all.
#:
#: Set above a normal leaf (`level_3_size`, 800 here) so an ordinary chunk is untouched
#: and only the outliers are trimmed.
_GRADER_CHUNK_CHARS = 1200


def _head(text: str, cap: int) -> str:
    """The first `cap` characters of `text`, cut at a line boundary."""
    if len(text) <= cap:
        return text
    kept: List[str] = []
    used = 0
    for line in text.splitlines():
        if used + len(line) + 1 > cap:
            break
        kept.append(line)
        used += len(line) + 1
    # A single line longer than the whole budget still has to give — a transcription
    # rendered as one enormous row is the case — and half of it beats none of it.
    return "\n".join(kept) if kept else text[:cap]


#: How much of a FIGURE's own text the grader sees, before its tags and questions.
#:
#: Tighter than `_GRADER_CHUNK_CHARS` because a figure's text is not prose that trails
#: off — it is a caption, then a description, then a transcription of every word printed
#: in the image, and only the first two say what the figure IS. The third is what the
#: ANSWER is written from, and sending it to a grader deciding "are these snippets about
#: school uniform" buys nothing while outweighing the rest of the prompt.
_GRADER_FIGURE_BODY_CHARS = 500


#: The compact lines `AssetDossier.render_surrogate` puts AFTER the transcription. Both
#: are one line, and both are close to exactly what a grade is made of — what the picture
#: is about, and which questions it can answer — so they are kept even though what sits
#: between them and the caption was cut.
_FIGURE_SUMMARY_PREFIXES = ("Tags:", "Answers:")


#: How `render_surrogate` marks the caption line, and therefore how a figure chunk is
#: recognised here without the dossier being in scope.
_FIGURE_MARKER = "[Figure]"


def _grading_view(text: str) -> str:
    """One chunk as the grader should see it: enough to judge it, and no transcription.

    Prose is simply capped — a leaf is `level_3_size` and the cap sits above it, so an
    ordinary chunk passes through whole.

    A figure is summarised instead, because truncating it from the top would keep the
    first rows of a table and drop the `Tags:` and `Answers:` lines underneath, which are
    the two most useful lines in it for this decision and one line each. So the head
    (section path, caption, description) is kept to a tight budget, the transcription in
    the middle is dropped, and the summary lines are put back.

    Heuristic in one respect and knowingly so: `render_surrogate` writes description and
    transcription as adjacent blocks with no marker between them, so "the head" is a
    character budget rather than a field. It is sized to hold a caption and a couple of
    sentences, which is what a description is.
    """
    text = text or ""
    lines = text.splitlines()
    if not any(line.startswith(_FIGURE_MARKER) for line in lines[:3]):
        return _head(text, _GRADER_CHUNK_CHARS)

    body: List[str] = []
    summary: List[str] = []
    for line in lines:
        if line.startswith(_FIGURE_SUMMARY_PREFIXES):
            summary.append(line)
        elif not summary:
            body.append(line)
    kept = _head("\n".join(body), _GRADER_FIGURE_BODY_CHARS)
    return "\n".join([kept, *summary]) if summary else kept


def format_docs_for_grading(docs: List[dict]) -> str:
    """The chunks as the GRADER sees them: the same list, each one topped and tailed.

    Deliberately not `format_docs`. The answer prompt needs a figure's transcription —
    a fee table read out of an image is only useful entire — while the grader is deciding
    whether the snippets are on the subject and whether they settle it, and neither
    question is answered by row 200 of a table. Sending it anyway made the size of the
    grading prompt a property of the corpus rather than of the retrieval.
    """
    if not docs:
        return ""
    chunks = []
    for i, doc in enumerate(docs, 1):
        source = doc.get("filename", "Unknown")
        page = doc.get("page_number", "N/A")
        text = _grading_view(doc.get("text", ""))
        chunks.append(f"[{i}] {source} (Page {page}):\n{text}")
    return "\n\n---\n\n".join(chunks)
