"""Retrieved chunks as a prompt sees them.

One rendering, used by the grader and the answer model alike. It was two: the answer
model got every chunk whole and the grader got each one topped and tailed — prose cut to
1,200 characters, a figure summarised to 500 plus its tags and questions.

That view is deleted, and what it was protecting is worth stating because it still has
to hold. `format_docs` passes every retrieved chunk in full, so the size of the grading
prompt is whatever the chunks happen to be; before this work one image could be a single
60 KB chunk, and eight of those is a prompt no grading model can answer inside its output
window. It comes back as `finish_reason: length` with no JSON at all, which costs the
whole turn: grading is the HIGH rung, so a ladder that reaches no rung routes to
`retrieval_error`, and the user is told the knowledge base has a technical problem while
its answer sits in the chunks already retrieved.

The bound now lives where the size is MADE rather than where the prompt is built:

  * every unit is smaller than its level's budget, so every chunk is bounded by the
    budget of the level it belongs to — this is what splitting a figure's transcription
    into passages bought, together with bounding an over-long table row and an over-long
    sentence (`backend/indexing/document_loader.py`);
  * merging cannot exceed `retrieval.evidence_window_chars`, which sits just above what
    correct chunking can produce, so a chunk that was never too big is untouched and one
    from an index built before figures were split is cut back (`backend/rag/utils.py`);
  * the evidence is selected under that budget BEFORE grading, so a rewrite cannot hand
    the grader the union of two passes (`backend/rag/graph_nodes.py`).

So the grading prompt is at most `top_k` times the evidence window, whatever the corpus
holds — the same property commit 1794762 protected by truncating, kept by bounding, and
now the grader can see the answer it is judging. Measured on the school corpus, 28% of
questions had their evidence hidden from the grader and nothing but the view was hiding
it.

Pure: nothing here reads the profile, a model or the retriever.
"""
from typing import List


def format_docs(docs: List[dict]) -> str:
    """Every retrieved chunk, whole and numbered.

    The `[n]` markers are the contract between three things that must agree: what the
    grader cites in `supporting_chunks`, what `select_context_indices` keeps, and what a
    citation in the answer points at. They are positions in this list, so anything that
    reorders or filters the list has to renumber with it.
    """
    if not docs:
        return ""
    chunks = []
    for i, doc in enumerate(docs, 1):
        source = doc.get("filename", "Unknown")
        page = doc.get("page_number", "N/A")
        text = doc.get("text", "")
        chunks.append(f"[{i}] {source} (Page {page}):\n{text}")
    return "\n\n---\n\n".join(chunks)
