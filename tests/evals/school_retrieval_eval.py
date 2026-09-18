"""Where a school question loses its evidence: the retrieval funnel, stage by stage.

    .venv/Scripts/python.exe tests/evals/school_retrieval_eval.py
    .venv/Scripts/python.exe tests/evals/school_retrieval_eval.py --split holdout
    .venv/Scripts/python.exe tests/evals/school_retrieval_eval.py --top-k 8 --pool 30 --verbose

Not a pass/fail test and not run by pytest: it needs Milvus up with the school corpus
indexed, and it reports numbers rather than asserting them.

## Why stages, and not one recall number

A retrieval score says "the chunk came back". It cannot say where a fact that the corpus
holds stopped being available to the model, and on this deployment that is the actual
question — a chunk can be recalled, ranked, and still reach the grader with the answer
truncated out of it. So each case is scored four times against the SAME retrieval:

    recalled   the candidate pool             — did retrieval find it at all?
    ranked     the top_k the turn keeps       — did ranking keep it?
    grader     format_docs_for_grading(...)   — can the grader SEE it?
    answer     format_docs(...)               — can the answer be written from it?

A case that passes `ranked` and fails `grader` is evidence hidden by the grading view,
not a retrieval failure, and the two want opposite fixes. That distinction is the point
of this harness.

`ranked` is the first `top_k` of the same ordered list rather than a second call, so the
two stages are the same retrieval seen at two depths. With the reranker off — its default
here — that is exactly what the turn does.

## Scoring

ALL of a case's `required` evidence has to be present for that stage to count as a pass,
because a question answered by a number and its year-group row is not answerable with one
of them. `hit` (any requirement satisfied) is reported beside it, since that is what the
older harness measured and the gap between the two is informative on its own.

Figure cases carry no spans — see school_dataset — so they pass when a figure chunk is
retrieved, and the `figure text kept` line reports how much of that chunk's text survives
into the grader's view. That percentage is what the image-chunking work has to move.

Unanswerable cases are listed but not scored: whether the turn correctly says "I don't
know" is decided after grading, not by retrieval, and belongs to the answer-level eval.

## Concurrency

Cases run on a thread pool (`--workers`, 8 by default) because a parent is not the only
one asking: what a single question costs on an idle machine says nothing about what it
costs when twenty arrive together. Running them in parallel measures the contended path —
the embedding forward pass, the Milvus round trip, and any remote reranker — and the
report separates per-question latency from wall-clock throughput so the two cannot be
confused. `--workers 1` restores the sequential run when a clean p50 is what is wanted.

Scoring is pure once the documents are in hand, so a case's result does not depend on how
many ran beside it; only its timing does.
"""
from __future__ import annotations

import argparse
import io
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Before `.env` is read, for the same reason conftest.py does it: a measurement run is not
# a conversation worth recording, and an over-quota LangSmith answers 429 on every call and
# prints the failure between the numbers being measured.
for _tracing in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
    os.environ.setdefault(_tracing, "false")

from backend.env import load_env  # noqa: E402

load_env()

from backend.rag import utils as u  # noqa: E402
from backend.rag.grading_view import format_docs, format_docs_for_grading  # noqa: E402
from tests.evals.school_dataset import (  # noqa: E402
    CORPUS_FILENAME,
    DATASET_VERSION,
    Case,
    cases,
    missing,
)

STAGES = ("recalled", "ranked", "grader", "answer")


def _corpus_is_indexed(filename: str) -> bool:
    """A cheap probe, so an empty index reports itself instead of scoring 0% everywhere."""
    try:
        probe = u.retrieve_documents("school", top_k=25)
    except Exception as exc:  # noqa: BLE001 — the message matters more than the type
        print(f"!! retrieval is not available: {exc}")
        return False
    return any(filename.lower() in str(doc.get("filename", "")).lower() for doc in probe["docs"])


def _figure_docs(docs: list[dict]) -> list[dict]:
    return [d for d in docs if str(d.get("modality", "")) == "figure"]


def _stage_text(case: Case, docs: list[dict], stage: str) -> str:
    if stage in ("recalled", "ranked"):
        return "\n\n".join(str(d.get("text", "")) for d in docs)
    if stage == "grader":
        return format_docs_for_grading(docs)
    return format_docs(docs)


def _passes(case: Case, docs: list[dict], stage: str) -> bool:
    if case.modality == "figure":
        figures = _figure_docs(docs)
        if not figures:
            return False
        # A figure case is about the picture being available at all; at the grader stage it
        # also has to still carry text, which is what the 500-character view takes away.
        return bool(_stage_text(case, figures, stage).strip()) if stage == "grader" else True
    return not missing(case, _stage_text(case, docs, stage))


def _figure_text_kept(docs: list[dict]) -> float | None:
    """How much of a retrieved figure's text the grader still sees, 0.0-1.0."""
    figures = _figure_docs(docs)
    if not figures:
        return None
    whole = sum(len(str(d.get("text", ""))) for d in figures)
    if not whole:
        return None
    return len(format_docs_for_grading(figures)) / whole


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=("dev", "holdout", "all"))
    ap.add_argument("--top-k", type=int, default=None, help="default: the profile/env value")
    ap.add_argument("--pool", type=int, default=30, help="depth scored as 'recalled'")
    ap.add_argument("--verbose", action="store_true", help="name the missing evidence per case")
    ap.add_argument("--workers", type=int, default=8, help="questions in flight at once (1 = sequential)")
    args = ap.parse_args()

    top_k = args.top_k or u.RETRIEVAL_TOP_K
    selected = cases(args.split)
    scored = [c for c in selected if c.kind != "unanswerable"]
    unanswerable = [c for c in selected if c.kind == "unanswerable"]

    print(f"dataset {DATASET_VERSION}  split={args.split}  cases={len(selected)} "
          f"({len(scored)} scored, {len(unanswerable)} unanswerable)")
    print(f"top_k={top_k}  pool={args.pool}  rerank={'ON' if u.RERANK_ENABLED else 'OFF'}  "
          f"workers={args.workers}\n")

    if not _corpus_is_indexed(CORPUS_FILENAME):
        print(f"!! {CORPUS_FILENAME} is not in the index — ingest it before scoring.")
        return 2

    results: dict[str, dict[str, bool]] = {}
    hits: dict[str, bool] = {}
    kept: list[float] = []
    lats: list[float] = []
    errors: dict[str, str] = {}
    ranked_by_case: dict[str, list[dict]] = {}

    def run(case: Case) -> tuple[Case, list[dict], float, str]:
        started = time.perf_counter()
        try:
            docs = u.retrieve_documents(case.question, top_k=args.pool)["docs"]
        except Exception as exc:  # noqa: BLE001 — one bad case must not lose the run
            return case, [], (time.perf_counter() - started) * 1000, f"{type(exc).__name__}: {exc}"
        return case, docs, (time.perf_counter() - started) * 1000, ""

    wall_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        finished = list(pool.map(run, scored))
    wall = time.perf_counter() - wall_started

    for case, pool_docs, elapsed, error in finished:
        lats.append(elapsed)
        ranked = pool_docs[:top_k]
        ranked_by_case[case.id] = ranked
        if error:
            errors[case.id] = error
            results[case.id] = {stage: False for stage in STAGES}
            hits[case.id] = False
            print(f"  ERR   {case.id:<28} {error[:60]}")
            continue

        results[case.id] = {
            "recalled": _passes(case, pool_docs, "recalled"),
            "ranked": _passes(case, ranked, "ranked"),
            "grader": _passes(case, ranked, "grader"),
            "answer": _passes(case, ranked, "answer"),
        }
        hits[case.id] = (
            len(missing(case, _stage_text(case, ranked, "ranked"))) < len(case.required)
            if case.required else results[case.id]["ranked"]
        )
        share = _figure_text_kept(ranked)
        if share is not None and case.modality == "figure":
            kept.append(share)

        marks = "".join("." if results[case.id][s] else "X" for s in STAGES)
        print(f"  {marks}  {case.id:<28} {case.kind:<12} {case.modality}")
        if args.verbose and not results[case.id]["answer"]:
            gaps = missing(case, _stage_text(case, ranked, "answer")) if case.required else ["(no figure retrieved)"]
            print(f"        missing: {gaps}")

    print("\n" + "=" * 70)
    print(f"{'stage':<12}{'all evidence':>16}{'change':>22}")
    previous = None
    for stage in STAGES:
        passed = sum(1 for r in results.values() if r[stage])
        # Signed, because the last stage GAINS: `answer` sees the whole chunk the grader
        # was shown a prefix of, so a positive number there is evidence the grading view
        # hid rather than evidence retrieval lost.
        change = "" if previous is None else f"{passed - previous:+d}"
        print(f"{stage:<12}{passed}/{len(scored)} ({passed / len(scored):.0%})".ljust(28) + f"{change:>22}")
        previous = passed

    any_hit = sum(1 for v in hits.values() if v)
    print(f"\nany evidence at ranked: {any_hit}/{len(scored)} ({any_hit / len(scored):.0%}) "
          f"— the gap to 'all evidence' is what a hit-rate score hides")

    for label, attr in (("kind", "kind"), ("modality", "modality")):
        print(f"\nby {label}:")
        groups = sorted({getattr(c, attr) for c in scored})
        for group in groups:
            members = [c for c in scored if getattr(c, attr) == group]
            row = "  ".join(
                f"{s}={sum(1 for c in members if results[c.id][s])}/{len(members)}" for s in STAGES
            )
            print(f"  {group:<12} {row}")

    if kept:
        print(f"\nfigure text kept in the grader's view: {statistics.mean(kept):.0%} "
              f"(min {min(kept):.0%})")

    def pct(values: list[float], share: float) -> float:
        return values[min(int(len(values) * share), len(values) - 1)]

    lats.sort()
    print(f"\nretrieval latency per question  p50={round(pct(lats, 0.5))}ms  "
          f"p95={round(pct(lats, 0.95))}ms  p99={round(pct(lats, 0.99))}ms  max={round(lats[-1])}ms")
    print(f"wall clock  {wall:.1f}s for {len(scored)} questions at {args.workers} worker(s) "
          f"= {len(scored) / wall:.1f} q/s  (sum of per-question time {sum(lats) / 1000:.1f}s)")
    if errors:
        print(f"!! {len(errors)} question(s) failed to retrieve: "
              f"{', '.join(f'{cid} ({err[:40]})' for cid, err in errors.items())}")

    # What each stage would COST a model, which is the input half of the grader's latency.
    grader_chars = [len(format_docs_for_grading(d)) for d in ranked_by_case.values() if d]
    answer_chars = [len(format_docs(d)) for d in ranked_by_case.values() if d]
    if grader_chars:
        grader_chars.sort()
        answer_chars.sort()
        print(f"prompt size  grader p50={round(pct(grader_chars, 0.5))} max={grader_chars[-1]} chars  |  "
              f"answer p50={round(pct(answer_chars, 0.5))} max={answer_chars[-1]} chars")

    for stage in STAGES:
        failed = [cid for cid, r in results.items() if not r[stage]]
        if failed:
            print(f"\nfailing at {stage}: {', '.join(failed)}")

    if unanswerable:
        print(f"\nnot scored here (answer-level eval decides them): "
              f"{', '.join(c.id for c in unanswerable)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
