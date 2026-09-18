---
noteId: "37e8a9a0b37b11f1966c69e3652709b5"
tags: []

---

# RAG fix plan

The retrieval pipeline (`backend/rag/`, `backend/indexing/`) is being fixed in phases, each
on its own branch and each reviewed before the next begins. This file is the working
agreement: what is agreed, what is measured, what is done, and what a session picks up next.

It exists because the plan changed twice after real measurement, and both changes reversed
what the original review recommended. Anyone continuing this work needs to know *why* the
order is what it is, or they will re-derive the wrong priority from `RAG_BACKEND_REVIEW.md`.

Companion documents:

- `RAG_BACKEND_REVIEW.md` — the original code review. Still accurate about mechanisms;
  **its priority order is superseded by the measurements below.** See "Corrections".
- `BACKEND_REFACTOR.md` — the finished backend refactor, and the conventions this work
  follows (one step per branch, behaviour preserved unless a fix is named and measured).

## How to use this file

1. Read "Where things stand" and "Rules".
2. Take the next phase from "The plan" — one phase, one branch.
3. Measure before and after with the evals named in "How to measure".
4. Update this file in the same branch: tick the phase, record the numbers, note anything
   the work proved wrong.

---

## Where things stand

| Branch | State | What is on it |
|---|---|---|
| `main` | merged through PR #100 | School is the only profile; no persistent note; no products tool |
| `eval/school-retrieval-dataset` | 2 commits, **not pushed** | Phase 0: the labelled Arabic dataset, the stage-by-stage retrieval eval, tracing off in tests |
| `rag/grader-optional` | 3 commits on top of the above, **not pushed, parked** | `grading_mode: never` made real, the answer-level eval, and the grader-on/off measurement |

**Do this first:** push `eval/school-retrieval-dataset`, open its PR, merge it. Every later
phase is judged with the evals it carries, so it belongs on `main` before anything else
lands. `rag/grader-optional` stays parked by the owner's decision — the grader question is
deferred, not settled (see Phase 2).

### The local environment a session inherits

- **The knowledge base is `Aurexis_Knowledge_Base_Mock_Egypt.docx`** — an English school KB,
  gitignored, at the repo root. Local Milvus holds only this: 153 leaves (139 text, 5 table,
  9 figure), 50 parents. BHCR and GS1 were removed on 2026-09-18.
  Re-ingest mirrors `backend/api/routes/documents.py::upload_document`: remove, load, upsert
  parents, write leaves.
- **The provider is Groq** (`LLM_PROVIDER=groq` in `.env`). Together's
  `openai/gpt-oss-20b` returns `400 model_not_available` — it is no longer serverless — so
  the previous setting is dead. Vision is pinned to Together explicitly on the generic
  `VISION_*` names, because `GROQ_VISION_MODEL` is empty and vision does not follow the
  provider switch.
- **Groq's free tier is 8,000 tokens per minute.** One school turn spends several thousand
  across classifier, resolver, grader and answer. Any live eval must run sequentially, and
  **latency measured under that cap is not a latency measurement.**
- **Production is NOT switched.** The server's `.env` is written only to bootstrap an empty
  estate, so it still names Together and the dead model — production model calls are
  presumed failing. Fixing it is `bash deploy/scripts/apply-env.sh backend` (DEVOPS.md).
  This is outside the plan and more urgent than it.
- **Test baseline on `main`:** `tests/general` has 3 failures unrelated to this work —
  `test_backend_auth`, `test_integration_cache_and_retrieval`, `test_serving_scalability`.
  Compare against that, not against zero.

---

## How we got here

**1. The review.** `RAG_BACKEND_REVIEW.md` listed 23 findings, read out of the code. Its
top priority was image evidence: transcriptions capped at ~3,200 characters at indexing,
and the grader shown only 500 characters of a figure.

**2. The review was corrected in discussion.** Re-reading the code against it found: the
"empty result skips recovery" finding is two different code paths with different fixes; the
reranker can end a turn on a bare score, which connects two separate findings into one
mechanism; and `policy.py` reports the capped count instead of the real one. Items 24 and 25
were added later, from measurement.

**3. Git archaeology answered "why did this get worse".** The owner remembered the system
grading whole chunks, 8 at a time, and was right. Until 2026-09-12 the grader received every
retrieved chunk in full (`_format_docs`), `top_k` was 8, and every leaf — figure text
included — went through the splitter at ≤800 characters. Then, in two hours:

| Time | Commit | Effect |
|---|---|---|
| 15:28 | `8b4e367` | A figure became one atomic chunk: no splitter, so a transcribed calendar reached 60 KB |
| 16:56 | `96a1610` | Figure leaves capped at 3,200 characters — evidence lost at **indexing** |
| 17:23 | `1794762` | `grading_view.py` added: 1,200 chars prose / 500 figure — evidence hidden at **grading**. Plus the 1,536-token grade ceiling and the truncation retry |

The size problem was treated twice and the cause — one image producing one enormous chunk —
was never fixed. `top_k: 4` is not in the code at all: profiles and `.env.example` say 8; only
the local `.env` overrides it.

**4. Measurement reversed the priority.** Phase 0 built a 220-question Arabic dataset against
the Aurexis KB and scored each question at four depths of the same retrieval. The images are
fine on this corpus; the grading view is the biggest single loss. See below.

**5. The grader question was opened, measured, and parked.** Removing the grader looked
compelling (it sees less than the answer model, and costs the turn's dominant call), but a
live comparison found the failure that matters is unaffected by it. Deferred by the owner.

---

## What was measured

### Retrieval, 176 scored questions, `tests/evals/school_retrieval_eval.py`

| Stage | All required evidence present | Change |
|---|---|---|
| recalled (pool of 30) | 167/176 — 95% | |
| ranked (kept `top_k` = 4) | 149/176 — 85% | −18 |
| **grader's view** | **99/176 — 56%** | **−50** |
| answer's view | 149/176 — 85% | +50 |

- **28% of questions have their evidence hidden from the grader**, not lost by retrieval —
  the `+50` on the last row is the same evidence returning when the whole chunk is read.
- **Cause**: auto-merged parents are 1,600–2,400 characters against a 1,200-character cap.
  Confirmed by inspecting the chunks (`class-capacity`: "18 students" present in the answer
  view, absent from the grader's).
- **Images are not the problem on this corpus**: figure questions score 4/4 at every stage,
  and the largest leaf is 1,170 characters, so the 3,200 cap never binds. It binds on
  calendar-shaped documents, which is what the production incident involved.
- **Cutting to `top_k` 4 costs 18 questions (10%).**
- Concurrency (`--workers`): 1 worker p50 109 ms, 9.0 q/s; 8 workers p50 513 ms, 14.1 q/s —
  4.7× the latency for 1.6× the throughput, because the embedding pass is local and CPU-bound.
- Prompt sizes: grader p50 3,662 chars (max 4,788); answer p50 5,527 (max 8,896).

### Grader on vs off, live turns, `tests/evals/school_answer_eval.py` (Groq)

Small sample — 6 questions can be scored for a numeric fact, 9 are unanswerable. Treat
one-case differences as noise.

| | grader on | grader off |
|---|---|---|
| answered | 5/6 | 4/6 |
| correct fact | 4/6 | 3/6 |
| **wrong fact** | **1/6** | **1/6** |
| refused when it should | 8/9 | 9/9 |
| latency p50 / mean | 2.1 s / 3.3 s | 1.5 s / 1.5 s |

Three defects this exposed, all real and none of them the grader's presence or absence:

1. **The grader fails on its own output**: live `400 json_validate_failed`, JSON cut off
   mid-`reason`, because the free-text `reason` plus reasoning tokens exhausts the 1,536-token
   ceiling. When it fails the turn becomes `retrieval_error` on a question the chunks answer.
2. **One graded turn hung for 3,320 seconds** before returning. There is no deadline anywhere.
3. **Both modes answered "Year 3 fees = 88,000 EGP"** — the FS1–FS2 row; the right answer is
   105,000. A confidently wrong fee is worse than any fallback, and grading neither caused
   nor prevented it.

---

## The plan

Item numbers are the review's, so they can be traced back. Status: ✅ done, ⏳ next, ⬜ later,
⏸ parked. **Phase numbers are fixed** — Phase 2 is parked rather than renumbered, so that
"Phase 4" means the same thing here as in the conversation this plan came from.

| Phase | Items | Status |
|---|---|---|
| 0 — Measure first | 22, 23 | ✅ done |
| 1 — Root cause | 1, 2, 5, 8, **24**, **25** | ⏳ next |
| 2 — Grader latency | 9, 14 | ⏸ parked by the owner |
| 3 — Wrong answers | 3, 4, 6, 7, **26** | ⬜ |
| 4 — Ranking and config | 10, 11, 12, 13 | ⬜ (12 already fixed on the parked branch) |
| 5 — Ingestion | 15, 16, 17 | ⬜ |
| 6 — Infrastructure | 18, 19, 20, 21 | ⬜ |

Items not in the original 23, added from measurement:

- **24** — put `RETRIEVAL_TOP_K` back to 8: splitting images makes 4 slots too few, and 4
  costs 18 of 176 questions today.
- **25** — cap how many of those slots one document or image may take, so one calendar
  cannot fill all of them.
- **26** — a local check that every number in the answer appears in the evidence it cited.
  Raised after both grader modes answered "Year 3 fees = 88,000 EGP".

### Phase 0 — measure first ✅ (branch `eval/school-retrieval-dataset`)

| # | What | State |
|---|---|---|
| 22 | 220 Egyptian-Arabic questions against the Aurexis KB, a fifth held out, scored at four stages, with the oracle itself tested in CI | ✅ |
| 23 | Tracing off in tests and evals (it also caused 6 spurious failures) | ✅ |

### Phase 1 — where the evidence is actually lost ⏳ NEXT

One branch. Measure with `school_retrieval_eval.py` before and after; the target is the
`grader` row rising toward the `answer` row, and `ranked` rising toward `recalled`.

| # | What | Acceptance |
|---|---|---|
| 2 | Stop truncating chunks for the grader, or select question-relevant spans instead of a prefix. The grading prompt must stay bounded — that is what `1794762` was protecting | `grader` stage within a few points of `answer` |
| 8 | Rank child chunks first, then expand; keep the match offset and send the text around it instead of the whole parent | No chunk over the grading budget reaches the grader |
| 24 | `RETRIEVAL_TOP_K` back to 8 (remove the local `.env` override; check the server's) | `ranked` ≈ `recalled` |
| 25 | Cap how many of the final slots one document or image may take | No regression in `ranked` |
| 5 | Select final evidence under a token budget **before** judging sufficiency; never cut an approved set by count. Fix the trace that reports the capped number as the real one | Approved evidence always reaches the answer |
| 1 | Split image transcriptions into ≤800-char passages (tables as row groups, header repeated, shared `asset_id`), then reindex | Figure questions stay 4/4; needed for calendar-shaped documents even though this corpus does not bind |

### Phase 2 — the 2-second grader (items 9, 14) ⏸ PARKED

Parked by the owner, with the work already specified. Nothing here is started; the switch
built while measuring it (`grading_mode: never`, default unchanged) sits on
`rag/grader-optional`, unpushed.

**Take one thing off the table first: the prompt is not what makes it slow.** The grader
receives about 4,800 characters today and is still slow. The time goes into the tokens it
*writes*, one after another, while the prompt is read in one parallel pass.

Where the ~2 seconds has to come from, in order of impact:

1. **Stop paying for reasoning tokens — the single biggest lever.** `gpt-oss-20b` is a
   reasoning model; its scratchpad is written before the JSON and billed against the same
   output budget. A few hundred scratchpad tokens are seconds. `low` is already the minimum
   that parameter offers, and `none` makes it *worse* — it omits the field, so the provider
   applies its own higher default, which commit `1794762` pinned with a test. The only real
   fix is a **non-reasoning instruct model** for this role on the same endpoint. Shortlist
   the provider's small instruct models that support structured output and measure them on
   the same evidence sets; do not take a model name on trust.
2. **Shrink what it writes.** The schema returns relevance, answerability, ambiguity, route,
   confidence, missing slots, supporting chunks, a free-text `reason`, and sometimes a
   question to ask the user. Cut it to what the code actually branches on: a verdict
   (`sufficient | partial | uncertain`), the supporting chunk numbers, and missing facts.
   Drop the free-text `reason`; take clarification wording from profile copy instead of
   generating it. Aim under ~80 output tokens. *(Measured since: the free-text `reason` is
   what truncates the JSON — live `400 json_validate_failed` on Groq.)*
3. **Never retry a big prompt.** A cut-off reply is retried today with double the ceiling,
   turning one slow call into two. Set the ceiling from the measured p99 output length, and
   on truncation answer from the evidence instead of paying again.
4. **Put a hard deadline on the call (~2.5 s)**, and proceed with the retrieved evidence
   when it passes. *(Measured since: one graded turn hung for 3,320 seconds. There is no
   deadline anywhere.)*
5. **Skip the call when the evidence is already clear** (item 14). The fastest grading is
   none: this moves the median, while 1–4 move the p95.
6. **Keep the prompt prefix stable and the client warm.** Instructions and schema first,
   evidence last, so provider caching can apply; confirm nothing builds an HTTP client per
   call (the model factory already caches them).

The budget that adds up to ~2 s, once reasoning is gone:

| Part | Expected |
|---|---|
| Network + queue | 0.1–0.3 s |
| Reading the prompt (~2k tokens, parallel) | 0.05–0.15 s |
| Writing ~80 output tokens (serial) | 0.4–0.8 s |
| **Total** | **~0.6–1.3 s typical, under 2 s at p95** |

That holds only with zero reasoning tokens and no retry. With reasoning left on, no amount
of prompt trimming reaches 2 seconds.

**Measure before changing the model** (item 9 — a prerequisite, not paperwork): log input
tokens, output tokens, reasoning tokens, time, finish reason and retry count for every
grading call. Two or three real turns show immediately whether the scratchpad or the retry
is the cost. Arabic inflates token counts, so measure on Arabic questions.

**Caveat on the 44 seconds**: grading is one call in a turn that also pays for
classification, follow-up resolution, retrieval and the answer. A 2-second grader does not
by itself make the turn fast — the same instrumentation should record every call's share,
so the next fix targets the real remainder.

### Phase 3 — wrong answers (items 3, 4, 6, 7, 26) ⬜

| # | What | Why |
|---|---|---|
| 26 | A local check: every number in the answer must appear in the evidence it cited. No model call | Would have caught "88,000" for Year 3. The highest-value fix found so far |
| 3 | An empty result runs one bounded recovery search before "no knowledge" | |
| 4 | A reranker score may order or admit, never end a turn | Enabling the reranker today arms the fallback path |
| 6 | Join tables across pages only on matching headers, section and layout | |
| 7 | Keep the first pass's evidence when a rewrite's retrieval fails | |

### Phase 4 — ranking and configuration (items 10, 11, 12, 13) ⬜

Item 10 (rerank the 30 candidates, not the final 4), 11 (relevance is not sufficiency —
check required facts separately), 13 (score scales are not interchangeable and the
conversion is not monotonic). **Item 12 is already done** on `rag/grader-optional`:
`grading_mode` was a dead switch read by a helper with no caller, and that helper is gone.

### Phase 5 — ingestion ⬜

Items 15 (visible partial-ingest status and a targeted retry), 16 (Arabic/English sentence
splitting; measure tokens instead of estimating four characters each), 17 (cache embeddings
and extractions by content hash).

### Phase 6 — retrieval and conversation infrastructure ⬜

Items 18 (cache retrieval by corpus version, language, constraints, access scope), 19 (load a
bounded history window, trimmed by tokens), 20 (persist the resolver's carried conditions —
the gap accepted when the persistent note was removed), 21 (durable background writes).

---

## How to measure

```bash
# Retrieval, four stages. Needs Milvus up and the Aurexis KB indexed.
.venv/Scripts/python.exe tests/evals/school_retrieval_eval.py --verbose
.venv/Scripts/python.exe tests/evals/school_retrieval_eval.py --workers 1   # clean p50
.venv/Scripts/python.exe tests/evals/school_retrieval_eval.py --split holdout  # only to confirm

# Live turns. Spends real model calls; sequential because of the Groq cap.
.venv/Scripts/python.exe tests/evals/school_answer_eval.py --limit 10 --workers 1

# The suites that guard this area
.venv/Scripts/python.exe -m pytest -q -p no:randomly tests/general
```

Record the four-stage table in this file with every phase. A phase that changes answers
without a before-and-after is not finished.

---

## Rules

- **One phase, one branch, one PR.** Wait for the owner between phases.
- **Measure anything that changes an answer** against `main`, with the evals above. Internals
  are free; observable behaviour changes need a named bug or a measured gain.
- **Never touch production.** No deploys, no server `.env` edits, no merges without the owner.
- **The corpus is the KB, not the chunks.** Questions are written from what a parent asks;
  chunking serves the questions, not the reverse.
- **The dataset is an oracle** — `tests/general/test_school_eval_dataset.py` guards it. If a
  gold span is wrong, fix the span and say so; do not loosen the check to make a run pass.
- Commit messages: what changed, why, and the measurement. Follow the repo's existing style.

## Corrections to `RAG_BACKEND_REVIEW.md`

Recorded so the next reader does not restore the wrong order:

1. **Image splitting is not the first fix for this corpus.** The review's P1 ranking came from
   the production calendar incident. On the Aurexis KB, figure questions pass every stage and
   the largest leaf is 1,170 characters. The grading view and parent merging cost 28%.
2. **"Empty or irrelevant results bypass recovery" is two findings.** `not has_docs` is a real
   gap; `relevance == "none"` is a deliberate, documented decision — and the reranker can
   trigger it from a bare score, which is the dangerous half.
3. **The grading view was a deliberate trade**, not an oversight: it bounds the grading prompt
   against corpus size. Any fix must keep that property.
4. **`policy.py` misreports its own cap** (the count is read after the slice) — not in the review.
5. **The memory findings are resolved**: the persistent note is gone (PR #100). What remains is
   item 20, the resolver's 6-message window.
