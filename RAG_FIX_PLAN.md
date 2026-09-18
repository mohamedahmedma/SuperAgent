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
2. Take the next phase from "The plan" — one phase, one branch. Every finding and its fix
   is spelled out in "Reference: every finding, and the fix for each", so nothing here
   depends on the conversation this came from.
3. Measure before and after with the evals named in "How to measure".
4. Update this file in the same branch: tick the phase, record the numbers, note anything
   the work proved wrong.

---

## Where things stand

`main` carries Phase 0 (PR #101): the labelled Arabic dataset, the stage-by-stage retrieval
eval, and tracing off in tests. Branch Phase 1 from it.

**The grader is out of scope.** Whether it stays, shrinks, or goes is decided *after* these
fixes land, on the numbers they produce — nothing in this plan depends on the answer, and no
phase here changes it. A branch called `rag/grader-optional` exists from that earlier
investigation; leave it alone.

### The local environment a session inherits

- **The knowledge base is `Aurexis_Knowledge_Base_Mock_Egypt.docx`** — an English school KB,
  gitignored, at the repo root. Local Milvus holds only this: 153 leaves (139 text, 5 table,
  9 figure), 50 parents.
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

**5. The grader question was deferred.** It is a decision to take on the numbers these
fixes produce, not before them. The grader stays on and unchanged meanwhile.

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

### From live turns, `tests/evals/school_answer_eval.py`

Three defects seen on real turns, kept here because each is a finding in its own right:

1. **A grading call can fail on its own output**: live `400 json_validate_failed`, the JSON
   cut off mid-`reason`. The turn then becomes `retrieval_error` on a question the chunks
   answer. (Item 9.)
2. **One turn hung for 3,320 seconds** before returning. There is no deadline anywhere.
   (Item 9.)
3. **The assistant answered "Year 3 fees = 88,000 EGP"** — the FS1–FS2 row; the right answer
   is 105,000. A confidently wrong fee is worse than any fallback, and it is a retrieval and
   answer-side failure, not a grading one. (Item 26.)

---

## The plan

Item numbers are the review's, so they can be traced back. Status: ✅ done, ⏳ next, ⬜ later,
⏸ out of scope. **Phase numbers are fixed** — Phase 2 keeps its number rather than the
others shifting up, so "Phase 4" means the same thing here as everywhere else.

| Phase | Items | Status |
|---|---|---|
| 0 — Measure first | 22, 23 | ✅ done |
| 1 — Root cause | 1, 2, 5, 8, **24**, **25** | ⏳ next |
| 2 — Grader | 9, 14 | ⏸ out of scope — decided after the fixes |
| 3 — Wrong answers | 3, 4, 6, 7, **26** | ⬜ |
| 4 — Ranking and config | 10, 11, 12, 13 | ⬜ |
| 5 — Ingestion | 15, 16, 17 | ⬜ |
| 6 — Infrastructure | 18, 19, 20, 21 | ⬜ |

Items not in the original 23, added from measurement:

- **24** — put `RETRIEVAL_TOP_K` back to 8: splitting images makes 4 slots too few, and 4
  costs 18 of 176 questions today.
- **25** — cap how many of those slots one document or image may take, so one calendar
  cannot fill all of them.
- **26** — a local check that every number in the answer appears in the evidence it cited.
  Raised after both grader modes answered "Year 3 fees = 88,000 EGP".

### Phase 0 — measure first ✅ (merged, PR #101)

| # | What | State |
|---|---|---|
| 22 | 220 Egyptian-Arabic questions against the Aurexis KB, a fifth held out, scored at four stages, with the oracle itself tested in CI | ✅ |
| 23 | Tracing off in tests and evals (it also caused 6 spurious failures) | ✅ |

### Phase 1 — where the evidence is actually lost ⏳ NEXT

One branch, in this order. Measure with `school_retrieval_eval.py` before and after; the
target is the `grader` row rising toward the `answer` row, and `ranked` rising toward
`recalled`. Reindex after step 1.

| Order | # | What to do | Acceptance |
|---|---|---|---|
| 1 | 1 | **Split image transcriptions into ≤800-character passages.** Tables become row groups with the header repeated in each. Every passage keeps a shared `asset_id`, so the picture is still cited and displayed once. Reindex afterwards | Figure questions stay 4/4 |
| 2 | 8 | **Limit what parent merging can return.** Two matching children still pull in a 1,600–2,400-character parent, which reintroduces the same oversized chunk. Keep the position of the match and send the text around it, not the whole parent | No chunk over the grading budget reaches the grader |
| 3 | 2 | **Delete the grader truncation** (`grading_view.py`), so the grader reads whole chunks as it originally did. Safe only once **no oversized chunk can reach it** — which is steps 1 and 2 together, not step 1 alone. That bound is what commit `1794762` was protecting and it must still hold | `grader` stage within a few points of `answer` |
| 4 | 24 | **`RETRIEVAL_TOP_K` back to 8** — remove the override from `.env` (local, and check the server's). 8 × 800 characters is 6,400, the size the system originally graded successfully | `ranked` ≈ `recalled` |
| 5 | 25 | **Cap how many of the final 8 slots one document or image may take.** New risk, and it only appears once one calendar can produce ten passages | No regression in `ranked` |
| 6 | 5 | **Fix the evidence cap**: select the final evidence under a token budget *before* judging sufficiency, so the grader approves exactly what the answer receives and approved chunks are never cut by count. Also fix the trace that reports "4 of 4" | Approved evidence always reaches the answer |

Why images are first even though they do not bind on this corpus: the split is what makes
step 3 safe for a calendar-shaped document, and doing it after would mean removing the
truncation twice — once safely here, once again the first time a real calendar is ingested.

### Phase 2 — grader (items 9, 14) ⏸ OUT OF SCOPE

The grader stays ON and unchanged while the fixes land. Whether it then keeps a deadline
and a smaller schema, gets skipped when the evidence is clear, or goes entirely is a
decision to take afterwards, on the numbers Phases 1 and 3 produce. Nothing in this plan
depends on the answer. Do not start it, and do not measure it, as part of these phases.

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
conversion is not monotonic), 12 (`grading_mode` is a dead switch read by a helper with no
caller — a fix for it exists on the unmerged `rag/grader-optional`, so check there before
writing it again).

### Phase 5 — ingestion ⬜

Items 15 (visible partial-ingest status and a targeted retry), 16 (Arabic/English sentence
splitting; measure tokens instead of estimating four characters each), 17 (cache embeddings
and extractions by content hash).

### Phase 6 — retrieval and conversation infrastructure ⬜

Items 18 (cache retrieval by corpus version, language, constraints, access scope), 19 (load a
bounded history window, trimmed by tokens), 20 (persist the resolver's carried conditions —
the gap accepted when the persistent note was removed), 21 (durable background writes).

---

## Reference: every finding, and the fix for each

The whole list in one place, so no session has to reconstruct it from the review plus a
chat log. Numbering is the review's. Status: ✅ done, ⏸ out of scope, ⬜ open.

### Evidence lost or wrong (originally P1)

| # | What is wrong | The fix | Phase |
|---|---|---|---|
| 1 | Image text is cut to about 3,200 characters before indexing, so facts near the end cannot be found. The cap also does not apply to the first line, so a single very long line passes through whole | Keep the full text in the asset store and index it as ≤800-character passages, or table-row groups with the header repeated, all sharing one `asset_id`. Caption and description stay a separate entry for discovery. Enforce the limit on every line. Reindex | 1 ⬜ |
| 2 | The grader sees only the start of each chunk — 1,200 characters, or 500 for a figure — so it can reject a chunk whose answer it never saw | Once no oversized chunk can reach it (items 1 and 8), delete the truncation and let it read whole chunks. The grading prompt must stay bounded by the retrieval, not by the corpus | 1 ⬜ |
| 3 | An empty search result goes straight to "no knowledge" without using the one allowed retry | One bounded recovery search first: wider pool, keyword-only variants, neighbouring passages, same access and language filters. Keep technical failure separate from "not found" | 3 ⬜ |
| 4 | The reranker can deny an answer on its own score: below its floor the turn ends as "no knowledge", with no grader and no retry. Enabling the reranker switches this path on | A score may order or admit, never end a turn. A low score hands over to the grader or the recovery search. Only something that read the chunks may call them off-subject | 3 ⬜ |
| 5 | The answer can lose evidence the grader approved: the kept set is capped at 4, so an approved 8 after a retry is cut to 4. The trace then reports the count *after* the cap, so it always reads "4 of 4" | Select the final evidence under a token budget before judging sufficiency, so the grader approves exactly what the answer receives. Never cut an approved set by count; if it does not fit, answer partially and say so. Record the count before the cap | 1 ⬜ |
| 6 | Tables on neighbouring pages are joined whenever their column counts match, even when they are different tables | Require matching normalised headers, the same section, and layout continuity (ends at the page foot, resumes at the head). When unsure keep them separate, and keep the source page per row group | 3 ⬜ |
| 7 | When the retry search fails, the first search's evidence is thrown away, even when it could have supported a partial answer | Keep the first pass's on-subject evidence and answer partially from it, naming what is missing. Do not invent the rest | 3 ⬜ |
| 8 | Parent-chunk merging happens before ranking, so the matched passage can land beyond what the grader or reranker reads | Rank the children first, then expand. Keep the match offset and send the text around it; grow within a budget instead of promoting a whole parent on two matching children | 1 ⬜ |

### Grader, ranking and cost (originally P2)

| # | What is wrong | The fix | Phase |
|---|---|---|---|
| 9 | Grading has no overall time limit, and a cut-off reply is retried with a larger allowance. No timings, token counts or retry counts are recorded | Instrument first: input, output and reasoning tokens, time, finish reason, retries, per call. Then a turn deadline and explicit HTTP timeouts; never resend a large prompt past the deadline. Keep the 1,536-token floor — that was a separate, real bug | 2 ⏸ |
| 10 | The local reranker only sees the final 4 chunks, not the 30 candidates, so it cannot recover a good chunk dropped earlier | Rerank the candidate pool, then select the answer set. Separate budgets for retrieval, reranking and answering. Measure p95 on the real hardware before enabling | 4 ⬜ |
| 11 | The reranker treats "relevant" as "enough to answer", so an on-topic passage passes without the requested fact | Scores order evidence. Check required facts (a year, a fee, a date) separately, and reserve the grader for uncertain or conflicting evidence. Calibrate on Arabic and English questions from this corpus | 4 ⬜ |
| 12 | `grading_mode` does nothing: it was read by a helper with no production caller while routing used a different setting | One setting, read in one place, and the dead `should_grade` helper removed. A fix exists on the unmerged `rag/grader-optional` | 4 ⬜ |
| 13 | Two latent score problems: the minimum-score check can compare against the retrieval score, which is a different scale, and the score conversion is not monotonic | Make each score type explicit and compare thresholds only against their own kind. Apply one documented monotonic conversion | 4 ⬜ |
| 14 | There is no fast path: every answerable question pays for the grader, even when the evidence is clear | Answer in one call when hybrid retrieval plus local checks already cover the question; grade only ambiguous, conflicting or incomplete evidence. Compare against `main` on the dataset before shipping | 2 ⏸ |

### Ingestion and chunking (originally P2)

| # | What is wrong | The fix | Phase |
|---|---|---|---|
| 15 | A failed image extraction silently indexes text only, with no visible status and no targeted retry | Record per document: complete, partial, or failed images. Show it to operators and retry only the failed images | 5 ⬜ |
| 16 | The splitter's separators suit CJK punctuation, not English or Arabic, and token budgets are estimated at four characters per token | Split on English and Arabic sentence punctuation; keep tables as row groups; measure with the tokenizer of the models actually in use; choose sizes from retrieval outcomes | 5 ⬜ |
| 17 | Unchanged documents are re-embedded on reindex, with no cache keyed by content hash | Cache embeddings by content hash plus embedding version, and extractions by image hash plus extractor version. Track text and metadata hashes separately so a metadata-only edit re-embeds nothing | 5 ⬜ |

### Retrieval and conversation infrastructure (originally P2)

| # | What is wrong | The fix | Phase |
|---|---|---|---|
| 18 | Search results are not cached across requests; only the last 64 query embeddings are, per process | Cache in Redis keyed by resolved query, carried conditions, language, corpus version, retrieval settings and access scope. Bump the corpus version on ingest or delete | 6 ⬜ |
| 19 | The whole conversation is loaded every turn, then trimmed by message count rather than tokens | Load a bounded recent window from the database and trim by a token budget | 6 ⬜ |
| 20 | The conditions the resolver carries forward (campus, document, year) come only from the last 6 messages, so older ones are lost | Persist them in session metadata beside the child pin, with the turn they came from and an expiry; feed them back to the resolver; let newer statements replace older ones. **This is the gap accepted when the persistent note was removed** | 6 ⬜ |
| 21 | Background saves are lost if the process crashes, and their ordering only holds on one worker | Store the turn before replying; move later work to a durable queue (outbox or Redis); add a session version so concurrent writers cannot overwrite each other. Not RAG — sequence it separately | 6 ⬜ |

### Evaluation

| # | What is wrong | The fix | Phase |
|---|---|---|---|
| 22 | The retrieval eval uses GS1 barcode questions, not school content, and reports the first matching snippet — a hit rate, not proof the required evidence arrived | A versioned labelled school set with required evidence spans, Arabic questions, images, tables, follow-ups, comparisons and unanswerables, scored per pipeline stage, with a holdout | 0 ✅ |
| 23 | Local test runs send traces to LangSmith, which also causes six spurious failures | `conftest.py` and the evals set `LANGSMITH_TRACING=false` / `LANGCHAIN_TRACING_V2=false` unless the shell says otherwise | 0 ✅ |

### Added from measurement

| # | What is wrong | The fix | Phase |
|---|---|---|---|
| 24 | `top_k` is 4, set only by the local `.env`; code and profiles say 8. It costs 18 of 176 questions, and splitting images makes 4 slots worse still — one calendar becomes ten passages competing for four slots | Put it back to 8. 8 × 800 characters is the size that originally graded successfully | 1 ⬜ |
| 25 | Nothing stops one document or image taking every final slot | Cap the slots one source may occupy | 1 ⬜ |
| 26 | Both grader modes answered "Year 3 fees = 88,000 EGP" — the FS1–FS2 row. Grading neither caused nor prevented it | A local check: every number in the answer must appear in the evidence it cited. No model call | 3 ⬜ |

---

## How to measure

```bash
# Retrieval, four stages. Needs Milvus up and the Aurexis KB indexed.
.venv/Scripts/python.exe tests/evals/school_retrieval_eval.py --verbose
.venv/Scripts/python.exe tests/evals/school_retrieval_eval.py --workers 1   # clean p50
.venv/Scripts/python.exe tests/evals/school_retrieval_eval.py --split holdout  # only to confirm

# Live turns — for Phase 3, where the answer's own correctness is what is being fixed.
# Spends real model calls; sequential because of the Groq cap. NOT on main: the script is
# on the unmerged rag/grader-optional, so cherry-pick it when Phase 3 starts.
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
