"""Answer quality against the live model: a score for the four defects, not a verdict.

`tests/general/` proves the mechanisms with a scripted model — deterministic, free, and a
merge gate. This is the other question, and pytest cannot answer it: with the REAL model
on the REAL provider, how often does a parent get a good answer?

That has to be sampled rather than asserted. `openai/gpt-oss-20b` at temperature 0.2
leaked its reasoning into `content` on 4 of 5 identical calls when this was written; a
single run proves nothing either way, and a pass/fail gate built on one sample would fail
the build at random. So this reports rates over N runs and never returns non-zero for a
quality score. It is the same contract `retrieval_eval.py` states: a number to watch, not
a build to break.

Run:
    .venv/Scripts/python.exe tests/evals/answer_quality_eval.py
    .venv/Scripts/python.exe tests/evals/answer_quality_eval.py --runs 5
    .venv/Scripts/python.exe tests/evals/answer_quality_eval.py --scenario fees-year-1
    .venv/Scripts/python.exe tests/evals/answer_quality_eval.py --no-judge   # checks only

## What is checked deterministically, and what is judged

Both, deliberately, because they answer different questions and the cheap one is the
one that must never be delegated:

  CHECKED   transcript markers in the answer; every figure present in the corpus that
            was actually handed over; citation indices inside range; how many searches
            ran. These are decidable, so a model is not asked — a judge that
            "assessed" them would be an expensive way to introduce error.

  JUDGED    whether the answer is about the right child's year, whether it hedges when
            the material was sufficient, whether it reads as reasoning rather than as
            an answer. These need reading comprehension.

The judge is prompted to quote the span it is judging before it rules, so a verdict can
be checked against the answer by hand. Its own failures are reported as INVALID rather
than folded into the score: a judge that errored is not evidence that the answer was
good, and a harness that scores it as a pass would report the opposite of the truth.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# `load_env`, not a bare `load_dotenv`: this deployment selects its provider with
# `LLM_PROVIDER` and collapses that block onto the generic `MODEL` / `BASE_URL` /
# `ARK_API_KEY` names inside `load_env`. Reading the .env without that step leaves the
# generic names unset and the eval reports "nothing to evaluate" on a machine that is
# configured perfectly well. See backend/llm_provider.py.
from backend.env import load_env

load_env()

from langchain.agents import create_agent  # noqa: E402
from langchain.chat_models import init_chat_model  # noqa: E402
from langchain_core.messages import (  # noqa: E402
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool  # noqa: E402

from backend.chat.finalize import Finalizer  # noqa: E402
from backend.chat.grounding import verify as verify_grounding  # noqa: E402
from backend.chat.model_output import TOKENS as HARMONY_TOKENS  # noqa: E402
from backend.prompts import render as render_prompt, resolve as resolve_prompt  # noqa: E402
import backend.chat.runtime as runtime  # noqa: E402

_RETRY_COPY = "__retry__"  # stands in for user_copy.retrieval_error; only its presence is scored

YEAR_1 = "الصف الأول الابتدائي"

# The school's material. Every figure a correct answer may state comes from here, which
# is what lets the grounding check below be an oracle rather than an opinion.
FEE_TABLE = [
    "رسوم الصف الأول الابتدائي للعام 2026: 30,000 جنيه على ثلاث دفعات.",
    "رسوم الصف الثاني الابتدائي: 35,000 جنيه على ثلاث دفعات.",
    "رسوم الصف الرابع الابتدائي: 45,000 جنيه على ثلاث دفعات.",
]
GENERAL_DOC = [
    "أوراق التحويل المطلوبة لكل الصفوف: شهادة الميلاد، آخر شهادة درجات، وصورة البطاقة."
]


class Scenario:
    """One parent's turn, and what a good answer to it looks like."""

    def __init__(self, key, question, chunks, *, child_year="", discriminate="yes",
                 must_contain=(), must_not_contain=(), judge_rubric=""):
        self.key = key
        self.question = question
        self.chunks = list(chunks)
        self.child_year = child_year
        self.discriminate = discriminate
        self.must_contain = tuple(must_contain)
        self.must_not_contain = tuple(must_not_contain)
        self.judge_rubric = judge_rubric


SCENARIOS = [
    Scenario(
        "fees-year-1",
        "مصاريف ابني كام",
        FEE_TABLE,
        child_year=YEAR_1,
        must_contain=("30,000", "30000", "30 ألف"),
        must_not_contain=("45,000", "45000", "35,000", "35000"),
        judge_rubric=(
            "The child is in الصف الأول الابتدائي (Year 1). A good answer states the "
            "Year 1 fee of 30,000 EGP. It is WRONG if it states another year's fee as "
            "this child's, and wrong if it refuses or asks which year — the year was "
            "given to it."
        ),
    ),
    Scenario(
        "fees-no-year-on-file",
        "مصاريف ابني كام",
        FEE_TABLE,
        child_year="",
        judge_rubric=(
            "No year is known for this child and the fees differ by year. A good answer "
            "either asks which year, or gives the fees per year clearly labelled. It is "
            "WRONG if it picks one year and presents it as this child's."
        ),
    ),
    Scenario(
        "parent-names-another-year",
        "مصاريف ابني لو دخل الصف الرابع كام",
        FEE_TABLE,
        child_year="",  # withheld: the parent named their own year
        must_contain=("45,000", "45000", "45 ألف"),
        judge_rubric=(
            "The parent asked specifically about الصف الرابع (Year 4). A good answer "
            "gives the Year 4 fee of 45,000. It is WRONG if it answers for a different "
            "year because the child is currently in one."
        ),
    ),
    Scenario(
        "general-document-list",
        "ايه الأوراق المطلوبة للتحويل",
        GENERAL_DOC,
        child_year=YEAR_1,
        discriminate="no",
        judge_rubric=(
            "This document list is written once for every year group. A good answer "
            "gives the list in full. It is WRONG if it withholds it, hedges, or claims "
            "the knowledge base has nothing for Year 1 — the list applies to Year 1 too."
        ),
    ),
    Scenario(
        "corpus-has-nothing",
        "مواعيد رحلة المدرسة الصيفية امتى",
        [],
        child_year=YEAR_1,
        must_not_contain=("30,000", "45,000", "2026"),
        judge_rubric=(
            "Nothing was retrieved. A good answer says the school's material does not "
            "cover this and suggests contacting the school. It is WRONG if it states "
            "any specific date, figure or arrangement — there is no source for one."
        ),
    ),
]


def _corpus_prompt(scenario):
    chunks = "\n\n---\n\n".join(
        f"[{i}] school_docs.pdf (Page {i}):\n{text}"
        for i, text in enumerate(scenario.chunks, 1)
    )
    kwargs = {
        "outcome": "chunks" if scenario.chunks else "no_knowledge",
        "chunks": chunks,
        "constraints": [],
        "discriminate": scenario.discriminate,
        "rewritten": False,
        "partial": False,
    }
    if scenario.child_year:
        kwargs["child_year"] = scenario.child_year
    return render_prompt("tools/knowledge_result.j2", **kwargs)


def run_turn(scenario, model):
    """One turn through the real agent and the real finalizer. Returns a result dict."""
    searches = []

    @tool("search_knowledge_base")
    def search_knowledge_base(query: str) -> str:
        """Search the knowledge base for documents that answer the user's question."""
        searches.append(query)
        return _corpus_prompt(scenario)

    ctx = _EvalContext()
    agent = create_agent(
        model=model,
        tools=[search_knowledge_base],
        system_prompt=(
            "You are a school assistant answering a signed-in parent. Use "
            "search_knowledge_base for anything the school's documents would know, and "
            "answer only from what it returns. Cite chunks inline as [1]. Answer in the "
            "parent's language."
        ),
        middleware=[runtime._collapse_duplicate_tool_calls(ctx)],
    )

    turn_context = resolve_prompt(
        "", "agent/turn_context.j2",
        resolved_question="", constraints=[],
        child_hint="علي", child_year=scenario.child_year, child_options=[],
    )
    messages = [SystemMessage(content=turn_context)] if turn_context.strip() else []
    messages.append(HumanMessage(content=scenario.question))

    finalizer = Finalizer()
    raw = ""
    for msg, _meta in agent.stream({"messages": messages}, stream_mode="messages"):
        if isinstance(msg, ToolMessage):
            finalizer.note_tool_result(msg)
            continue
        if not isinstance(msg, AIMessage):
            continue
        raw += msg.content if isinstance(msg.content, str) else ""
        finalizer.consider(msg)
    finalizer.finish()

    # The service serves retry copy when the finalizer withheld everything, so a harness
    # that stopped at the finalizer reported "no answer" for a turn a parent would see a
    # message on. Mirrored here, or the score describes a product that does not exist.
    answer = finalizer.answer.strip()
    suppressed = not answer and (raw.strip() or finalizer.as_trace()["finalize_dropped_chars"])
    if suppressed:
        answer = _RETRY_COPY

    return {
        "answer": answer,
        "suppressed_to_retry_copy": bool(suppressed),
        "raw": raw,
        "searches": searches,
        "duplicates_collapsed": ctx.duplicate_tool_calls,
        "trace": finalizer.as_trace(),
    }


class _EvalContext:
    """The two hooks the middleware touches. Not a ChatRequestContext: this harness has
    no request, and standing one up would drag in storage the eval does not use."""

    def __init__(self):
        self.duplicate_tool_calls = 0

    def note_duplicate_tool_calls(self, count):
        self.duplicate_tool_calls += count


# ---------------------------------------------------------------------------
# Deterministic checks. Never delegated to the judge — these are decidable.
# ---------------------------------------------------------------------------

#: Three or more consecutive English words — prose, not a borrowed term.
_ENGLISH_PROSE = re.compile(r"(?:[A-Za-z']{2,}[ ,.]+){2}[A-Za-z']{2,}")
_ARABIC = re.compile(r"[؀-ۿ]")


def deterministic_checks(scenario, result):
    answer = result["answer"]
    findings = {}

    tokens = [t for t in HARMONY_TOKENS if t in answer]
    # A RUN of English words inside an answer to an Arabic question is this model's
    # reasoning showing through — "We need to see the result" — and that is the
    # signature the leak was first measured by.
    #
    # Three consecutive words, not one: this corpus holds English documents, so an
    # Arabic answer may legitimately quote an English policy term or a document name.
    # Flagging a single Latin token would mark a correct bilingual answer as a leak,
    # which is the failure mode that gets a check ignored.
    asked_in_arabic = bool(_ARABIC.search(scenario.question))
    reasoning = asked_in_arabic and bool(_ENGLISH_PROSE.search(answer))
    findings["bug1_clean_output"] = not tokens and not reasoning

    report = verify_grounding(answer, scenario.chunks)
    findings["bug2_every_figure_grounded"] = report.ok
    findings["bug2_reason"] = report.reason

    # DISTINCT searches, not total. The middleware collapses repeats of the same call
    # and the graph memoises repeats of the same query; neither claims to stop a model
    # from issuing two genuinely different searches, and scoring that as a defect would
    # report a failure the fixes were never for. `searches` is printed raw alongside, so
    # a run that fanned out is still visible.
    findings["bug3_no_repeated_search"] = len(set(result["searches"])) == len(result["searches"])

    if scenario.must_contain:
        findings["bug4_states_the_right_figure"] = any(
            value in answer for value in scenario.must_contain
        )
    if scenario.must_not_contain:
        findings["bug4_avoids_other_years"] = not any(
            value in answer for value in scenario.must_not_contain
        )
    return findings


# ---------------------------------------------------------------------------
# The judge. Only for what reading comprehension is needed for.
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are grading a school assistant's reply to a parent.

THE PARENT ASKED:
{question}

THE ONLY MATERIAL THE ASSISTANT WAS GIVEN:
{corpus}

{year_line}
THE ASSISTANT REPLIED:
{answer}

WHAT A GOOD REPLY LOOKS LIKE:
{rubric}

Grade it. Before each verdict, quote the exact span of the reply you are judging, so
your ruling can be checked against the text. If the reply is empty, every verdict is
false.

Return ONLY a JSON object, no other text:
{{
  "quote": "<the span your ruling rests on, verbatim, or empty>",
  "answers_the_question": true|false,
  "uses_only_the_material": true|false,
  "correct_year_group": true|false,
  "reads_as_an_answer_not_as_reasoning": true|false,
  "hedges_unnecessarily": true|false,
  "verdict": "good"|"acceptable"|"bad",
  "why": "<one sentence>"
}}"""


def judge(scenario, result, judge_model):
    corpus = "\n".join(f"- {c}" for c in scenario.chunks) or "(nothing was retrieved)"
    year_line = (
        f"THE SCHOOL'S RECORDS SAY THIS CHILD IS IN: {scenario.child_year}\n"
        if scenario.child_year else ""
    )
    prompt = JUDGE_PROMPT.format(
        question=scenario.question,
        corpus=corpus,
        year_line=year_line,
        answer=result["answer"] or "(empty)",
        rubric=scenario.judge_rubric,
    )
    try:
        raw = judge_model.invoke([HumanMessage(content=prompt)]).content or ""
        match = re.search(r"\{.*\}", raw, re.S)
        if not match:
            return {"error": "judge returned no JSON"}
        return json.loads(match.group(0))
    except Exception as exc:  # a judge failure is INVALID, never a pass
        return {"error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3,
                        help="samples per scenario; one proves nothing at temperature>0")
    parser.add_argument("--scenario", default="", help="run one scenario by key")
    parser.add_argument("--no-judge", action="store_true",
                        help="deterministic checks only; no judge calls")
    parser.add_argument("--judge-model", default=os.getenv("GRADE_MODEL") or os.getenv("MODEL"))
    parser.add_argument("--verbose", action="store_true", help="print every answer")
    args = parser.parse_args()

    api_key, base_url = os.getenv("ARK_API_KEY"), os.getenv("BASE_URL")
    if not api_key or not os.getenv("MODEL"):
        print("ARK_API_KEY / MODEL are not set — nothing to evaluate.")
        return 2

    model = init_chat_model(
        model=os.getenv("MODEL"), model_provider="openai", api_key=api_key,
        base_url=base_url, stream_usage=True, temperature=0.2,
    )
    judge_model = None if args.no_judge else init_chat_model(
        model=args.judge_model, model_provider="openai", api_key=api_key,
        base_url=base_url, temperature=0.0,
    )

    scenarios = [s for s in SCENARIOS if not args.scenario or s.key == args.scenario]
    if not scenarios:
        print(f"no scenario named {args.scenario!r}; known: "
              f"{', '.join(s.key for s in SCENARIOS)}")
        return 2

    print(f"model={os.getenv('MODEL')}  provider={os.getenv('LLM_PROVIDER') or 'generic'}  "
          f"runs={args.runs}  judge={'off' if args.no_judge else args.judge_model}\n")

    totals, invalid, suppressed = {}, 0, 0
    for scenario in scenarios:
        print(f"── {scenario.key}: «{scenario.question}»")
        for run_index in range(args.runs):
            try:
                result = run_turn(scenario, model)
            except Exception as exc:
                invalid += 1
                print(f"   run {run_index + 1}: INVALID — {type(exc).__name__}: {exc}")
                continue

            if result.get("suppressed_to_retry_copy"):
                # The model emitted a transcript with no answer channel, and the service
                # serves retry copy for that. There is nothing here for a judge to read
                # and no figure for the checks to weigh, so it is counted as its own
                # outcome: folding it into "bad answer" overstates how often the
                # assistant says something WRONG, and folding it into "good" hides how
                # often it says nothing at all.
                suppressed += 1
                print(f"   run {run_index + 1}: searches={len(result['searches'])} "
                      f"collapsed={result['duplicates_collapsed']}  "
                      f"SUPPRESSED — no answer channel; retry copy served")
                continue

            checks = deterministic_checks(scenario, result)
            for name, value in checks.items():
                if isinstance(value, bool):
                    hit, total = totals.get(name, (0, 0))
                    totals[name] = (hit + int(value), total + 1)

            line = "  ".join(
                f"{name.split('_', 1)[0]}:{'ok' if value else 'FAIL'}"
                for name, value in checks.items() if isinstance(value, bool)
            )
            verdict = ""
            if judge_model is not None:
                ruling = judge(scenario, result, judge_model)
                if "error" in ruling:
                    invalid += 1
                    verdict = f"  judge:INVALID({ruling['error'][:40]})"
                else:
                    verdict = f"  judge:{ruling.get('verdict', '?')}"
                    key = f"judge_{ruling.get('verdict', 'unknown')}"
                    hit, total = totals.get(key, (0, 0))
                    totals[key] = (hit + 1, total + 1)
                    if ruling.get("verdict") == "bad":
                        verdict += f" — {ruling.get('why', '')[:80]}"
            print(f"   run {run_index + 1}: searches={len(result['searches'])} "
                  f"collapsed={result['duplicates_collapsed']}  {line}{verdict}")
            if args.verbose:
                print(f"      answer: {result['answer'][:200]}")
            if not checks.get("bug2_every_figure_grounded", True):
                print(f"      grounding: {checks['bug2_reason']}")

    print("\n── rates")
    for name in sorted(totals):
        hit, total = totals[name]
        if name.startswith("judge_"):
            print(f"   {name:34} {hit}")
        else:
            print(f"   {name:34} {hit}/{total}"
                  f"   {'100%' if hit == total else f'{100 * hit / total:.0f}%'}")
    if invalid:
        print(f"\n   {invalid} run(s) INVALID — not counted as passes. "
              f"Treat the rates above as provisional.")
    # A quality score never fails the build. See the module docstring.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
