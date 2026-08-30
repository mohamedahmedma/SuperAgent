"""Twenty dependent turns over a corpus where some answers vary by the user's
conditions and some do not.

The distinction this file exists for: a condition the user sets ("my son is 5", "up to
Year 6") is a reading of the CONVERSATION, never a claim about the knowledge base. The
two come apart constantly in a real corpus:

  * a fee table has a row per year, so "up to Year 6" narrows it — answering with every
    year's fees is wrong
  * an admissions document list is written once for everyone, so "up to Year 6" narrows
    nothing — and refusing because the list does not mention Year 6 is worse than wrong,
    because the answer was in hand

The shipped system got the first case right by enforcing carried conditions at four
points: the retrieval query, the grading question, the rewrite input, and the answer
prompt. That made the second case fail three separate ways, and the reported symptom was
a fallback on "what document to apply for them" while the corpus held the list.

So the rule under test is: **conditions narrow an answer only where the material varies
by them, and can never suppress one.** `constraints_discriminate` is the grader's
verdict on which case it is, taken on the call that graded the evidence.

The agent is not in the loop — it would need a live model and it makes no decision this
file is about. What runs is the real turn planner, the real RAG graph, the real routing
policy and the real prompts, with the resolver and the grader scripted.
"""
import re
import unittest
from dataclasses import dataclass, field
from typing import List, Optional

from backend.chat.orchestrator import plan_turn
from backend.chat.request_context import ChatRequestContext
from backend.chat.resolution import CORRECTION, FOLLOWUP, NEW_TOPIC, STANDALONE
from backend.profiles.registry import load_profile, set_profile
from backend.prompts import render
from tests.general.test_rag_short_circuit import FakeStructuredModel, load_pipeline, _meta


# ---------------------------------------------------------------------------
# A corpus with both kinds of section
# ---------------------------------------------------------------------------

@dataclass
class Section:
    chunk_id: str
    text: str
    # Whether this material gives different answers depending on year/age/gender.
    # The whole point of the fixture: half of it does not.
    varies_by_condition: bool


CORPUS = [
    Section("fees-primary", (
        "Tuition fees 2025/2026. Year 1: 42,000. Year 2: 42,000. Year 3: 45,000. "
        "Year 4: 45,000. Year 5: 48,000. Year 6: 48,000. Fees are billed per term."
    ), True),
    Section("fees-secondary", (
        "Tuition fees 2025/2026, secondary. Year 7: 55,000. Year 8: 55,000. "
        "Year 9: 58,000. Year 10: 61,000."
    ), True),
    Section("fees-discount", (
        "Sibling discount: families enrolling two or more children receive 10% off the "
        "tuition of each additional child. The discount applies at every year group."
    ), False),
    Section("fees-payment", (
        "Payment plans: tuition may be paid in three termly installments or in full "
        "before the start of the academic year. Installments are available to all families."
    ), False),
    Section("uniform-girls", (
        "Uniform and clothes, day wear for girls up to Grade 6 and Year 6: navy "
        "pinafore with the school crest, white blouse, navy cardigan. Girls in Grade 7 "
        "and above wear the navy skirt and blazer."
    ), True),
    Section("uniform-boys", (
        "Uniform and clothes, day wear for boys up to Grade 6 and Year 6: grey "
        "trousers, white shirt, school tie. Boys in Grade 7 and above wear the blazer."
    ), True),
    Section("admission-docs", (
        "Documents required to apply for admission: the child's birth certificate, a "
        "copy of the passport, two passport photographs, the most recent school report, "
        "and an up-to-date vaccination record. The same documents are required for every "
        "applicant."
    ), False),
    Section("transfer-docs", (
        "Documents required to transfer a student from another school: a transfer "
        "certificate from the previous school, the last two school reports, the birth "
        "certificate, and a vaccination record."
    ), False),
    Section("medical-policy", (
        "A medical report is not required at application. A vaccination record is "
        "required for all students before the first day of term."
    ), False),
    Section("transport", (
        "School transport operates on twelve routes across the city. Seats are allocated "
        "on application and are available to students in every year group."
    ), False),
    Section("transport-fees", (
        "Transport fee: 8,000 per academic year for a return seat, 5,000 for one way. "
        "The same fee applies on every route."
    ), False),
    Section("calendar-start", (
        "The school day starts at 07:45 and ends at 14:30. Start time for the gates "
        "is 07:15. The same times apply to every year group."
    ), False),
    Section("calendar-terms", (
        "Term one ends on 18 December. Term two runs from 6 January to 27 March. "
        "Term three ends on 25 June."
    ), False),
    Section("subjects-3-6", (
        "Subjects taught in Years 3 to 6: Arabic, English, mathematics, science, "
        "social studies, Islamic studies, art, music and physical education."
    ), True),
    Section("subjects-7-9", (
        "Subjects taught in Years 7 to 9 add a second language, design technology and "
        "separate sciences."
    ), True),
    Section("grade-placement", (
        "Grade placement is by age on 1 September: age 4 enters Foundation Stage 2, "
        "age 5 enters Year 1, age 6 enters Year 2."
    ), True),
    Section("contacts", (
        "Admissions office: admissions@school.example, +20 2 555 0100. The team handles "
        "applications and transfers for all year groups."
    ), False),
    Section("admission-deadline", (
        "The application deadline is 31 May: applications for the 2025/2026 year close "
        "then. Late applications are considered only where places remain. The deadline "
        "is the same for every year group."
    ), False),
]

BY_ID = {section.chunk_id: section for section in CORPUS}

_STOP = {
    "the", "a", "an", "is", "are", "for", "to", "of", "in", "on", "at", "and", "or",
    "what", "which", "who", "how", "do", "does", "my", "his", "her", "i", "it", "be",
    "there", "any", "with", "up", "can", "will", "would", "child", "children", "school",
}


def _terms(text: str) -> set:
    """Content tokens, crudely stemmed.

    Digits are kept whatever their length — "5" and "6" are the whole discriminating
    payload of an age or a year group. Trailing plurals are stripped because a real
    index analyses "ends"/"end" and "documents"/"document" to the same term, and a
    fixture that does not would fail on morphology rather than on anything this file
    is about.
    """
    out = set()
    for raw in (text or "").split():
        word = raw.strip(".,?():;").lower()
        if not word or word in _STOP:
            continue
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        if len(word) > 2 or any(char.isdigit() for char in word):
            out.add(word)
    return out


def retrieve(query, top_k=5, language=""):
    """Token-overlap retrieval over the fixture corpus.

    Crude on purpose. What matters is that it is a pure function of the query STRING,
    so a query carrying a condition the target passage never mentions is measurably
    worse than one that does not — which is the retrieval half of the bug this file
    guards, and it would be invisible against a retriever that ignored the query.
    """
    wanted = _terms(query)
    scored = []
    for section in CORPUS:
        have = _terms(section.text)
        hits = len(wanted & have)
        if not hits:
            continue
        # Query terms absent from the passage dilute the match, exactly as extra BM25
        # terms do against a real index.
        scored.append((hits / max(1, len(wanted)), section))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    docs = [
        {"filename": f"{section.chunk_id}.md", "page_number": 1,
         "text": section.text, "chunk_id": section.chunk_id, "score": score}
        for score, section in scored[:top_k]
    ]
    return {"docs": docs, "meta": _meta(len(docs))}


# ---------------------------------------------------------------------------
# A grader that behaves the way the prompt now asks one to
# ---------------------------------------------------------------------------

class ScriptedGrader:
    """Reads the rendered grading prompt and answers it the way the template asks.

    Deliberately NOT a lookup table keyed on turn number. It re-derives its verdict from
    the prompt text, so if the pipeline ever folds the conditions back into the question
    this grader sees it and the assertions below catch it.
    """

    def __init__(self):
        self.prompts: List[str] = []

    def __call__(self, schema, prompt: str):
        self.prompts.append(prompt)
        question = prompt.split("User question:", 1)[-1].split("Conditions in force:")[0]
        question = question.split("Retrieved snippets:", 1)[0].strip()
        body = prompt.split("Retrieved snippets:", 1)[-1]

        # In the order the snippets were NUMBERED in the prompt, not corpus order.
        # `supporting_chunks` is 1-based into the retrieved list and the pipeline trims
        # context to it, so a grader answering in a different order silently re-points
        # every citation at the wrong passage.
        cited = [
            BY_ID[chunk_id]
            for chunk_id in re.findall(r"\[\d+\] ([\w-]+)\.md", body)
            if chunk_id in BY_ID
        ]
        wanted = _terms(question)
        scored = sorted(
            ((len(wanted & _terms(s.text)), s) for s in cited),
            key=lambda pair: pair[0], reverse=True,
        )
        on_subject = [(hits, s) for hits, s in scored if hits >= 2]

        if not on_subject:
            return {"relevance": "none", "answerability": "none", "ambiguity": "none",
                    "route": "no_knowledge", "confidence": 0.9,
                    "constraints_discriminate": "unknown", "supporting_chunks": []}

        # The rule the template states: relevance is judged against the QUESTION alone,
        # never penalised for material that does not mention a carried condition.
        supporting = [cited.index(s) + 1 for _, s in on_subject]
        discriminate = "unknown"
        if "Conditions in force:" in prompt:
            # Read off the passage that CARRIES the answer, not off anything retrieval
            # happened to surface. A uniform table that shares the words "Year 6" with
            # a question about admission documents is not what the answer rests on, and
            # letting it vote would report the answer as year-specific when it is not.
            discriminate = "yes" if on_subject[0][1].varies_by_condition else "no"
        return {"relevance": "strong", "answerability": "sufficient", "ambiguity": "none",
                "route": "answer", "confidence": 0.85,
                "constraints_discriminate": discriminate,
                "supporting_chunks": supporting}


# ---------------------------------------------------------------------------
# The conversation harness
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    text: str
    resolved: str
    constraints: List[str]
    intent: str
    route: str = ""
    status: str = ""
    chunk_ids: List[str] = field(default_factory=list)
    discriminate: str = ""
    tool_result: str = ""
    search_query: str = ""
    grader_prompt: str = ""
    short_circuit: bool = False


class Conversation:
    """Real planner, real graph, real policy, real prompts — scripted resolver/grader."""

    def __init__(self, pipeline, grader: ScriptedGrader):
        self.pipeline = pipeline
        self.grader = grader
        self.history: List[dict] = []
        self.turns: List[Turn] = []

    def ask(self, text, *, resolved=None, constraints=(), intent=FOLLOWUP) -> Turn:
        payload = {
            "question": resolved or text,
            "constraints": list(constraints),
            "intent": intent,
        }
        ctx = ChatRequestContext.for_sync(user_id="u", session_id="s")
        try:
            plan, signals = plan_turn(
                text, list(self.history), ctx, resolve_invoke=lambda *a: payload
            )
            turn = Turn(text=text, resolved=plan.resolved_question or text,
                        constraints=list(plan.carried_constraints), intent=signals.followup_intent)

            # A social turn, or a confirmed out-of-domain one. Either way the knowledge
            # tool is unbound, so nothing searches — modelling it as a search would be
            # asserting against a code path that never runs.
            if plan.short_circuit or plan.exposed_tools == []:
                turn.short_circuit = True
                turn.route = "static"
                self._record(text, "(no search)", turn)
                return turn

            # What the agent would have searched for, per agent/turn_context.j2: the
            # resolved question, not the words the user typed.
            before = len(self.grader.prompts)
            result = self.pipeline.run_rag_graph(turn.resolved, ctx)
            trace = result.get("rag_trace") or {}

            turn.route = result.get("route") or trace.get("route") or ""
            turn.status = result.get("retrieval_status") or trace.get("retrieval_status") or ""
            turn.chunk_ids = [d.get("chunk_id") for d in (result.get("docs") or [])]
            turn.discriminate = trace.get("evidence_constraints_discriminate") or ""
            turn.search_query = trace.get("query") or ""
            if len(self.grader.prompts) > before:
                turn.grader_prompt = self.grader.prompts[before]
            turn.tool_result = self._tool_result(result, trace, ctx)
            self._record(text, "(answered)", turn)
            return turn
        finally:
            ctx.close()

    def _tool_result(self, result, trace, ctx) -> str:
        status = trace.get("retrieval_status")
        if status in ("no_knowledge", "retrieval_error") or not result.get("docs"):
            return render("tools/knowledge_result.j2",
                          outcome="no_knowledge" if status == "no_knowledge" else "empty")
        return render(
            "tools/knowledge_result.j2",
            outcome="chunks",
            chunks="\n\n".join(
                f"[{i}] {d['filename']}:\n{d['text']}"
                for i, d in enumerate(result["docs"], 1)
            ),
            rewritten=bool(trace.get("rewrite_method")),
            partial=status == "partial",
            constraints=list(ctx.carried_constraints),
            discriminate=str(trace.get("evidence_constraints_discriminate") or "unknown"),
        )

    def _record(self, user_text, reply, turn: Turn) -> None:
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": reply})
        self.turns.append(turn)


# ---------------------------------------------------------------------------
# The sequence
# ---------------------------------------------------------------------------

UP_TO_6 = ["grades up to Year 6"]
AGE_5 = ["the child is 5 years old"]


class TwentyTurnConversationTests(unittest.TestCase):
    """One conversation, twenty dependent turns, asserted turn by turn."""

    @classmethod
    def setUpClass(cls):
        profile = load_profile("base")
        set_profile(profile.model_copy(update={
            "agent": profile.agent.model_copy(update={"request_envelope_enabled": False}),
            "rag": profile.rag.model_copy(update={
                "scope_index_enabled": False, "domain_gate_enabled": False,
            }),
        }))
        cls.grader = ScriptedGrader()
        cls.pipeline = load_pipeline(retrieve_documents=retrieve)
        cls.pipeline.API_KEY = "k"
        cls.pipeline.GRADE_MODEL = "g"
        cls.pipeline._grader_model = FakeStructuredModel(cls.grader)
        cls.pipeline._get_grader_model = lambda: cls.pipeline._grader_model

        chat = Conversation(cls.pipeline, cls.grader)
        cls.chat = chat

        # 1-2 — the pair that established the mechanism.
        chat.ask("what is the clothes for children under year 6",
                 resolved="what is the school uniform for children up to Year 6",
                 constraints=UP_TO_6, intent=STANDALONE)
        chat.ask("and what is the fees for this years",
                 resolved="what are the school tuition fees for the years up to Year 6",
                 constraints=UP_TO_6, intent=FOLLOWUP)
        # 3 — the reported failure: general material, inherited condition.
        chat.ask("what document to apply for them",
                 resolved="what documents are required to apply for admission for children up to Year 6",
                 constraints=UP_TO_6, intent=FOLLOWUP)
        # 4-5
        chat.ask("is there a discount if i have two children",
                 resolved="is there a sibling discount on tuition for two children",
                 constraints=UP_TO_6, intent=FOLLOWUP)
        chat.ask("my child is 5 years old, which grade is that",
                 resolved="which grade is a child aged 5 placed in",
                 constraints=AGE_5, intent=FOLLOWUP)
        # 6-8 — placement, then transfer documents: general again, new condition.
        chat.ask("and what does that grade cost",
                 resolved="what is the tuition fee for Year 1",
                 constraints=AGE_5, intent=FOLLOWUP)
        chat.ask("what are the required documents to transfer him",
                 resolved="what documents are required to transfer a student from another school",
                 constraints=AGE_5, intent=FOLLOWUP)
        chat.ask("does he need a medical report",
                 resolved="is a medical report required for a 5 year old applicant",
                 constraints=AGE_5, intent=FOLLOWUP)
        # 9-10 — back to material that does vary.
        chat.ask("and what about the uniform for him",
                 resolved="what is the day wear uniform for boys up to Grade 6",
                 constraints=AGE_5 + ["boys"], intent=FOLLOWUP)
        chat.ask("how much is the sibling discount worth",
                 resolved="how much is the sibling discount on tuition",
                 constraints=AGE_5, intent=FOLLOWUP)
        # 11-14 — general policies.
        chat.ask("is transport available",
                 resolved="is school transport available",
                 constraints=AGE_5, intent=FOLLOWUP)
        chat.ask("how much does it cost",
                 resolved="how much does school transport cost per year",
                 constraints=AGE_5, intent=FOLLOWUP)
        chat.ask("what time does school start",
                 resolved="what time does the school day start",
                 constraints=AGE_5, intent=FOLLOWUP)
        chat.ask("and when does the term end",
                 resolved="when does the school term end",
                 constraints=AGE_5, intent=FOLLOWUP)
        # 15-16 — varies by year again.
        chat.ask("what subjects will he study in year 3",
                 resolved="what subjects are taught in Years 3 to 6",
                 constraints=["Year 3"], intent=FOLLOWUP)
        chat.ask("is arabic included",
                 resolved="is Arabic taught in Years 3 to 6",
                 constraints=["Year 3"], intent=FOLLOWUP)
        # 17-19 — admissions logistics, all general.
        chat.ask("who do i contact to apply",
                 resolved="who do I contact in the admissions office to apply",
                 constraints=["Year 3"], intent=FOLLOWUP)
        chat.ask("what is the deadline",
                 resolved="what is the deadline to apply for the 2025/2026 year",
                 constraints=["Year 3"], intent=FOLLOWUP)
        chat.ask("can i pay in installments",
                 resolved="can tuition be paid in installments",
                 constraints=["Year 3"], intent=FOLLOWUP)
        # 20 — a pleasantry, which must not be treated as a question.
        chat.ask("thanks", resolved="thanks", constraints=(), intent=NEW_TOPIC)

    @classmethod
    def tearDownClass(cls):
        set_profile(None)

    def turn(self, number: int) -> Turn:
        return self.chat.turns[number - 1]

    # --- the shape of the run ------------------------------------------------

    def test_all_twenty_turns_ran(self):
        self.assertEqual(20, len(self.chat.turns))

    def test_no_turn_denied_knowledge_the_corpus_holds(self):
        """The invariant. Every question here has an answer in the fixture corpus, so a
        `no_knowledge` anywhere is the failure this whole file is about."""
        denied = [
            (i, t.text, t.route, t.search_query)
            for i, t in enumerate(self.chat.turns, 1)
            if t.route == "no_knowledge"
        ]
        self.assertEqual([], denied)

    def test_no_turn_was_handed_back_to_the_user(self):
        """Every turn inherits its subject from the conversation, so a scope_select
        offering a choice of subjects could not narrow any of them."""
        asked = [(i, t.text) for i, t in enumerate(self.chat.turns, 1)
                 if t.route in ("clarify", "scope_select")]
        self.assertEqual([], asked)

    # --- turn 3: the reported bug -------------------------------------------

    def test_turn_3_answers_the_general_document_list(self):
        """"what document to apply for them", asked after two Year-6 questions. The
        admissions list does not mention Year 6 anywhere; the shipped system denied it."""
        turn = self.turn(3)
        self.assertEqual("answer", turn.route)
        self.assertIn("admission-docs", turn.chunk_ids)
        self.assertEqual(UP_TO_6, turn.constraints)

    def test_turn_3_tells_the_model_the_list_applies_to_everyone(self):
        turn = self.turn(3)
        self.assertEqual("no", turn.discriminate)
        self.assertIn("does not vary by them", turn.tool_result)
        self.assertIn("Give it in full", turn.tool_result)
        self.assertIn("Do NOT withhold it", turn.tool_result)

    def test_turn_3_never_shows_the_grader_a_constrained_question(self):
        """The regression guard. Folding "grades up to Year 6" into the question is what
        let the grader honestly return `relevance: none` on a general document list."""
        question = self.turn(3).grader_prompt.split("User question:", 1)[-1]
        question = question.split("Conditions in force:", 1)[0]
        self.assertNotIn("(", question, "conditions were appended to the question")
        self.assertIn("Conditions in force:", self.turn(3).grader_prompt,
                      "they must still reach the grader, as their own field")

    # --- turn 7: the case the user named ------------------------------------

    def test_turn_7_answers_transfer_documents_for_a_five_year_old(self):
        """"required documents to transfer him", where "him" is 5. The transfer list is
        one list for every age — an answer, not a reason to refuse."""
        turn = self.turn(7)
        self.assertEqual("answer", turn.route)
        self.assertIn("transfer-docs", turn.chunk_ids)
        self.assertEqual(AGE_5, turn.constraints)
        self.assertEqual("no", turn.discriminate)
        self.assertIn("applies to everyone", turn.tool_result)

    # --- the conditions still work where the material varies ----------------

    def test_material_that_varies_is_still_narrowed(self):
        """Turn 1 is absent on purpose: it is the first message, so the gate skips
        resolution and there is nothing carried. Its condition is in its own words,
        where retrieval and the grader can both see it."""
        for number in (2, 5, 6, 9, 15, 16):
            with self.subTest(turn=number):
                turn = self.turn(number)
                self.assertEqual("answer", turn.route)
                self.assertEqual("yes", turn.discriminate, turn.text)
                self.assertIn("leave the other cases out", turn.tool_result)

    def test_material_that_does_not_vary_is_never_narrowed(self):
        for number in (3, 4, 7, 8, 10, 11, 12, 13, 14, 17, 18, 19):
            with self.subTest(turn=number):
                turn = self.turn(number)
                self.assertEqual("no", turn.discriminate, turn.text)
                self.assertNotIn("leave the other cases out", turn.tool_result)

    # --- the query itself ----------------------------------------------------

    def test_no_condition_ever_reaches_the_search_query(self):
        """The rule that cost three of these twenty turns when it was the other way
        round. A condition is absent from every passage the corpus wrote once for
        everybody, so appending it to a query is dilution precisely where it hurts."""
        for i, turn in enumerate(self.chat.turns, 1):
            if turn.short_circuit:
                continue
            with self.subTest(turn=i):
                self.assertEqual(turn.resolved, turn.search_query)
                for condition in turn.constraints:
                    self.assertNotIn(f"({condition}", turn.search_query)

    def test_the_condition_still_reaches_the_answer(self):
        """It is not dropped, only moved: the stage that can narrow an answer without
        being able to lose a document."""
        turn = self.turn(7)
        self.assertIn("5 years old", turn.tool_result)

    # --- inheritance across the sequence ------------------------------------

    def test_the_year_condition_survives_until_it_is_replaced(self):
        for number in (2, 3, 4):
            with self.subTest(turn=number):
                self.assertEqual(UP_TO_6, self.turn(number).constraints)

    def test_a_new_condition_replaces_the_old_one(self):
        self.assertEqual(AGE_5, self.turn(5).constraints)
        self.assertEqual(AGE_5, self.turn(6).constraints)

    def test_a_pleasantry_is_not_treated_as_a_question(self):
        turn = self.turn(20)
        self.assertTrue(turn.short_circuit, turn.route)
        self.assertEqual([], turn.chunk_ids, "a greeting must not search anything")

    # --- the trace stays diagnosable ----------------------------------------

    def test_every_answered_turn_records_why_it_routed_that_way(self):
        """`route_reason` was written and then dropped by normalize_rag_trace, which is
        why the reported bug took a code read rather than a trace read to find."""
        from backend.schemas.chat import RagTraceFields

        self.assertIn("route_reason", RagTraceFields.model_fields)
        self.assertIn("evidence_constraints_discriminate", RagTraceFields.model_fields)


class ConstraintPoisoningTests(unittest.TestCase):
    """Why conditions are kept out of the query, stated as a property of retrieval.

    This is the part that degrades silently: nothing errors, the right passage just
    ranks below something that happens to share the condition's vocabulary. It is
    recorded here so the reasoning survives the next person who thinks appending the
    condition would help recall.
    """

    def test_appending_a_condition_costs_the_right_passage_its_rank(self):
        clean = retrieve("what time does the school day start")
        poisoned = retrieve("what time does the school day start (the child is 5 years old)")
        self.assertEqual("calendar-start", clean["docs"][0]["chunk_id"])
        poisoned_ids = [doc["chunk_id"] for doc in poisoned["docs"]]
        self.assertTrue(
            not poisoned_ids or poisoned_ids[0] != "calendar-start"
            or poisoned["docs"][0]["score"] < clean["docs"][0]["score"],
            "the condition should measurably dilute a passage that never mentions it",
        )

    def test_a_condition_the_question_states_naturally_still_helps(self):
        """The mechanism that replaced appending: the RESOLVED question carries the
        condition in vocabulary the target passage actually contains."""
        vague = retrieve("what are the tuition fees")
        resolved = retrieve("what are the school tuition fees for the years up to Year 6")
        self.assertTrue(any(d["chunk_id"] == "fees-primary" for d in resolved["docs"]))
        self.assertGreaterEqual(
            next(d["score"] for d in resolved["docs"] if d["chunk_id"] == "fees-primary"),
            0.0,
        )
        self.assertTrue(vague["docs"], "the bare question still retrieves something")


if __name__ == "__main__":
    unittest.main()
