"""What happens when the system does NOT cleanly know the parent, or their children.

The whole child-selection feature rests on one claim: no model ever chooses which child a
question is about, because deterministic code picks from a list of REAL children read
under the caller's own verified identity. That claim is only worth as much as its failure
modes, and every one of them lives here:

  * nobody is signed in as a parent at all — staff, a background job, an expired session;
  * the roster read did not answer (an outage), or was refused (a stale sign-in);
  * the guardian is real but has no student linked to their account;
  * the roster read blew up in a way nobody anticipated.

Each of these must degrade the same way: the planner narrows nothing, asks nothing, and
leaves the turn exactly as it ran before the feature existed — and the ONLY component that
words the failure to a parent is the records tool, which has separate, careful copy for
"we could not reach the school", "your sign-in expired" and "no student is linked". Those
three sentences must never be swapped for each other, and none of them may ever be
delivered as "your child has no grades".

The rest of the file covers the identity boundary itself: a context whose caller
contradicts its storage key is refused outright, the planner's chosen child never changes
whose token is used or whose guardian is read, and the per-message trace — persisted and
streamed to a browser — reports booleans about a child, never a name, an id or a year.
"""
import os
import unittest
from unittest.mock import patch

import requests

import backend.chat.child_roster as child_roster
from backend.chat.caller_identity import CallerIdentity
from backend.chat.child_resolution import resolve_child
from backend.chat.child_roster import ChildOption
from backend.chat.orchestrator import _settle_child, _start_roster, plan_turn
from backend.chat.request_context import ChatRequestContext
from backend.chat.resolution import unresolved
from backend.chat.signals import RequestSignals
from backend.chat.turn_policy import resolve_turn
from backend.profiles.registry import load_profile, set_profile
from backend.tools.records import make_get_student_records

PARENT_TOKEN = "signed.identity.token"
GUARDIAN = "G-77"
SESSION = "turn-77"

KNOWLEDGE_TOOL = "search_knowledge_base"
RECORDS_TOOL = "get_student_records"

LAYLA = ChildOption(student_id="S-1", label="ليلى أحمد", gender="female", year_level="Year 4")
OMAR = ChildOption(student_id="S-2", label="عمر أحمد", gender="male")


# --- fixtures, deliberately local ------------------------------------------------
#
# Nothing here is imported from a sibling test file: these rows and stubs describe the
# identity edges this file is about, and sharing them would tie two suites' idea of a
# roster together.


class _Response:
    """Only what `requests` callers in this repo actually read off a response."""

    def __init__(self, status_code: int, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload


TWO_CHILDREN = _Response(
    200,
    {
        "guardian_id": GUARDIAN,
        "students": [
            {"student_id": "S-1", "full_name_ar": "ليلى أحمد", "gender": "female"},
            {"student_id": "S-2", "full_name_ar": "عمر أحمد", "gender": "male"},
        ],
    },
)
ONE_CHILD = _Response(
    200,
    {
        "guardian_id": GUARDIAN,
        "students": [{"student_id": "S-1", "full_name_ar": "ليلى أحمد", "gender": "female"}],
    },
)
GRADES = _Response(
    200,
    {
        "term": {"term_id": "2026-T1", "name_ar": "الفصل الأول"},
        "courses": [
            {
                "course_id": "9001",
                "subject_name_ar": "الرياضيات",
                "subject_name_en": "Mathematics",
                "computed_percentage": 91.0,
                "letter_grade": "A",
                "excused_count": 0,
                "missing_count": 0,
                "is_complete": True,
            }
        ],
    },
)


def _route(responses: dict, seen=None):
    """Serve canned payloads by URL suffix, recording every call when asked to."""

    def fake_get(url, headers=None, params=None, timeout=None):
        if seen is not None:
            seen.append((url, dict(headers or {})))
        for suffix, response in responses.items():
            if url.endswith(suffix):
                return response
        return _Response(404)

    return fake_get


def _parent_ctx(*, guardian_id=GUARDIAN, token=PARENT_TOKEN, children=()) -> ChatRequestContext:
    return ChatRequestContext(
        user_id="user-77",
        session_id=SESSION,
        caller=CallerIdentity(
            user_id="user-77",
            guardian_id=guardian_id,
            guardian_token=token,
            children=tuple(children),
        ),
    )


def _staff_ctx() -> ChatRequestContext:
    """A signed-in user who is not a parent. The default identity shape."""
    return ChatRequestContext(user_id="staff-1", session_id=SESSION)


def _envelope(kind="records", reference="child"):
    """A classifier verdict, without a classifier.

    The envelope rung reports what the MESSAGE said and is never told who is asking —
    which is exactly why every case below can hold it constant and vary only the
    identity underneath it.
    """

    def invoke(question, history, config):
        return {
            "scope": "in_domain",
            "about_child": True,
            "child_reference": reference,
            "child_question_kind": kind,
        }

    return invoke


def _fetch(outcome, rows=()):
    def fetch(guardian_id, token, request_id):
        return outcome, list(rows)

    return fetch


class _Agent:
    """The school profile's shape for the pure-policy cases, one knob at a time."""

    tools = [KNOWLEDGE_TOOL, RECORDS_TOOL]
    social_phrases = []
    social_reply_mode = "model"
    narrow_tools_to_the_turn = True
    year_reference_markers = ()


class _Copy:
    social = None
    out_of_domain = None
    which_child = "Which child do you mean?"


class _NoRosterCache(unittest.TestCase):
    """The roster sits behind a cache shared with whatever Redis is running.

    Guardian ids collide across suites, so without this a case here can be answered from
    children another file cached — which is how a test about an outage starts reporting
    somebody's daughter. Zero turns the cache off for reading and writing both.
    """

    def setUp(self):
        env = patch.dict(os.environ, {"CHILD_ROSTER_TTL_SECONDS": "0"})
        env.start()
        self.addCleanup(env.stop)


class _SchoolTurn(_NoRosterCache):
    """`plan_turn` reads the process profile; these need the deployment that has children."""

    def setUp(self):
        super().setUp()
        set_profile(load_profile("school"))
        self.addCleanup(set_profile, None)

    def plan(self, ctx, *, roster_fetch=None, question="كيف حال ابني في الدراسة؟", **kw):
        """One planned turn with no model call anywhere in it.

        `resolution` is supplied so the resolver never reaches for a model, and the
        envelope is injected for the same reason.
        """
        return plan_turn(
            question,
            [],
            ctx,
            envelope_invoke=_envelope(**kw),
            resolution=unresolved(question, "test"),
            roster_fetch=roster_fetch,
        )


class ASessionThatIsNotAParent(_SchoolTurn):
    """Staff, a background job, a signed-out visitor, a half-built identity.

    Every one of them must cost nothing: no thread, no socket, no question put to
    somebody who has no children on file to be asked about.
    """

    def test_no_roster_read_is_started_without_a_guardian(self):
        """Not "started and discarded" — started at all would mean an HTTP call under a
        blank guardian id, against the unroutable path `/v1/guardians//students`."""
        called = []

        def fetch(*args):
            called.append(args)
            return child_roster.OK, []

        for label, ctx in (
            ("staff", _staff_ctx()),
            ("no token", _parent_ctx(token="")),
            ("no guardian id", _parent_ctx(guardian_id="")),
            ("no context at all", None),
        ):
            with self.subTest(session=label):
                self.assertIsNone(_start_roster(ctx, fetch))
        self.assertEqual([], called)

    def test_a_turn_with_no_roster_behind_it_settles_on_nobody(self):
        """`about_child` is a fact about the MESSAGE, and a staff member can perfectly
        well ask "how is my son doing" — the identity, not the wording, is what makes
        this unanswerable."""
        child = _settle_child(_staff_ctx(), RequestSignals(question="q", about_child=True), None)

        self.assertFalse(child.resolved)
        self.assertFalse(child.ask)
        self.assertEqual((), child.options)

    def test_a_staff_turn_narrows_nothing_and_asks_nothing(self):
        """The degraded path has to be the pre-feature path exactly: every tool bound,
        no short circuit, and nothing pinned onto the context for the graph to read."""
        ctx = _staff_ctx()
        plan, signals = self.plan(ctx)

        self.assertTrue(signals.about_child)
        self.assertIsNone(plan.exposed_tools)
        self.assertEqual("", plan.forced_tool)
        self.assertEqual([], plan.child_options)
        self.assertEqual("", plan.child_hint)
        self.assertFalse(plan.short_circuit)
        self.assertEqual("", ctx.planned_child_id)

    def test_the_records_tool_refuses_a_non_parent_before_touching_the_network(self):
        """A refusal that first calls the facade under an empty guardian would be an
        unauthenticated read of somebody's route, and a slower "no" for no benefit."""

        def forbidden(*args, **kwargs):
            raise AssertionError("a session with no guardian must not call the facade")

        ctx = _staff_ctx()
        with patch.object(requests, "get", forbidden):
            result = make_get_student_records(ctx).invoke({"record_type": "grades"})

        self.assertIn("NOT_A_PARENT_SESSION", result)
        self.assertEqual([("get_student_records", "not_a_parent")], ctx.tool_outcomes)

    def test_the_refusal_never_asks_the_user_to_identify_a_student(self):
        """Collecting a name, a student number or a birthdate authorises nothing — it
        only teaches a signed-out user that typing an identifier is how you get in."""
        result = make_get_student_records(_staff_ctx()).invoke({"record_type": "grades"})

        self.assertIn("NOT_A_PARENT_SESSION", result)
        for invitation in ("Do not ask", "signing in"):
            self.assertIn(invitation, result)

    def test_half_an_identity_is_not_a_parent_session(self):
        """A guardian id with no token cannot be proved; a token with no id has no
        subject. Either one alone reaching the tool as "a parent" would surface as a
        confusing failure deep inside a records read instead of a clear refusal."""
        for label, ctx in (
            ("id only", _parent_ctx(token="")),
            ("token only", _parent_ctx(guardian_id="")),
        ):
            with self.subTest(session=label):
                self.assertFalse(ctx.is_parent)
                result = make_get_student_records(ctx).invoke({"record_type": "grades"})
                self.assertIn("NOT_A_PARENT_SESSION", result)


class WhenTheRosterCannotBeRead(_SchoolTurn):
    """An outage and a refusal are different events with different remedies.

    Collapsing them tells a parent whose sign-in expired to wait for a service that is
    working fine — and tells a parent whose network blipped that the school has no record
    of their child.
    """

    def test_a_roster_that_did_not_answer_settles_on_nobody(self):
        for outcome in (child_roster.UNAVAILABLE, child_roster.NOT_AUTHORIZED, child_roster.NONE):
            with self.subTest(outcome=outcome):
                ctx = _parent_ctx()
                ahead = child_roster.prefetch(ctx, fetch=_fetch(outcome))
                child = _settle_child(
                    ctx, RequestSignals(question="q", about_child=True), ahead
                )

                self.assertFalse(child.resolved)
                self.assertFalse(child.ask)

    def test_an_outage_binds_every_tool_rather_than_narrowing(self):
        """The narrowing is an optimisation over a KNOWN child. Narrowing on a roster
        nobody answered for would bet a whole turn on a guess."""
        ctx = _parent_ctx()
        plan, _ = self.plan(ctx, roster_fetch=_fetch(child_roster.UNAVAILABLE))

        self.assertIsNone(plan.exposed_tools)
        self.assertEqual("", plan.forced_tool)
        self.assertEqual([], plan.child_options)
        self.assertFalse(plan.short_circuit)

    def test_an_outage_never_ends_the_turn_with_a_question(self):
        """Asking "which child?" here would offer a parent an empty choice, or worse a
        choice the turn cannot then act on — the tool is the only thing that can word
        what actually went wrong."""
        plan, _ = self.plan(_parent_ctx(), roster_fetch=_fetch(child_roster.NOT_AUTHORIZED))

        self.assertEqual([], plan.child_options)
        self.assertFalse(plan.static_reply)
        self.assertFalse(plan.short_circuit)

    def test_an_outage_falls_back_to_the_children_the_signed_token_asserts(self):
        """The token already said who this parent's children were when it was minted.

        Only on an outage, and only because it costs nothing in authorisation terms: the
        read that follows goes to the same facade, which re-checks the guardian link. What
        it buys is "I can't reach Layla's records right now" instead of "I don't know who
        your children are".
        """
        ctx = _parent_ctx(
            children=[{"student_id": "S-1", "full_name_ar": "ليلى أحمد", "gender": "female"}]
        )
        plan, _ = self.plan(ctx, roster_fetch=_fetch(child_roster.UNAVAILABLE))

        self.assertEqual("ليلى أحمد", plan.child_hint)
        self.assertEqual("S-1", ctx.planned_child_id)

    def test_a_refused_session_is_never_answered_from_the_tokens_own_claim(self):
        """The distinction that makes the fallback above safe. `not_authorized` is the
        facade saying this session may not read this guardian; leaning on the token's
        claim there would answer with exactly the children access was just refused for."""
        ctx = _parent_ctx(
            children=[{"student_id": "S-1", "full_name_ar": "ليلى أحمد", "gender": "female"}]
        )
        plan, _ = self.plan(ctx, roster_fetch=_fetch(child_roster.NOT_AUTHORIZED))

        self.assertEqual("", plan.child_hint)
        self.assertEqual([], plan.child_options)
        self.assertEqual("", ctx.planned_child_id)

    def test_the_tool_reports_the_outage_and_keeps_the_pin(self):
        """A timeout is not evidence the pinned child is wrong, and dropping the pin
        would re-ask the parent which child for a reason they could never see."""
        ctx = _parent_ctx()
        ctx.remember_child("S-1", label="ليلى أحمد")
        with patch.object(requests, "get", _route({"/students": _Response(503)})):
            result = make_get_student_records(ctx).invoke({"record_type": "grades"})

        self.assertIn("RECORDS_UNAVAILABLE", result)
        self.assertEqual([("get_student_records", "unavailable")], ctx.tool_outcomes)
        self.assertEqual("S-1", ctx.remembered_child)

    def test_a_refusal_drops_the_pin_because_it_is_evidence_the_hint_is_stale(self):
        """The one case that IS evidence: the facade refused this guardian, so whatever
        the conversation settled on may no longer be readable."""
        ctx = _parent_ctx()
        ctx.remember_child("S-1", label="ليلى أحمد")
        with patch.object(requests, "get", _route({"/students": _Response(403)})):
            result = make_get_student_records(ctx).invoke({"record_type": "grades"})

        self.assertIn("NOT_AUTHORIZED", result)
        self.assertEqual("", ctx.remembered_child)

    def test_neither_failure_is_ever_worded_as_missing_records(self):
        """The sentence this whole three-valued outcome exists to prevent."""
        for status, marker in ((503, "RECORDS_UNAVAILABLE"), (401, "NOT_AUTHORIZED")):
            with self.subTest(status=status):
                with patch.object(requests, "get", _route({"/students": _Response(status)})):
                    result = make_get_student_records(_parent_ctx()).invoke(
                        {"record_type": "grades"}
                    )

                self.assertIn(marker, result)
                self.assertNotIn("NO_RECORDS", result)
                self.assertNotIn("NO_STUDENTS_LINKED", result)


class AGuardianWithNoChildrenOnFile(_SchoolTurn):
    """A real, authorised parent whose account simply has nothing linked yet.

    Distinct from every failure above: nothing is broken and nothing expired, so the
    answer is a plain administrative fact — and only the tool has copy for it.
    """

    def test_an_empty_roster_resolves_to_nobody_without_asking(self):
        """Asking would offer a choice of nothing at all."""
        child = resolve_child(reference="child", roster=[])

        self.assertFalse(child.resolved)
        self.assertFalse(child.ask)
        self.assertEqual((), child.options)

    def test_the_planner_says_nothing_about_an_empty_roster(self):
        ctx = _parent_ctx()
        plan, _ = self.plan(ctx, roster_fetch=_fetch(child_roster.OK, []))

        self.assertEqual([], plan.child_options)
        self.assertEqual("", plan.child_hint)
        self.assertIsNone(plan.exposed_tools)
        self.assertFalse(plan.short_circuit)

    def test_a_two_hundred_with_no_students_is_not_an_outage(self):
        """`load_roster` re-labels it, so a working facade with an empty list can never
        be relayed to a parent as "the school's records are unavailable"."""
        ctx = _parent_ctx()
        outcome, children = child_roster.load_roster(ctx, fetch=_fetch(child_roster.OK, []))

        self.assertEqual(child_roster.NONE, outcome)
        self.assertEqual([], children)

    def test_only_the_tool_words_it_and_it_names_nobody(self):
        ctx = _parent_ctx()
        with patch.object(
            requests,
            "get",
            _route({"/students": _Response(200, {"guardian_id": GUARDIAN, "students": []})}),
        ):
            result = make_get_student_records(ctx).invoke({"record_type": "grades"})

        self.assertIn("NO_STUDENTS_LINKED", result)
        self.assertEqual([("get_student_records", "no_students")], ctx.tool_outcomes)


class ThePlannerNeverCostsTheTurn(_SchoolTurn):
    """`plan_turn` promises a planner fault costs the SAVING, never the turn.

    Every case here breaks something inside the roster path on purpose and asserts a
    usable plan still comes back — one that runs the turn exactly as it ran before any
    of this existed.
    """

    def test_a_roster_read_that_raises_still_produces_a_usable_plan(self):
        def boom(*_args):
            raise RuntimeError("facade on fire")

        ctx = _parent_ctx()
        plan, signals = self.plan(ctx, roster_fetch=boom)

        self.assertIsNone(plan.exposed_tools)
        self.assertFalse(plan.short_circuit)
        self.assertEqual("", plan.child_hint)
        self.assertEqual([], plan.child_options)
        self.assertEqual("", ctx.planned_child_id)
        self.assertTrue(signals.about_child)

    def test_a_roster_that_explodes_on_collection_is_contained(self):
        """The read succeeded and reading the RESULT failed — a different failure, and
        one the prefetch's own guard does not cover."""

        class _Exploding:
            def result(self, timeout=None):
                raise RuntimeError("collected a corpse")

        with patch.object(child_roster, "prefetch", lambda ctx, fetch=None: _Exploding()):
            plan, _ = self.plan(_parent_ctx())

        self.assertEqual("", plan.child_hint)
        self.assertEqual([], plan.child_options)
        self.assertIsNone(plan.exposed_tools)

    def test_a_roster_of_the_wrong_shape_does_not_reach_the_resolver(self):
        """A facade that changes its payload should degrade to "no child", not to a
        traceback halfway through building the turn."""
        rows = [{"full_name_ar": "ليلى"}, None, {"student_id": "S-9"}]
        ctx = _parent_ctx()
        plan, _ = self.plan(ctx, roster_fetch=_fetch(child_roster.OK, rows))

        # The nameless row survives as its own id; the broken rows are dropped rather
        # than raising, which is the only property this asserts.
        self.assertFalse(plan.short_circuit)
        self.assertNotIn("ليلى", plan.child_hint)

    def test_a_context_that_rejects_every_hint_still_gets_a_plan(self):
        """An integrating deployment's older context. Losing the hints is a degradation;
        losing the turn is the regression this module says it cannot cause."""

        class _OldContext:
            is_parent = False
            child = None
            planned_child_id = ""

            def note_turn_plan(self, *args, **kwargs):
                raise TypeError("unexpected keyword argument")

            def emit_rag_step(self, *args, **kwargs):
                return None

        plan, signals = self.plan(_OldContext())

        self.assertFalse(plan.short_circuit)
        self.assertIsNone(plan.exposed_tools)

    def test_a_policy_failure_returns_the_empty_plan_rather_than_raising(self):
        """The outermost guarantee, asserted from the outside: whatever breaks between
        the classifier and the plan, the caller gets defaults and runs the turn."""

        def boom(*args, **kwargs):
            raise RuntimeError("policy exploded")

        with patch("backend.chat.orchestrator.resolve_turn", boom):
            plan, signals = self.plan(_parent_ctx(), roster_fetch=_fetch(child_roster.OK, []))

        self.assertFalse(plan.short_circuit)
        self.assertIsNone(plan.exposed_tools)
        self.assertEqual("", plan.child_hint)
        self.assertEqual([], plan.child_options)


class TheSessionsOwnIdentityIsTheOnlyAuthority(_NoRosterCache):
    """The planner picks WHICH child. It can never widen WHOSE children are readable."""

    def test_a_context_whose_caller_names_another_user_is_refused(self):
        """`user_id` is the storage key. A caller naming somebody else would write one
        user's conversation under another's name while reading a third party's records,
        so the state is made impossible rather than merely unlikely."""
        other = CallerIdentity(user_id="someone-else", guardian_id=GUARDIAN, guardian_token=PARENT_TOKEN)

        with self.assertRaises(ValueError):
            ChatRequestContext(user_id="user-77", session_id=SESSION, caller=other)
        with self.assertRaises(ValueError):
            ChatRequestContext.for_sync(user_id="user-77", session_id=SESSION, caller=other)

    def test_the_refusal_does_not_leak_the_other_sessions_token(self):
        """The error text reaches logs and trackers; a live bearer credential must not."""
        other = CallerIdentity(user_id="someone-else", guardian_id=GUARDIAN, guardian_token=PARENT_TOKEN)

        with self.assertRaises(ValueError) as caught:
            ChatRequestContext(user_id="user-77", session_id=SESSION, caller=other)
        self.assertNotIn(PARENT_TOKEN, str(caught.exception))

    def test_the_records_read_carries_the_sessions_token_and_guardian(self):
        """The planner supplied a child id. It must change nothing about WHO is asking:
        the same bearer token, the same guardian in the path, the same audit id."""
        seen = []
        with patch.object(
            requests, "get", _route({"/students": TWO_CHILDREN, "/grades": GRADES}, seen=seen)
        ):
            ctx = _parent_ctx()
            ctx.note_turn_plan([], [], child_id="S-1", child_label="ليلى أحمد")
            result = make_get_student_records(ctx).invoke({"record_type": "grades"})

        self.assertIn("STUDENT_GRADES", result)
        self.assertTrue(seen)
        for url, headers in seen:
            with self.subTest(url=url):
                self.assertIn(f"/v1/guardians/{GUARDIAN}/", url)
                self.assertEqual(f"Bearer {PARENT_TOKEN}", headers.get("Authorization"))
                self.assertEqual(SESSION, headers.get("X-Request-Id"))

    def test_no_tool_argument_can_carry_an_identity(self):
        """Structural, not behavioural: there is nothing to inject into. If a future edit
        adds a guardian, a token or a student id as a parameter, the model can name whose
        records to read and the whole defence is gone."""
        args = set(make_get_student_records(_parent_ctx()).args.keys())

        self.assertEqual({"record_type", "student_name", "subject"}, args)
        for forbidden in ("guardian", "token", "student_id", "user"):
            self.assertFalse(
                [name for name in args if forbidden in name], f"{forbidden} is addressable"
            )

    def test_a_planned_child_missing_from_this_calls_roster_is_not_read(self):
        """A child withdrawn mid-conversation, or two reads either side of a change.

        The planner's id is a hint about the roster IT read; answering about it here would
        be answering about nobody. The ordinary resolver runs instead — and with an only
        child on file that resolves, rather than asking a parent about their one child.
        """
        seen = []
        with patch.object(
            requests, "get", _route({"/students": ONE_CHILD, "/grades": GRADES}, seen=seen)
        ):
            ctx = _parent_ctx()
            ctx.note_turn_plan([], [], child_id="S-404", child_label="طفل غير موجود")
            result = make_get_student_records(ctx).invoke({"record_type": "grades"})

        self.assertIn("STUDENT_GRADES", result)
        read = [url for url, _ in seen if url.endswith("/grades")]
        self.assertEqual(1, len(read))
        self.assertIn("/students/S-1/grades", read[0])
        self.assertNotIn("S-404", read[0])

    def test_a_planned_child_missing_from_a_family_of_two_asks_rather_than_guessing(self):
        ctx = _parent_ctx()
        with patch.object(requests, "get", _route({"/students": TWO_CHILDREN})):
            ctx.note_turn_plan([], [], child_id="S-404", child_label="طفل غير موجود")
            result = make_get_student_records(ctx).invoke({"record_type": "grades"})

        self.assertIn("NEEDS_STUDENT_CHOICE", result)
        self.assertEqual([("get_student_records", "which_student")], ctx.tool_outcomes)

    def test_a_context_nobody_planned_reads_under_no_child_at_all(self):
        """The default every non-planned caller is in — a sync call, a resumed turn, a
        test. It must be indistinguishable from "the planner settled nobody"."""
        ctx = _parent_ctx()

        self.assertEqual("", ctx.planned_child_id)
        self.assertEqual("", ctx.planned_child_label)
        self.assertEqual("", ctx.forced_tool)


class TheTraceNamesNoChild(_SchoolTurn):
    """The plan's trace is persisted per message and streamed to a browser.

    A name plus a year group narrows a child to a handful of real people, so this reports
    that a decision was MADE and never what it was. Asserted on all three shapes, because
    each fills a different field and a leak in any one of them is the same leak.
    """

    def _plan(self, child, **signal_kwargs):
        signals = RequestSignals(question="q", **signal_kwargs)
        return resolve_turn(
            signals, agent_config=_Agent(), copy_config=_Copy(), child=child
        )

    def test_no_shape_of_plan_puts_a_child_in_its_trace(self):
        cases = {
            "resolved": (resolve_child(reference="context", roster=[LAYLA]), True, False, True),
            "asking": (resolve_child(reference="child", roster=[LAYLA, OMAR]), False, True, False),
            "no child": (resolve_child(reference="plural", roster=[LAYLA, OMAR]), False, False, False),
        }
        for label, (child, resolved, asked, year_applied) in cases.items():
            with self.subTest(plan=label):
                trace = self._plan(
                    child, about_child=True, child_question_kind="records"
                ).as_trace()

                self.assertEqual(resolved, trace["turn_child_resolved"])
                self.assertEqual(asked, trace["turn_child_asked"])
                self.assertEqual(year_applied, trace["turn_child_year_applied"])
                rendered = repr(trace)
                for secret in ("ليلى", "عمر", "S-1", "S-2", "Year 4"):
                    self.assertNotIn(secret, rendered)

    def test_the_trace_of_a_real_planned_turn_names_nobody_either(self):
        """The pure-policy cases above cannot see what `plan_turn` adds on its way out —
        the reasons list, which is assembled from the resolver's own explanations."""
        ctx = _parent_ctx()
        plan, _ = self.plan(
            ctx,
            roster_fetch=_fetch(
                child_roster.OK,
                [{"student_id": "S-1", "full_name_ar": "ليلى أحمد", "year_level": "Year 4"}],
            ),
        )
        trace = plan.as_trace()

        self.assertTrue(trace["turn_child_resolved"])
        self.assertTrue(trace["turn_child_year_applied"])
        for secret in ("ليلى", "S-1", "Year 4"):
            self.assertNotIn(secret, repr(trace))


if __name__ == "__main__":
    unittest.main()
