"""The question the assistant is still waiting on, and what the next message does to it.

A turn can end in a question instead of an answer: which of these did you mean, which year
group, which of your children. The next message is then read against that pending question
— answering it, replacing it, or settling it by naming a child — before anything expensive
runs. This module holds that state and the reading of it.

It was part of `service.py`, where both chat entry points reached it. Nothing here builds
an agent or calls a model, except through the resolver `enter_turn` is handed.
"""
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from backend.chat.orchestrator import resolve_turn_question
from backend.chat.resolution import ResolvedQuestion
from backend.profiles import get_profile
from backend.schemas.chat import PendingHitlState
from backend.text_matching import name_key

logger = logging.getLogger(__name__)


PENDING_HITL_KEY = "pending_hitl"


HITL_STATUSES = {"needs_clarification", "needs_scope_selection"}


HITL_ROUTES = {"clarify", "scope_select"}


def is_hitl_trace(rag_trace: dict | None) -> bool:
    if not isinstance(rag_trace, dict):
        return False
    status = rag_trace.get("retrieval_status")
    route = rag_trace.get("route")
    return status in HITL_STATUSES or route in HITL_ROUTES


def _hitl_route_from_trace(rag_trace: dict) -> str:
    status = rag_trace.get("retrieval_status")
    route = rag_trace.get("route")
    if status == "needs_scope_selection" or route == "scope_select":
        return "scope_select"
    return "clarify"


def _hitl_prompt_from_trace(rag_trace: dict) -> str:
    prompt = (rag_trace.get("hitl_prompt") or "").strip()
    if prompt:
        return prompt
    route = _hitl_route_from_trace(rag_trace)
    if route == "scope_select":
        return get_profile().user_copy.hitl_scope_agent
    return get_profile().user_copy.hitl_clarify_agent


def _hitl_options_from_trace(rag_trace: dict) -> list[str]:
    options = rag_trace.get("hitl_options") or []
    if not isinstance(options, list):
        return []
    return [str(option).strip() for option in options if str(option).strip()]


def format_hitl_message(prompt: str, options: list[str] | None = None) -> str:
    clean_prompt = prompt.strip()
    clean_options = [item for item in (options or []) if item]
    if not clean_options:
        return clean_prompt
    option_lines = "\n".join(f"- {item}" for item in clean_options)
    return f"{clean_prompt}\n\nAvailable options:\n{option_lines}"


def _existing_hitl_answers(pending_hitl: dict | None) -> list[str]:
    if not isinstance(pending_hitl, dict):
        return []
    answers = pending_hitl.get("answers") or []
    if not isinstance(answers, list):
        return []
    return [str(answer).strip() for answer in answers if str(answer).strip()]


def build_pending_hitl(
    rag_trace: dict,
    original_question: str,
    previous_answers: list[str] | None = None,
    resume_state: dict | None = None,
) -> dict:
    prompt = _hitl_prompt_from_trace(rag_trace)
    options = _hitl_options_from_trace(rag_trace)
    route = _hitl_route_from_trace(rag_trace)
    return PendingHitlState(
        id=uuid4().hex,
        original_question=original_question,
        prompt=prompt,
        options=options,
        route=route,
        retrieval_status=(
            "needs_scope_selection" if route == "scope_select" else "needs_clarification"
        ),
        answers=previous_answers or [],
        resume_state=resume_state,
        created_at=datetime.now(timezone.utc).isoformat(),
    ).model_dump()


def child_choice_pending(turn_plan, original_question: str) -> dict | None:
    """The clarification for "which of your children?", when the planner asked one.

    Built here rather than by `build_pending_hitl` because that one reads a `rag_trace`,
    and this question never went near retrieval — it comes from a roster and a length
    check. Everything the client needs to render selectable names is on the plan already.

    `resume_state` is None, and that is the whole difference between this route and the
    other two: there is no half-finished search to pick up. The answer pins a child to
    the session, and the original question is planned again from the start — which is
    also why `original_question` has to be carried, since the next message will be a
    name and nothing else.
    """
    options = [str(name) for name in (getattr(turn_plan, "child_options", None) or []) if name]
    question = (getattr(turn_plan, "static_reply", "") or "").strip()
    if not options or not question:
        return None
    return PendingHitlState(
        id=uuid4().hex,
        original_question=original_question or question,
        prompt=question,
        options=options,
        route="child_select",
        retrieval_status="needs_child_choice",
        answers=[],
        resume_state=None,
        created_at=datetime.now(timezone.utc).isoformat(),
    ).model_dump()


def pin_the_child_the_parent_named(ctx, chosen: str) -> bool:
    """Settle the pending "which child?" against the roster, with no model involved.

    The reply is matched by the same resolver every other route uses, so a parent who
    typed the name rather than tapping the option is read the same way, and a reply
    matching nobody pins nothing — the next plan then asks again rather than answering
    about a child nobody chose.

    Pinning is what makes the answer stick: `resolve_child` consults the pin only among
    the candidates a stated sex already allows, so "my son" asked again later still
    means a son, and the pin only breaks the tie it was created to break.
    """
    if not chosen or ctx is None:
        return False
    try:
        from backend.chat.child_resolution import resolve_child
        from backend.chat.child_roster import OK, load_roster

        outcome, roster = load_roster(ctx)
        if outcome != OK or not roster:
            return False
        # The option was offered in the roster's own words, so a reply equal to one of
        # them is that child and needs no matching at all. Tried first because the
        # substring matcher below cannot settle it: where two children share a name,
        # every offered label contains the other's, so a tapped option matches both and
        # resolves to neither — which is the very question this reply is answering.
        tapped = [c for c in roster if name_key(c.label) == name_key(chosen)]
        picked = tapped[0] if len(tapped) == 1 else None

        if picked is None:
            # Typed rather than tapped, or spelled differently. The shared resolver reads
            # it exactly as every other route does.
            found = resolve_child(reference="named", child_name=chosen, roster=roster)
            if not found.resolved:
                return False
            picked = next((c for c in roster if c.student_id == found.student_id), None)
            if picked is None:
                return False

        ctx.remember_child(
            picked.student_id,
            label=picked.label,
            gender=getattr(picked, "gender", "") or "",
            # They were shown their own children and chose one. That is what lets this
            # pin settle a name matching two children later in the conversation.
            chosen_by_parent=True,
        )
        return True
    except Exception:  # pragma: no cover - a pin must never break a turn
        logger.warning("could not settle the child the parent chose", exc_info=True)
        return False


def build_hitl_event(pending_hitl: dict) -> dict:
    return {
        "id": pending_hitl["id"],
        "prompt": pending_hitl["prompt"],
        "options": pending_hitl["options"],
        "route": pending_hitl["route"],
        "retrieval_status": pending_hitl["retrieval_status"],
        "original_question": pending_hitl["original_question"],
    }


def _build_hitl_resume_query(pending_hitl: dict, user_text: str) -> str:
    original_question = pending_hitl.get("original_question") or ""
    prompt = pending_hitl.get("prompt") or ""
    previous_answers = _existing_hitl_answers(pending_hitl)

    lines = [
        "This is a continuation request after HITL clarification in the previous RAG flow.",
        "Do not treat the user's follow-up as a standalone new question; return to the original question and continue answering it.",
        f"Original question: {original_question}",
    ]
    if prompt:
        lines.append(f"HITL question: {prompt}")
    if previous_answers:
        lines.append("The user has previously provided:")
        lines.extend(f"- {answer}" for answer in previous_answers)
    lines.extend([
        f"User's input this round: {user_text}",
        "Please form a complete query based on the above and continue with the original Agent/RAG flow.",
    ])
    return "\n".join(lines)


@dataclass
class TurnEntry:
    """How this message relates to a clarification the assistant is still waiting on.

    Computed once, before anything expensive, because the answer changes which of two
    paths the turn takes — and both entry points need the same answer.
    """

    pending_hitl: dict | None = None
    invalid_pending_hitl: bool = False
    is_hitl_resume: bool = False
    resume_state: dict | None = None
    resolution: ResolvedQuestion | None = None
    superseded: bool = False
    effective_user_text: str = ""
    hitl_answers: list = field(default_factory=list)
    original_question: str = ""
    # The child the parent just chose, when this message answers a "which child?"
    # question. A name as they typed or tapped it — resolved against the roster later,
    # by code holding a verified identity this dataclass deliberately does not.
    child_choice: str = ""

    def spends_the_pending_question(self, *, agent_error: bool = False) -> bool:
        """Whether the clarification that was waiting is finished with after this turn.

        There are three ways a pending question ends, and every save path has to agree
        on all three or the question outlives its answer. Asked here, once, rather than
        re-derived at each site — which is how the third one came to be missing from two
        of them.

          * **Answered** — `is_hitl_resume`, the retrieval clarifications.
          * **Replaced** — `superseded`, the user corrected the question instead.
          * **Settled** — `child_choice`, this message named which child.

        The third reached production. A `child_select` reply is deliberately neither a
        resume nor a supersession (there is no search to continue, and the parent did
        not change the subject), so a rule written as `is_hitl_resume or superseded`
        left it stored — and every later message was then read as another child's name,
        re-answering the ORIGINAL question. A father asking for Sunday's lessons, for
        Monday's, and for the term dates was told his daughter's subjects each time.

        `agent_error` keeps a RETRIEVAL clarification alive, because the answer never
        got used and the parent should be able to retry it. A child choice is already
        spent whatever happens afterwards: the pin is written before the agent runs, so
        keeping the question would cost the parent their next message for nothing.
        """
        if self.child_choice:
            return True
        if agent_error:
            return False
        return bool(self.is_hitl_resume or self.superseded)


def enter_turn(
    user_text: str,
    messages: list,
    metadata: dict,
    *,
    resolve: Callable[..., ResolvedQuestion] | None = None,
) -> TurnEntry:
    """Decide whether this message answers the pending clarification or replaces it.

    The resume path assumes the reply either picks one of the offered options or fills
    a named slot. A reply that does neither — "no, I meant the fees", or a change of
    subject entirely — is not something to fold into the old query, because folding is
    concatenation and concatenation retrieves both readings. Those replies abandon the
    pending clarification and start a fresh turn from the resolved question instead.

    The resolution computed here is handed to `plan_turn`, so a turn that takes the
    fresh path pays for exactly one resolver call rather than two.

    `resolve` is the question resolver, handed in rather than reached for: it is the one
    collaborator here that can cost a model call, so a caller that wants a different one
    — a test, most usually — passes it instead of patching a module attribute. None means
    the real resolver.
    """
    entry = TurnEntry(effective_user_text=user_text, original_question=user_text)

    stored = metadata.get(PENDING_HITL_KEY)
    entry.pending_hitl = _current_pending_hitl(stored)
    entry.invalid_pending_hitl = stored is not None and entry.pending_hitl is None
    if not isinstance(entry.pending_hitl, dict):
        return entry

    if entry.pending_hitl.get("route") == "child_select":
        # No resolver call and no model. The pending question was "which of your
        # children", the reply is a name, and matching it belongs to the roster — so this
        # turn simply becomes the ORIGINAL question again, planned from the start now
        # that the pin can settle it. Deliberately not a `hitl_resume`: there is no
        # search to continue, and folding the name into the old query as the other routes
        # do would retrieve for "علي" rather than for what the parent actually asked.
        entry.child_choice = user_text
        entry.original_question = entry.pending_hitl.get("original_question") or user_text
        entry.effective_user_text = entry.original_question
        return entry

    entry.resolution = (resolve or resolve_turn_question)(
        user_text,
        messages,
        hitl_prompt=entry.pending_hitl.get("prompt") or "",
        hitl_options=entry.pending_hitl.get("options") or [],
    )
    entry.superseded = entry.resolution.supersedes_pending_question
    entry.is_hitl_resume = not entry.superseded
    entry.resume_state = pending_resume_state(entry.pending_hitl)
    entry.hitl_answers = [*_existing_hitl_answers(entry.pending_hitl), user_text]

    if entry.superseded:
        # A fresh turn: the pending state is dropped rather than resumed, and the
        # question this turn is about is the one the user has just corrected it to.
        entry.original_question = entry.resolution.question or user_text
        return entry

    entry.effective_user_text = _build_hitl_resume_query(entry.pending_hitl, user_text)
    entry.original_question = entry.pending_hitl.get("original_question") or user_text
    return entry


def _current_pending_hitl(value: dict | None) -> dict | None:
    if not isinstance(value, dict):
        return None
    try:
        return PendingHitlState.model_validate(value).model_dump()
    except ValueError:
        return None


def pending_resume_state(pending_hitl: dict | None) -> dict | None:
    if not isinstance(pending_hitl, dict):
        return None
    resume_state = pending_hitl.get("resume_state")
    return dict(resume_state) if isinstance(resume_state, dict) else None
