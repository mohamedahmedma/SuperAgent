"""Sampling settings for one model role, shaped for `init_chat_model`.

Every chat model in the request path is built the same way — model id and credentials
from env, sampling from the active profile — and this is the sampling half. It exists
so that the six roles cannot drift: before it, temperature was passed at each call
site and the two knobs that actually govern spend (reasoning effort and the output
ceiling) were passed at none of them.

Why the knobs are per role. On a reasoning model the output side dominates: those
tokens bill several times input, and each is produced serially while the entire prompt
prefills in one parallel pass. A grader or a classifier emits a fixed structured shape,
so deliberation buys it nothing — but left at the default effort it will still spend
the bulk of a turn's wall-clock thinking before it answers. `answer` is the only role
doing open-ended work and the only one that should be paying for depth.

Two values mean "omit the parameter" rather than "send this":

  reasoning_effort  — empty (or none/off in a profile) sends no effort field. Required
                      for a non-reasoning model behind BASE_URL, which rejects the
                      field rather than ignoring it.
  max_tokens        — 0 sends no ceiling. A cap that truncates a structured response
                      turns a priced call into a parse failure, so it is opt-in.

Resolution order is the backend's usual one — env > profile > schema default — and is
implemented once, in backend/profiles/registry.py. Nothing here reads the environment.
"""
from __future__ import annotations

from typing import Any, Dict

from backend.profiles import get_profile

# Each entry is a `<role>_*` field group on ModelConfig, and names the node it serves.
ROLES = (
    "answer",    # the agent's answering call — the only open-ended role
    "fast",      # rolling persistent-note summariser
    "planner",   # question-complexity classification / decomposition
    "grade",     # evidence grading
    "rewrite",   # step-back / HyDE query planning
    "scope",     # in-domain / out-of-domain scope check
    "resolve",   # rewriting a follow-up into a standalone question
)


def sampling(role: str) -> Dict[str, Any]:
    """Sampling kwargs for `role`, ready to splat into `init_chat_model`.

    Always carries `temperature`. Carries `reasoning_effort` and `max_tokens` only when
    they are set, because for both of them the unset state has to reach the provider as
    an absent field rather than as a default value.
    """
    if role not in ROLES:
        raise ValueError(f"Unknown model role {role!r}; expected one of {', '.join(ROLES)}")

    models = get_profile().models
    kwargs: Dict[str, Any] = {"temperature": getattr(models, f"{role}_temperature")}

    # Validated and normalised by ModelConfig, so it is either "" or a level the
    # provider accepts.
    effort = getattr(models, f"{role}_reasoning_effort")
    if effort:
        kwargs["reasoning_effort"] = effort

    ceiling = getattr(models, f"{role}_max_tokens")
    if ceiling and ceiling > 0:
        kwargs["max_tokens"] = ceiling

    return kwargs
