"""The chat models the retrieval path calls, one per role, built once each.

Three roles reached for a model through the same shape of module global — a `_model =
None`, a `global` statement, and a function that filled it in on first use. They sat in
three files (`rag/pipeline.py` twice, `rag/utils.py` once), so the rule they all encode
— which model id, whose credentials, which sampling — was stated three times and could
drift three ways.

`ChatModelFactory` states it once. What is per role is the model id and the sampling
settings; everything else is shared, and a role that is not configured yields `None`
rather than a broken client, because every caller here has a working path without a
model: the grader falls through to the rung below it, the planner skips decomposition,
the rewriter leaves the query alone.

Credentials and model ids are read when the factory is built, not per call, matching
what the module globals did: one read per process, early, so a deployment cannot change
which model it is calling halfway through serving a request.
"""
from __future__ import annotations

import os
import threading
from typing import Any, Callable, Dict, Mapping, Optional

from backend.llm import sampling

#: Which environment variable names the model for each role. `planner` and `rewrite`
#: deliberately share one: both are short structured calls on the cheap model, and a
#: deployment that wanted them apart would be choosing to pay twice for the same work.
_MODEL_VARIABLE = {
    "grade": "GRADE_MODEL",
    "planner": "FAST_MODEL",
    "rewrite": "FAST_MODEL",
}

#: Output ceiling for the second attempt at a grade, when the first was cut off.
#:
#: Generous on purpose. The grade itself is a handful of enums and a short reason —
#: under 200 tokens — so a ceiling this size is not for the JSON. It is for the
#: scratchpad in front of it: GRADE_MODEL on this deployment is a reasoning model, and
#: its reasoning is billed against the SAME completion budget as the answer, produced
#: first. Run out there and the call ends at `finish_reason: length` having emitted no
#: JSON at all, which the OpenAI client raises as `LengthFinishReasonError`.
GRADE_RETRY_MAX_TOKENS = 3072


def _headroom(settings: Dict[str, Any]) -> Dict[str, Any]:
    """The grading settings, with room for the model to finish.

    Minimum effort the parameter offers, NOT absent. An absent `reasoning_effort` is not
    "no reasoning": `ModelConfig._validate_effort` maps none/off onto "" so the field is
    omitted, and a reasoning model then applies its OWN default — which is higher than
    `low`, so switching it "off" here would buy more reasoning and less room, the
    opposite of the fix.
    """
    adjusted = dict(settings)
    adjusted["reasoning_effort"] = "low"
    adjusted["max_tokens"] = max(int(adjusted.get("max_tokens") or 0), GRADE_RETRY_MAX_TOKENS)
    return adjusted


class ChatModelFactory:
    """Chat models by role, built on first use and shared thereafter.

    `build` is the constructor to call, injectable so a test can assert which model and
    credentials a role reaches for without standing up a client.
    """

    def __init__(
        self,
        *,
        environ: Optional[Mapping[str, str]] = None,
        build: Optional[Callable[..., Any]] = None,
        http_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        source = environ if environ is not None else os.environ
        self._api_key = source.get("ARK_API_KEY")
        self._base_url = source.get("BASE_URL")
        self._model_ids = {
            role: source.get(variable) for role, variable in _MODEL_VARIABLE.items()
        }
        self._build = build
        # The shared provider clients (backend/llm_http.py), when the composition root
        # supplies them. Empty in a factory built on its own, which then behaves as before.
        self._http_kwargs = dict(http_kwargs or {})
        self._models: Dict[tuple, Any] = {}
        self._lock = threading.RLock()

    def _builder(self) -> Callable[..., Any]:
        if self._build is not None:
            return self._build
        # Imported here rather than at module scope: this module is imported by the
        # retrieval path, and langchain's model registry is not cheap to import in a
        # process that never builds a model.
        from langchain.chat_models import init_chat_model

        return init_chat_model

    def for_role(self, role: str, *, variant: str = "") -> Any:
        """The model for `role`, or `None` when this deployment has not configured one.

        `variant` names a second instance of the same role built with different sampling
        — today only the grader's `headroom`, because sampling is fixed when a model is
        built and the truncation retry has to change it.
        """
        model_id = self._model_ids.get(role)
        if not self._api_key or not model_id:
            return None

        key = (role, variant)
        try:
            return self._models[key]
        except KeyError:
            pass

        with self._lock:
            if key not in self._models:
                settings = sampling(role)
                if variant == "headroom":
                    settings = _headroom(settings)
                self._models[key] = self._builder()(
                    model=model_id,
                    model_provider="openai",
                    api_key=self._api_key,
                    base_url=self._base_url,
                    stream_usage=True,
                    **self._http_kwargs,
                    **settings,
                )
            return self._models[key]

    # -- the roles, named ---------------------------------------------------------
    # Callers ask for the job rather than for a string, so a typo is an AttributeError
    # here instead of a silent `None` that reads as "not configured".

    def grader(self, *, headroom: bool = False) -> Any:
        """Evidence grading. `headroom` is reached only from the truncation handler."""
        return self.for_role("grade", variant="headroom" if headroom else "")

    def planner(self) -> Any:
        """Question-complexity classification and sub-question decomposition."""
        return self.for_role("planner")

    def rewriter(self) -> Any:
        """Step-back and HyDE query planning."""
        return self.for_role("rewrite")


__all__ = ["ChatModelFactory", "GRADE_RETRY_MAX_TOKENS"]
