"""The base class for every schema a model is asked to answer in.

Profiled under load (RAG_FIX_PLAN item 47), 10.4% of the serving process's CPU was
pydantic regenerating the SAME JSON schemas on every model call. The OpenAI SDK derives a
strict schema from the response model on every structured request
(`openai.lib._pydantic.to_strict_json_schema` -> `model.model_json_schema()`), and
LangChain does the same when it turns a model into a function definition. Neither caches
it, and pydantic does not either. A schema class does not change after it is defined, so
its JSON schema is computed once per class and argument set and handed out as a COPY: the
SDK's `_ensure_strict_json_schema` edits the dict it is given in place, and a shared cached
dict would be edited by every caller.

With one process serving on one core (the GIL), every millisecond of Python a turn spends
is a millisecond no other parent's turn can use — this is throughput, not tidiness.
"""
from __future__ import annotations

import copy
import threading
from typing import Any

from pydantic import BaseModel

_cache: dict[tuple, dict[str, Any]] = {}
_lock = threading.Lock()


class StructuredOutput(BaseModel):
    """A pydantic model whose JSON schema is generated once per class."""

    @classmethod
    def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        key = (cls, args, tuple(sorted(kwargs.items(), key=lambda item: item[0])))
        try:
            cached = _cache.get(key)
        except TypeError:  # an unhashable argument: nothing to key on, so just compute it
            return super().model_json_schema(*args, **kwargs)
        if cached is None:
            cached = super().model_json_schema(*args, **kwargs)
            with _lock:
                _cache[key] = cached
        return copy.deepcopy(cached)
