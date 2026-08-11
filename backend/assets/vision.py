"""Vision model credentials, resolved in one place.

Three call sites need a vision-capable model — figure extraction, on-demand figure
reading, and entity extraction — and all three previously resolved the environment
themselves. That is three chances for the fallback rules to drift apart, and three
places to change when a deployment puts its vision model somewhere else.

Resolution, most specific first:

    model     VISION_MODEL    -> MODEL
    api key   VISION_API_KEY  -> ARK_API_KEY
    base url  VISION_BASE_URL -> BASE_URL

The independent VISION_* overrides matter: a deployment whose text model is on one
provider and whose vision model is on another is common, and the earlier code could
not express it — it always sent the main provider's key to whatever VISION_MODEL
named, which fails confusingly rather than clearly.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Optional, Type

logger = logging.getLogger(__name__)

# Reasoning models emit their scratchpad inline, ahead of the answer. It has to come
# off before the text is parsed as JSON or stored as an observation.
_REASONING_BLOCK_RE = re.compile(
    r"<\s*(think|thinking|reasoning|scratchpad)\s*>.*?<\s*/\s*\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
_OPEN_REASONING_RE = re.compile(
    r"<\s*(think|thinking|reasoning|scratchpad)\s*>.*\Z", re.IGNORECASE | re.DOTALL
)


def strip_reasoning(text: str) -> str:
    """Remove a reasoning model's inline scratchpad from its answer."""
    cleaned = _REASONING_BLOCK_RE.sub("", text or "")
    if "<" in cleaned:
        cleaned = _OPEN_REASONING_RE.sub("", cleaned)
    return cleaned.strip()


@dataclass(frozen=True)
class VisionCredentials:
    """What a vision call needs, plus where each piece came from."""

    model_id: str = ""
    api_key: str = ""
    base_url: str = ""
    model_source: str = ""
    key_source: str = ""
    base_url_source: str = ""

    @property
    def available(self) -> bool:
        """A model id and a key are the minimum; base_url is optional because some
        providers have a default endpoint."""
        return bool(self.model_id and self.api_key)

    def missing(self) -> list[str]:
        gaps = []
        if not self.model_id:
            gaps.append("VISION_MODEL (or MODEL)")
        if not self.api_key:
            gaps.append("VISION_API_KEY (or ARK_API_KEY)")
        return gaps

    def describe(self) -> str:
        """Diagnostic line. Never includes the key itself — this reaches logs."""
        if not self.available:
            return f"unavailable (missing: {', '.join(self.missing())})"
        endpoint = self.base_url or "provider default"
        return (
            f"model={self.model_id} (from {self.model_source}), "
            f"key from {self.key_source}, endpoint={endpoint}"
        )


def _first_env(*names: str) -> tuple[str, str]:
    """First non-blank value among `names`, with the variable it came from."""
    for name in names:
        value = (os.getenv(name) or "").strip()
        if value:
            return value, name
    return "", ""


def resolve_vision_credentials() -> VisionCredentials:
    model_id, model_source = _first_env("VISION_MODEL", "MODEL")
    api_key, key_source = _first_env("VISION_API_KEY", "ARK_API_KEY")
    base_url, base_url_source = _first_env("VISION_BASE_URL", "BASE_URL")
    return VisionCredentials(
        model_id=model_id,
        api_key=api_key,
        base_url=base_url,
        model_source=model_source,
        key_source=key_source,
        base_url_source=base_url_source,
    )


def build_vision_model(
    temperature: float = 0.0,
    credentials: Optional[VisionCredentials] = None,
    max_tokens: Optional[int] = None,
    extra_params: Optional[dict] = None,
    reasoning_effort: str = "",
):
    """A configured chat model, or None when credentials are incomplete.

    None rather than an exception: every caller has a working non-vision fallback, and
    a missing key should degrade the deployment rather than break it.

    `reasoning_effort` is empty by default so that no effort field is sent at all — a
    vision model without the parameter rejects the key rather than ignoring it.
    """
    resolved = credentials or resolve_vision_credentials()
    if not resolved.available:
        return None

    from langchain.chat_models import init_chat_model

    kwargs = {}
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    if extra_params:
        # Passed straight through: these are provider knobs this layer deliberately
        # does not model. Applied last so that a profile already suppressing reasoning
        # through this dict keeps winning — the typed field above is the preferred
        # spelling, not a replacement that would silently change existing behaviour.
        kwargs.update(extra_params)
    return init_chat_model(
        model=resolved.model_id,
        model_provider="openai",
        api_key=resolved.api_key,
        base_url=resolved.base_url or None,
        temperature=temperature,
        **kwargs,
    )


# Providers signal a quota problem with different status codes — 429 conventionally,
# but Groq uses 413 for "this single request exceeds your per-minute allowance". The
# error body is the reliable discriminator.
_RATE_LIMIT_MARKERS = ("rate_limit", "rate limit", "tokens per minute", "tpm", "too many requests")
_RETRY_AFTER_RE = re.compile(r"try again in\s*([\d.]+)\s*(ms|s|m)?", re.IGNORECASE)


def is_rate_limit_error(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    if status in (429, 413):
        return True
    text = str(exc).lower()
    return any(marker in text for marker in _RATE_LIMIT_MARKERS)


def retry_after_seconds(exc: BaseException, default: float) -> float:
    """How long the provider asked us to wait, if it said.

    Honouring the stated delay matters more than any backoff curve: a token-per-minute
    window refills on a schedule, and guessing shorter just burns another attempt.
    """
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    for key in ("retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        raw = headers.get(key)
        if raw:
            try:
                return float(str(raw).rstrip("s"))
            except ValueError:
                pass

    match = _RETRY_AFTER_RE.search(str(exc))
    if match:
        value = float(match.group(1))
        unit = (match.group(2) or "s").lower()
        return value / 1000 if unit == "ms" else value * 60 if unit == "m" else value
    return default


def call_with_rate_limit_retry(operation, config, description: str = "vision call"):
    """Run `operation`, waiting out transient quota rejections.

    Only rate limits are retried. A malformed request or an unsupported feature will
    fail identically every time, and retrying it just multiplies the delay before the
    caller gets to fall back.
    """
    import time

    attempts = max(int(getattr(config, "vision_retry_attempts", 1)), 1)
    base = float(getattr(config, "vision_retry_base_seconds", 5.0))
    ceiling = float(getattr(config, "vision_retry_max_seconds", 90.0))

    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:
            if not is_rate_limit_error(exc) or attempt == attempts:
                raise
            delay = min(retry_after_seconds(exc, base * attempt), ceiling)
            logger.warning(
                "%s hit a rate limit (attempt %d/%d); waiting %.1fs",
                description, attempt, attempts, delay,
            )
            time.sleep(delay)
    raise RuntimeError(f"{description} exhausted its retries")  # pragma: no cover


def _message_text(result: Any) -> str:
    content = getattr(result, "content", result)
    if isinstance(content, list):
        content = "".join(
            block.get("text", "") for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return strip_reasoning(str(content or ""))


def _extract_json_object(text: str) -> Optional[dict]:
    """First balanced JSON object in a free-text response.

    Brace matching rather than a regex: model output routinely wraps the object in
    prose or a ```json fence, and nested objects defeat a non-recursive pattern.
    """
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:index + 1])
                    except ValueError:
                        break
        start = text.find("{", start + 1)
    return None


def _json_instructions(schema: Type) -> str:
    fields = json.dumps(schema.model_json_schema().get("properties", {}), ensure_ascii=False)
    return (
        "\n\nRespond with ONE JSON object and NOTHING else.\n"
        "Begin your reply with the opening brace. Do not write reasoning, explanation, "
        "a preamble, or a code fence first — the output budget is finite, and a reply "
        "that spends it on commentary gets truncated before the JSON is ever written.\n"
        "Include every key, using an empty string or empty list when a value does not apply.\n"
        f"Keys and their types: {fields}"
    )


def invoke_structured(model, schema: Type, messages: list, prompt_index: int = 0, config=None):
    """Structured output with provider fallbacks.

    `with_structured_output` defaults to the provider's `json_schema` response format,
    which many models simply do not support — Groq returns HTTP 400 for exactly that
    reason on several of its vision models, and the resulting exception silently
    demotes a vision extraction to the heuristic path.

    So the call cascades, cheapest and most faithful first:

        1. json_schema      — provider validates, best fidelity
        2. function_calling — far wider support, same guarantees in practice
        3. prompted JSON    — works on any model that can emit text at all

    Rung 3 is the one that matters operationally: it means a vision model without any
    structured-output support still produces a usable extraction instead of none.
    """
    last_error: Optional[Exception] = None

    for method in (None, "function_calling"):
        label = method or "json_schema"
        try:
            bound = (
                model.with_structured_output(schema)
                if method is None
                else model.with_structured_output(schema, method=method)
            )
            return call_with_rate_limit_retry(
                lambda: bound.invoke(messages), config, f"structured output ({label})"
            )
        except Exception as exc:  # provider rejection, or an unparsable response
            if is_rate_limit_error(exc):
                # Quota, not capability. The next method sends the same payload against
                # the same allowance, so falling through would just burn the budget
                # again and report the wrong cause.
                raise
            last_error = exc
            logger.debug("Structured output via %s failed: %s", label, exc)

    # Rung 3: ask in the prompt, parse what comes back.
    patched = [dict(message) for message in messages]
    try:
        target = patched[prompt_index]
        content = target.get("content")
        if isinstance(content, list):
            blocks = [dict(block) for block in content]
            for block in blocks:
                if block.get("type") == "text":
                    block["text"] = block.get("text", "") + _json_instructions(schema)
                    break
            target["content"] = blocks
        else:
            target["content"] = str(content or "") + _json_instructions(schema)
    except (IndexError, KeyError, TypeError):
        logger.debug("Could not append JSON instructions to the prompt")

    result = call_with_rate_limit_retry(
        lambda: model.invoke(patched), config, "structured output (prompted JSON)"
    )
    payload = _extract_json_object(_message_text(result))
    if payload is None:
        # Truncation is the most common cause and is otherwise invisible: a response
        # cut off mid-scratchpad has no JSON and no closing tag, which is
        # indistinguishable from a model that simply refused.
        finish = (getattr(result, "response_metadata", None) or {}).get("finish_reason")
        usage = getattr(result, "usage_metadata", None) or {}
        hint = (
            " The response was TRUNCATED — raise assets.figures.vision_max_output_tokens."
            if finish == "length" else ""
        )
        raise RuntimeError(
            "Structured output failed on every method and no JSON object was found "
            f"(finish_reason={finish}, output_tokens={usage.get('output_tokens')})."
            f"{hint} Last provider error: {last_error}"
        )
    return schema.model_validate(payload)


def vision_status(profile=None) -> dict:
    """Whether vision is actually usable, and for what.

    Surfaced at startup because the failure mode is silent: a profile can enable
    vision, the credentials can be absent, and the system will quietly extract with
    the heuristic path forever. This turns that into one line in the boot log.
    """
    if profile is None:
        from backend.profiles import get_profile

        profile = get_profile()

    assets = profile.assets
    credentials = resolve_vision_credentials()
    wanted = {
        "figures": assets.figures.enabled and assets.figures.vision_enabled,
        "entities": assets.entities.enabled and assets.entities.vision_enabled,
    }
    return {
        "credentials_available": credentials.available,
        "detail": credentials.describe(),
        "requested_by": [name for name, on in wanted.items() if on],
        # The state worth shouting about: asked for, not configured.
        "degraded": [name for name, on in wanted.items() if on and not credentials.available],
    }


def log_vision_status(profile=None) -> dict:
    status = vision_status(profile)
    if status["degraded"]:
        logger.warning(
            "Vision extraction is enabled for %s but no vision credentials are configured "
            "(%s). Falling back to heuristic extraction — captions will be weak and "
            "scanned pages will not be readable. Set VISION_MODEL and VISION_API_KEY.",
            ", ".join(status["degraded"]),
            status["detail"],
        )
    elif status["requested_by"]:
        logger.info(
            "Vision extraction active for %s — %s",
            ", ".join(status["requested_by"]),
            status["detail"],
        )
    else:
        logger.info("Vision extraction is disabled by profile; using heuristic extraction.")
    return status
