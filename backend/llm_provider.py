"""Which LLM provider the deployment talks to, selected by one variable.

The backend reads its model settings from a fixed set of generic names — `ARK_API_KEY`,
`BASE_URL`, `MODEL`, `FAST_MODEL`, `GRADE_MODEL`, and the `VISION_*` trio — and it reads
them at import time, in eight different modules. That shape is fine for a deployment that
only ever talks to one provider, and it is exactly wrong for switching between two: every
switch meant editing six values at once and commenting out the six they replaced, which is
how `.env` ended up carrying three commented-out API keys and two commented-out endpoints
with nothing recording which set belonged together.

So the sets are named here instead. Each provider owns a prefixed block —

    GROQ_API_KEY      GROQ_BASE_URL      GROQ_MODEL      ...
    TOGETHER_API_KEY  TOGETHER_BASE_URL  TOGETHER_MODEL  ...

— and `LLM_PROVIDER` picks which block is live. Both blocks stay in `.env` at all times;
switching back is one word, not six edits, and neither set can be half-applied.

Resolution for each setting, most specific first:

    <PREFIX>_<NAME>   the selected provider's block
    <NAME>            the generic variable, shared by every provider
    built-in default  base_url only, since a provider's endpoint is a fact about it

The provider block WINS over the generic name. That ordering is the point: `.env` already
sets `MODEL` and `BASE_URL`, and if the generic name won, `LLM_PROVIDER` would be a
variable that changes nothing. The generic names keep their old meaning as the value used
when the live block does not name one — which is what lets a setting that is genuinely the
same everywhere (a model id both providers serve, a vision endpoint deliberately left on a
third) stay written once.

Leaving `LLM_PROVIDER` unset is a supported state and a total no-op: nothing is resolved,
nothing is written, and the generic variables reach the modules exactly as they always
have. That is what a deployment predating this file gets.

The resolved values are written back into `os.environ` by `apply_provider_env()`, called
from `backend.env.load_env()` before anything imports a module that reads them. Publishing
through the environment rather than through an accessor is deliberate: the eight reading
modules capture their values at import, several of them at module scope, and an accessor
would have to reach all of them to be true. One write, before the first import, reaches
all of them by construction.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, Mapping, MutableMapping, Optional, Tuple

logger = logging.getLogger(__name__)

#: The variable that picks a block. Unset means "use the generic names as-is".
SELECTOR = "LLM_PROVIDER"


@dataclass(frozen=True)
class ProviderSpec:
    """One provider's block: what to call it, and what it defaults to."""

    name: str
    label: str
    prefix: str
    default_base_url: str = ""


#: Every provider the backend knows how to be pointed at. All of them speak the
#: OpenAI-compatible protocol `init_chat_model(model_provider="openai")` sends, which is
#: the only reason a single set of variables can serve them all — adding one is a row
#: here plus its block in `.env`, with no code change anywhere else.
PROVIDERS: Dict[str, ProviderSpec] = {
    "groq": ProviderSpec(
        name="groq",
        label="Groq",
        prefix="GROQ",
        default_base_url="https://api.groq.com/openai/v1",
    ),
    "together": ProviderSpec(
        name="together",
        label="Together AI",
        prefix="TOGETHER",
        default_base_url="https://api.together.xyz/v1",
    ),
}

#: The generic names a block can override, in the order they are reported.
SETTINGS: Tuple[str, ...] = (
    "ARK_API_KEY",
    "BASE_URL",
    "MODEL",
    "FAST_MODEL",
    "GRADE_MODEL",
    "VISION_MODEL",
    "VISION_API_KEY",
    "VISION_BASE_URL",
)

#: `ARK_API_KEY` is the one generic name whose prefixed spelling is not the name with a
#: prefix bolted on: `GROQ_ARK_API_KEY` would be nonsense, and every provider's own docs
#: call the variable `<PROVIDER>_API_KEY`.
_PREFIXED_SUFFIX: Dict[str, str] = {"ARK_API_KEY": "API_KEY"}

#: Settings a provider spec can supply a built-in default for. Only the endpoint: a model
#: id is a choice the deployment makes and a key is a secret, so neither has a default
#: that could be right.
_SPEC_DEFAULTS: Dict[str, str] = {"BASE_URL": "default_base_url"}


def prefixed_name(provider: ProviderSpec, setting: str) -> str:
    """The block variable that overrides `setting` for `provider`."""
    return f"{provider.prefix}_{_PREFIXED_SUFFIX.get(setting, setting)}"


def _clean(value: Optional[str]) -> str:
    """Blank counts as unset, matching `backend.env.env_value`.

    `.env` files carry `FOO=` for a value someone meant to disable, and a block whose
    `TOGETHER_MODEL=` is empty must fall through to the generic `MODEL` rather than
    resolve to `""` and reach a provider as a request for a model with no name.
    """
    return (value or "").strip()


@dataclass(frozen=True)
class Resolution:
    """What the selected block resolved to, and where each value came from.

    `values` holds only the settings the block (or its built-in default) supplies —
    anything left to the generic variable is already in the environment under its own
    name and is deliberately not repeated here, so that `apply_provider_env()` writes the
    minimum. `sources` covers every setting that has a value at all, block or generic,
    because the startup line's job is to say where each one came from.
    """

    provider: ProviderSpec
    values: Dict[str, str]
    sources: Dict[str, str]

    def effective(self, setting: str, environ: Optional[Mapping[str, str]] = None) -> str:
        """The value `setting` ends up with, block or generic."""
        env = os.environ if environ is None else environ
        return self.values.get(setting) or _clean(env.get(setting))

    def describe(self, environ: Optional[Mapping[str, str]] = None) -> str:
        """Diagnostic line. Never includes a key — this reaches logs."""
        model = self.effective("MODEL", environ) or "unset"
        base_url = self.effective("BASE_URL", environ) or "provider default"
        return (
            f"provider={self.provider.name} ({self.provider.label}), "
            f"model={model}, endpoint={base_url} "
            f"(from {self.sources.get('BASE_URL', 'unset')}), "
            f"key from {self.sources.get('ARK_API_KEY', 'unset')}"
        )


class UnknownProviderError(ValueError):
    """`LLM_PROVIDER` names a provider with no block defined."""


def active_provider(environ: Optional[Mapping[str, str]] = None) -> Optional[ProviderSpec]:
    """The selected provider, or None when `LLM_PROVIDER` is unset.

    Raises on a name with no block rather than falling back to the generic variables. The
    fallback would be a defined state — it is the pre-`LLM_PROVIDER` behaviour — but a
    deployment that wrote `LLM_PROVIDER=togather` asked for a specific provider and would
    instead get whichever one the generic variables happen to describe, discovering it as
    a 401 from an endpoint it never named. A typo is a boot-time error here.
    """
    env = os.environ if environ is None else environ
    raw = _clean(env.get(SELECTOR))
    if not raw:
        return None
    provider = PROVIDERS.get(raw.lower())
    if provider is None:
        known = ", ".join(sorted(PROVIDERS))
        raise UnknownProviderError(
            f"{SELECTOR}={raw!r} names no known provider; expected one of {known} "
            f"(or leave {SELECTOR} unset to use the generic variables directly)"
        )
    return provider


def resolve(provider: ProviderSpec, environ: Optional[Mapping[str, str]] = None) -> Resolution:
    """Apply `provider`'s block over the generic variables, without writing anything."""
    env = os.environ if environ is None else environ
    values: Dict[str, str] = {}
    sources: Dict[str, str] = {}

    for setting in SETTINGS:
        block_name = prefixed_name(provider, setting)
        from_block = _clean(env.get(block_name))
        if from_block:
            values[setting] = from_block
            sources[setting] = block_name
            continue

        if _clean(env.get(setting)):
            # Already in the environment under this name; nothing to write.
            sources[setting] = setting
            continue

        spec_field = _SPEC_DEFAULTS.get(setting)
        default = getattr(provider, spec_field, "") if spec_field else ""
        if default:
            values[setting] = default
            sources[setting] = f"{provider.name} default"

    return Resolution(provider=provider, values=values, sources=sources)


def apply_provider_env(
    environ: Optional[MutableMapping[str, str]] = None,
) -> Optional[Resolution]:
    """Resolve `LLM_PROVIDER` into the generic variables. Returns None when unset.

    Must run before the first import of a module that reads a model variable, which in
    practice means from `backend.env.load_env()` and nowhere else.
    """
    env = os.environ if environ is None else environ
    provider = active_provider(env)
    if provider is None:
        return None

    resolution = resolve(provider, env)
    for setting, value in resolution.values.items():
        env[setting] = value
    return resolution


def log_provider_status(resolution: Optional[Resolution] = None) -> None:
    """One line at boot saying which provider the model calls are going to.

    Worth a line of its own: with two blocks in `.env` and one selector, the failure this
    guards against is a deployment that believes it switched. Every model call in the
    request path goes wherever this says.
    """
    if resolution is None:
        try:
            provider = active_provider()
        except UnknownProviderError as exc:
            logger.error("LLM provider: %s", exc)
            return
        if provider is None:
            logger.info(
                "LLM provider: none selected (%s unset) — "
                "using MODEL/BASE_URL/ARK_API_KEY as set",
                SELECTOR,
            )
            return
        resolution = resolve(provider)
    logger.info("LLM provider: %s", resolution.describe())


__all__ = [
    "PROVIDERS",
    "SELECTOR",
    "SETTINGS",
    "ProviderSpec",
    "Resolution",
    "UnknownProviderError",
    "active_provider",
    "apply_provider_env",
    "log_provider_status",
    "prefixed_name",
    "resolve",
]
