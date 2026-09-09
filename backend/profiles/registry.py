"""Profile discovery, composition, and environment overrides.

Resolution order for any single value (highest wins):

    environment variable  >  profile YAML (incl. inherited)  >  schema default

Environment stays on top so that an existing `.env` deployment keeps behaving exactly
as it did before profiles existed, and so a single container can be re-tuned without
editing a packaged YAML file. Profiles express *what a domain is*; env expresses
*how this one deployment is dialled in*.
"""
from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from backend.profiles.schema import DomainProfile

logger = logging.getLogger(__name__)

DEFINITIONS_DIR = Path(__file__).resolve().parent / "definitions"
DEFAULT_PROFILE = "supermew"
PROFILE_ENV_VAR = "ACTIVE_PROFILE"

# Environment variable -> dotted path in the profile. The retrieval and chunking rows
# predate profiles, which makes this table a backward-compatibility contract as well as
# a convenience: removing one of those rows silently changes a live deployment. Later
# rows are values a deployment is expected to tune per environment rather than per
# domain — model sampling above all, since it is what a turn costs.
ENV_OVERRIDES: Dict[str, str] = {
    # Identity
    "REDIS_KEY_PREFIX": "identity.redis_key_prefix",
    "LANGSMITH_PROJECT": "identity.langsmith_project",
    # Retrieval
    "RETRIEVAL_TOP_K": "retrieval.top_k",
    "RETRIEVAL_CANDIDATE_K": "retrieval.candidate_k",
    "RETRIEVAL_CANDIDATE_MULTIPLIER": "retrieval.candidate_multiplier",
    "LEAF_RETRIEVE_LEVEL": "retrieval.leaf_retrieve_level",
    "AUTO_MERGE_ENABLED": "retrieval.auto_merge_enabled",
    "AUTO_MERGE_THRESHOLD": "retrieval.auto_merge_threshold",
    "RERANK_MIN_SCORE": "retrieval.rerank_min_score",
    "RERANK_DOC_CHAR_LIMIT": "retrieval.rerank_doc_char_limit",
    "RERANK_TIMEOUT_SECONDS": "retrieval.rerank_timeout_seconds",
    # Models — sampling per role. Effort and the output ceiling are the two knobs that
    # decide what a turn costs on a reasoning model, so they are dialled per deployment
    # rather than baked into the domain profile.
    "ANSWER_TEMPERATURE": "models.answer_temperature",
    "FAST_TEMPERATURE": "models.fast_temperature",
    "PLANNER_TEMPERATURE": "models.planner_temperature",
    "GRADE_TEMPERATURE": "models.grade_temperature",
    "REWRITE_TEMPERATURE": "models.rewrite_temperature",
    "SCOPE_TEMPERATURE": "models.scope_temperature",
    "RESOLVE_TEMPERATURE": "models.resolve_temperature",
    "ANSWER_REASONING_EFFORT": "models.answer_reasoning_effort",
    "FAST_REASONING_EFFORT": "models.fast_reasoning_effort",
    "PLANNER_REASONING_EFFORT": "models.planner_reasoning_effort",
    "GRADE_REASONING_EFFORT": "models.grade_reasoning_effort",
    "REWRITE_REASONING_EFFORT": "models.rewrite_reasoning_effort",
    "SCOPE_REASONING_EFFORT": "models.scope_reasoning_effort",
    "RESOLVE_REASONING_EFFORT": "models.resolve_reasoning_effort",
    "ANSWER_MAX_TOKENS": "models.answer_max_tokens",
    "FAST_MAX_TOKENS": "models.fast_max_tokens",
    "PLANNER_MAX_TOKENS": "models.planner_max_tokens",
    "GRADE_MAX_TOKENS": "models.grade_max_tokens",
    "REWRITE_MAX_TOKENS": "models.rewrite_max_tokens",
    "SCOPE_MAX_TOKENS": "models.scope_max_tokens",
    "RESOLVE_MAX_TOKENS": "models.resolve_max_tokens",
    # Agent
    "CONTEXT_WINDOW_MESSAGES": "agent.context_window_messages",
    # Contextual query resolution. Reachable from the environment because it is the one
    # switch worth flipping without a redeploy: it costs a small call per referential
    # turn, and a deployment hitting a quota wall needs to be able to stop paying it.
    "QUERY_RESOLUTION_ENABLED": "agent.query_resolution_enabled",
    # Vision — the model that reads a figure into text AT INGEST. Nothing reads pixels
    # at query time any more, so this is an ingest cost, but it is still tuned per
    # deployment for the same reason as the rest.
    #
    # NOTE: this cannot express "no reasoning". FigurePipelineConfig folds "none"/"off"
    # to "", which means SEND NO effort parameter and leaves the provider's default on.
    # To actually turn a scratchpad off, set assets.figures.vision_extra_params in the
    # profile — see school.yaml.
    "VISION_REASONING_EFFORT": "assets.figures.vision_reasoning_effort",
    # Chunking
    "CHUNK_STRATEGY": "chunking.strategy",
    "CHUNK_SIZE": "chunking.chunk_size",
    "CHUNK_OVERLAP": "chunking.chunk_overlap",
    "CHUNK_L1_SIZE": "chunking.l1_size",
    "CHUNK_L2_SIZE": "chunking.l2_size",
    "CHUNK_SENTENCE_OVERLAP": "chunking.sentence_overlap",
    "LAYOUT_PARSER_ENABLED": "chunking.layout_parser_enabled",
    "SEMANTIC_DEDUP_ENABLED": "chunking.semantic_dedup_enabled",
    "SEMANTIC_DEDUP_THRESHOLD": "chunking.semantic_dedup_threshold",
}

_MAX_INHERITANCE_DEPTH = 8
_cached_profile: Optional[DomainProfile] = None


class ProfileError(RuntimeError):
    """Raised for an unusable profile. Always fatal: a misconfigured profile changes
    what the assistant says and how it retrieves, so failing at startup is strictly
    better than serving wrong behaviour."""


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursive dict merge. Lists REPLACE rather than concatenate: a profile that
    narrows `agent.tools` or swaps a marker vocabulary must be able to shrink the
    inherited list, which append semantics would make impossible."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_yaml(name: str) -> Dict[str, Any]:
    path = DEFINITIONS_DIR / f"{name}.yaml"
    if not path.exists():
        raise ProfileError(
            f"Profile {name!r} not found at {path}. Available: {', '.join(available_profiles()) or '(none)'}"
        )
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ProfileError(f"Profile {name!r} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ProfileError(f"Profile {name!r} must be a YAML mapping, got {type(data).__name__}")
    return data


def _inheritance_chain(name: str) -> List[str]:
    """The profile and everything it extends, most specific first.

    Split out because the prompt packs need the same walk: a profile with no pack of its
    own should fall back to the pack of whatever it extends, exactly as its YAML values
    do. Deriving the order twice from the same `extends` links is what keeps the two
    from disagreeing.
    """
    chain: List[str] = []
    seen: set[str] = set()
    current: Optional[str] = name

    while current:
        if current in seen:
            raise ProfileError(
                f"Circular profile inheritance: {' -> '.join([*chain, current])}"
            )
        if len(chain) >= _MAX_INHERITANCE_DEPTH:
            raise ProfileError(
                f"Profile inheritance deeper than {_MAX_INHERITANCE_DEPTH}: {' -> '.join(chain)}"
            )
        seen.add(current)
        chain.append(current)
        current = _load_yaml(current).get("extends")

    return chain


def _resolve_inheritance(name: str) -> Dict[str, Any]:
    """Walk the `extends` chain to its root, then merge back down so the requested
    profile wins over everything it inherits."""
    chain = _inheritance_chain(name)
    merged: Dict[str, Any] = {}
    for profile_name in reversed(chain):
        merged = _deep_merge(merged, _load_yaml(profile_name))
    # `extends` is a composition instruction, not inherited state; the resolved
    # profile records only the name that was actually requested.
    merged["name"] = name
    merged.pop("extends", None)
    return merged


def _coerce(raw: str, current: Any, env_name: str) -> Any:
    """Coerce an env string to the type of the value it overrides.

    `current` may legitimately be None (an unset Optional such as candidate_k), so the
    target type is taken from the schema default when the composed value is None.
    """
    text = raw.strip()
    if isinstance(current, bool):
        if text.lower() in ("true", "1", "yes", "on"):
            return True
        if text.lower() in ("false", "0", "no", "off"):
            return False
        logger.warning("Invalid boolean %s=%r — keeping %r", env_name, raw, current)
        return current
    if isinstance(current, int) and not isinstance(current, bool):
        try:
            return int(text)
        except ValueError:
            logger.warning("Invalid integer %s=%r — keeping %r", env_name, raw, current)
            return current
    if isinstance(current, float):
        try:
            return float(text)
        except ValueError:
            logger.warning("Invalid float %s=%r — keeping %r", env_name, raw, current)
            return current
    if isinstance(current, list):
        return [item.strip() for item in text.split(",") if item.strip()]
    return text


def _apply_env_overrides(data: Dict[str, Any], profile: DomainProfile) -> Dict[str, Any]:
    """Overlay environment variables onto composed profile data.

    `profile` is the already-validated composed profile, used only to learn each
    target's type so coercion works even when the YAML omitted the key.
    """
    result = copy.deepcopy(data)
    for env_name, dotted in ENV_OVERRIDES.items():
        raw = os.getenv(env_name)
        if raw is None or not raw.strip():
            continue

        # Paths are walked to arbitrary depth rather than split once: nested sections
        # are real (assets.figures.*), and a two-level-only resolver would silently
        # ignore an override rather than fail, which is the worst of both outcomes.
        *parents, field_name = dotted.split(".")
        section = profile
        for part in parents:
            section = getattr(section, part, None)
            if section is None:
                break
        if section is None or not hasattr(section, field_name):
            logger.warning("ENV_OVERRIDES maps %s to unknown path %r — ignoring", env_name, dotted)
            continue

        current = getattr(section, field_name)
        if current is None:
            # Unset Optional: fall back to the declared annotation via the field default
            # of a fresh section instance, then to string.
            current = type(section).model_fields[field_name].default

        bucket = result
        for part in parents:
            existing = bucket.get(part)
            if not isinstance(existing, dict):
                # The YAML omitted the section, or holds a scalar where the schema
                # declares one. Either way the override still has to land, and
                # validation downstream is what reports a genuinely malformed profile.
                existing = {}
                bucket[part] = existing
            bucket = existing
        bucket[field_name] = _coerce(raw, current, env_name)
    return result


#: Prompt text a profile can carry, and the pack template each field is loaded from.
#: Dotted path -> filename under `backend/prompts/templates/packs/<profile>/`.
#:
#: These are the last prompts that still lived in profile YAML. Everything else already
#: reaches its template through `backend.prompts.resolve`; see that module's docstring
#: for why a profile answers "what is this deployment" and a prompt answers "how does
#: the system talk", and why only the first belongs in YAML.
PROMPT_PACK_FIELDS: Dict[str, str] = {
    "identity.persona": "persona.j2",
    "agent.system_prompt": "system.j2",
    "agent.arabic_style_prompt": "arabic_style.j2",
}


def _read_at(data: Dict[str, Any], dotted: str) -> Any:
    section: Any = data
    for part in dotted.split("."):
        if not isinstance(section, dict) or part not in section:
            return None
        section = section[part]
    return section


def _write_at(data: Dict[str, Any], dotted: str, value: Any) -> None:
    *parents, field = dotted.split(".")
    bucket = data
    for part in parents:
        existing = bucket.get(part)
        if not isinstance(existing, dict):
            existing = {}
            bucket[part] = existing
        bucket = existing
    bucket[field] = value


def _apply_prompt_packs(data: Dict[str, Any], chain: List[str]) -> Dict[str, Any]:
    """Fill prompt fields from `packs/<profile>/*.j2`, most specific profile first.

    YAML still wins. A deployment that has already tuned a prompt in its profile keeps
    that text and never notices the pack exists — the same reversibility
    `backend.prompts.resolve` promises for every other prompt.

    Rendered through the Jinja environment rather than read as bytes, so a pack is a
    template like every other prompt in that folder: comments are stripped, and the
    files are covered by the test that compiles all of them.

    A missing pack is not an error. Most profiles carry only a persona, and a custom
    profile that ships none at all falls through to the schema defaults exactly as it
    did before packs existed.
    """
    from backend.prompts import TEMPLATE_ROOT, render

    result = copy.deepcopy(data)
    for dotted, filename in PROMPT_PACK_FIELDS.items():
        if _read_at(result, dotted):
            continue  # the profile set it in YAML; that is still the override
        for profile_name in chain:
            relative = f"packs/{profile_name}/{filename}"
            if not (TEMPLATE_ROOT / relative).exists():
                continue
            try:
                _write_at(result, dotted, render(relative))
            except Exception as exc:  # a broken pack must name itself, not the profile
                raise ProfileError(f"Prompt pack {relative!r} failed to render: {exc}") from exc
            break
    return result


def available_profiles() -> List[str]:
    if not DEFINITIONS_DIR.exists():
        return []
    return sorted(path.stem for path in DEFINITIONS_DIR.glob("*.yaml"))


def load_profile(name: Optional[str] = None) -> DomainProfile:
    """Compose, validate, and env-overlay a profile. Not cached — use `get_profile`."""
    profile_name = (name or os.getenv(PROFILE_ENV_VAR) or DEFAULT_PROFILE).strip()

    composed = _resolve_inheritance(profile_name)
    composed = _apply_prompt_packs(composed, _inheritance_chain(profile_name))
    try:
        profile = DomainProfile.model_validate(composed)
    except Exception as exc:
        raise ProfileError(f"Profile {profile_name!r} failed validation: {exc}") from exc

    with_env = _apply_env_overrides(composed, profile)
    try:
        return DomainProfile.model_validate(with_env)
    except Exception as exc:
        raise ProfileError(
            f"Profile {profile_name!r} became invalid after environment overrides: {exc}"
        ) from exc


def get_profile() -> DomainProfile:
    """The active profile for this process.

    Cached because it is read at module import time across the backend; the cost of a
    reload is a YAML parse plus validation, and callers treat it as a constant.
    """
    global _cached_profile
    if _cached_profile is None:
        _cached_profile = load_profile()
        logger.info(
            "Loaded domain profile %r (v%d) — %s",
            _cached_profile.name,
            _cached_profile.profile_version,
            _cached_profile.identity.display_name,
        )
    return _cached_profile


def set_profile(profile: Optional[DomainProfile]) -> None:
    """Replace the cached profile. For tests and for the CLI profile inspector —
    production code reads `get_profile()` and never swaps it mid-process."""
    global _cached_profile
    _cached_profile = profile


def reload_profile(name: Optional[str] = None) -> DomainProfile:
    set_profile(None)
    if name is not None:
        os.environ[PROFILE_ENV_VAR] = name
    return get_profile()
