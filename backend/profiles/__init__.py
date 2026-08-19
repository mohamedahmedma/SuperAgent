"""Domain profiles: the single source of truth for identity, prompts, and behaviour.

Import `get_profile` and read the section you need:

    from backend.profiles import get_profile

    profile = get_profile()
    prompt = profile.render_system_prompt()
    top_k = profile.retrieval.top_k

Select the active profile with the ACTIVE_PROFILE environment variable (default:
"supermew" — the profile id/filename, not the display identity it composes to, which
is "SuperAgent"). Individual values can still be overridden by their original environment
variables — see ENV_OVERRIDES in registry.py for the full list.
"""
from backend.profiles.registry import (
    DEFAULT_PROFILE,
    ENV_OVERRIDES,
    PROFILE_ENV_VAR,
    ProfileError,
    available_profiles,
    get_profile,
    load_profile,
    reload_profile,
    set_profile,
)
from backend.profiles.schema import (
    AgentConfig,
    ChunkingConfig,
    CopyConfig,
    DomainProfile,
    IdentityConfig,
    IngestConfig,
    ModelConfig,
    RagConfig,
    RetrievalConfig,
)

__all__ = [
    "DEFAULT_PROFILE",
    "ENV_OVERRIDES",
    "PROFILE_ENV_VAR",
    "ProfileError",
    "available_profiles",
    "get_profile",
    "load_profile",
    "reload_profile",
    "set_profile",
    "AgentConfig",
    "ChunkingConfig",
    "CopyConfig",
    "DomainProfile",
    "IdentityConfig",
    "IngestConfig",
    "ModelConfig",
    "RagConfig",
    "RetrievalConfig",
]
