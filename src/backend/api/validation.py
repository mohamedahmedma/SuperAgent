"""Request checks that belong at the HTTP boundary rather than in a service.

Two kinds, both about what THIS deployment will accept, and both reading the profile - which the
application layer may not import, because the profile module reaches the tool registry and,
through it, the graph framework:

  * whether an uploaded file is one this profile ingests;
  * whether a feature this profile can switch off is switched on.

Each raises a domain error, so the status is decided in `backend/api/errors.py` like any other.
The shape of a request body is Pydantic's to check (see `backend/api/schemas/`); these cover
what a schema cannot know, because the answer depends on the deployment.
"""

from __future__ import annotations

from backend.api.resources import is_supported_document
from backend.domain.errors import FeatureDisabled, InvalidInput
from backend.profiles import get_profile


def require_supported_document(filename: str) -> str:
    """The filename, when it names a file this profile ingests. Otherwise `InvalidInput`."""
    if not filename:
        raise InvalidInput("Filename cannot be empty")
    if not is_supported_document(filename):
        raise InvalidInput(get_profile().user_copy.unsupported_file_type)
    return filename


def require_assets_enabled() -> None:
    """Refuse, as though the route did not exist, when this deployment has assets off."""
    if not get_profile().assets.enabled:
        raise FeatureDisabled("Asset support is disabled for this deployment.")


__all__ = ["require_assets_enabled", "require_supported_document"]
