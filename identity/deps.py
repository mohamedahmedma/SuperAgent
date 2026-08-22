"""Composition for the request-scoped pieces of the WhatsApp flow.

Small on purpose, and separate from `routes.py` so a test can replace the service through
`app.dependency_overrides` without reaching into a module global. The two seams themselves
are chosen once at startup — see `identity/app.py` — because which gateway this process
talks to is a property of the deployment, not of a request.

Configuration is read here and in `app.py` and nowhere else. `verification.py` takes its
TTL and its business number as constructor arguments for the reason every service in this
estate does: a use case that reads the environment cannot be unit-tested without arranging
one, and an expiry rule cannot be tested at all without either injecting the clock or
sleeping for ten minutes.
"""
from __future__ import annotations

import os

from identity import guardians, whatsapp as wa
from identity.verification import DEFAULT_TTL_MINUTES, VerificationService


def _ttl_minutes() -> int:
    """Minutes a challenge stays usable, from the environment, falling back on junk.

    A typo'd `IDENTITY_VERIFICATION_TTL_MINUTES=ten` must not take parent login down; it
    should run with the documented default, which is what `sis.config._int_env` does for
    the same reason.
    """
    raw = os.getenv("IDENTITY_VERIFICATION_TTL_MINUTES")
    if not raw:
        return DEFAULT_TTL_MINUTES
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_TTL_MINUTES
    return value if value > 0 else DEFAULT_TTL_MINUTES


def get_verification_service() -> VerificationService:
    """One service per request, over the process-wide gateway and directory.

    Cheap to build — it holds no connection of its own; both seams keep their own pooled
    clients — so there is nothing to gain from caching it and something to lose: a cached
    instance would pin whichever gateway was selected the first time a request arrived.
    """
    return VerificationService(
        gateway=wa.get_gateway(),
        directory=guardians.get_directory(),
        business_number=wa.get_business_number(),
        ttl_minutes=_ttl_minutes(),
    )
