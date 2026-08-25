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

from identity import guardians, schools, whatsapp as wa
from identity.verification import DEFAULT_TTL_MINUTES, SchoolChannel, VerificationService


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


def _channel_for(school_code: str | None) -> SchoolChannel:
    """Everything the flow needs to talk to one school: number, gateway, directory.

    In single-school mode `school_code` is `None` and this is the process-wide trio, which
    is exactly what the service was handed before schools were separated.

    In multi-school mode the number and the gateway come from that school's own settings,
    so a code goes back out through the number the parent messaged. The directory is the
    shared HTTP client either way — one SIS base URL — because the school travels on the
    request as `X-School-Code` rather than by pointing at a different service. That is what
    lets a school be moved to its own database, or later its own server, by changing SIS's
    registry and nothing here.
    """
    if school_code is None:
        return SchoolChannel(
            code=None,
            business_number=wa.get_business_number(),
            gateway=wa.get_gateway(),
            directory=guardians.get_directory(),
        )
    school = schools.get_registry().by_code(school_code)
    return SchoolChannel(
        code=school.code,
        business_number=school.number,
        gateway=wa.get_gateway(school.code),
        directory=guardians.get_directory(),
    )


def get_verification_service() -> VerificationService:
    """One service per request, over the process-wide gateways and directory.

    Cheap to build — it holds no connection of its own; both seams keep their own pooled
    clients — so there is nothing to gain from caching it and something to lose: a cached
    instance would pin whichever gateways were selected the first time a request arrived.

    The resolver is passed rather than a fixed channel, so the school is chosen per call
    from what the request actually proves — the login page for `start`, the WhatsApp number
    the message arrived on for `claim` — instead of being fixed when this object is built.
    """
    return VerificationService(
        channel_for=_channel_for,
        ttl_minutes=_ttl_minutes(),
    )
