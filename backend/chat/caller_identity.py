"""Who a turn is being served for.

One immutable value object instead of three loose strings threaded through the route,
the service and the context. That is the whole design decision, and it buys three
things worth having:

**Adding a fourth identity fact touches one class.** A school year, a preferred
language, a staff department — each would otherwise mean editing two service
signatures, two context constructors and two routes, and every one of those edits is a
chance to drop the value silently on the streaming path but not the sync one.

**It cannot be mutated mid-turn.** Frozen, so a tool handed the context cannot rewrite
whose records the turn is allowed to read. The identity is decided once, at the HTTP
boundary, by code that has verified a signature.

**It does not know what produced it.** `from_principal` reads attributes off whatever
the auth layer hands it rather than importing a concrete user type, so `backend.chat`
keeps no dependency on `backend.infra.auth` and swapping the authentication provider
changes nothing here.

The token deserves particular care: it is a live bearer credential for another
service. The generated `__repr__` is overridden so it cannot reach a log line, a
traceback, or a captured test failure, and `close()` on the context drops it.
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class CallerIdentity:
    """The verified subject of one chat turn.

    Every field is server-derived. None of it is model-visible: no prompt renders it,
    no tool takes it as an argument, and the model has no way to read or influence it.

    `guardian_id` and `guardian_token` are empty for anyone who is not a signed-in
    parent — staff, an ordinary user, a test, an unauthenticated turn. Empty is the
    correct and safe default: the records tool refuses rather than guessing.
    """

    user_id: str
    guardian_id: str = ""
    guardian_token: str = ""

    @property
    def is_parent(self) -> bool:
        """Whether this turn can read student records at all.

        Both halves are required. A guardian id without a token cannot be proved to
        the records facade, and a token without an id has no subject to ask about — so
        neither alone is a parent session, and treating one as such would produce a
        confusing failure deep in a tool instead of a clear refusal up front.
        """
        return bool(self.guardian_id) and bool(self.guardian_token)

    @classmethod
    def for_user(cls, user_id: str) -> CallerIdentity:
        """An identity with no guardian binding.

        The default for every existing caller — background jobs, tests, and any turn
        that was never authenticated as a parent.
        """
        return cls(user_id=user_id or "")

    @classmethod
    def from_principal(cls, principal) -> CallerIdentity:
        """Build from whatever the authentication layer returned.

        Structural, not nominal: anything exposing `username`, `guardian_id` and
        `access_token` works. That keeps this module free of any import from
        `backend.infra`, so the chat layer does not depend on the authentication
        implementation and replacing the identity provider is a change in one place.

        Missing attributes degrade to empty rather than raising. A principal type that
        predates guardians is a legitimate caller; it simply is not a parent.
        """
        return cls(
            user_id=str(getattr(principal, "username", "") or ""),
            guardian_id=str(getattr(principal, "guardian_id", "") or ""),
            guardian_token=str(getattr(principal, "access_token", "") or ""),
        )

    def without_credentials(self) -> CallerIdentity:
        """The same identity with the bearer token dropped.

        For anywhere the identity outlives the request that authorised it — a retained
        context, a queued job, a diagnostic dump. Who the caller was is still useful;
        the ability to act as them is not.
        """
        return replace(self, guardian_token="")

    def __repr__(self) -> str:
        """Redacted. The token is a live credential for another service.

        The dataclass-generated repr would print it in full, and a repr reaches places
        nobody audits: log lines, tracebacks, captured test output, error trackers.
        Presence is reported because "is there a token" is the question worth asking
        while debugging; the value never is.
        """
        return (
            f"CallerIdentity(user_id={self.user_id!r}, "
            f"guardian_id={self.guardian_id!r}, "
            f"guardian_token={'<redacted>' if self.guardian_token else ''!r})"
        )
