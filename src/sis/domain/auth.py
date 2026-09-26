"""API-key identity: a prefix that *names* a key, a hash that *authenticates* it.

**The full secret is never stored.** What the caller sends is hashed and compared; what
this service keeps is the hash plus the leading `prefix`. Those two fields do different
jobs and neither can do the other's.

The prefix is the human-readable handle. It is what an audit row, a log line and the
registrar UI show, so a person reading "key `sis_a1b2c3d4` imported 300 grades at 02:00"
can point at one row and revoke it. Storing only a hash would make that impossible: a
hash is not something an operator can recognise or match against the key in a colleague's
config file, so the response to a leak becomes "rotate everything", which is the response
nobody takes until it is too late. The prefix is deliberately not secret — it identifies,
it does not authorise.

The hash is the verifier and is never shown, never logged, and never returned by the API.
It is excluded from this dataclass's `repr` for that reason: a `logger.info("auth: %s",
key)` written in a hurry must not put the verifier into a log file that is backed up,
shipped and read by more people than the database ever will be.

**The domain never reads the clock.** Expiry is answered against a `now` the caller
supplies, so a service that decides "is this key still valid" is unit-testable with a
fixed timestamp and no patching of `datetime`.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Final

from sis.domain.errors import ValidationError

# Enough leading characters to be unambiguous in a log while remaining useless to an
# attacker: it names the key, it does not narrow the search for the rest of it.
PREFIX_LENGTH: Final[int] = 12


class Scope(StrEnum):
    """What a key may do. Two levels, checked by exact equality.

    `StrEnum` so the member serialises to its own literal (`"registrar"`, `"reader"`) in
    JSON and in a database column without a translation table that can drift from the
    enum it mirrors.
    """

    REGISTRAR = "registrar"
    READER = "reader"

    def permits(self, required: "Scope") -> bool:
        """Exact equality. `registrar` does **not** implicitly satisfy `reader`.

        The tempting alternative is an ordering — registrar outranks reader, so a
        registrar key passes a reader check. It reads as convenience and it is how a
        read-only integration quietly gains write access: someone hands the reporting
        dashboard a registrar key "just to unblock it", every endpoint accepts it because
        registrar satisfies everything, and the dashboard now has a credential that can
        rewrite a term's grades. Nothing in the logs distinguishes that from a registrar
        doing their job.

        With equality, a key that is meant to read is given `reader` and is *refused* by
        every write route, so the mistake is caught at the first request instead of at
        the first incident. A caller that legitimately needs both holds two keys.
        """
        return self is required


@dataclass(frozen=True, slots=True)
class ApiKey:
    """A stored credential: how it is named, how it is verified, and when it stops.

    Frozen because revocation and expiry are *new state written by a service*, not a
    mutation of the object a request is currently being authorised against. A key that
    could flip `is_active` mid-request is a key whose authorisation decision depends on
    when in the handler it was asked.
    """

    prefix: str
    key_hash: str = field(repr=False)  # verifier; kept out of every accidental log line
    label: str
    scope: Scope
    is_active: bool
    expires_at: datetime | None
    created_at: datetime
    last_used_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.prefix.strip():
            raise ValidationError("api key prefix is required", field="prefix")
        if not self.key_hash.strip():
            raise ValidationError("api key hash is required", field="key_hash")
        if not self.label.strip():
            # A key nobody can attribute to a system is a key nobody dares revoke.
            raise ValidationError("api key label is required", field="label")
        for name in ("expires_at", "created_at", "last_used_at"):
            moment: datetime | None = getattr(self, name)
            # A naive datetime and an aware `now` raise TypeError on comparison, so a
            # single timezone-less column would turn every authenticated request into a
            # 500 rather than a refusal. Refuse it here, where the value is constructed.
            if moment is not None and moment.tzinfo is None:
                raise ValidationError(
                    f"{name} must be timezone-aware", field=name
                )

    def is_expired_at(self, now: datetime) -> bool:
        """`None` expiry means no expiry — a key that must be revoked deliberately."""
        return self.expires_at is not None and self.expires_at <= now

    def is_usable_at(self, now: datetime) -> bool:
        """Revoked or expired, independent of what the caller is trying to do."""
        return self.is_active and not self.is_expired_at(now)

    def authorises(self, required: Scope, *, now: datetime) -> bool:
        """The whole decision: live key, exact scope. Both, or nothing."""
        return self.is_usable_at(now) and self.scope.permits(required)

    def __str__(self) -> str:
        # Printing a key prints its handle. See the module docstring.
        return self.prefix
