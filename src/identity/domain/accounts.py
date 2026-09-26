"""What an account may be, and the rules that govern one.

Pure. No SQLAlchemy, no FastAPI, no environment — the lockout rule is a function of a
count, a threshold and a clock, and it is tested by calling it with three values rather
than by arranging a database and eight failed HTTP requests.

The invariant this module exists to state is `guardian_external_id`. That single value is
the mapping from "someone who signed in" to "a guardian the records facade will answer
about", and it exists in exactly one place in the whole estate. Nothing may derive it,
infer it, or accept it from a request body. **Two authorities may write it, and only
two** — an administrator through the admin route, and the WhatsApp verification flow,
which does not take the value from anybody's request either: it proves control of a phone
number, asks the school's own system of record which guardian that number belongs to, and
writes the answer it is given. Both share the property the rule is actually about: the
account never names its own guardian.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final

from identity.domain.errors import Forbidden

#: What an administrator may assign through the admin route.
ASSIGNABLE_ROLES: Final[frozenset[str]] = frozenset({"user", "admin", "parent", "staff"})

#: What an account created by the WhatsApp flow, or by an admin who named nothing, gets.
DEFAULT_ROLE: Final[str] = "parent"

#: The username a WhatsApp-verified parent signs in under.
#:
#: Keyed on the guardian handle rather than on the phone number, so a parent who verifies
#: her second number lands in the account she already had instead of acquiring a duplicate
#: holding half her history. A username built from a phone would have to change when she
#: changes number, and a username is a join key elsewhere.
_GUARDIAN_USERNAME_PREFIX: Final[str] = "guardian:"


def guardian_username(guardian_external_id: str) -> str:
    """The stable username for a guardian who signs in over WhatsApp."""
    return f"{_GUARDIAN_USERNAME_PREFIX}{guardian_external_id}"


def is_guardian_username(username: str) -> bool:
    return username.startswith(_GUARDIAN_USERNAME_PREFIX)


def assignable_role(requested: str | None) -> str:
    """The role an administrator asked for, or the default when it is not one we assign.

    Falls back rather than refusing, because the admin route's job is to create the
    account: a typo'd role should produce a parent who can sign in and read nothing, not
    a 400 that leaves a bulk import half-done.
    """
    role = (requested or "").strip().lower()
    return role if role in ASSIGNABLE_ROLES else DEFAULT_ROLE


def guard_last_administrator(
    *, removing_an_active_admin: bool, other_active_admins: int
) -> None:
    """Refuse a change that would leave the estate with no administrator at all.

    The three ways to commit that mistake are deleting the last admin, demoting them, and
    deactivating them — so this is checked by all three rather than by the delete route
    alone, which is the version of this guard everybody writes first and which the other
    two walk straight past.

    Why it matters more here than in most systems: the admin routes are the ONLY way to
    bind a parent to their children. Lose every administrator and nobody can onboard a
    family until somebody edits the database by hand. Recovery is possible — the seeded
    account returns on the next restart — but only if `IDENTITY_BOOTSTRAP_ADMIN_USER` and
    `IDENTITY_BOOTSTRAP_ADMIN_PASSWORD` are still set, and a school that has since removed
    them from `.env` has no way back in at all.

    Counted rather than named, so this stays a rule about the estate and not about who is
    calling. An administrator IS allowed to demote or delete themselves while another one
    exists — that is ordinary staff turnover, and forbidding it would mean the last person
    to leave can never tidy up after themselves.
    """
    if removing_an_active_admin and other_active_admins == 0:
        raise Forbidden(
            "This is the only administrator left. Create another one before removing, "
            "demoting or deactivating this account."
        )


@dataclass(frozen=True, slots=True)
class LockoutPolicy:
    """How many bad passwords an account tolerates, and for how long it then refuses.

    **Per account, not per IP.** The threat is credential stuffing against a known parent,
    and an attacker with a botnet has more IPs than the school has parents. The cost is
    that an attacker can lock a parent out deliberately, which is the better failure: a
    locked-out parent phones the school, whereas a breached one does not know to.
    """

    max_failed_attempts: int
    lockout_minutes: int

    def next_failure(self, failed_attempts: int, *, now: datetime) -> tuple[int, datetime | None]:
        """The `(failed_attempts, locked_until)` an account takes after one bad password.

        The counter resets when the lock is applied rather than continuing to climb, so a
        second lockout takes the same number of attempts as the first. A counter that kept
        rising would make each subsequent lockout arrive sooner, which reads as flakiness
        to a parent who fat-fingers a password twice a month.
        """
        attempts = (failed_attempts or 0) + 1
        if attempts >= self.max_failed_attempts:
            return 0, now + timedelta(minutes=self.lockout_minutes)
        return attempts, None

    @staticmethod
    def is_locked(locked_until: datetime | None, *, now: datetime) -> bool:
        """Whether a lock is still in force. `None` means it never was."""
        if locked_until is None:
            return False
        return as_aware(locked_until) > now


def as_aware(moment: datetime) -> datetime:
    """Re-attach UTC to a datetime SQLite handed back naive.

    SQLite returns naive datetimes even from a timezone-aware column, so every comparison
    against `now` raises `TypeError` — and only under SQLite, which means only in
    development and in tests, which is the worst place for a type error to live.
    """
    from datetime import timezone

    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


__all__ = [
    "guard_last_administrator",
    "ASSIGNABLE_ROLES",
    "DEFAULT_ROLE",
    "LockoutPolicy",
    "as_aware",
    "assignable_role",
    "guardian_username",
    "is_guardian_username",
]
