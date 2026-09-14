"""Refresh-token rotation, at the use-case layer with a clock we control.

`test_auth.py` proves the same rules over HTTP against SQLite. These run in microseconds
with fakes, which is what lets them move time: a session's inactivity window is a year,
and the interesting behaviour — a token that slides, a replay caught days later, a family
pruned — is all about what happens after longer than a test can wait.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from identity.application.ports.repositories import RefreshTokenRecord
from identity.application.services.sessions import SessionService
from identity.domain.accounts import LockoutPolicy
from identity.domain.errors import NotAuthorized

START = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
REFRESH_TTL = timedelta(days=365)


class Clock:
    def __init__(self, now: datetime = START) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs) -> None:
        self.now += timedelta(**kwargs)


@dataclass
class _Row:
    account_id: int
    token_hash: str
    family_id: str
    expires_at: datetime
    revoked_at: datetime | None = None
    rotated_at: datetime | None = None
    replaced_by_hash: str | None = None


class FakeRefreshTokens:
    def __init__(self) -> None:
        self.rows: dict[str, _Row] = {}

    def issue(self, *, account_id, token_hash, expires_at, family_id):
        self.rows[token_hash] = _Row(account_id, token_hash, family_id, expires_at)

    def find(self, token_hash):
        row = self.rows.get(token_hash)
        if row is None:
            return None
        return RefreshTokenRecord(
            row.account_id, row.token_hash, row.family_id, row.expires_at, row.revoked_at, row.rotated_at
        )

    def mark_rotated(self, token_hash, *, replaced_by_hash, at):
        self.rows[token_hash].rotated_at = at
        self.rows[token_hash].replaced_by_hash = replaced_by_hash

    def revoke(self, token_hash):
        row = self.rows.get(token_hash)
        if row is None or row.revoked_at is not None:
            return False
        row.revoked_at = START
        return True

    def revoke_family(self, account_id, family_id):
        count = 0
        for row in self.rows.values():
            if row.account_id == account_id and row.family_id == family_id and row.revoked_at is None:
                row.revoked_at = START
                count += 1
        return count

    def revoke_all_for_account(self, account_id):
        return self.revoke_family(account_id, next(iter(self.rows.values())).family_id)

    def prune(self, account_id, *, dead_before, now):
        dead = [
            key for key, row in self.rows.items()
            if row.account_id == account_id and (
                row.expires_at <= now
                or (row.revoked_at is not None and row.revoked_at < dead_before)
                or (row.rotated_at is not None and row.rotated_at < dead_before)
            )
        ]
        for key in dead:
            del self.rows[key]
        return len(dead)


class FakeIssuer:
    """Mints predictable tokens: refresh tokens are numbered, hashes are sha256."""

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self.minted = 0

    def mint_access_token(self, **claims):
        return f"access-for-{claims['subject']}", self._clock() + timedelta(minutes=30)

    def mint_refresh_token(self):
        self.minted += 1
        raw = f"refresh-{self.minted}"
        return raw, self.hash_refresh_token(raw), self._clock() + REFRESH_TTL

    @staticmethod
    def hash_refresh_token(raw):
        return hashlib.sha256(raw.encode()).hexdigest()

    def decode_own_token(self, token):
        raise ValueError("not used here")


class FakeAudit:
    def __init__(self) -> None:
        self.events = []

    def write(self, **event):
        self.events.append(event)


PARENT = SimpleNamespace(
    id=7, username="0501234567", phone="", password_hash="x", role="parent",
    guardian_external_id="G-1", display_name="Umm Layla", preferred_language="ar",
    is_active=True, failed_attempts=0, locked_until=None,
)


class FakeAccounts:
    def by_id(self, account_id):
        return PARENT if account_id == PARENT.id else None

    def by_username(self, username):
        return PARENT if username == PARENT.username else None


@pytest.fixture()
def world():
    clock = Clock()
    tokens = FakeRefreshTokens()
    audit = FakeAudit()
    service = SessionService(
        accounts=FakeAccounts(),
        refresh_tokens=tokens,
        audit=audit,
        hasher=SimpleNamespace(verify=lambda *a: True, needs_rehash=lambda h: False, dummy_hash="d"),
        issuer=FakeIssuer(clock),
        lockout=LockoutPolicy(max_failed_attempts=8, lockout_minutes=15),
        refresh_reuse_grace_seconds=60,
        clock=clock,
    )
    session = service.issue_session(PARENT)
    return SimpleNamespace(service=service, tokens=tokens, audit=audit, clock=clock, first=session.refresh_token)


def test_a_refresh_spends_the_token_and_issues_a_new_one_in_the_same_family(world):
    issued = world.service.refresh(refresh_token=world.first)

    assert issued.refresh_token != world.first
    first = world.tokens.find(FakeIssuer.hash_refresh_token(world.first))
    second = world.tokens.find(FakeIssuer.hash_refresh_token(issued.refresh_token))
    assert first.rotated_at == world.clock.now
    assert second.family_id == first.family_id
    assert second.rotated_at is None


def test_the_session_slides_with_every_refresh(world):
    """Used on day 300 of a 365-day window, the session runs to day 665 — using the app is
    what keeps a parent signed in, not when they last typed a password."""
    world.clock.advance(days=300)
    issued = world.service.refresh(refresh_token=world.first)

    assert issued.refresh_expires_at == START + timedelta(days=300) + REFRESH_TTL

    world.clock.advance(days=300)  # day 600: the ORIGINAL token would be long expired
    again = world.service.refresh(refresh_token=issued.refresh_token)
    assert again.refresh_expires_at == START + timedelta(days=600) + REFRESH_TTL


def test_an_unused_session_ends_when_its_window_closes(world):
    world.clock.advance(days=366)
    with pytest.raises(NotAuthorized):
        world.service.refresh(refresh_token=world.first)
    assert world.audit.events[-1]["reason"] == "expired_refresh"


def test_a_retry_inside_the_grace_window_is_honoured(world):
    """The response to a refresh was lost, or a second tab still held the old token."""
    world.service.refresh(refresh_token=world.first)
    world.clock.advance(seconds=20)

    retried = world.service.refresh(refresh_token=world.first)

    assert retried.refresh_token
    family = world.tokens.find(FakeIssuer.hash_refresh_token(world.first)).family_id
    assert world.tokens.find(FakeIssuer.hash_refresh_token(retried.refresh_token)).family_id == family


def test_a_replay_after_the_grace_window_revokes_the_whole_family(world):
    issued = world.service.refresh(refresh_token=world.first)
    world.clock.advance(seconds=61)

    with pytest.raises(NotAuthorized):
        world.service.refresh(refresh_token=world.first)

    assert world.audit.events[-1]["reason"] == "refresh_reuse"
    # The thief's copy and the parent's live token are both dead: the parent signs in again,
    # the thief cannot.
    with pytest.raises(NotAuthorized):
        world.service.refresh(refresh_token=issued.refresh_token)


def test_the_grace_window_is_measured_from_the_first_exchange(world):
    """A copy presented every fifty seconds must not stay alive by always being 'inside
    the window' of its own last use."""
    world.service.refresh(refresh_token=world.first)
    world.clock.advance(seconds=50)
    world.service.refresh(refresh_token=world.first)  # a retry, honoured
    world.clock.advance(seconds=50)                    # 100s after the FIRST exchange

    with pytest.raises(NotAuthorized):
        world.service.refresh(refresh_token=world.first)
    assert world.audit.events[-1]["reason"] == "refresh_reuse"


def test_logout_revokes_the_family_not_just_the_token_presented(world):
    issued = world.service.refresh(refresh_token=world.first)
    world.service.logout(refresh_token=issued.refresh_token)

    # The predecessor is inside its grace window and would otherwise still mint.
    with pytest.raises(NotAuthorized):
        world.service.refresh(refresh_token=world.first)
    with pytest.raises(NotAuthorized):
        world.service.refresh(refresh_token=issued.refresh_token)


def test_logout_with_an_unknown_token_still_reports_success(world):
    assert world.service.logout(refresh_token="never-issued") is True


def test_spent_tokens_are_pruned_once_a_replay_would_no_longer_be_recognised(world):
    """Rotation writes a row per refresh. A week of them is kept for replay detection;
    older spent rows are removed on the next refresh, so the table stays the size of the
    live sessions plus a week of history."""
    token = world.first
    for _ in range(3):
        token = world.service.refresh(refresh_token=token).refresh_token
    assert len(world.tokens.rows) == 4

    world.clock.advance(days=8)
    world.service.refresh(refresh_token=token)

    live = {row.token_hash for row in world.tokens.rows.values()}
    assert FakeIssuer.hash_refresh_token(world.first) not in live, "spent eight days ago: pruned"
    assert FakeIssuer.hash_refresh_token(token) in live, "spent just now: kept, to catch a replay"
    assert len(world.tokens.rows) == 2


def test_a_token_from_before_rotation_existed_is_a_family_of_one(world):
    """Rows written by the previous release carry no family. They keep working, and are
    rotated into a family on their first refresh."""
    legacy_hash = FakeIssuer.hash_refresh_token("legacy")
    world.tokens.issue(account_id=PARENT.id, token_hash=legacy_hash, expires_at=START + REFRESH_TTL, family_id=legacy_hash)

    issued = world.service.refresh(refresh_token="legacy")

    assert world.tokens.find(FakeIssuer.hash_refresh_token(issued.refresh_token)).family_id == legacy_hash
