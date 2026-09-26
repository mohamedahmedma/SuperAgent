"""Signing in with a password, refreshing, and signing out.

The whole of what used to sit inline in `routes.py`, with no FastAPI in it. That matters
beyond tidiness: the rules below — an unknown user and a wrong password are
indistinguishable, a refresh re-reads the binding, unbinding kills the session — are the
security posture of the estate, and they were previously stated in the middle of HTTP
handlers where the only way to test one was to make a request.

## Two rules that read like implementation details and are not

**Wrong password and unknown user are the same answer, in the same time.** Otherwise this
endpoint confirms which parents are registered at the school, one guess at a time. The
message is identical, the status is identical, and — see `_reject_unknown_user` — so is
the CPU cost.

**Refresh re-reads the binding** from the account rather than copying it out of the old
token. A custody change then takes effect within one access-token lifetime instead of
persisting until the parent happens to log out. The case that matters is a court order,
and it must not wait a month.

## Staying signed in, and how a stolen token is caught

A parent stays signed in until they sign out. The mechanism is the one the OAuth 2.0
Security Best Current Practice (RFC 9700, §4.14) describes as refresh token rotation:

* Every refresh **spends** the token presented and issues a new one, valid for a full
  inactivity window from now. Using the app is what keeps the session alive; a phone that
  has not opened it for the whole window is no longer signed in.
* Every token issued from one sign-in shares a **family**. Signing out revokes the family,
  and so does the one thing a spent token can still tell us: that somebody presented it
  again. A browser that lost the response to its refresh, or a second tab holding the
  token a moment longer than the first, does that within seconds, so a short **grace
  window** honours the retry. Outside it the token was copied, and revoking the family
  signs out the thief and the parent alike — the parent signs in again; the thief cannot.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

from identity.application.dto import IssuedAccessToken, IssuedSession, TokenSubject
from identity.application.ports.repositories import (
    Account,
    AccountRepository,
    AuditSink,
    RefreshTokenRecord,
    RefreshTokenRepository,
)
from identity.application.ports.security import PasswordHasher, TokenIssuer
from identity.domain.accounts import LockoutPolicy
from identity.domain.errors import (
    AccountLocked,
    BadRequest,
    Conflict,
    NotAuthorized,
)

logger = logging.getLogger(__name__)


class SessionService:
    """Everything that turns a credential into a token, and back out again.

    Holds no connection and no state; the repositories it is given are already bound to
    one request's transaction. Building one per request is a few attribute assignments.
    """

    #: How long a spent refresh token is kept past its grace window, so that presenting it
    #: is still recognised as a replay and ends the family. Beyond this an old token is
    #: merely unknown: refused, but without the detection. A week catches a token copied
    #: from a device days before it is used; keeping every spent token for the whole
    #: inactivity window would grow the table by one row per half hour per parent.
    REPLAY_MEMORY = timedelta(days=7)

    def __init__(
        self,
        *,
        accounts: AccountRepository,
        refresh_tokens: RefreshTokenRepository,
        audit: AuditSink,
        hasher: PasswordHasher,
        issuer: TokenIssuer,
        lockout: LockoutPolicy,
        refresh_reuse_grace_seconds: int = 60,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._accounts = accounts
        self._refresh = refresh_tokens
        self._audit = audit
        self._hasher = hasher
        self._issuer = issuer
        self._lockout = lockout
        self._reuse_grace = timedelta(seconds=refresh_reuse_grace_seconds)
        self._clock = clock

    # -- signing in ---------------------------------------------------------

    def login(self, *, username: str, password: str, client_ip: str = "") -> IssuedSession:
        """Exchange credentials for tokens, or refuse without saying why.

        The order of the checks below is load-bearing. Lockout is tested before the
        password, so an account under attack stops doing key derivation entirely rather
        than paying 100ms of PBKDF2 per guess — which is the difference between a lockout
        that protects the account and one that merely annoys the attacker while the
        service burns CPU on their behalf.
        """
        account = self._accounts.by_username(username)
        if account is None:
            return self._reject_unknown_user(username, password, client_ip)

        if self._lockout.is_locked(account.locked_until, now=self._clock()):
            self._audit.write(
                username=username, event="login", reason="locked", succeeded=False, client_ip=client_ip
            )
            raise AccountLocked()

        if not account.is_active:
            self._audit.write(
                username=username, event="login", reason="inactive", succeeded=False, client_ip=client_ip
            )
            raise NotAuthorized()

        if not self._hasher.verify(password, account.password_hash):
            self._accounts.register_failure(account, self._lockout, now=self._clock())
            self._audit.write(
                username=username, event="login", reason="bad_password", succeeded=False, client_ip=client_ip
            )
            raise NotAuthorized()

        self._accounts.register_success(account)
        self._upgrade_hash_if_needed(account, password)

        session = self.issue_session(account)
        self._audit.write(
            username=username, event="login", reason="ok", succeeded=True, client_ip=client_ip
        )
        return session

    def _reject_unknown_user(self, username: str, password: str, client_ip: str) -> IssuedSession:
        """Refuse a username we do not hold, in the time it takes to refuse one we do.

        **Exactly one key derivation**, against a hash precomputed at startup. The obvious
        spelling of this — hashing a throwaway string and then verifying against it — runs
        the derivation twice on a miss and once on a hit. That does not equalise the
        timing; it inverts it, making an unknown username measurably *slower* than a known
        one, leaving the enumeration oracle open in the other direction and doubling the
        CPU cost of the most-attacked endpoint in the estate.
        """
        self._hasher.verify(password, self._hasher.dummy_hash)
        self._audit.write(
            username=username, event="login", reason="unknown_user", succeeded=False, client_ip=client_ip
        )
        raise NotAuthorized()

    def _upgrade_hash_if_needed(self, account: Account, password: str) -> None:
        """Re-hash a just-verified password into the current format.

        The only moment a legacy bcrypt hash imported from the old backend can be
        upgraded: the plaintext is in hand and already proven correct. This is what let
        accounts migrate from the old system without a forced password reset — a migration
        that reset every family's password would have been abandoned halfway and left the
        old auth running forever.

        A failure here must not fail the login. The user authenticated; the worst case is
        that the upgrade happens on their next sign-in instead.
        """
        if not self._hasher.needs_rehash(account.password_hash):
            return
        try:
            self._accounts.set_password_hash(account, self._hasher.hash(password))
        except Exception:  # noqa: BLE001 - an upgrade must never cost a successful login
            logger.exception("Password hash upgrade failed for %s", account.username)

    # -- staying signed in --------------------------------------------------

    def refresh(self, *, refresh_token: str, client_ip: str = "") -> IssuedAccessToken:
        """Exchange a refresh token for a fresh access token and a fresh refresh token.

        The guardian binding is re-read from the account here, not carried over from the
        old token — see the module docstring for why that is the point of the endpoint
        rather than a detail of it.

        The token presented is spent by this call; the one returned lives a full
        inactivity window from now. A spent token presented again inside the grace window
        is a retry and is honoured, in the same family. Outside it, it is a replay: the
        whole family is revoked and the caller is refused — with the same message every
        other refusal here uses, because which of them applied is not for the caller to
        learn.
        """
        now = self._clock()
        presented = self._issuer.hash_refresh_token(refresh_token)
        found = self._refresh.find(presented)
        if found is None or found.revoked_at is not None or found.expires_at <= now:
            self._audit.write(
                username="", event="refresh", reason="expired_refresh", succeeded=False, client_ip=client_ip
            )
            raise NotAuthorized("Invalid or expired refresh token.")

        account = self._accounts.by_id(found.account_id)
        if account is None or not account.is_active:
            raise NotAuthorized("Invalid or expired refresh token.")

        if self._is_replay(found, now):
            self._refresh.revoke_family(found.account_id, found.family_id)
            self._audit.write(
                username=account.username, event="refresh", reason="refresh_reuse", succeeded=False, client_ip=client_ip
            )
            logger.warning(
                "Refresh token replayed for %s; the session family was revoked", account.username
            )
            raise NotAuthorized("Invalid or expired refresh token.")

        access_token, expires_at = self._issuer.mint_access_token(
            subject=account.username,
            role=account.role,
            guardian_external_id=account.guardian_external_id,
            display_name=account.display_name,
        )
        raw_refresh, refresh_hash, refresh_expires = self._issuer.mint_refresh_token()
        self._refresh.issue(
            account_id=account.id,
            token_hash=refresh_hash,
            expires_at=refresh_expires,
            family_id=found.family_id,
        )
        if found.rotated_at is None:
            # A retry inside the grace window leaves the first rotation on record: the
            # window is measured from the FIRST exchange, or a thief could keep a copy
            # alive by presenting it every fifty seconds.
            self._refresh.mark_rotated(presented, replaced_by_hash=refresh_hash, at=now)
        self._refresh.prune(account.id, dead_before=now - self.REPLAY_MEMORY, now=now)
        self._audit.write(
            username=account.username, event="refresh", reason="ok", succeeded=True, client_ip=client_ip
        )
        return IssuedAccessToken(
            access_token=access_token,
            expires_at=expires_at,
            refresh_token=raw_refresh,
            refresh_expires_at=refresh_expires,
        )

    def _is_replay(self, token: RefreshTokenRecord, now: datetime) -> bool:
        """A spent token presented again after its grace window was copied."""
        return token.rotated_at is not None and now - token.rotated_at > self._reuse_grace

    def logout(self, *, refresh_token: str) -> bool:
        """Sign out: revoke every refresh token of the session this one belongs to.

        The family and not the one token, because the browser may hold a token whose
        predecessor is still inside its grace window — revoking only what it presented
        would leave that predecessor able to mint. The access token already issued stays
        valid until it expires: offline verification is the trade made for not calling
        this service on every request, and keeping access tokens short is what bounds that
        window.

        Always reports success. Whether that particular token existed is not something a
        caller holding it needs told, and not something a caller *not* holding it should
        be able to find out.
        """
        found = self._refresh.find(self._issuer.hash_refresh_token(refresh_token))
        if found is not None:
            self._refresh.revoke_family(found.account_id, found.family_id)
        return True

    def describe_token(self, token: str) -> TokenSubject:
        """Decode the caller's own token. Useful to a front end; used by nothing critical."""
        from datetime import datetime as _datetime

        claims = self._issuer.decode_own_token(token)
        return TokenSubject(
            username=claims.get("sub", ""),
            role=claims.get("role", ""),
            guardian_external_id=claims.get("guardian_id"),
            display_name=claims.get("name", ""),
            expires_at=_datetime.fromtimestamp(claims["exp"], tz=timezone.utc),
        )

    # -- issuing ------------------------------------------------------------

    def issue_session(self, account: Account, **claims) -> IssuedSession:
        """One access token, one refresh token, recorded. Every door ends here.

        Public because the WhatsApp door is a separate use case —
        `ParentSessionService` — and a parent's token is not a second kind of token. It is
        the same token with two more claims on it, and the two paths must not drift into
        minting subtly different ones.
        """
        access_token, expires_at = self._issuer.mint_access_token(
            subject=account.username,
            role=account.role,
            guardian_external_id=account.guardian_external_id,
            display_name=account.display_name,
            **claims,
        )
        raw_refresh, refresh_hash, refresh_expires = self._issuer.mint_refresh_token()
        # The first token names the family every later rotation of this sign-in joins.
        self._refresh.issue(
            account_id=account.id,
            token_hash=refresh_hash,
            expires_at=refresh_expires,
            family_id=refresh_hash,
        )
        return IssuedSession(
            access_token=access_token,
            refresh_token=raw_refresh,
            expires_at=expires_at,
            username=account.username,
            role=account.role,
            guardian_external_id=account.guardian_external_id,
            display_name=account.display_name,
        )


__all__ = ["SessionService"]
