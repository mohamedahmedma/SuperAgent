"""Turning a verified WhatsApp challenge into a parent's session.

The step between `WhatsAppLoginService.verify` and a token. It exists as its own use case
because it does three things that are each a policy decision, and none of them belong in
an HTTP handler:

**The account is created here, on first use, rather than by an administrator.** The
binding written onto it does not come from the request: it came from the school's own
records, keyed on a number WhatsApp proved. That is the invariant `domain/accounts.py`
states — an account never names its own guardian — held through a second authority rather
than broken by one.

**The binding is re-asserted on every sign-in.** A registrar who corrects a guardian record
in `sis/` should see it take effect the next time that parent signs in, without an
administrator having to touch this service as well.

**The children claim is best-effort and never blocks the sign-in.** See `_children_claim`.
"""
from __future__ import annotations

import logging

from identity.application.dto import IssuedSession
from identity.application.ports.directory import GuardianDirectory
from identity.application.ports.repositories import Account, AccountRepository, AuditSink
from identity.application.services.sessions import SessionService
from identity.domain.accounts import DEFAULT_ROLE, guardian_username
from identity.domain.errors import GuardianDirectoryUnavailable, NotAuthorized

logger = logging.getLogger(__name__)


class ParentSessionService:
    """Signing in a parent whose identity WhatsApp and the SIS have jointly established."""

    def __init__(
        self,
        *,
        accounts: AccountRepository,
        sessions: SessionService,
        directory: GuardianDirectory,
        audit: AuditSink,
    ) -> None:
        self._accounts = accounts
        self._sessions = sessions
        self._directory = directory
        self._audit = audit

    def sign_in(self, challenge, *, client_ip: str = "") -> IssuedSession:
        """The tokens a verified challenge earns.

        `school_code` is read off the stored challenge rather than from anything the
        browser sent: by this point it has been agreed by both halves of the flow — the
        page that started it, and the WhatsApp number the parent's message arrived on.
        """
        account = self._account_for(challenge)
        school_code = (challenge.school_code or "") or None

        session = self._sessions.issue_session(
            account,
            children=self._children_claim(account.guardian_external_id, school_code),
            school_code=school_code,
        )
        self._audit.write(
            username=account.username,
            event="whatsapp_verify",
            reason="ok",
            succeeded=True,
            client_ip=client_ip,
        )
        return session

    def audit_failure(self, *, event: str, reason: str, client_ip: str = "") -> None:
        """Record a verification that did not succeed.

        Exposed because the router is the only place that knows a refusal happened *and*
        holds a transaction to write it in — `api/errors.py`, which turns the error into a
        status, has neither. A failed verification is precisely the event an incident
        review looks for, so it is recorded with the same weight as a successful one.
        """
        self._audit.write(
            username="", event=event, reason=reason, succeeded=False, client_ip=client_ip
        )

    def _account_for(self, challenge) -> Account:
        """Find or create the account this guardian signs in through.

        Keyed on the guardian handle rather than on the phone number, so a parent who
        verifies her second number lands in the account she already had instead of
        acquiring a duplicate holding half her history.

        The account carries no password. No hash verification can succeed against the
        empty string stored here, so the password route stays shut for parents — this is
        the only door they have, and it is one the school closes by removing a guardian
        link rather than by resetting anything.
        """
        username = guardian_username(challenge.guardian_external_id)
        account = self._accounts.by_username(username)

        if account is None:
            return self._accounts.create(
                username=username,
                password_hash="",
                role=DEFAULT_ROLE,
                guardian_external_id=challenge.guardian_external_id,
                display_name=challenge.display_name,
                preferred_language=challenge.preferred_language or "ar",
            )

        # An account an administrator disabled stays disabled: re-verifying a phone must
        # not become a way to walk back that decision.
        if not account.is_active:
            raise NotAuthorized("That account is disabled.")

        self._accounts.set_guardian_binding(account, challenge.guardian_external_id)
        if challenge.display_name:
            self._accounts.set_display_name(account, challenge.display_name)
        return account

    def _children_claim(
        self, guardian_external_id: str | None, school_code: str | None
    ) -> list[dict]:
        """The children to stamp into a token, or nothing at all.

        **Never raises, and never returns a partial answer as though it were complete.** A
        directory outage yields `[]`, which the claim assembly then omits — so the token
        says nothing about this parent's family rather than saying they have none. The chat
        backend reads the roster itself and will simply do so a moment later.

        Signing in must not depend on this. A parent whose sign-in failed because a
        convenience claim could not be assembled would be locked out by a feature that
        exists to save them one HTTP call — which is why the directory this service is
        given for `children_of` carries a **tighter timeout** than the one the sign-in
        itself uses. See `config.children_timeout_seconds`: the call sits inside the
        latency a parent is waiting on, and a slow SIS must cost them a claim, not a wait.
        """
        if not guardian_external_id:
            return []
        try:
            found = self._directory.children_of(
                guardian_external_id, school_code=school_code
            )
        except GuardianDirectoryUnavailable:
            logger.warning(
                "Could not read this parent's children while minting a token; the token "
                "will carry none and the chat backend will look them up itself."
            )
            return []
        except Exception:  # noqa: BLE001 - a convenience claim must never break sign-in
            logger.exception("Unexpected failure assembling the children claim")
            return []
        return [child.as_claim() for child in found]


__all__ = ["ParentSessionService"]
