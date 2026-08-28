"""Creating accounts, and the one write that decides which family somebody can read.

Two operations, kept apart on purpose.

**Creation and binding are two separate calls.** A bulk parent import that runs only the
first produces accounts that can log in and read nothing, which is the safe half-finished
state. Fused into one call, the same interrupted import produces accounts bound to
whatever guardian id the row happened to carry, and nobody knows which rows got that far.

**Unbinding revokes refresh tokens.** It is the urgent custody path: the binding stops
applying to new access tokens immediately, existing ones die within their remaining
lifetime, and the session itself dies now.
"""
from __future__ import annotations

from identity.application.dto import AccountSummary
from identity.application.ports.repositories import (
    AccountRepository,
    AuditSink,
    RefreshTokenRepository,
)
from identity.application.ports.security import PasswordHasher
from identity.domain.accounts import assignable_role
from identity.domain.errors import Conflict, NotFound


class AdministrationService:
    """The admin-key routes, with no HTTP in them.

    Every method here is also reachable from a script — `import_legacy_accounts.py` uses
    the same repositories — which is the practical reason none of them raises
    `HTTPException`.
    """

    def __init__(
        self,
        *,
        accounts: AccountRepository,
        refresh_tokens: RefreshTokenRepository,
        audit: AuditSink,
        hasher: PasswordHasher,
    ) -> None:
        self._accounts = accounts
        self._refresh = refresh_tokens
        self._audit = audit
        self._hasher = hasher

    def create_account(
        self,
        *,
        username: str,
        password: str,
        role: str | None = None,
        phone: str = "",
        display_name: str = "",
        preferred_language: str = "ar",
    ) -> AccountSummary:
        """Create a login. Note what this cannot do: bind a guardian.

        The returned `guardian_external_id` is always `None`, and that is not an
        approximation — there is no argument to this method that could make it anything
        else.
        """
        if self._accounts.by_username(username) is not None:
            raise Conflict("Username already exists.")

        account = self._accounts.create(
            username=username,
            password_hash=self._hasher.hash(password),
            role=assignable_role(role),
            phone=phone,
            display_name=display_name,
            preferred_language=preferred_language,
        )
        return AccountSummary(
            username=account.username, role=account.role, guardian_external_id=None
        )

    def bind_guardian(self, *, username: str, guardian_external_id: str) -> AccountSummary:
        """Bind a login to a guardian. The single most sensitive write in the system.

        Audited as its own event type, because "who decided this parent is that guardian"
        is the first question anyone asks after a records leak, and it must be answerable
        without correlating two ordinary account-update lines.
        """
        account = self._accounts.by_username(username)
        if account is None:
            raise NotFound("No such account.")

        self._accounts.set_guardian_binding(account, guardian_external_id)
        self._audit.write(
            username=username, event="guardian_bind", reason="ok", succeeded=True
        )
        return AccountSummary(
            username=username,
            role=account.role,
            guardian_external_id=account.guardian_external_id,
        )

    def unbind_guardian(self, *, username: str) -> AccountSummary:
        """Remove a binding, and kill the sessions that were using it.

        Both halves matter and neither is sufficient. Clearing the column alone leaves an
        access token in a browser asserting the old binding for up to its full lifetime,
        and revoking the refresh tokens alone lets the next login re-establish it.
        """
        account = self._accounts.by_username(username)
        if account is None:
            raise NotFound("No such account.")

        self._accounts.set_guardian_binding(account, None)
        self._refresh.revoke_all_for_account(account.id)
        self._audit.write(
            username=username, event="guardian_unbind", reason="ok", succeeded=True
        )
        return AccountSummary(
            username=username, role=account.role, guardian_external_id=None
        )


__all__ = ["AdministrationService"]
