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
from identity.domain.accounts import assignable_role, guard_last_administrator
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

    # -- managing accounts --------------------------------------------------
    #
    # These replace what self-registration and a shared admin key used to cover between
    # them. Note what none of them can do: write `guardian_external_id`. There is no
    # parameter for it on any of them, so "update the account" cannot become "grant access
    # to a family" through a field somebody added to a form — the binding keeps its own
    # route, its own audit event, and its own deliberate act.

    def list_accounts(self, *, limit: int = 50, offset: int = 0) -> tuple[list, int]:
        """One page of accounts and the total, for a management screen.

        Returns `AccountSummary` objects, which carry no password hash and no token. That
        is not a filtering step that could be forgotten — the DTO has nowhere to put one.
        """
        limit = max(1, min(limit, 200))
        offset = max(0, offset)
        accounts = self._accounts.list_page(limit=limit, offset=offset)
        return (
            [
                AccountSummary(
                    username=account.username,
                    role=account.role,
                    guardian_external_id=account.guardian_external_id,
                    is_active=account.is_active,
                    display_name=account.display_name,
                )
                for account in accounts
            ],
            self._accounts.count(),
        )

    def update_account(
        self,
        *,
        username: str,
        password: str | None = None,
        role: str | None = None,
        display_name: str | None = None,
        phone: str | None = None,
        preferred_language: str | None = None,
        is_active: bool | None = None,
    ) -> AccountSummary:
        """Change what an account is. Every field is optional; absent means unchanged.

        Two rules are enforced here rather than at the route, because both must hold no
        matter who calls and a script is as capable of breaking them as a form is:

        **The last administrator cannot be demoted or deactivated.** Losing every admin
        means nobody can bind a parent to their children until somebody edits the database.

        **Changing the password revokes existing sessions.** A password is changed either
        because it leaked or because somebody left; in both cases the refresh tokens issued
        under the old one are exactly what an attacker would still be holding. Leaving them
        alive would make the change cosmetic for up to a full refresh lifetime.
        """
        account = self._accounts.by_username(username)
        if account is None:
            raise NotFound("No such account.")

        was_active_admin = account.role == "admin" and account.is_active
        stays_admin = account.role == "admin" if role is None else assignable_role(role) == "admin"
        stays_active = account.is_active if is_active is None else is_active
        guard_last_administrator(
            removing_an_active_admin=was_active_admin and not (stays_admin and stays_active),
            other_active_admins=self._accounts.count_active_admins(
                excluding_id=account.id
            ),
        )

        if role is not None:
            self._accounts.set_role(account, assignable_role(role))
        if display_name is not None:
            self._accounts.set_display_name(account, display_name)
        if phone is not None:
            self._accounts.set_phone(account, phone)
        if preferred_language is not None:
            self._accounts.set_preferred_language(account, preferred_language)
        if is_active is not None:
            self._accounts.set_active(account, is_active)
        if password is not None:
            self._accounts.set_password_hash(account, self._hasher.hash(password))
            self._refresh.revoke_all_for_account(account.id)

        self._audit.write(
            username=username, event="account_update", reason="ok", succeeded=True
        )
        return AccountSummary(
            username=account.username,
            role=account.role,
            guardian_external_id=account.guardian_external_id,
            is_active=account.is_active,
            display_name=account.display_name,
        )

    def delete_account(self, *, username: str) -> None:
        """Remove an account, and the sessions it is holding.

        The revocation is not tidiness. An access token already minted stays valid until it
        expires — that is the trade offline verification makes — so deleting the row alone
        leaves a browser able to act as a person who no longer exists for up to one access
        lifetime. Revoking the refresh tokens bounds it to that, instead of letting the
        session renew itself indefinitely against an account nobody can see any more.

        A seeded administrator deleted here returns on the next restart. That is the
        bootstrap guarantee doing its job rather than a bug, and it is why removing one for
        good means clearing IDENTITY_BOOTSTRAP_ADMIN_USER as well.
        """
        account = self._accounts.by_username(username)
        if account is None:
            raise NotFound("No such account.")

        guard_last_administrator(
            removing_an_active_admin=account.role == "admin" and account.is_active,
            other_active_admins=self._accounts.count_active_admins(
                excluding_id=account.id
            ),
        )

        self._refresh.revoke_all_for_account(account.id)
        self._accounts.delete(account)
        self._audit.write(
            username=username, event="account_delete", reason="ok", succeeded=True
        )


__all__ = ["AdministrationService"]
