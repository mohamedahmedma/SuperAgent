"""Managing accounts through the API: list, update, delete.

Only create, bind and unbind existed before. An operator could bring an account into being
and could never afterwards change its password, change its role, suspend it, or remove it —
so the answer to "somebody left the school" was to edit the database by hand.

Two rules run through all of it, and both are enforced in the service rather than at the
route, because a script is as capable of breaking them as a form is:

**Nothing here can write `guardian_external_id`.** Binding keeps its own route and its own
audit event. An update path that accepted the field would reopen from one side the door that
`AccountIn` already closes on the other — and it is the one write that decides which family
somebody can read.

**The last active administrator cannot be removed, demoted, or deactivated.** The admin
routes are the only way to bind a parent to their children, so losing every administrator
locks the school out of onboarding until the seeded account is restored by a restart.
"""
import pytest

from tests.identity.conftest import BOOTSTRAP_ADMIN_PASSWORD, BOOTSTRAP_ADMIN_USER


@pytest.fixture()
def staff(client, admin_headers):
    """An ordinary account to operate on, so tests never mutate the suite's administrator."""
    client.post(
        "/v1/admin/accounts",
        headers=admin_headers,
        json={
            "username": "0501234567",
            "password": "correct-horse-battery",
            "role": "staff",
            "display_name": "Ms Warda",
        },
    )
    return "0501234567"


class TestListing:
    def test_it_returns_the_accounts_and_a_total(self, client, admin_headers, staff):
        response = client.get("/v1/admin/accounts", headers=admin_headers)

        assert response.status_code == 200
        body = response.json()
        assert body["total"] >= 2  # the seeded administrator, and `staff`
        assert staff in [a["username"] for a in body["accounts"]]

    def test_it_never_returns_a_password_hash(self, client, admin_headers, staff):
        """Asserted over the raw body, not the parsed model.

        A response model that happens to omit the field today is not the same guarantee as
        the bytes never containing it, and this is the assertion that would catch somebody
        swapping in a dict.
        """
        response = client.get("/v1/admin/accounts", headers=admin_headers)

        assert "password" not in response.text.lower()
        assert "pbkdf2" not in response.text.lower()

    def test_it_pages(self, client, admin_headers, staff):
        first = client.get("/v1/admin/accounts?limit=1&offset=0", headers=admin_headers).json()
        second = client.get("/v1/admin/accounts?limit=1&offset=1", headers=admin_headers).json()

        assert len(first["accounts"]) == 1
        assert len(second["accounts"]) == 1
        assert first["accounts"][0]["username"] != second["accounts"][0]["username"]

    def test_an_absurd_limit_is_refused_rather_than_served(self, client, admin_headers):
        """A management screen must not be able to ask for the whole school in one call."""
        assert client.get("/v1/admin/accounts?limit=100000", headers=admin_headers).status_code == 422

    def test_a_non_admin_cannot_list_accounts(self, client, admin_headers, staff):
        token = client.post(
            "/v1/auth/login",
            json={"username": staff, "password": "correct-horse-battery"},
        ).json()["access_token"]

        response = client.get(
            "/v1/admin/accounts", headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 403


class TestUpdating:
    def test_a_password_can_be_changed_and_the_new_one_works(
        self, client, admin_headers, staff
    ):
        response = client.patch(
            f"/v1/admin/accounts/{staff}",
            headers=admin_headers,
            json={"password": "a-brand-new-password"},
        )
        assert response.status_code == 200

        assert client.post(
            "/v1/auth/login", json={"username": staff, "password": "a-brand-new-password"}
        ).status_code == 200
        assert client.post(
            "/v1/auth/login", json={"username": staff, "password": "correct-horse-battery"}
        ).status_code == 401

    def test_changing_a_password_revokes_existing_sessions(
        self, client, admin_headers, staff
    ):
        """Otherwise the change is cosmetic for a full refresh lifetime.

        A password is changed because it leaked or because somebody left. In both cases the
        refresh token issued under the old one is exactly what the attacker still holds.
        """
        tokens = client.post(
            "/v1/auth/login", json={"username": staff, "password": "correct-horse-battery"}
        ).json()

        client.patch(
            f"/v1/admin/accounts/{staff}",
            headers=admin_headers,
            json={"password": "a-brand-new-password"},
        )

        refreshed = client.post(
            "/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]}
        )
        assert refreshed.status_code == 401

    def test_a_role_can_be_changed(self, client, admin_headers, staff):
        response = client.patch(
            f"/v1/admin/accounts/{staff}", headers=admin_headers, json={"role": "admin"}
        )

        assert response.status_code == 200
        assert response.json()["role"] == "admin"

    def test_deactivating_refuses_login_without_deleting_the_account(
        self, client, admin_headers, staff
    ):
        """The softer alternative to deletion, and the one that keeps the audit trail."""
        client.patch(
            f"/v1/admin/accounts/{staff}", headers=admin_headers, json={"is_active": False}
        )

        assert client.post(
            "/v1/auth/login", json={"username": staff, "password": "correct-horse-battery"}
        ).status_code == 401
        listed = client.get("/v1/admin/accounts", headers=admin_headers).json()
        assert staff in [a["username"] for a in listed["accounts"]]

    def test_an_absent_field_is_left_alone(self, client, admin_headers, staff):
        """PATCH, not PUT. A form that forgets `display_name` must not erase it."""
        client.patch(
            f"/v1/admin/accounts/{staff}", headers=admin_headers, json={"phone": "0500000000"}
        )

        response = client.patch(
            f"/v1/admin/accounts/{staff}", headers=admin_headers, json={"role": "user"}
        )
        assert response.json()["display_name"] == "Ms Warda"

    def test_it_cannot_bind_a_guardian(self, client, admin_headers, staff):
        """The rule the whole update surface is shaped around.

        Refused rather than ignored: a caller who believes they bound a guardian and did
        not is worse off than one who is told no.
        """
        response = client.patch(
            f"/v1/admin/accounts/{staff}",
            headers=admin_headers,
            json={"guardian_external_id": "G-1"},
        )

        assert response.status_code == 422

    def test_updating_an_unknown_account_is_404(self, client, admin_headers):
        response = client.patch(
            "/v1/admin/accounts/nobody", headers=admin_headers, json={"role": "user"}
        )

        assert response.status_code == 404


class TestDeleting:
    def test_an_account_is_removed_and_can_no_longer_log_in(
        self, client, admin_headers, staff
    ):
        assert client.delete(
            f"/v1/admin/accounts/{staff}", headers=admin_headers
        ).status_code == 204

        assert client.post(
            "/v1/auth/login", json={"username": staff, "password": "correct-horse-battery"}
        ).status_code == 401
        listed = client.get("/v1/admin/accounts", headers=admin_headers).json()
        assert staff not in [a["username"] for a in listed["accounts"]]

    def test_deleting_revokes_the_sessions_it_was_holding(
        self, client, admin_headers, staff
    ):
        """Deleting the row alone would leave a browser renewing itself indefinitely.

        An access token already minted stays valid until it expires — that is the trade
        offline verification makes — but a live refresh token would let the session carry
        on forever against an account nobody can see any more.
        """
        tokens = client.post(
            "/v1/auth/login", json={"username": staff, "password": "correct-horse-battery"}
        ).json()

        client.delete(f"/v1/admin/accounts/{staff}", headers=admin_headers)

        assert client.post(
            "/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]}
        ).status_code == 401

    def test_deleting_an_unknown_account_is_404(self, client, admin_headers):
        assert client.delete(
            "/v1/admin/accounts/nobody", headers=admin_headers
        ).status_code == 404


class TestTheLastAdministratorIsProtected:
    """Three ways to lock the school out of its own onboarding, all refused.

    Everybody writes the delete guard first; demotion and deactivation walk straight past
    it, which is why the rule lives in the domain and all three paths consult it.
    """

    def test_the_last_admin_cannot_be_deleted(self, client, admin_headers):
        response = client.delete(
            f"/v1/admin/accounts/{BOOTSTRAP_ADMIN_USER}", headers=admin_headers
        )

        assert response.status_code == 403
        assert "only administrator" in response.json()["detail"]["message"]

    def test_the_last_admin_cannot_be_demoted(self, client, admin_headers):
        response = client.patch(
            f"/v1/admin/accounts/{BOOTSTRAP_ADMIN_USER}",
            headers=admin_headers,
            json={"role": "user"},
        )

        assert response.status_code == 403

    def test_the_last_admin_cannot_be_deactivated(self, client, admin_headers):
        response = client.patch(
            f"/v1/admin/accounts/{BOOTSTRAP_ADMIN_USER}",
            headers=admin_headers,
            json={"is_active": False},
        )

        assert response.status_code == 403

    def test_the_admin_can_still_change_their_own_password(self, client, admin_headers):
        """The guard must not become "the last admin is frozen".

        Rotating the seeded password is the single most likely thing an operator does after
        first boot — it is how the value in `.env` stops being the live credential.
        """
        response = client.patch(
            f"/v1/admin/accounts/{BOOTSTRAP_ADMIN_USER}",
            headers=admin_headers,
            json={"password": "rotated-away-from-the-env-value"},
        )

        assert response.status_code == 200
        assert client.post(
            "/v1/auth/login",
            json={
                "username": BOOTSTRAP_ADMIN_USER,
                "password": "rotated-away-from-the-env-value",
            },
        ).status_code == 200

    def test_an_admin_may_be_removed_once_another_exists(self, client, admin_headers, staff):
        """Ordinary staff turnover must still work.

        Forbidding self-removal outright would mean the last person to leave can never tidy
        up after themselves — so the rule counts administrators rather than naming them.
        """
        client.patch(
            f"/v1/admin/accounts/{staff}", headers=admin_headers, json={"role": "admin"}
        )

        assert client.delete(
            f"/v1/admin/accounts/{BOOTSTRAP_ADMIN_USER}", headers=admin_headers
        ).status_code == 204

    def test_an_inactive_admin_does_not_count_as_cover(self, client, admin_headers, staff):
        """A suspended administrator cannot log in, so they cannot unlock anything.

        Counting them would let an operator deactivate one admin, then delete the other,
        and end up with two accounts and no way in.
        """
        client.patch(
            f"/v1/admin/accounts/{staff}",
            headers=admin_headers,
            json={"role": "admin", "is_active": False},
        )

        response = client.delete(
            f"/v1/admin/accounts/{BOOTSTRAP_ADMIN_USER}", headers=admin_headers
        )

        assert response.status_code == 403


class TestTheSeededAdminReturns:
    def test_a_deleted_seeded_admin_comes_back_on_the_next_boot(
        self, client, admin_headers, staff
    ):
        """The bootstrap guarantee doing its job, not a bug.

        It is also the reason removing a seeded administrator for good means clearing
        IDENTITY_BOOTSTRAP_ADMIN_USER as well as deleting the row.
        """
        from identity.infrastructure.db.bootstrap import seed_bootstrap_admin

        client.patch(
            f"/v1/admin/accounts/{staff}", headers=admin_headers, json={"role": "admin"}
        )
        client.delete(f"/v1/admin/accounts/{BOOTSTRAP_ADMIN_USER}", headers=admin_headers)

        seed_bootstrap_admin(
            username=BOOTSTRAP_ADMIN_USER,
            password=BOOTSTRAP_ADMIN_PASSWORD,
            pbkdf2_rounds=2000,
        )

        assert client.post(
            "/v1/auth/login",
            json={
                "username": BOOTSTRAP_ADMIN_USER,
                "password": BOOTSTRAP_ADMIN_PASSWORD,
            },
        ).status_code == 200
