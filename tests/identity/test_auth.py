"""Authentication behaviour.

The guardian-binding tests are the ones that matter most: that claim is what the
records facade trusts, so anything that lets it be set by the wrong party defeats
every check downstream.
"""
from identity.app import app


def decode_own_token(token: str) -> dict:
    """Decode through the issuer the running app built.

    The issuer is no longer a module-level function reading `IDENTITY_AUDIENCE` at
    import; it is an object on `app.state`, built by the lifespan from resolved
    settings. Every call below happens after a request, so it is there.
    """
    return app.state.token_issuer.decode_own_token(token)


def test_login_returns_a_token_carrying_the_guardian_claim(client, parent):
    response = client.post("/v1/auth/login", json=parent)
    assert response.status_code == 200

    body = response.json()
    claims = decode_own_token(body["access_token"])
    assert claims["guardian_id"] == "G-1"
    assert claims["role"] == "parent"
    assert claims["sub"] == parent["username"]


def test_unbound_account_gets_a_token_with_no_guardian_claim(client, unbound_parent):
    """A bulk import that created accounts but bound none leaks nothing.

    The account logs in fine and can read no records at all, because the claim the
    records facade requires simply is not there.
    """
    response = client.post("/v1/auth/login", json=unbound_parent)
    assert response.status_code == 200
    assert response.json()["guardian_id"] is None
    assert "guardian_id" not in decode_own_token(response.json()["access_token"])


def test_wrong_password_and_unknown_user_are_indistinguishable(client, parent):
    """Otherwise this endpoint confirms which parents are registered at the school."""
    wrong = client.post("/v1/auth/login", json={"username": parent["username"], "password": "nope"})
    unknown = client.post("/v1/auth/login", json={"username": "0500000000", "password": "nope"})

    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json()["detail"]["message"] == unknown.json()["detail"]["message"]


def test_account_locks_after_repeated_failures(client, parent):
    for _ in range(3):
        client.post("/v1/auth/login", json={"username": parent["username"], "password": "nope"})

    # Correct password now, and still refused — the lockout is on the account.
    response = client.post("/v1/auth/login", json=parent)
    assert response.status_code == 423
    assert response.json()["detail"]["code"] == "locked"


def test_refresh_returns_a_fresh_access_token(client, parent):
    tokens = client.post("/v1/auth/login", json=parent).json()
    response = client.post("/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]})

    assert response.status_code == 200
    assert decode_own_token(response.json()["access_token"])["guardian_id"] == "G-1"


def test_refresh_re_reads_the_binding_rather_than_copying_it(client, parent, admin_headers):
    """A custody change must take effect without waiting for the parent to log out."""
    tokens = client.post("/v1/auth/login", json=parent).json()

    client.put(
        f"/v1/admin/accounts/{parent['username']}/guardian-binding",
        headers=admin_headers,
        json={"guardian_external_id": "G-99"},
    )

    refreshed = client.post("/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]}).json()
    assert decode_own_token(refreshed["access_token"])["guardian_id"] == "G-99"


def test_unbinding_revokes_existing_sessions(client, parent, admin_headers):
    """The urgent custody path: remove the binding and the session dies."""
    tokens = client.post("/v1/auth/login", json=parent).json()

    client.delete(f"/v1/admin/accounts/{parent['username']}/guardian-binding", headers=admin_headers)

    response = client.post("/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert response.status_code == 401


def test_logout_revokes_the_refresh_token(client, parent):
    tokens = client.post("/v1/auth/login", json=parent).json()
    client.post("/v1/auth/logout", json={"refresh_token": tokens["refresh_token"]})

    response = client.post("/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert response.status_code == 401


def test_binding_requires_the_admin_key(client, parent):
    """The most sensitive write in the system has no self-service path.

    An account that could name its own guardian id could read any family's records.
    """
    response = client.put(
        f"/v1/admin/accounts/{parent['username']}/guardian-binding",
        json={"guardian_external_id": "G-2"},
    )
    assert response.status_code == 401


def test_account_creation_requires_the_admin_key(client):
    response = client.post(
        "/v1/admin/accounts",
        json={"username": "intruder", "password": "whatever"},
    )
    assert response.status_code == 401


def test_creating_an_account_cannot_set_a_guardian_binding(client, admin_headers):
    """Creation and binding are two calls on purpose.

    Even holding the admin key, the create route must not accept a guardian id — an
    extra field in an import CSV should not silently become a records grant.
    """
    client.post(
        "/v1/admin/accounts",
        headers=admin_headers,
        json={
            "username": "0507777777",
            "password": "correct-horse-battery",
            "guardian_external_id": "G-1",
        },
    )
    tokens = client.post(
        "/v1/auth/login", json={"username": "0507777777", "password": "correct-horse-battery"}
    ).json()
    assert tokens["guardian_id"] is None


def test_me_decodes_the_callers_own_token(client, parent):
    tokens = client.post("/v1/auth/login", json=parent).json()
    response = client.get(
        "/v1/auth/me", headers={"Authorization": f"Bearer {tokens['access_token']}"}
    )
    assert response.status_code == 200
    assert response.json()["guardian_id"] == "G-1"


def test_jwks_publishes_a_usable_public_key(client):
    """What every other service verifies against."""
    response = client.get("/.well-known/jwks.json")
    assert response.status_code == 200

    key = response.json()["keys"][0]
    assert key["kty"] == "RSA"
    assert key["alg"] == "RS256"
    assert key["n"] and key["e"] and key["kid"]


def test_jwks_never_exposes_the_private_key(client):
    """A JWKS document containing 'd' would hand out the signing key."""
    key = client.get("/.well-known/jwks.json").json()["keys"][0]
    assert "d" not in key
    assert "p" not in key and "q" not in key
