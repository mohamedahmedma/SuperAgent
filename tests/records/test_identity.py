"""Identity verification.

The API key proves which system is calling. These tests cover the second, independent
proof: a token signed by the identity service naming the guardian. Together they mean
a fully compromised chat backend still cannot read a family it holds no token for.

Each test forges a token that is wrong in exactly one way, because a verifier that
checks only the signature passes a frightening number of attacks.
"""
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from tests.records.conftest import agent_headers, mint_token


def _foreign_key_pem() -> str:
    """A valid RSA key that is not the identity service's."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")


def test_api_key_alone_is_not_enough(client):
    """The central claim of the design.

    A valid agent key with no identity token reads nothing. If this test fails, a
    leaked key is a student-body dump.
    """
    response = client.get("/v1/guardians/G-1/students", headers={"X-API-Key": agent_headers()["X-API-Key"]})
    assert response.status_code == 401


def test_token_for_another_guardian_is_rejected(client):
    """G-1's token cannot be used to ask about G-2.

    This is what stops the calling system choosing whose records it reads. It holds a
    parent's token; it cannot produce a signature for a different parent.
    """
    response = client.get("/v1/guardians/G-2/students", headers=agent_headers("G-1"))
    assert response.status_code == 403


def test_guardian_mismatch_is_reported_under_its_own_reason(client, caplog):
    """A relay attempt is the loudest signal in the system. It must be alertable.

    Reported as a structured line rather than a row: this request never becomes a SIS
    call, so if this service does not say so, nothing does. `sis/` keeps the trail of
    answers about children; `records.audit` keeps the record of callers who never got
    that far.
    """
    with caplog.at_level("WARNING"):
        client.get(
            "/v1/guardians/G-2/students/S-1001/grades", headers=agent_headers("G-1")
        )

    assert "guardian_mismatch" in caplog.text


def test_token_signed_by_a_foreign_key_is_rejected(client):
    """Anyone can mint a well-formed JWT. Only the identity service can sign one."""
    forged = mint_token("G-1", key_pem=_foreign_key_pem())
    response = client.get("/v1/guardians/G-1/students", headers=agent_headers(token=forged))
    assert response.status_code == 401


def test_token_for_another_audience_is_rejected(client):
    """Stops a token minted for one service being replayed against this one."""
    wrong_audience = mint_token("G-1", audience="some-other-service")
    response = client.get("/v1/guardians/G-1/students", headers=agent_headers(token=wrong_audience))
    assert response.status_code == 401


def test_token_from_another_issuer_is_rejected(client):
    wrong_issuer = mint_token("G-1", issuer="not-our-identity-service")
    response = client.get("/v1/guardians/G-1/students", headers=agent_headers(token=wrong_issuer))
    assert response.status_code == 401


def test_expired_token_is_rejected(client):
    expired = mint_token("G-1", expired=True)
    response = client.get("/v1/guardians/G-1/students", headers=agent_headers(token=expired))
    assert response.status_code == 401


def test_token_without_a_guardian_binding_reads_nothing(client):
    """A staff token, or a parent the registrar has not bound yet.

    Both arrive as the same absent claim, and both must read nothing rather than
    falling through to some default.
    """
    unbound = mint_token(None)
    response = client.get("/v1/guardians/G-1/students", headers=agent_headers(token=unbound))
    assert response.status_code == 401


def test_malformed_token_is_rejected(client):
    response = client.get("/v1/guardians/G-1/students", headers=agent_headers(token="not.a.jwt"))
    assert response.status_code == 401


def test_valid_token_and_key_together_succeed(client):
    """The happy path, stated once so the denials above mean something."""
    response = client.get("/v1/guardians/G-1/students", headers=agent_headers("G-1"))
    assert response.status_code == 200
