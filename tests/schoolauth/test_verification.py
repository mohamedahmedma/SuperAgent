"""The verifier, against tokens that are wrong in exactly one way each.

A verifier that checks only the signature passes a frightening number of attacks, so
every case here forges a token that is valid apart from the single thing under test.
"""
import base64
import hashlib
import hmac
import json
import time

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from jose import jwt

import schoolauth.verification as verification
from schoolauth import (
    IdentityConfig,
    IdentityError,
    IdentityNotConfigured,
    children_from_claims,
    guardian_id_from_claims,
    reset_key_cache,
    school_from_claims,
    verify_token,
)


def _keypair() -> tuple[str, str]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")
    public = (
        key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("utf-8")
    )
    return private, public


PRIVATE_PEM, PUBLIC_PEM = _keypair()
FOREIGN_PRIVATE_PEM, _ = _keypair()

ISSUER = "test-identity"
AUDIENCE = "test-services"


def _config(**overrides) -> IdentityConfig:
    settings = {"issuer": ISSUER, "audience": AUDIENCE, "public_key_pem": PUBLIC_PEM}
    settings.update(overrides)
    return IdentityConfig(**settings)


def _mint(
    *,
    issuer=ISSUER,
    audience=AUDIENCE,
    guardian_id="G-1",
    key_pem=PRIVATE_PEM,
    expired=False,
    extra=None,
) -> str:
    now = int(time.time())
    claims = {
        "iss": issuer,
        "aud": audience,
        "sub": "parent-one",
        "role": "user",
        "iat": now,
        "exp": now - 300 if expired else now + 1800,
    }
    if guardian_id:
        claims["guardian_id"] = guardian_id
    claims.update(extra or {})
    return jwt.encode(claims, key_pem, algorithm="RS256")


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _forged_claims() -> dict:
    return {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "sub": "parent-one",
        "guardian_id": "G-1",
        "exp": int(time.time()) + 1800,
    }


def _assemble(header: dict, claims: dict, signature: bytes) -> str:
    """A JWT built by hand.

    `python-jose` refuses to *mint* the two forgeries below, which is exactly the
    position an attacker is in — and no obstacle at all to them, since a JWT is three
    base64 segments and a dot. Minting them here is what proves the verifier refuses
    them rather than proving the signing library has scruples.
    """
    return "{}.{}.{}".format(
        _b64(json.dumps(header, separators=(",", ":")).encode("utf-8")),
        _b64(json.dumps(claims, separators=(",", ":")).encode("utf-8")),
        _b64(signature),
    )


@pytest.fixture(autouse=True)
def _clean_cache(monkeypatch):
    """No pinned PEM leaking in from another suite, and no keys cached across tests."""
    monkeypatch.delenv("IDENTITY_PUBLIC_KEY_PEM", raising=False)
    reset_key_cache()
    yield
    reset_key_cache()


class TestVerification:
    def test_a_genuine_token_is_accepted(self):
        claims = verify_token(_mint(), _config())
        assert claims["sub"] == "parent-one"
        assert claims["guardian_id"] == "G-1"

    def test_an_empty_token_is_refused(self):
        with pytest.raises(IdentityError):
            verify_token("", _config())

    def test_a_foreign_signature_is_refused(self):
        """Anyone can mint a well-formed JWT. Only identity can sign one."""
        with pytest.raises(IdentityError):
            verify_token(_mint(key_pem=FOREIGN_PRIVATE_PEM), _config())

    def test_another_audience_is_refused(self):
        """Without this a token minted for one service is replayable against another."""
        with pytest.raises(IdentityError):
            verify_token(_mint(audience="someone-else"), _config())

    def test_another_issuer_is_refused(self):
        with pytest.raises(IdentityError):
            verify_token(_mint(issuer="not-our-identity"), _config())

    def test_an_expired_token_is_refused(self):
        with pytest.raises(IdentityError):
            verify_token(_mint(expired=True), _config())

    def test_a_garbled_token_is_refused_rather_than_crashing(self):
        with pytest.raises(IdentityError):
            verify_token("not.a.jwt", _config())

    def test_an_unsigned_token_is_refused(self):
        """`alg: none` is the classic forgery. Algorithms are declared, never read."""
        forged = _assemble({"alg": "none", "typ": "JWT"}, _forged_claims(), b"")
        with pytest.raises(IdentityError):
            verify_token(forged, _config())

    def test_a_token_signed_with_the_public_key_as_an_hmac_secret_is_refused(self):
        """The other half of the algorithm-confusion attack.

        The public key is public. If the verifier honoured the token's own `alg`, anyone
        could sign HS256 with the very PEM this service publishes and be believed.
        """
        header = {"alg": "HS256", "typ": "JWT"}
        signing_input = "{}.{}".format(
            _b64(json.dumps(header, separators=(",", ":")).encode("utf-8")),
            _b64(json.dumps(_forged_claims(), separators=(",", ":")).encode("utf-8")),
        ).encode("ascii")
        signature = hmac.new(
            PUBLIC_PEM.encode("utf-8"), signing_input, hashlib.sha256
        ).digest()
        forged = f"{signing_input.decode('ascii')}.{_b64(signature)}"
        with pytest.raises(IdentityError):
            verify_token(forged, _config())


class TestFailingClosed:
    def test_no_key_material_at_all_is_not_configured(self):
        """The distinction that matters: this is a 503, never an unverified read."""
        with pytest.raises(IdentityNotConfigured):
            verify_token(_mint(), IdentityConfig(issuer=ISSUER, audience=AUDIENCE))

    def test_not_configured_is_not_an_identity_error(self):
        """Separate types so a caller cannot collapse them into one 401.

        A misconfigured service telling a parent "your sign-in is invalid" sends them to
        re-authenticate against a service that will refuse them again.
        """
        assert not issubclass(IdentityNotConfigured, IdentityError)

    def test_an_unreachable_jwks_with_a_cold_cache_is_not_configured(self):
        config = IdentityConfig(
            issuer=ISSUER, audience=AUDIENCE, jwks_url="http://127.0.0.1:1/jwks.json"
        )
        with pytest.raises(IdentityNotConfigured):
            verify_token(_mint(), config)

    def test_a_pinned_pem_wins_over_a_jwks_url(self):
        """An operator who pinned a key meant it, so no fetch is attempted at all."""
        config = _config(jwks_url="http://127.0.0.1:1/jwks.json")
        assert verify_token(_mint(), config)["sub"] == "parent-one"

    def test_the_environment_supplies_the_pem_when_the_config_does_not(self, monkeypatch):
        """Read per call, so a variable set after import still takes effect."""
        monkeypatch.setenv("IDENTITY_PUBLIC_KEY_PEM", PUBLIC_PEM)
        config = IdentityConfig(issuer=ISSUER, audience=AUDIENCE)
        assert verify_token(_mint(), config)["sub"] == "parent-one"


def _serve(monkeypatch, document, calls):
    """Stand in for the JWKS endpoint, recording every URL actually fetched."""

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(document).encode("utf-8")

    def _urlopen(request, timeout=None):
        calls.append(getattr(request, "full_url", request))
        return _Response()

    monkeypatch.setattr(verification.urllib.request, "urlopen", _urlopen)


def _break(monkeypatch):
    def _urlopen(request, timeout=None):
        raise OSError("identity is down")

    monkeypatch.setattr(verification.urllib.request, "urlopen", _urlopen)


class TestJwksCaching:
    """The stale-key rule: an identity outage must not become an estate outage.

    Exercised against `_fetch_jwks` directly rather than through `verify_token`, because
    what is under test is the caching contract — how often the network is touched and what
    happens when it fails — not whether a particular key document verifies a signature.
    """

    URL = "https://identity.test/jwks.json"
    DOCUMENT = {"keys": [{"kid": "one", "kty": "RSA"}]}

    def test_keys_are_fetched_once_within_the_ttl(self, monkeypatch):
        calls: list = []
        _serve(monkeypatch, self.DOCUMENT, calls)

        assert verification._fetch_jwks(self.URL, 600) == self.DOCUMENT
        assert verification._fetch_jwks(self.URL, 600) == self.DOCUMENT
        assert len(calls) == 1, "the second call refetched a key set that was still fresh"

    def test_an_expired_ttl_refetches(self, monkeypatch):
        calls: list = []
        _serve(monkeypatch, self.DOCUMENT, calls)

        verification._fetch_jwks(self.URL, 0)
        verification._fetch_jwks(self.URL, 0)
        assert len(calls) == 2, "a stale entry was served past its TTL without a refresh"

    def test_a_stale_key_survives_a_failed_refresh(self, monkeypatch):
        """The whole point: identity down means grades unavailable, not everyone locked out."""
        calls: list = []
        _serve(monkeypatch, self.DOCUMENT, calls)
        verification._fetch_jwks(self.URL, 0)

        _break(monkeypatch)
        assert verification._fetch_jwks(self.URL, 0) == self.DOCUMENT

    def test_a_cold_cache_is_fatal(self, monkeypatch):
        _break(monkeypatch)
        with pytest.raises(IdentityNotConfigured):
            verification._fetch_jwks(self.URL, 600)

    def test_a_body_that_is_not_an_object_is_refused(self, monkeypatch):
        _serve(monkeypatch, ["not", "a", "key", "set"], [])
        with pytest.raises(IdentityNotConfigured):
            verification._fetch_jwks(self.URL, 600)

    def test_two_identity_services_do_not_share_a_cache_entry(self, monkeypatch):
        calls: list = []
        _serve(monkeypatch, self.DOCUMENT, calls)

        verification._fetch_jwks("https://one.test/jwks.json", 600)
        verification._fetch_jwks("https://two.test/jwks.json", 600)
        assert len(calls) == 2, "one service's keys were served for the other's tokens"

    def test_resetting_the_cache_forces_a_refetch(self, monkeypatch):
        calls: list = []
        _serve(monkeypatch, self.DOCUMENT, calls)

        verification._fetch_jwks(self.URL, 600)
        reset_key_cache()
        verification._fetch_jwks(self.URL, 600)
        assert len(calls) == 2


class TestClaims:
    def test_the_guardian_binding_is_read(self):
        assert guardian_id_from_claims({"guardian_id": "G-7"}) == "G-7"

    def test_a_token_with_no_binding_raises_rather_than_returning_blank(self):
        """A staff token and an unbound parent arrive as the same absence."""
        with pytest.raises(IdentityError):
            guardian_id_from_claims({"sub": "teacher-one"})

    def test_a_non_string_binding_is_refused(self):
        with pytest.raises(IdentityError):
            guardian_id_from_claims({"guardian_id": ["G-1"]})

    def test_the_school_is_optional(self):
        assert school_from_claims({}) is None
        assert school_from_claims({"school": "branch-a"}) == "branch-a"

    def test_children_are_a_tuple_so_they_cannot_be_accumulated_into(self):
        claims = {"children": [{"id": "S-1"}, {"id": "S-2"}]}
        assert children_from_claims(claims) == ({"id": "S-1"}, {"id": "S-2"})

    def test_absent_children_read_as_empty(self):
        assert children_from_claims({}) == ()

    def test_junk_in_the_children_claim_is_dropped_rather_than_trusted(self):
        assert children_from_claims({"children": ["S-1", {"id": "S-2"}]}) == ({"id": "S-2"},)
