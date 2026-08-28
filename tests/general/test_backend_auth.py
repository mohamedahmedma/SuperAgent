"""The chat backend's half of authentication, after consolidation.

The backend no longer hashes passwords, mints tokens, or serves login. What remains is
a verifier, and these cover the properties that verifier must hold — above all that it
fails closed and that authority comes from the signed token rather than from a local
row anyone could edit.
"""
import os
import time
import unittest

import backend.infra.auth as backend_auth
import backend.infra.identity as backend_identity
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException
from jose import jwt
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.db.models import User

_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
PRIVATE_PEM = _KEY.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption(),
).decode("utf-8")
PUBLIC_PEM = (
    _KEY.public_key()
    .public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    .decode("utf-8")
)


def mint(
    username="parent-one",
    role="user",
    guardian_id=None,
    *,
    issuer=None,
    audience=None,
    expired=False,
    key_pem=PRIVATE_PEM,
):
    """Mint against whatever issuer/audience the module was imported with.

    Not the hardcoded defaults: `records/tests/conftest.py` sets `IDENTITY_ISSUER`
    and `IDENTITY_AUDIENCE` at import time, so the values baked into
    `backend.infra.identity` depend on test collection order. Reading them back is what
    makes this file pass alone and in the full suite alike.
    """
    issuer = backend_identity.ISSUER if issuer is None else issuer
    audience = backend_identity.AUDIENCE if audience is None else audience
    now = int(time.time())
    claims = {
        "iss": issuer,
        "aud": audience,
        "sub": username,
        "role": role,
        "iat": now,
        "exp": now - 300 if expired else now + 1800,
    }
    if guardian_id:
        claims["guardian_id"] = guardian_id
    return jwt.encode(claims, key_pem, algorithm="RS256")


class BackendAuthTests(unittest.TestCase):
    """`get_current_user` against a throwaway SQLite projection table."""

    def setUp(self):
        # Only the users table is needed: it is the sole thing this module touches.
        self.engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
        User.__table__.create(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.db = self.Session()

        self._saved = os.environ.get("IDENTITY_PUBLIC_KEY_PEM")
        os.environ["IDENTITY_PUBLIC_KEY_PEM"] = PUBLIC_PEM

    def tearDown(self):
        self.db.close()
        if self._saved is None:
            os.environ.pop("IDENTITY_PUBLIC_KEY_PEM", None)
        else:
            os.environ["IDENTITY_PUBLIC_KEY_PEM"] = self._saved

    def test_a_valid_token_identifies_the_subject(self):
        user = backend_auth.get_current_user(token=mint(), db=self.db)
        self.assertEqual("parent-one", user.username)
        self.assertEqual("user", user.role)

    def test_the_guardian_claim_is_carried_through(self):
        user = backend_auth.get_current_user(token=mint(guardian_id="G-1"), db=self.db)
        self.assertEqual("G-1", user.guardian_id)

    def test_a_token_without_a_guardian_claim_yields_an_empty_guardian(self):
        """Staff, and unbound parents. Must be empty, never None-shaped surprises."""
        user = backend_auth.get_current_user(token=mint(), db=self.db)
        self.assertEqual("", user.guardian_id)

    def test_an_invalid_token_is_rejected(self):
        with self.assertRaises(HTTPException) as caught:
            backend_auth.get_current_user(token="not.a.jwt", db=self.db)
        self.assertEqual(401, caught.exception.status_code)

    def test_a_token_signed_by_a_foreign_key_is_rejected(self):
        other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        forged = other.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode("utf-8")

        with self.assertRaises(HTTPException) as caught:
            backend_auth.get_current_user(token=mint(key_pem=forged), db=self.db)
        self.assertEqual(401, caught.exception.status_code)

    def test_an_expired_token_is_rejected(self):
        with self.assertRaises(HTTPException) as caught:
            backend_auth.get_current_user(token=mint(expired=True), db=self.db)
        self.assertEqual(401, caught.exception.status_code)

    def test_a_token_for_another_audience_is_rejected(self):
        with self.assertRaises(HTTPException) as caught:
            backend_auth.get_current_user(token=mint(audience="somewhere-else"), db=self.db)
        self.assertEqual(401, caught.exception.status_code)

    def test_a_token_from_another_issuer_is_rejected(self):
        with self.assertRaises(HTTPException) as caught:
            backend_auth.get_current_user(token=mint(issuer="not-ours"), db=self.db)
        self.assertEqual(401, caught.exception.status_code)

    def test_unconfigured_verification_fails_closed(self):
        """No public key must mean "reject", never "trust"."""
        os.environ.pop("IDENTITY_PUBLIC_KEY_PEM", None)
        with self.assertRaises(HTTPException) as caught:
            backend_auth.get_current_user(token=mint(), db=self.db)
        self.assertEqual(503, caught.exception.status_code)

    def test_a_projection_row_is_created_on_first_sight(self):
        """Session ownership hangs off this row, so it must exist before the first save."""
        self.assertIsNone(self.db.query(User).filter(User.username == "parent-one").first())

        backend_auth.get_current_user(token=mint(), db=self.db)

        row = self.db.query(User).filter(User.username == "parent-one").first()
        self.assertIsNotNone(row)
        self.assertEqual(backend_auth.EXTERNAL_CREDENTIAL_SENTINEL, row.password_hash)

    def test_the_projection_row_is_not_duplicated(self):
        for _ in range(3):
            backend_auth.get_current_user(token=mint(), db=self.db)
        self.assertEqual(1, self.db.query(User).filter(User.username == "parent-one").count())

    def test_authority_comes_from_the_token_not_the_local_row(self):
        """The projection is written with role "user" and never updated.

        Reading the role from it would silently strip an administrator of their
        privileges the moment they were first seen here.
        """
        backend_auth.get_current_user(token=mint(username="boss", role="admin"), db=self.db)
        row = self.db.query(User).filter(User.username == "boss").first()
        self.assertEqual("user", row.role)

        user = backend_auth.get_current_user(token=mint(username="boss", role="admin"), db=self.db)
        self.assertEqual("admin", user.role)

    def test_require_admin_accepts_an_admin_and_refuses_everyone_else(self):
        admin = backend_auth.AuthenticatedUser(username="boss", role="admin")
        self.assertIs(admin, backend_auth.require_admin(admin))

        with self.assertRaises(HTTPException) as caught:
            backend_auth.require_admin(backend_auth.AuthenticatedUser(username="u", role="user"))
        self.assertEqual(403, caught.exception.status_code)

    def test_a_parent_is_not_an_admin(self):
        """The new roles must not accidentally satisfy the old admin gate."""
        for role in ("parent", "staff", "user"):
            with self.assertRaises(HTTPException):
                backend_auth.require_admin(
                    backend_auth.AuthenticatedUser(username="p", role=role)
                )


class RemovedSurfaceTests(unittest.TestCase):
    """What consolidation deleted must actually be gone, not merely unused."""

    def test_the_backend_no_longer_mints_or_hashes(self):
        for name in (
            "create_access_token",
            "get_password_hash",
            "verify_password",
            "authenticate_user",
            "resolve_role",
        ):
            self.assertFalse(
                hasattr(backend_auth, name),
                f"{name} still exists in backend.infra.auth — credentials belong to identity/",
            )

    def test_the_auth_routes_are_gone(self):
        from backend.api.router import router

        paths = {route.path for route in router.routes}
        for removed in ("/auth/login", "/auth/register", "/auth/me"):
            self.assertNotIn(removed, paths)

    def test_the_auth_schemas_are_gone(self):
        import backend.schemas as schemas

        for name in ("LoginRequest", "RegisterRequest", "AuthResponse", "CurrentUserResponse"):
            self.assertNotIn(name, schemas.__all__)


if __name__ == "__main__":
    unittest.main()
