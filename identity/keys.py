"""Signing keys, and the public half other services verify with.

RS256 rather than HS256, and the reason is the whole architecture. With a shared
secret, every service that *verifies* a token also holds the key that *mints* one —
so a bug in the records facade, or a leaked config file on the chat backend, forges
parent identities. With an asymmetric pair, only this service can sign. Everyone
else holds a public key and can do nothing with it but check a signature.

That is what makes "authentication is handled at the authentication layer" a
structural fact rather than a convention someone has to remember.
"""
import base64
import hashlib
import logging
import os
import pathlib

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

logger = logging.getLogger(__name__)

_ALGORITHM = "RS256"
_DEV_KEY_PATH = pathlib.Path(os.getenv("IDENTITY_DEV_KEY_FILE", "./identity-dev-key.pem"))

_private_pem: str | None = None
_public_pem: str | None = None
_kid: str | None = None


def _b64url_uint(value: int) -> str:
    raw = value.to_bytes((value.bit_length() + 7) // 8, "big")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _generate_dev_key() -> str:
    """Create a development keypair and persist it.

    Persisted rather than held in memory because a key regenerated on every restart
    invalidates every token in flight, which during development looks exactly like an
    authentication bug and wastes an afternoon.

    Never reached in production: `IDENTITY_PRIVATE_KEY_PEM` is required there, and
    the loud warning below is what makes an accidental dev key obvious in logs.
    """
    if _DEV_KEY_PATH.exists():
        return _DEV_KEY_PATH.read_text(encoding="utf-8")

    logger.warning(
        "No IDENTITY_PRIVATE_KEY_PEM set — generating a DEVELOPMENT signing key at %s. "
        "Do not use this in production.",
        _DEV_KEY_PATH,
    )
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")
    _DEV_KEY_PATH.write_text(pem, encoding="utf-8")
    return pem


def _load() -> None:
    global _private_pem, _public_pem, _kid
    if _private_pem is not None:
        return

    pem = os.getenv("IDENTITY_PRIVATE_KEY_PEM")
    if not pem:
        key_file = os.getenv("IDENTITY_PRIVATE_KEY_FILE")
        pem = pathlib.Path(key_file).read_text(encoding="utf-8") if key_file else _generate_dev_key()

    private_key = serialization.load_pem_private_key(pem.encode("utf-8"), password=None)
    public_key = private_key.public_key()

    _private_pem = pem
    _public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("utf-8")

    # Key id derived from the public key itself, so it is stable across restarts and
    # changes exactly when the key does. A random kid would force every verifier to
    # refetch JWKS after an unrelated restart.
    _kid = hashlib.sha256(_public_pem.encode("utf-8")).hexdigest()[:16]


def private_pem() -> str:
    _load()
    assert _private_pem is not None
    return _private_pem


def public_pem() -> str:
    _load()
    assert _public_pem is not None
    return _public_pem


def kid() -> str:
    _load()
    assert _kid is not None
    return _kid


def algorithm() -> str:
    return _ALGORITHM


def jwks() -> dict:
    """The public key, in the format every JWT library already knows how to consume.

    Publishing JWKS rather than asking operators to copy a PEM into each service is
    what makes key rotation survivable: verifiers refetch, and nothing needs
    redeploying.
    """
    _load()
    public_key = serialization.load_pem_public_key(public_pem().encode("utf-8"))
    numbers = public_key.public_numbers()
    return {
        "keys": [
            {
                "kty": "RSA",
                "use": "sig",
                "alg": _ALGORITHM,
                "kid": kid(),
                "n": _b64url_uint(numbers.n),
                "e": _b64url_uint(numbers.e),
            }
        ]
    }
