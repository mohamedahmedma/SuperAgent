"""Signing keys, and the public half other services verify with.

RS256 rather than HS256, and the reason is the whole architecture. With a shared secret,
every service that *verifies* a token also holds the key that *mints* one — so a bug in
the records facade, or a leaked config file on the chat backend, forges parent identities.
With an asymmetric pair, only this service can sign. Everyone else holds a public key and
can do nothing with it but check a signature.

That is what makes "authentication is handled at the authentication layer" a structural
fact rather than a convention someone has to remember.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import pathlib
from typing import Final

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

logger = logging.getLogger(__name__)

ALGORITHM: Final[str] = "RS256"


class SigningKey:
    """One RSA keypair, loaded once and held for the life of the process.

    An object rather than module globals, because the module-global version read
    `IDENTITY_DEV_KEY_FILE` **at import** — capturing whatever the environment held at
    collection time, which in a test run was reliably the wrong thing. The composition
    root now builds one of these from resolved settings, and a test builds its own.
    """

    def __init__(
        self,
        *,
        private_key_pem: str = "",
        private_key_file: str = "",
        dev_key_file: str = "./identity-dev-key.pem",
    ) -> None:
        self._configured_pem = private_key_pem
        self._private_key_file = private_key_file
        self._dev_key_path = pathlib.Path(dev_key_file)
        self._private_pem: str | None = None
        self._public_pem: str | None = None
        self._kid: str | None = None

    # -- loading ------------------------------------------------------------

    def _load(self) -> None:
        if self._private_pem is not None:
            return

        pem = self._configured_pem
        if not pem:
            pem = (
                pathlib.Path(self._private_key_file).read_text(encoding="utf-8")
                if self._private_key_file
                else self._generate_dev_key()
            )

        private_key = serialization.load_pem_private_key(pem.encode("utf-8"), password=None)
        public_key = private_key.public_key()

        self._private_pem = pem
        self._public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode("utf-8")

        # Key id derived from the public key itself, so it is stable across restarts and
        # changes exactly when the key does. A random kid would force every verifier to
        # refetch JWKS after an unrelated restart.
        self._kid = hashlib.sha256(self._public_pem.encode("utf-8")).hexdigest()[:16]

    def _generate_dev_key(self) -> str:
        """Create a development keypair and persist it.

        Persisted rather than held in memory, because a key regenerated on every restart
        invalidates every token in flight — which during development looks exactly like an
        authentication bug and wastes an afternoon.

        Never reached in production: `IDENTITY_PRIVATE_KEY_PEM` is required there, and the
        loud warning below is what makes an accidental dev key obvious in logs.
        """
        if self._dev_key_path.exists():
            return self._dev_key_path.read_text(encoding="utf-8")

        logger.warning(
            "No IDENTITY_PRIVATE_KEY_PEM set — generating a DEVELOPMENT signing key at %s. "
            "Do not use this in production.",
            self._dev_key_path,
        )
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        pem = key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode("utf-8")
        self._dev_key_path.parent.mkdir(parents=True, exist_ok=True)
        self._dev_key_path.write_text(pem, encoding="utf-8")
        return pem

    # -- the key ------------------------------------------------------------

    @property
    def private_pem(self) -> str:
        self._load()
        assert self._private_pem is not None
        return self._private_pem

    @property
    def public_pem(self) -> str:
        self._load()
        assert self._public_pem is not None
        return self._public_pem

    @property
    def kid(self) -> str:
        self._load()
        assert self._kid is not None
        return self._kid

    @property
    def algorithm(self) -> str:
        return ALGORITHM

    def jwks(self) -> dict:
        """The public key, in the format every JWT library already knows how to consume.

        Publishing JWKS rather than asking operators to copy a PEM into each service is
        what makes key rotation survivable: verifiers refetch, and nothing needs
        redeploying.
        """
        public_key = serialization.load_pem_public_key(self.public_pem.encode("utf-8"))
        numbers = public_key.public_numbers()
        return {
            "keys": [
                {
                    "kty": "RSA",
                    "use": "sig",
                    "alg": ALGORITHM,
                    "kid": self.kid,
                    "n": _b64url_uint(numbers.n),
                    "e": _b64url_uint(numbers.e),
                }
            ]
        }


def signing_key_from(settings) -> SigningKey:
    """Build the key a deployment's settings describe.

    Takes the settings object rather than importing `identity.config`, so this module
    stays free of configuration and a test can hand it any object with the three
    attributes. It exists because the composition root and the cross-service journey
    test both need the same key, and two places spelling out the same three arguments
    is two places to get one of them wrong.
    """
    return SigningKey(
        private_key_pem=settings.private_key_pem,
        private_key_file=settings.private_key_file,
        dev_key_file=settings.dev_key_file,
    )


def _b64url_uint(value: int) -> str:
    raw = value.to_bytes((value.bit_length() + 7) // 8, "big")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


__all__ = ["ALGORITHM", "SigningKey", "signing_key_from"]
