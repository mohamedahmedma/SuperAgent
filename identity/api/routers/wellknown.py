"""The public signing key.

Its own router, and its own module, because it is the one endpoint here that is not
authentication: it is what makes authentication possible everywhere else.
"""
from fastapi import APIRouter

from identity.api.deps import SigningKeyDep

router = APIRouter(tags=["keys"])


@router.get("/.well-known/jwks.json")
def jwks(key: SigningKeyDep) -> dict:
    """The public signing key.

    Every other service verifies tokens against this and holds nothing that could mint
    one. Public by design — a public key is not a secret.

    Verification elsewhere is **offline**: no service calls this one per request, so
    identity being down does not take records down, and this service is not in the latency
    path of a parent's question.
    """
    return key.jwks()


__all__ = ["router"]
