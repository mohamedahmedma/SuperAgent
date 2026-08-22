"""The identity application.

    uvicorn identity.app:app --port 8200

Owns authentication for the whole system and nothing else. It stores no grades, no
students, and no chat history — a service holding credentials should be the least
interesting target in the estate, not the most.
"""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from identity import guardians, whatsapp
from identity.db import init_db
from identity.routes import (
    admin_router,
    public_router,
    wellknown_router,
    whatsapp_router,
)

logger = logging.getLogger(__name__)


def _configure_whatsapp() -> None:
    """Choose the gateway this process will send through, and install the webhook secrets.

    The environment is read here and in `identity/deps.py`, and nowhere else. This is the
    composition root, so a misconfiguration is meant to stop the deploy rather than surface
    as a parent's login failing at eight in the morning.

    `e164_or_raise` is applied to the school's number at startup for exactly that reason:
    the national spelling `01288339613` produces `wa.me/01288339613`, which is a different
    number that does not exist, and the resulting failure is completely silent — the link
    opens, the chat is empty, no message ever arrives, and nothing logs anything.

    With no credentials the recording gateway stays in place and the flow still runs end to
    end, which is what makes this developable without a Meta account. It is a loud warning
    rather than a hard failure because that is also the state every test runs in.
    """
    number = os.getenv("IDENTITY_WHATSAPP_NUMBER") or ""
    if number:
        number = whatsapp.e164_or_raise(number, setting="IDENTITY_WHATSAPP_NUMBER")

    whatsapp.configure(
        verify_token=os.getenv("IDENTITY_WHATSAPP_VERIFY_TOKEN") or "",
        app_secret=os.getenv("IDENTITY_WHATSAPP_APP_SECRET") or "",
        business_number=number,
    )

    phone_number_id = os.getenv("IDENTITY_WHATSAPP_PHONE_NUMBER_ID") or ""
    token = os.getenv("IDENTITY_WHATSAPP_TOKEN") or ""
    if phone_number_id and token:
        whatsapp.set_gateway(
            whatsapp.CloudApiWhatsAppGateway(
                phone_number_id=phone_number_id, access_token=token
            )
        )
        return

    logger.warning(
        "WhatsApp is not configured (IDENTITY_WHATSAPP_PHONE_NUMBER_ID and "
        "IDENTITY_WHATSAPP_TOKEN); verification codes will be discarded rather than sent. "
        "Parent login by WhatsApp cannot work in this state."
    )


def _configure_guardians() -> None:
    """Point the guardian directory at the school's system of record.

    Left as the in-memory fake when `IDENTITY_SIS_BASE_URL` is unset, which means an
    unconfigured deployment refuses every parent rather than authenticating them against
    nothing. That is the safe direction: a login that cannot succeed is a support call, and
    a login that succeeds against an empty directory is a stranger holding a token.
    """
    base_url = os.getenv("IDENTITY_SIS_BASE_URL") or ""
    if not base_url:
        logger.warning(
            "IDENTITY_SIS_BASE_URL is not set; guardian lookups use an empty in-memory "
            "directory and every parent will be told their number is not registered."
        )
        return
    guardians.set_directory(
        guardians.SisGuardianDirectory(
            base_url=base_url, api_key=os.getenv("IDENTITY_SIS_API_KEY") or ""
        )
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # Force key load at startup so a misconfigured signing key fails the deploy,
    # rather than the first parent's login.
    from identity import keys

    keys.kid()
    _configure_whatsapp()
    _configure_guardians()
    yield


app = FastAPI(
    title="School Identity Service",
    version="0.1.0",
    description=(
        "Authentication for the school assistant. The only service that decides who "
        "someone is, and the only one that can mint a token.\n\n"
        "**Tokens are RS256.** Other services verify against `/.well-known/jwks.json` "
        "with a public key and hold nothing that could forge one. That is what keeps "
        "identity resolution out of the chat backend and the records facade.\n\n"
        "**The `guardian_id` claim is the whole integration.** It is set only by an "
        "administrator through `PUT /v1/admin/accounts/{username}/guardian-binding`, "
        "never at self-registration and never from a request body on a public route. "
        "A token without the claim can read no student records at all.\n\n"
        "**Parents sign in through WhatsApp, not with a password.** They ask for a "
        "challenge, send a pre-filled message from their own WhatsApp, and type back a "
        "code. WhatsApp proves they control the number; the school's own records decide "
        "whether that number belongs to a parent. Neither the browser nor the parent ever "
        "states which guardian they are."
    ),
    lifespan=lifespan,
)

app.include_router(wellknown_router)
app.include_router(public_router)
app.include_router(whatsapp_router)
app.include_router(admin_router)


@app.get("/health", tags=["ops"])
def health() -> dict:
    return {"status": "ok", "service": "identity"}
