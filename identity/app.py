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

from identity import guardians, schools, whatsapp
from identity.db import init_db
from identity.env import load_env

# Before anything below reads the environment. This service is deployed as its own
# process, so nothing else has loaded the project's `.env` for it — see identity/env.py
# for what that cost.
load_env()
from identity.routes import (
    admin_router,
    public_router,
    wellknown_router,
    whatsapp_router,
)

logger = logging.getLogger(__name__)


def _configure_whatsapp() -> None:
    """Choose the gateways this process sends through, and install the webhook secrets.

    The environment is read here and in `identity/deps.py`, and nowhere else. This is the
    composition root, so a misconfiguration is meant to stop the deploy rather than surface
    as a parent's login failing at eight in the morning.

    `e164_or_raise` is applied to every school's number at startup for exactly that reason:
    the national spelling `01288339613` produces `wa.me/01288339613`, which is a different
    number that does not exist, and the resulting failure is completely silent — the link
    opens, the chat is empty, no message ever arrives, and nothing logs anything.

    With no credentials the recording gateway stays in place and the flow still runs end to
    end, which is what makes this developable without a Meta account. It is a loud warning
    rather than a hard failure because that is also the state every test runs in.
    """
    registry = schools.get_registry()
    if registry.is_multi_school:
        _configure_whatsapp_per_school(registry)
        return

    number = os.getenv("IDENTITY_WHATSAPP_NUMBER") or ""
    if number:
        number = whatsapp.e164_or_raise(number, setting="IDENTITY_WHATSAPP_NUMBER")
    else:
        logger.warning(
            "IDENTITY_WHATSAPP_NUMBER is not set. Parent sign-in is DISABLED: without "
            "the school's number a click-to-chat link opens WhatsApp's contact picker "
            "instead of a chat, so /v1/auth/whatsapp/start will refuse rather than hand "
            "a parent a link that cannot work."
        )

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

    # Without a Meta account the code is generated and then thrown away, which makes the
    # flow impossible to try end to end on a laptop. This turns it into a log line
    # instead. Off unless asked for, because the body IS the verification code: anywhere
    # a real parent can be verified, this writes their credential into a file that is
    # backed up, shipped to a log aggregator, and read by people who are not them.
    log_codes = (os.getenv("IDENTITY_WHATSAPP_LOG_CODES") or "").strip().lower() in (
        "1", "true", "yes", "on",
    )
    whatsapp.set_gateway(whatsapp.RecordingWhatsAppGateway(log_bodies=log_codes))
    logger.warning(
        "WhatsApp is not configured (IDENTITY_WHATSAPP_PHONE_NUMBER_ID and "
        "IDENTITY_WHATSAPP_TOKEN); verification codes are %s. Parent login by WhatsApp "
        "cannot reach a real phone in this state.",
        "WRITTEN TO THIS LOG (IDENTITY_WHATSAPP_LOG_CODES is on — never do this in "
        "production)" if log_codes
        else "discarded. Set IDENTITY_WHATSAPP_LOG_CODES=true to read them here while "
             "developing",
    )


def _configure_whatsapp_per_school(registry: schools.SchoolRegistry) -> None:
    """One number, one gateway, one webhook — several schools.

    The webhook secrets stay estate-wide: one Meta app delivers every school's messages to
    one endpoint, and which school a delivery belongs to is read from its own
    `phone_number_id` rather than from a separate endpoint per school. What is per school
    is the pair that has to travel together — the number a parent messages, and the
    credentials a reply goes back out through. Split those and a code for one school's
    parent is sent from another school's number, arriving in a conversation the parent is
    not looking at.

    A school with no credentials gets no gateway of its own and falls back to the recording
    gateway, exactly as an unconfigured single-school deployment does. It is logged per
    school rather than once, because "WhatsApp is configured" stops being a single fact the
    moment there are several schools, and an estate where one branch silently cannot
    deliver codes is the failure worth naming.
    """
    whatsapp.configure(
        verify_token=os.getenv("IDENTITY_WHATSAPP_VERIFY_TOKEN") or "",
        app_secret=os.getenv("IDENTITY_WHATSAPP_APP_SECRET") or "",
        # No single business number exists here. `start` resolves each school's own from
        # the registry; this stays empty so anything still reading the process-wide value
        # in a multi-school deployment refuses rather than handing out one school's number
        # to every school's parents.
        business_number="",
    )

    log_codes = _log_codes_enabled()
    live: list[str] = []
    recording: list[str] = []
    for school in registry.schools:
        if school.can_send:
            whatsapp.set_gateway(
                whatsapp.CloudApiWhatsAppGateway(
                    phone_number_id=school.phone_number_id,
                    access_token=school.access_token,
                ),
                school.code,
            )
            live.append(school.code)
        else:
            whatsapp.set_gateway(
                whatsapp.RecordingWhatsAppGateway(log_bodies=log_codes), school.code
            )
            recording.append(school.code)

    logger.info(
        "WhatsApp is configured for %d school(s): %s deliver to real phones.",
        len(registry.schools),
        ", ".join(live) or "none",
    )
    if recording:
        logger.warning(
            "These schools have no WhatsApp credentials and cannot deliver a verification "
            "code to a real phone: %s. Parent login is effectively DISABLED for them.",
            ", ".join(recording),
        )


def _log_codes_enabled() -> bool:
    """Whether the recording gateway may write verification codes into the log.

    Off unless asked for, because the body IS the verification code: anywhere a real parent
    can be verified, this writes their credential into a file that is backed up, shipped to
    a log aggregator, and read by people who are not them.
    """
    return (os.getenv("IDENTITY_WHATSAPP_LOG_CODES") or "").strip().lower() in (
        "1", "true", "yes", "on",
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

    api_key = os.getenv("IDENTITY_SIS_API_KEY") or ""
    if not api_key:
        # Demanded at startup rather than discovered at the first sign-in. SIS
        # authenticates every caller now, so an unkeyed directory does not degrade — it
        # gets a 401 for every parent, which reads downstream as "the school has no such
        # number" and tells every family in the school they are not registered.
        #
        # A `reader` key, deliberately not a registrar one: this service asks whether a
        # number belongs to a parent and never writes a thing.
        raise RuntimeError(
            "IDENTITY_SIS_BASE_URL is set without IDENTITY_SIS_API_KEY. SIS authenticates "
            "its callers; mint a reader-scoped key there and set it here."
        )

    guardians.set_directory(
        guardians.SisGuardianDirectory(base_url=base_url, api_key=api_key)
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
