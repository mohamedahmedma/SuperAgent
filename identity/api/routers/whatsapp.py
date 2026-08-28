"""Parent sign-in by WhatsApp.

Why this exists beside the password login: a parent has no password and should never be
given one. The school already holds their phone number, entered by a registrar from
paperwork, and WhatsApp can prove somebody controls that number for nothing. See
`application/services/whatsapp_login.py` for the flow, and for why it takes two secrets
rather than one.

## The webhook does its work off the event loop

`receive_whatsapp_webhook` must be `async def`, because reading the raw request body is
the only way to check Meta's signature — re-serialising the parsed JSON produces different
bytes and breaks the check for exactly the parents whose names are not ASCII.

But everything *after* that read is blocking: SQLAlchemy queries, and an outbound HTTP call
to Meta that is allowed up to eight seconds. Run inline in an `async def`, those block the
whole event loop — so one slow send to one parent stalls every other request this process
is serving, including JWKS and every password login, for as long as it takes. The claim
work therefore goes to a threadpool, which is where FastAPI would have run it anyway had
the handler been a plain `def`.
"""
import logging

from fastapi import APIRouter, Query, Request, Response, status
from fastapi.responses import PlainTextResponse
from starlette.concurrency import run_in_threadpool

from identity.api.deps import (
    ChannelsDep,
    ClientIp,
    ParentSessionServiceDep,
    WhatsAppServiceDep,
)
from identity.api.routers.auth import AUTH_RESPONSES
from identity.api.schemas.auth import TokenOut
from identity.api.schemas.whatsapp import (
    WhatsAppStartOut,
    WhatsAppStatusIn,
    WhatsAppStatusOut,
    WhatsAppVerifyIn,
)
from identity.infrastructure.whatsapp.inbound import (
    inbound_text_messages,
    signature_is_valid,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/auth/whatsapp", tags=["auth"])


@router.post("/start", response_model=WhatsAppStartOut, status_code=201)
def start_whatsapp_verification(
    service: WhatsAppServiceDep,
    school: str | None = Query(
        default=None,
        description=(
            "Which school the parent is signing in to. Required where this server holds "
            "several; omitted where it holds one. It selects the WhatsApp number the "
            "link points at, and is checked again when the parent's message arrives."
        ),
    ),
) -> WhatsAppStartOut:
    """Begin a verification. Takes no phone number, and that is the point.

    Because the caller states nothing about a *parent*, there is nothing here to probe:
    this endpoint cannot be used to ask whether a given number belongs to a parent. That
    question is answered only to somebody who can actually send a WhatsApp message from
    the number, and it is answered over WhatsApp rather than in this response.

    The school is not a secret and does not weaken that: it names the login page the parent
    is standing on, which is public. What it buys is the pairing — the school recorded here
    must match the school that owns the number the parent's message arrives on, so a nonce
    steered to the wrong school's number is refused rather than resolved against a database
    the parent's children are not in.

    A missing number is a 503 and an unknown school a 404, both decided in
    `api/errors.py`: the first is the operator's to fix, the second the caller's.
    """
    started = service.start(school_code=school)
    return WhatsAppStartOut(
        poll_secret=started.poll_secret,
        link=started.link,
        message=started.message,
        business_number=started.business_number,
        expires_at=started.expires_at,
    )


@router.get("/webhook", include_in_schema=False)
def verify_whatsapp_webhook(request: Request, channels: ChannelsDep) -> Response:
    """Meta's subscription handshake.

    Answered with the bare `hub.challenge` as plain text — no JSON, no quotes. Meta
    compares the body byte for byte, and a JSON-wrapped answer fails the subscription with
    no explanation beyond "the callback URL could not be validated".
    """
    params = request.query_params
    expected = channels.verify_token
    if (
        params.get("hub.mode") == "subscribe"
        and expected
        and params.get("hub.verify_token") == expected
    ):
        return PlainTextResponse(params.get("hub.challenge") or "")
    return PlainTextResponse("", status_code=status.HTTP_403_FORBIDDEN)


@router.post("/webhook", include_in_schema=False)
async def receive_whatsapp_webhook(
    request: Request,
    service: WhatsAppServiceDep,
    channels: ChannelsDep,
) -> dict:
    """Inbound WhatsApp messages.

    **Always answers 200 once the signature checks out.** Meta retries any delivery it does
    not see acknowledged, for up to seven days, so returning an error for a message we
    simply cannot use would have that message replayed for a week.

    The signature is computed over the raw bytes. Re-serialising the parsed JSON produces
    different bytes — Meta escapes non-ASCII — so an Arabic name in a parent's WhatsApp
    profile is enough to break a signature checked against re-encoded JSON, and it breaks
    for only some parents, which is the worst way to find out.
    """
    from fastapi import HTTPException

    raw = await request.body()

    if not channels.app_secret:
        # Unsigned deliveries are being accepted. Logged on EVERY message rather than once
        # at startup, because this is the state that must not persist unnoticed: without
        # the secret, anyone who finds this URL can claim any phone number sent the code
        # phrase, and sign in as that parent.
        logger.warning(
            "Accepting an unverified WhatsApp webhook: IDENTITY_WHATSAPP_APP_SECRET is "
            "not set, so the sender cannot be proved to be Meta. Fine while testing; set "
            "it (App settings -> Basic -> App Secret) before anyone else can reach this."
        )
    elif not signature_is_valid(
        raw_body=raw,
        header=request.headers.get("X-Hub-Signature-256"),
        app_secret=channels.app_secret,
    ):
        # 403 and nothing more. An unsigned caller is not Meta, and Meta does not retry a
        # 403 the way it retries a 5xx.
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "not_authorized", "message": "Bad signature."},
        )

    messages = inbound_text_messages(raw)
    if messages:
        # Off the event loop: see the module docstring. Every message in one delivery is
        # handled in a single hop rather than one hop each, because they are all addressed
        # to the same number and a hop per message would multiply the switching for a
        # payload that is almost always one message long.
        await run_in_threadpool(_claim_all, service, channels, messages)
    return {"received": True}


def _claim_all(service, channels, messages) -> None:
    """Run every claim in this delivery. Blocking, and called only from a worker thread."""
    for message in messages:
        school_code = channels.school_for_delivery(message.phone_number_id)
        outcome = service.claim(
            wa_id=message.wa_id,
            body=message.text,
            message_id=message.message_id,
            school_code=school_code,
        )
        # The number is deliberately not logged: it is a parent's phone and this line goes
        # wherever logs go. The message id is enough to trace one delivery, and the school
        # is safe to name — it is ours, not the parent's.
        logger.info(
            "WhatsApp verification: outcome=%s message_id=%s school=%s",
            outcome,
            message.message_id,
            school_code or "-",
        )


@router.post("/status", response_model=WhatsAppStatusOut)
def whatsapp_verification_status(
    body: WhatsAppStatusIn, service: WhatsAppServiceDep
) -> WhatsAppStatusOut:
    """Where has this verification got to? Polled while the parent goes to tap send."""
    found = service.status(poll_secret=body.poll_secret)
    return WhatsAppStatusOut(
        status=found.status,
        display_name=found.display_name,
        expires_at=found.expires_at,
    )


@router.post("/verify", response_model=TokenOut, responses=AUTH_RESPONSES)
def complete_whatsapp_verification(
    body: WhatsAppVerifyIn,
    service: WhatsAppServiceDep,
    parents: ParentSessionServiceDep,
    ip: ClientIp,
) -> TokenOut:
    """The code, and the tokens if it is right.

    The account is created here on first use rather than by an administrator, and the
    binding written onto it does not come from this request: it came from the school's own
    records, keyed on a number WhatsApp proved. That is the invariant
    `domain/accounts.py` states — an account never names its own guardian — held through a
    second authority rather than broken by one.
    """
    from identity.domain.errors import VerificationError

    try:
        challenge = service.verify(poll_secret=body.poll_secret, code=body.code)
    except VerificationError as error:
        # Audited before it is re-raised. A failed verification is exactly the event an
        # incident review needs, and `api/errors.py` — which has no session — cannot write
        # it.
        parents.audit_failure(event="whatsapp_verify", reason=error.code, client_ip=ip)
        raise

    session = parents.sign_in(challenge, client_ip=ip)
    return TokenOut(
        access_token=session.access_token,
        refresh_token=session.refresh_token,
        expires_at=session.expires_at,
        username=session.username,
        role=session.role,
        guardian_id=session.guardian_external_id,
        display_name=session.display_name,
    )


__all__ = ["router"]
