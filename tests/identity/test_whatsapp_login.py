"""Parent login by WhatsApp, end to end, with no Meta account and no SIS running.

Both seams are Protocols, so the whole flow runs against a list and a dict. That is the
point of the ports: the rules below — that a nonce alone is worthless, that a stranger's
number is refused, that Meta's retries do not send three codes — are properties of the
flow, and a test that needed a real WhatsApp number to assert them would not exist.

The two tests that matter most are the pair near the bottom that hold one half of the
proof each. Every other test here checks that the flow works; those two check that it
cannot be made to work by somebody who should not be able to.
"""
import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone

import pytest

from identity import guardians as directory_module
from identity import whatsapp as wa
from identity.guardians import FakeGuardianDirectory, GuardianRef
from identity.models import Account, VerificationChallenge
from identity.whatsapp import RecordingWhatsAppGateway

APP_SECRET = "test-app-secret"
VERIFY_TOKEN = "test-verify-token"
SCHOOL_NUMBER = "+201288339613"

MOTHER_WA_ID = "201001234567"
MOTHER_E164 = "+201001234567"
MOTHER_ALT_E164 = "+201119998888"
STRANGER_WA_ID = "201110000000"

MOTHER = GuardianRef(
    public_id="guardian-public-id-1",
    full_name_ar="فاطمة علي",
    full_name_en="Fatma Ali",
    preferred_language="ar",
)


@pytest.fixture()
def gateway() -> RecordingWhatsAppGateway:
    """Every message the school would have sent, kept instead of sent."""
    sender = RecordingWhatsAppGateway()
    wa.set_gateway(sender)
    wa.configure(
        verify_token=VERIFY_TOKEN, app_secret=APP_SECRET, business_number=SCHOOL_NUMBER
    )
    return sender


@pytest.fixture()
def directory() -> FakeGuardianDirectory:
    """The school's records: one mother, reachable on either of her two numbers."""
    fake = FakeGuardianDirectory({MOTHER_E164: MOTHER, MOTHER_ALT_E164: MOTHER})
    directory_module.set_directory(fake)
    return fake


def _signed(payload: dict) -> tuple[bytes, dict[str, str]]:
    """A payload and the header Meta would have signed it with.

    `ensure_ascii=False` so Arabic survives as UTF-8 bytes rather than `\\uXXXX` escapes.
    That is what Meta actually sends, and signing the escaped form is the mistake this
    helper exists to keep the tests honest about.
    """
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    signature = hmac.new(APP_SECRET.encode(), raw, hashlib.sha256).hexdigest()
    return raw, {
        "X-Hub-Signature-256": f"sha256={signature}",
        "Content-Type": "application/json",
    }


def _inbound(wa_id: str, text: str, *, message_id: str = "wamid.TEST1") -> dict:
    """One inbound text message, in the shape Meta documents."""
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "waba-1",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "201288339613",
                                "phone_number_id": "phone-number-id-1",
                            },
                            "contacts": [
                                {"profile": {"name": "فاطمة علي"}, "wa_id": wa_id}
                            ],
                            "messages": [
                                {
                                    "from": wa_id,
                                    "id": message_id,
                                    "timestamp": "1750000000",
                                    "type": "text",
                                    "text": {"body": text},
                                }
                            ],
                        },
                    }
                ],
            }
        ],
    }


def _deliver(client, wa_id: str, text: str, *, message_id: str = "wamid.TEST1"):
    raw, headers = _signed(_inbound(wa_id, text, message_id=message_id))
    return client.post("/v1/auth/whatsapp/webhook", content=raw, headers=headers)


def _start(client) -> dict:
    response = client.post("/v1/auth/whatsapp/start")
    assert response.status_code == 201, response.text
    return response.json()


def _code_from(gateway: RecordingWhatsAppGateway) -> str:
    """The six digits out of the last message the school sent."""
    body = gateway.sent[-1][1]
    digits = "".join(character for character in body if character.isdigit())
    assert len(digits) == 6, f"expected a six-digit code, got {body!r}"
    return digits


# ---------------------------------------------------------------------------
# The flow
# ---------------------------------------------------------------------------


def test_a_parent_signs_in_without_ever_having_a_password(
    client, gateway, directory
) -> None:
    """The whole point, start to finish."""
    started = _start(client)
    assert started["business_number"] == SCHOOL_NUMBER
    assert started["link"].startswith("https://wa.me/201288339613?text=")
    # The link never sends itself, so the message has to be visible for a parent whose
    # in-app browser swallowed the handoff.
    assert started["message"] in ("SCHOOL VERIFY: " + started["message"].split()[-1],)

    assert _deliver(client, MOTHER_WA_ID, started["message"]).status_code == 200

    status = client.post(
        "/v1/auth/whatsapp/status", json={"poll_secret": started["poll_secret"]}
    ).json()
    assert status["status"] == "code_sent"
    # She is greeted by name before she has an account at all.
    assert status["display_name"] == "فاطمة علي"

    verified = client.post(
        "/v1/auth/whatsapp/verify",
        json={"poll_secret": started["poll_secret"], "code": _code_from(gateway)},
    )
    assert verified.status_code == 200, verified.text
    body = verified.json()
    assert body["guardian_id"] == MOTHER.public_id
    assert body["role"] == "parent"
    assert body["access_token"] and body["refresh_token"]


def test_the_token_carries_the_guardian_claim_other_services_verify(
    client, gateway, directory
) -> None:
    """`guardian_id` is the whole integration; a token without it reads nothing anywhere."""
    started = _start(client)
    _deliver(client, MOTHER_WA_ID, started["message"])
    token = client.post(
        "/v1/auth/whatsapp/verify",
        json={"poll_secret": started["poll_secret"], "code": _code_from(gateway)},
    ).json()["access_token"]

    me = client.get("/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200, me.text
    assert me.json()["guardian_id"] == MOTHER.public_id


def test_starting_a_verification_asks_for_no_phone_number(client, gateway, directory) -> None:
    """There is nothing here to probe.

    The password login spends a whole PBKDF2 hash making an unknown username
    indistinguishable from a known one. This flow gets the same property for free by never
    accepting a number: the only way to ask "is this number a parent" is to send a message
    from it, and the answer goes to that number over WhatsApp.
    """
    started = _start(client)

    assert "phone" not in started
    assert directory.asked == []


def test_a_number_the_school_does_not_hold_is_refused_and_told_so(
    client, gateway, directory
) -> None:
    """Refused politely over WhatsApp, and told nothing about which numbers do work."""
    started = _start(client)

    assert _deliver(client, STRANGER_WA_ID, started["message"]).status_code == 200

    sent = gateway.sent[-1][1]
    assert "not registered" in sent
    # No code, and no hint about the school's data.
    assert not any(character.isdigit() for character in sent.replace(" ", ""))

    status = client.post(
        "/v1/auth/whatsapp/status", json={"poll_secret": started["poll_secret"]}
    ).json()
    assert status["status"] == "rejected"


def test_this_flow_never_creates_a_guardian(client, gateway, directory) -> None:
    """Whose parent somebody is, is the registrar's fact and not a claim to self-assert."""
    started = _start(client)
    _deliver(client, STRANGER_WA_ID, started["message"])

    verified = client.post(
        "/v1/auth/whatsapp/verify",
        json={"poll_secret": started["poll_secret"], "code": "000000"},
    )
    assert verified.status_code == 400
    assert verified.json()["detail"]["code"] == "not_ready"


def test_either_of_her_numbers_reaches_the_same_account(client, gateway, directory) -> None:
    """A mother who verifies her WhatsApp line is the woman who verified her mobile.

    Keyed on the guardian handle rather than the number precisely so this holds — a second
    account would carry half her children and none of her history.
    """
    for wa_id in (MOTHER_WA_ID, "201119998888"):
        started = _start(client)
        _deliver(client, wa_id, started["message"], message_id=f"wamid.{wa_id}")
        signed_in = client.post(
            "/v1/auth/whatsapp/verify",
            json={"poll_secret": started["poll_secret"], "code": _code_from(gateway)},
        )
        assert signed_in.status_code == 200, signed_in.text

    accounts = client.app  # noqa: F841 - the assertion below reads the database directly
    from identity.db import new_session

    session = new_session()
    try:
        rows = session.query(Account).filter(Account.role == "parent").all()
        assert len(rows) == 1
        assert rows[0].guardian_external_id == MOTHER.public_id
    finally:
        session.close()


# ---------------------------------------------------------------------------
# The two secrets. Each of these holds one half and must fail.
# ---------------------------------------------------------------------------


def test_a_stolen_nonce_alone_cannot_sign_anyone_in(client, gateway, directory) -> None:
    """Someone screenshots the link and sends it from their own phone.

    The code is then delivered to *their* WhatsApp — but the poll secret never left the
    browser that asked, so they have nothing to type it into. They learn only about their
    own number, which they already knew.
    """
    victim = _start(client)

    # The attacker sends the victim's nonce from the attacker's own registered number.
    _deliver(client, MOTHER_WA_ID, victim["message"])

    # They hold the code, and no poll secret.
    code = _code_from(gateway)
    attacker_attempt = client.post(
        "/v1/auth/whatsapp/verify",
        json={"poll_secret": "a-secret-the-attacker-invented", "code": code},
    )
    assert attacker_attempt.status_code == 400
    assert attacker_attempt.json()["detail"]["code"] == "not_found"


def test_a_poll_secret_alone_cannot_sign_anyone_in(client, gateway, directory) -> None:
    """The mirror image: a parent is tricked into sending the attacker's nonce.

    The code goes to the *parent's* WhatsApp, which the attacker cannot read. They hold the
    poll secret and must still guess six digits, five times.
    """
    attackers = _start(client)
    _deliver(client, MOTHER_WA_ID, attackers["message"])

    guessed = client.post(
        "/v1/auth/whatsapp/verify",
        json={"poll_secret": attackers["poll_secret"], "code": "000000"},
    )
    assert guessed.status_code == 400
    assert guessed.json()["detail"]["code"] in {"bad_code", "too_many_attempts"}


# ---------------------------------------------------------------------------
# Codes
# ---------------------------------------------------------------------------


def test_a_wrong_code_is_refused_and_counted(client, gateway, directory) -> None:
    started = _start(client)
    _deliver(client, MOTHER_WA_ID, started["message"])

    wrong = client.post(
        "/v1/auth/whatsapp/verify",
        json={"poll_secret": started["poll_secret"], "code": "000000"},
    )
    assert wrong.status_code == 400
    assert wrong.json()["detail"]["code"] == "bad_code"

    # The right code still works afterwards: one slip is not a lockout.
    right = client.post(
        "/v1/auth/whatsapp/verify",
        json={"poll_secret": started["poll_secret"], "code": _code_from(gateway)},
    )
    assert right.status_code == 200, right.text


def test_guessing_is_stopped_long_before_a_million_tries(client, gateway, directory) -> None:
    """Six digits is only safe because the counter exists.

    The counter lives on the challenge rather than on an account because at this point in
    the flow there may be no account — which is exactly why the password lockout cannot
    serve here.
    """
    started = _start(client)
    _deliver(client, MOTHER_WA_ID, started["message"])
    real = _code_from(gateway)

    # Five guesses are allowed and the sixth is refused, so the loop runs one past the
    # limit: asserting on exactly `_MAX_ATTEMPTS` guesses would pass against an
    # implementation that allowed four and against one that allowed five.
    outcomes = [
        client.post(
            "/v1/auth/whatsapp/verify",
            json={"poll_secret": started["poll_secret"], "code": f"{n:06d}"},
        ).json()["detail"]["code"]
        for n in range(6)
    ]
    assert outcomes[:5] == ["bad_code"] * 5
    assert outcomes[5] == "too_many_attempts"

    # Even the correct code is now worthless.
    assert (
        client.post(
            "/v1/auth/whatsapp/verify",
            json={"poll_secret": started["poll_secret"], "code": real},
        ).status_code
        == 400
    )


def test_a_challenge_mints_one_token_and_no_more(client, gateway, directory) -> None:
    """Consumed on success, so a replayed request cannot produce a second session."""
    started = _start(client)
    _deliver(client, MOTHER_WA_ID, started["message"])
    code = _code_from(gateway)

    assert (
        client.post(
            "/v1/auth/whatsapp/verify",
            json={"poll_secret": started["poll_secret"], "code": code},
        ).status_code
        == 200
    )
    replayed = client.post(
        "/v1/auth/whatsapp/verify",
        json={"poll_secret": started["poll_secret"], "code": code},
    )
    assert replayed.status_code == 400
    assert replayed.json()["detail"]["code"] == "already_used"


def test_an_expired_challenge_is_dead(client, gateway, directory, db) -> None:
    """A link shared later is worthless, which is why the TTL is short."""
    started = _start(client)
    _deliver(client, MOTHER_WA_ID, started["message"])
    code = _code_from(gateway)

    challenge = db.query(VerificationChallenge).one()
    challenge.expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
    db.commit()

    expired = client.post(
        "/v1/auth/whatsapp/verify",
        json={"poll_secret": started["poll_secret"], "code": code},
    )
    assert expired.status_code == 400
    assert expired.json()["detail"]["code"] == "expired"


# ---------------------------------------------------------------------------
# The webhook
# ---------------------------------------------------------------------------


def test_the_subscription_handshake_returns_the_bare_challenge(client, gateway) -> None:
    """Meta compares the body byte for byte; a JSON wrapper fails with no explanation."""
    response = client.get(
        "/v1/auth/whatsapp/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": VERIFY_TOKEN,
            "hub.challenge": "1158201444",
        },
    )
    assert response.status_code == 200
    assert response.text == "1158201444"


def test_a_wrong_verify_token_is_refused(client, gateway) -> None:
    response = client.get(
        "/v1/auth/whatsapp/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "not-the-token",
            "hub.challenge": "1158201444",
        },
    )
    assert response.status_code == 403


def test_an_unsigned_payload_is_refused(client, gateway, directory) -> None:
    """The webhook is public. Without the signature check, anyone could claim to be
    any parent's phone by posting a payload that says so."""
    started = _start(client)
    unsigned = client.post(
        "/v1/auth/whatsapp/webhook",
        content=json.dumps(_inbound(MOTHER_WA_ID, started["message"])).encode(),
        headers={"Content-Type": "application/json"},
    )
    assert unsigned.status_code == 403
    assert gateway.sent == []


def test_a_tampered_payload_is_refused(client, gateway, directory) -> None:
    started = _start(client)
    raw, headers = _signed(_inbound(MOTHER_WA_ID, started["message"]))

    tampered = client.post(
        "/v1/auth/whatsapp/webhook", content=raw + b" ", headers=headers
    )
    assert tampered.status_code == 403
    assert gateway.sent == []


def test_the_signature_is_checked_over_the_raw_bytes(client, gateway, directory) -> None:
    """An Arabic profile name is enough to break a signature computed over re-encoded JSON.

    Meta escapes non-ASCII on the wire. A handler that parses and re-serialises produces
    different bytes, so the HMAC fails — for some parents only, which is the worst way to
    discover the bug. The payload here carries Arabic for exactly that reason.
    """
    started = _start(client)
    assert _deliver(client, MOTHER_WA_ID, started["message"]).status_code == 200
    assert gateway.sent, "an Arabic-carrying payload must verify like any other"


def test_metas_retries_do_not_send_a_second_code(client, gateway, directory) -> None:
    """Meta replays a delivery for up to seven days when it is not acknowledged.

    Without deduplication on the message id, one parent tap becomes several codes in the
    thread and several burnt challenges, and the parent types the one that no longer works.
    """
    started = _start(client)

    for _ in range(3):
        assert (
            _deliver(
                client, MOTHER_WA_ID, started["message"], message_id="wamid.SAME"
            ).status_code
            == 200
        )

    assert len(gateway.sent) == 1


def test_a_message_with_no_code_is_ignored_silently(client, gateway, directory) -> None:
    """People message a number they can see. Replying to strangers is not this flow's job."""
    assert _deliver(client, STRANGER_WA_ID, "Hello, is this the school?").status_code == 200
    assert gateway.sent == []


def test_a_nonce_survives_a_parent_typing_around_it(client, gateway, directory) -> None:
    """Parents add words, and keyboards capitalise. Only the token has to survive."""
    started = _start(client)
    nonce = started["message"].split()[-1]

    _deliver(client, MOTHER_WA_ID, f"hello please verify me: {nonce.lower()} thanks")

    assert gateway.sent, "an edited message should still carry its nonce"


def test_a_non_text_message_is_ignored(client, gateway, directory) -> None:
    """Stickers and voice notes carry no nonce and must not raise on the retry path."""
    payload = _inbound(MOTHER_WA_ID, "irrelevant")
    payload["entry"][0]["changes"][0]["value"]["messages"][0]["type"] = "sticker"
    raw, headers = _signed(payload)

    assert client.post("/v1/auth/whatsapp/webhook", content=raw, headers=headers).status_code == 200
    assert gateway.sent == []


# ---------------------------------------------------------------------------
# When the school's records cannot be reached
# ---------------------------------------------------------------------------


def test_an_unreachable_directory_leaves_the_challenge_usable(client, gateway) -> None:
    """Our problem, not the parent's — so the challenge survives for a second attempt."""
    directory_module.set_directory(FakeGuardianDirectory(unavailable=True))
    started = _start(client)

    assert _deliver(client, MOTHER_WA_ID, started["message"]).status_code == 200
    assert "try again" in gateway.sent[-1][1]

    status = client.post(
        "/v1/auth/whatsapp/status", json={"poll_secret": started["poll_secret"]}
    ).json()
    assert status["status"] == "pending"


def test_an_undeliverable_code_does_not_leave_a_parent_waiting(client, directory) -> None:
    """If the code could not be sent, the challenge is dead rather than unsatisfiable.

    A parent staring at a code box that can never be filled is worse than one told plainly
    to start again.
    """

    class BrokenGateway:
        def send_text(self, to_wa_id: str, body: str) -> None:
            raise wa.WhatsAppUnavailable("no")

    wa.set_gateway(BrokenGateway())
    wa.configure(
        verify_token=VERIFY_TOKEN, app_secret=APP_SECRET, business_number=SCHOOL_NUMBER
    )
    started = _start(client)

    assert _deliver(client, MOTHER_WA_ID, started["message"]).status_code == 200

    status = client.post(
        "/v1/auth/whatsapp/status", json={"poll_secret": started["poll_secret"]}
    ).json()
    assert status["status"] == "rejected"


def test_a_disabled_account_cannot_be_revived_by_re_verifying(
    client, gateway, directory, db
) -> None:
    """Re-proving a phone must not walk back an administrator's decision."""
    started = _start(client)
    _deliver(client, MOTHER_WA_ID, started["message"])
    client.post(
        "/v1/auth/whatsapp/verify",
        json={"poll_secret": started["poll_secret"], "code": _code_from(gateway)},
    )

    account = db.query(Account).filter(Account.role == "parent").one()
    account.is_active = False
    db.commit()

    again = _start(client)
    _deliver(client, MOTHER_WA_ID, again["message"], message_id="wamid.SECOND")
    refused = client.post(
        "/v1/auth/whatsapp/verify",
        json={"poll_secret": again["poll_secret"], "code": _code_from(gateway)},
    )
    assert refused.status_code == 401
