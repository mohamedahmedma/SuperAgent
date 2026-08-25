"""Play Meta's part, so parent sign-in can be tested without a Cloud API account.

The click-to-chat link works with any WhatsApp number — it only opens a chat. What needs
Cloud API is the other direction: Meta calling YOUR server to say a message arrived. With
`IDENTITY_WHATSAPP_PHONE_NUMBER_ID` and `IDENTITY_WHATSAPP_TOKEN` unset, no webhook is
registered, so a parent can send the message and nothing on this machine ever hears about
it. That is not a bug; it is the half of the flow that Meta owns.

This posts the webhook Meta would have posted, signed the way Meta signs it.

    python scripts/simulate_whatsapp.py +201093887199

It requires `IDENTITY_WHATSAPP_APP_SECRET` to be set, because the webhook verifies an
HMAC over the raw body and refuses anything else with a 403 — correctly, since an
unsigned caller is not Meta. Any string will do locally, as long as identity was started
with the same one.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import httpx  # noqa: E402

IDENTITY = (os.getenv("IDENTITY_JWKS_URL", "http://localhost:8200/x")
            .rsplit("/.well-known", 1)[0].rstrip("/"))
SECRET = (os.getenv("IDENTITY_WHATSAPP_APP_SECRET") or "").strip()

def _guardian_numbers(limit: int = 8) -> list[tuple[str, str, int]]:
    """`(phone, name, child_count)` straight from the SIS database.

    Read directly rather than over HTTP because there is no route that lists guardians —
    deliberately, since one would be a way to enumerate every parent's phone number. This
    is a developer's own machine reading its own file.
    """
    import sqlite3

    url = os.getenv("SIS_DATABASE_URL", "sqlite:///./sis.db")
    if not url.startswith("sqlite:///"):
        return []
    try:
        con = sqlite3.connect(f"file:{url[10:]}?mode=ro", uri=True)
        rows = con.execute(
            "SELECT MIN(p.phone), g.full_name_ar, COUNT(DISTINCT sg.id) AS n "
            "FROM guardians g "
            "JOIN guardian_phones p ON p.guardian_id = g.id "
            "LEFT JOIN student_guardians sg ON sg.guardian_id = g.id "
            "GROUP BY g.id ORDER BY n DESC, g.id LIMIT ?",
            (limit,),
        ).fetchall()
        con.close()
        return rows
    except Exception:  # noqa: BLE001 - a convenience listing must never be the failure
        return []


if len(sys.argv) < 2:
    print("Give the parent's phone number:")
    print()
    print("    python scripts/simulate_whatsapp.py +201093887199")
    print()
    found = _guardian_numbers()
    if found:
        print("Numbers this school will recognise (any other is refused):")
        for phone, name, children in found:
            print(f"    {phone:<16} {name or '(unnamed)':<14} {children} child(ren)")
    else:
        print("No guardians found in sis.db — upload a guardians sheet at /ui first.")
    print()
    print("This stands in for the webhook Meta would send. It is needed because the")
    print("number is not registered on the WhatsApp Business Platform (Cloud API), so")
    print("nothing delivers a parent's message to this machine.")
    raise SystemExit(2)

phone = sys.argv[1].strip()
if not phone.startswith("+"):
    print(f"'{phone}' must be international form with a leading + — the school's records")
    print("store E.164, and a national spelling matches nobody.")
    raise SystemExit(2)

if not SECRET:
    print("IDENTITY_WHATSAPP_APP_SECRET is empty.")
    print()
    print("The webhook verifies an HMAC over the raw body and refuses anything else, so")
    print("there is nothing to sign with. Put any string in .env, restart identity so it")
    print("reads the new value, and run this again:")
    print()
    print("    IDENTITY_WHATSAPP_APP_SECRET=local-dev-secret")
    raise SystemExit(1)

# 1. Begin a verification, exactly as the sign-in screen does.
started = httpx.post(f"{IDENTITY}/v1/auth/whatsapp/start", timeout=10)
if started.status_code != 201:
    print(f"identity refused to start a verification: {started.status_code}")
    print(started.text[:300])
    raise SystemExit(1)
challenge = started.json()
print(f"1. the parent is shown : {challenge['link']}")
print(f"   they send           : {challenge['message']}")

# 2. Deliver it, the way Meta would.
wa_id = phone.lstrip("+")
payload = {
    "object": "whatsapp_business_account",
    "entry": [{
        "id": "SIMULATED",
        "changes": [{
            "field": "messages",
            "value": {
                "messaging_product": "whatsapp",
                "metadata": {"display_phone_number": "sim", "phone_number_id": "sim"},
                "contacts": [{"profile": {"name": "simulated parent"}, "wa_id": wa_id}],
                "messages": [{
                    "from": wa_id,
                    # Unique per run: the webhook de-duplicates on this, because Meta
                    # retries a delivery for up to seven days and a replay must not
                    # burn a second challenge.
                    "id": f"wamid.SIM{os.urandom(6).hex()}",
                    "timestamp": "1",
                    "type": "text",
                    "text": {"body": challenge["message"]},
                }],
            },
        }],
    }],
}

# The signature is over the bytes as sent. Re-serialising the parsed JSON produces a
# different byte string — Meta escapes non-ASCII — so the body is built once and both
# signed and posted from that exact object.
raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
signature = hmac.new(SECRET.encode(), raw, hashlib.sha256).hexdigest()

delivered = httpx.post(
    f"{IDENTITY}/v1/auth/whatsapp/webhook",
    content=raw,
    headers={
        "Content-Type": "application/json",
        "X-Hub-Signature-256": f"sha256={signature}",
    },
    timeout=10,
)
print(f"2. webhook delivered   : HTTP {delivered.status_code}")
if delivered.status_code == 403:
    print("   Bad signature — identity is running with a different APP_SECRET than .env.")
    print("   Restart identity so it picks up the current value.")
    raise SystemExit(1)

# 3. Ask what identity made of it, as the browser polls.
status = httpx.post(
    f"{IDENTITY}/v1/auth/whatsapp/status",
    json={"poll_secret": challenge["poll_secret"]},
    timeout=10,
).json()
print(f"3. verification status : {status.get('status')}"
      + (f"  ({status.get('display_name')})" if status.get("display_name") else ""))

if status.get("status") == "rejected":
    print()
    print(f"   {phone} is not a guardian sis knows.")
    print("   Check the number against sis, and that IDENTITY_SIS_BASE_URL is set and sis")
    print("   is running — with it unset, every parent is rejected.")
    raise SystemExit(1)

if status.get("status") != "code_sent":
    print()
    print(f"   Expected 'code_sent'. Identity said '{status.get('status')}'.")
    raise SystemExit(1)

print()
print("The code was 'sent'. With no Cloud API it went to identity's own log instead of a")
print("phone — look in the identity :8200 window for the line containing it")
print("(IDENTITY_WHATSAPP_LOG_CODES must be true), then type those six digits into the")
print("sign-in screen.")
