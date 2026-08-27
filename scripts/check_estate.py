"""Walk a parent's whole sign-in against the services you actually have running.

Not a test. Tests prove the code is right against fakes; this proves *this machine* is
wired right — which is a different question and the one that has failed every time so
far. Every bug this feature hit in production was a seam where two correct services
disagreed about a name, a port, or an environment variable.

    python scripts/check_estate.py

It reads `.env`, talks to the running services over HTTP, and prints what it found. It
writes nothing except one verification challenge, which expires on its own.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import httpx  # noqa: E402

SIS = os.getenv("SIS_BASE_URL", "http://localhost:8300").rstrip("/")
RECORDS = os.getenv("RECORDS_BASE_URL", "http://localhost:8100").rstrip("/")
IDENTITY = (os.getenv("IDENTITY_JWKS_URL", "http://localhost:8200/x")
            .rsplit("/.well-known", 1)[0].rstrip("/"))

OK, BAD, WARN = "  OK  ", " FAIL ", " WARN "
problems: list[str] = []


def report(mark: str, what: str, detail: str = "") -> None:
    print(f"[{mark}] {what}")
    if detail:
        print(f"        {detail}")
    if mark is BAD:
        problems.append(what)


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def reachable(name: str, url: str) -> bool:
    try:
        response = httpx.get(f"{url}/docs", timeout=4, follow_redirects=True)
    except httpx.HTTPError as error:
        report(BAD, f"{name} is not reachable at {url}", str(error)[:100])
        return False
    report(OK, f"{name} is up at {url}", f"HTTP {response.status_code}")
    return True


# --------------------------------------------------------------------------
section("1. Are the services running?")
# --------------------------------------------------------------------------
up = {
    "sis": reachable("sis", SIS),
    "records": reachable("records", RECORDS),
    "identity": reachable("identity", IDENTITY),
}
if not all(up.values()):
    print()
    print("Start what is missing with run_all.bat, then run this again.")
    raise SystemExit(1)

# --------------------------------------------------------------------------
section("2. Configuration a parent's sign-in depends on")
# --------------------------------------------------------------------------
for name, why in (
    ("IDENTITY_WHATSAPP_NUMBER", "without it the wa.me link opens the contact picker"),
    ("IDENTITY_SIS_BASE_URL", "without it every parent is told their number is unknown"),
    ("RECORDS_BASE_URL", "without it the chat backend cannot read any records"),
    ("RECORDS_API_KEY", "without it records refuses the chat backend"),
    ("ACTIVE_PROFILE", "must be `school`, or the records tool is not bound at all"),
):
    value = (os.getenv(name) or "").strip()
    if not value:
        report(BAD, f"{name} is not set", why)
    else:
        shown = value if "KEY" not in name else value[:8] + "…"
        report(OK, f"{name} = {shown}")

if (os.getenv("ACTIVE_PROFILE") or "").strip() != "school":
    report(BAD, "ACTIVE_PROFILE is not `school`", "the records tool is bound by that profile only")

# --------------------------------------------------------------------------
section("3. Does the school have a parent to sign in?")
# --------------------------------------------------------------------------
guardian_phone = ""
public_id = ""
try:
    # Any guardian will do; this only needs one to prove the chain.
    import sqlite3

    url = os.getenv("SIS_DATABASE_URL", "sqlite:///./sis.db")
    if url.startswith("sqlite:///"):
        con = sqlite3.connect(f"file:{url[10:]}?mode=ro", uri=True)
        total = con.execute("SELECT COUNT(*) FROM students").fetchone()[0]
        report(OK if total else BAD, f"sis holds {total} student(s)",
               "" if total else "upload a roster at /ui first")
        # Prefer a parent with SEVERAL children: that is the case the whole feature is
        # about, and a one-child parent proves nothing about it.
        row = con.execute(
            # DISTINCT, because the join to guardian_phones fans out: a guardian with
            # two numbers on file would otherwise be reported as having twice as many
            # children as she has.
            "SELECT g.public_id, MIN(p.phone), g.full_name_ar, COUNT(DISTINCT sg.id) AS n "
            "FROM guardians g "
            "JOIN guardian_phones p ON p.guardian_id = g.id "
            "LEFT JOIN student_guardians sg ON sg.guardian_id = g.id "
            "GROUP BY g.id ORDER BY n DESC LIMIT 1"
        ).fetchone()
        con.close()
        if row:
            public_id, guardian_phone, name, children = row
            report(OK, f"a parent to test with: {name or public_id}",
                   f"{guardian_phone} — {children} child(ren)"
                   + ("  <- several: 'my son' will have to ask" if children > 1
                      else "  <- an only child: she is never asked which one"))
            if children < 2:
                report(WARN, "no parent here has more than one child",
                       "the ask-which-child path cannot be exercised on this data; "
                       "seed a sibling pair to try it")
        else:
            report(BAD, "sis holds no guardians", "upload a guardians sheet at /ui first")
except Exception as error:  # noqa: BLE001
    report(WARN, "could not read a sample guardian directly", str(error)[:90])

# --------------------------------------------------------------------------
section("4. The sign-in link a parent is handed")
# --------------------------------------------------------------------------
try:
    started = httpx.post(f"{IDENTITY}/v1/auth/whatsapp/start", timeout=8)
    if started.status_code == 503:
        report(BAD, "identity refuses to start a sign-in",
               started.json().get("detail", {}).get("message", ""))
    elif started.status_code != 201:
        report(BAD, f"unexpected status {started.status_code}", started.text[:120])
    else:
        body = started.json()
        link = body["link"]
        digits = link.split("wa.me/", 1)[1].split("?", 1)[0]
        if digits:
            report(OK, "the link opens a chat, not the contact picker", link)
        else:
            report(BAD, "the link has NO NUMBER — it opens the contact picker", link)
        report(OK, "the message a parent sends", body["message"])

        # The check that catches a stale process. A service reads its settings ONCE, at
        # startup, so editing `.env` changes nothing until it restarts — and if the old
        # one is still holding the port, the new one fails to bind and the old keeps
        # answering. Nothing anywhere reports that: the link is valid, the endpoint
        # returns 201, and it names a number you replaced an hour ago.
        configured = (os.getenv("IDENTITY_WHATSAPP_NUMBER") or "").strip()
        serving = (body.get("business_number") or "").strip()
        if configured and serving and configured != serving:
            report(
                BAD,
                "identity is serving a number that is not the one in .env",
                f".env says {configured}, the running service says {serving} — it is an "
                f"older process. Stop whatever is listening on that port and start it "
                f"again; a second uvicorn cannot bind a port that is already taken, and "
                f"it says so only in its own window.",
            )
        elif configured and serving:
            report(OK, "the running service matches .env", serving)
except Exception as error:  # noqa: BLE001
    report(BAD, "could not begin a verification", str(error)[:120])

# --------------------------------------------------------------------------
section("5. Can identity resolve that number to a parent?")
# --------------------------------------------------------------------------
if guardian_phone:
    try:
        # SIS authenticates every caller. Reading `SIS_API_KEY` rather than skipping the
        # header means this check fails when the key is wrong — which is the whole point,
        # since a wrong key here looks downstream like "the school has no such number".
        resolved = httpx.post(
            f"{SIS}/v1/guardians/resolve",
            json={"phone": guardian_phone},
            headers={"X-API-Key": (os.getenv("SIS_API_KEY") or "").strip()},
            timeout=6,
        )
        if resolved.status_code == 200:
            handle = resolved.json().get("public_id", "")
            report(OK, "sis resolves the number to a guardian handle", handle)
            kids = httpx.get(
                f"{SIS}/v1/guardians/by-id/{handle}/students",
                headers={"X-API-Key": (os.getenv("SIS_API_KEY") or "").strip()},
                timeout=6,
            ).json()
            for child in kids.get("students", []):
                report(OK, f"  child: {child.get('full_name_ar') or child.get('full_name_en')}",
                       f"gender={child.get('gender')}  year={child.get('year_level') or '(none)'}")
            if not kids.get("students"):
                report(BAD, "that guardian has no readable children",
                       "check can_view_records on the link")
        else:
            report(BAD, f"sis could not resolve {guardian_phone}", resolved.text[:120])
    except Exception as error:  # noqa: BLE001
        report(BAD, "the guardian lookup failed", str(error)[:120])

# --------------------------------------------------------------------------
section("6. Can a real phone reach you? (WhatsApp Cloud API)")
# --------------------------------------------------------------------------
# The click-to-chat link works with any number — it only opens a chat. Everything below
# is the OTHER direction: Meta calling this server to say a message arrived. Without it a
# parent can send the message and nothing here ever hears about it.
cloud = {
    name: (os.getenv(name) or "").strip()
    for name in (
        "IDENTITY_WHATSAPP_PHONE_NUMBER_ID",
        "IDENTITY_WHATSAPP_TOKEN",
        "IDENTITY_WHATSAPP_APP_SECRET",
        "IDENTITY_WHATSAPP_VERIFY_TOKEN",
    )
}
# The app secret is the one that is legitimately blank: unset means identity skips the
# signature check and accepts the delivery, which is how the flow is tested before the
# real secret is in place. Reported as a warning on its own rather than counted as
# missing, because "you chose this" and "you forgot this" deserve different words.
signing = cloud.pop("IDENTITY_WHATSAPP_APP_SECRET")
missing = [name for name, value in cloud.items() if not value]

if missing == list(cloud) and not signing:
    report(WARN, "Cloud API is not configured — real phones cannot sign in",
           "This is the only part a laptop cannot fake. Until it is set up, use "
           "scripts/simulate_whatsapp.py, which posts the webhook Meta would post.")
elif missing:
    report(BAD, f"Cloud API is half configured — missing {', '.join(missing)}",
           "Sending needs the phone number id and the token; the subscription handshake "
           "needs the verify token.")
else:
    report(OK, "Cloud API can send: phone number id, token and verify token are set")

if not signing:
    report(WARN, "webhook signatures are NOT being verified",
           "IDENTITY_WHATSAPP_APP_SECRET is blank, so identity accepts any caller that "
           "reaches the webhook URL — including one claiming any phone number sent the "
           "code phrase. Fine while testing; set it (App settings -> Basic -> App Secret) "
           "before this URL is reachable by anyone else.")
else:
    report(OK, "webhook signatures are verified")
    # Ask Meta whether the credentials actually work. A token that has expired, or a
    # phone number id from a different WABA, looks identical to a correct one in .env.
    try:
        probe = httpx.get(
            f"https://graph.facebook.com/v21.0/{cloud['IDENTITY_WHATSAPP_PHONE_NUMBER_ID']}",
            params={"fields": "display_phone_number,verified_name,quality_rating"},
            headers={"Authorization": f"Bearer {cloud['IDENTITY_WHATSAPP_TOKEN']}"},
            timeout=10,
        )
        if probe.status_code == 200:
            meta = probe.json()
            report(OK, "Meta accepts the token and knows this number",
                   f"{meta.get('display_phone_number')} — {meta.get('verified_name')} "
                   f"(quality {meta.get('quality_rating', 'n/a')})")
            # The number Meta has must be the number the link points at, or parents are
            # sent to one chat while the webhook listens on another.
            configured_number = (os.getenv("IDENTITY_WHATSAPP_NUMBER") or "").strip()
            meta_digits = "".join(ch for ch in str(meta.get("display_phone_number", "")) if ch.isdigit())
            if configured_number and meta_digits and meta_digits != configured_number.lstrip("+"):
                report(BAD, "the registered number is not the one in .env",
                       f"Meta has {meta.get('display_phone_number')}, .env says "
                       f"{configured_number}. Parents would be sent to a chat nobody is "
                       f"listening on.")
        elif probe.status_code in (401, 403):
            report(BAD, "Meta rejected the token",
                   "A dashboard token expires in under 24 hours — generate a System User "
                   "token instead. " + probe.text[:120])
        else:
            report(BAD, f"Meta answered {probe.status_code} for that phone number id",
                   probe.text[:160])
    except Exception as error:  # noqa: BLE001
        report(WARN, "could not reach Meta to verify the credentials", str(error)[:110])

# --------------------------------------------------------------------------
section("Result")
# --------------------------------------------------------------------------
if problems:
    print(f"{len(problems)} problem(s) to fix before a parent can sign in:")
    for problem in problems:
        print(f"  - {problem}")
    raise SystemExit(1)

print("Everything a parent's sign-in needs is wired.")
print()
print("To try it by hand:")
print("  1. open http://localhost:3000 and pick the parent tab")
print("  2. tap the WhatsApp button; send the prefilled message from the parent's number")
print("     (no Meta account? the code is in identity's window —")
print("      IDENTITY_WHATSAPP_LOG_CODES=true puts it there)")
print("  3. type the six digits, and ask about the child")
