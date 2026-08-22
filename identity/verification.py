"""Proving that a browser and a WhatsApp number belong to the same person.

The rules of the flow, with no HTTP client and no FastAPI in sight. Routes hand this
module a session and a clock; it talks to the two seams (`WhatsAppGateway`,
`GuardianDirectory`) through their Protocols, so every rule below is testable with a dict
and a list.

## The flow

    browser   POST /v1/auth/whatsapp/start
              -> a challenge, a wa.me link carrying `nonce`, and a `poll_secret`
    parent    taps the link, WhatsApp opens with the message pre-filled, parent taps send
    Meta      POST /v1/auth/whatsapp/webhook  { from: wa_id, text: "... NONCE ..." }
    here      wa_id -> sis -> a guardian handle, or a polite refusal over WhatsApp
    here      reply over WhatsApp with a six-digit code
    browser   POST /v1/auth/whatsapp/verify   { poll_secret, code }  -> tokens

## Two secrets, and why

`nonce` goes out in a URL and comes back over WhatsApp. `poll_secret` never leaves the
browser. Holding one without the other is worth nothing:

  * A nonce lifted from a screenshot and sent from the attacker's own phone delivers the
    code to *the attacker's* WhatsApp — but they have no poll secret, so they cannot
    finish, and the parent whose nonce it was is merely blocked.
  * A parent tricked into sending an attacker's nonce delivers the code to *the parent's*
    WhatsApp, which the attacker cannot read.

## What proves what

WhatsApp proves control of a number. `sis/` decides whether that number is a parent's.
Neither alone is enough and neither is inferred from the browser: the sender's number comes
from Meta's payload, never from anything the caller said. A number the school does not hold
is refused — this flow never creates a guardian, because whose parent somebody is, is the
registrar's fact and not a claim to be self-asserted.

## What is deliberately not here

No enumeration surface. `start` takes no phone number at all, so there is nothing to probe:
a caller learns only about the number they can actually send a WhatsApp message from.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Final

from sqlalchemy.orm import Session

from identity import whatsapp as wa
from identity.guardians import GuardianDirectory, GuardianDirectoryUnavailable
from identity.models import VerificationChallenge

logger = logging.getLogger(__name__)

#: Crockford-ish: no I, O, U, or digits that look like letters at a glance. The parent may
#: have to read this off a screen and type it into WhatsApp by hand when an in-app browser
#: swallows the link, and `0` against `O` is the pair that costs a support call.
_NONCE_ALPHABET: Final[str] = "ABCDEFGHJKLMNPQRSTVWXYZ23456789"
_NONCE_LENGTH: Final[int] = 8

#: Six digits, because it is typed on a phone keypad by somebody holding a second phone.
#: Short enough to be usable and useless without `_MAX_ATTEMPTS` and the TTL below.
_CODE_DIGITS: Final[int] = 6

#: Guesses allowed against one challenge before it dies. Six digits is a million
#: possibilities; five guesses makes brute force pointless, and a parent who fat-fingers
#: the code twice still gets in.
_MAX_ATTEMPTS: Final[int] = 5

#: Long enough for a parent to find the message, short enough that a screenshotted link
#: shared later is worthless.
DEFAULT_TTL_MINUTES: Final[int] = 10

#: How many challenges one number may claim in the window below. A parent who taps the
#: link repeatedly is normal; a script walking nonces is not, and WhatsApp itself throttles
#: replies to one user to roughly one every six seconds.
_MAX_PER_PHONE: Final[int] = 5
_RATE_WINDOW_MINUTES: Final[int] = 15

STATUS_PENDING: Final[str] = "pending"
STATUS_CODE_SENT: Final[str] = "code_sent"
STATUS_VERIFIED: Final[str] = "verified"
STATUS_REJECTED: Final[str] = "rejected"


class VerificationError(RuntimeError):
    """A challenge cannot proceed. Carries a machine-readable `code` like every refusal here."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class StartedChallenge:
    """What the browser is handed. `poll_secret` is shown exactly once and never stored."""

    nonce: str
    poll_secret: str
    link: str
    message: str
    expires_at: datetime


def _hash(raw: str) -> str:
    """The stored form of every bearer value in this module.

    SHA-256 rather than PBKDF2, which is right here and wrong for a password: these are
    high-entropy machine-generated values with no dictionary behind them, and stretching
    would add latency to a request a parent is waiting on without adding strength.
    """
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _aware(moment: datetime | None) -> datetime | None:
    """Re-attach UTC to a datetime SQLite handed back naive.

    SQLite returns naive datetimes even from a timezone-aware column, so every comparison
    against `now` raises `TypeError` — and only under SQLite, which means only in
    development and tests. `auth.is_locked` does the same dance for the same reason.
    """
    if moment is None:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _new_nonce() -> str:
    return "".join(secrets.choice(_NONCE_ALPHABET) for _ in range(_NONCE_LENGTH))


def _new_code() -> str:
    """A zero-padded six-digit code. `secrets`, never `random`."""
    return f"{secrets.randbelow(10 ** _CODE_DIGITS):0{_CODE_DIGITS}d}"


def prefilled_message(nonce: str) -> str:
    """The text the parent sends us.

    The nonce is fenced by a word on each side so it survives the two things parents
    actually do: adding "hello" in front, and letting a keyboard capitalise or autocorrect
    around it. `extract_nonce` looks for the pattern rather than the whole string, so a
    message that has been edited still verifies.
    """
    return f"SCHOOL VERIFY: {nonce}"


def extract_nonce(body: str) -> str | None:
    """Find a nonce anywhere in what the parent actually sent.

    Deliberately lenient about everything except the nonce itself. Parents type ahead of
    the pre-filled text, WhatsApp capitalises after a full stop, and an in-app browser may
    have dropped the link entirely so the parent typed the code from a screen. The nonce
    alphabet excludes lookalike characters precisely so that this scan can be strict about
    the token while forgiving about its surroundings.
    """
    if not body:
        return None
    upper = body.upper()
    candidates = []
    current = []
    for character in upper:
        if character in _NONCE_ALPHABET:
            current.append(character)
        else:
            if len(current) >= _NONCE_LENGTH:
                candidates.append("".join(current))
            current = []
    if len(current) >= _NONCE_LENGTH:
        candidates.append("".join(current))
    for candidate in candidates:
        # A run longer than the nonce is a word that happens to be spelled from the
        # alphabet; only an exact-length run is a nonce.
        if len(candidate) == _NONCE_LENGTH:
            return candidate
    return None


class VerificationService:
    """The whole flow. Depends on two Protocols, a session factory's session, and a clock."""

    def __init__(
        self,
        *,
        gateway: wa.WhatsAppGateway,
        directory: GuardianDirectory,
        business_number: str,
        ttl_minutes: int = DEFAULT_TTL_MINUTES,
        clock=lambda: datetime.now(timezone.utc),
    ) -> None:
        self._gateway = gateway
        self._directory = directory
        self._business_number = business_number
        self._ttl = timedelta(minutes=ttl_minutes)
        self._clock = clock

    @property
    def business_number(self) -> str:
        """The school's own number, for the page's manual fallback.

        Shown beside the link because an in-app browser will sometimes swallow the
        handoff, and a parent who can see the number and the message can send it by hand.
        """
        return self._business_number

    # -- the browser's first step -------------------------------------------

    def start(self, db: Session) -> StartedChallenge:
        """Mint a challenge. Takes no phone number, which is what makes it unprobeable."""
        now = self._clock()
        nonce = _new_nonce()
        poll_secret = secrets.token_urlsafe(32)

        challenge = VerificationChallenge(
            nonce=nonce,
            poll_secret_hash=_hash(poll_secret),
            status=STATUS_PENDING,
            expires_at=now + self._ttl,
        )
        db.add(challenge)
        db.commit()

        message = prefilled_message(nonce)
        return StartedChallenge(
            nonce=nonce,
            poll_secret=poll_secret,
            link=wa.click_to_chat_link(self._business_number, message),
            message=message,
            expires_at=challenge.expires_at,
        )

    # -- what the webhook calls ---------------------------------------------

    def claim(self, db: Session, *, wa_id: str, body: str, message_id: str) -> str:
        """Handle one inbound WhatsApp message. Returns a short outcome for the log.

        **Never raises for a bad message.** Meta retries any webhook it does not see
        acknowledged, for up to seven days, so an exception here becomes a delivery that is
        replayed indefinitely. Every outcome — no nonce, unknown nonce, expired, not a
        parent — is a recorded string and a 200.
        """
        now = self._clock()

        if message_id:
            # Meta's retries are guaranteed, not hypothetical. Without this one parent tap
            # sends several conflicting codes and burns several challenges.
            already = (
                db.query(VerificationChallenge)
                .filter(VerificationChallenge.wa_message_id == message_id)
                .first()
            )
            if already is not None:
                return "duplicate"

        nonce = extract_nonce(body)
        if nonce is None:
            # Somebody messaged the school's number without a code in it — a parent saying
            # hello, or a wrong number. Not our conversation; say nothing at all rather
            # than reply to strangers.
            return "no_nonce"

        challenge = (
            db.query(VerificationChallenge)
            .filter(VerificationChallenge.nonce == nonce)
            .first()
        )
        if challenge is None:
            return "unknown_nonce"
        if challenge.status != STATUS_PENDING:
            return "already_claimed"
        if _aware(challenge.expires_at) <= now:
            self._reject(db, challenge, "expired", message_id)
            self._say(wa_id, "That link has expired. Please start again from the school app.")
            return "expired"

        if self._too_many_recently(db, wa_id=wa_id, now=now):
            self._reject(db, challenge, "rate_limited", message_id)
            return "rate_limited"

        phone = _e164(wa_id)
        try:
            guardian = self._directory.resolve(phone)
        except GuardianDirectoryUnavailable as error:
            # Left pending on purpose: the school's records were unreachable, which is our
            # problem and not the parent's, and the challenge is still good if they try
            # again inside its window.
            logger.warning("Guardian lookup failed during verification: %s", error)
            self._say(
                wa_id,
                "We could not reach the school's records just now. Please try again in a "
                "few minutes.",
            )
            return "directory_unavailable"

        if guardian is None:
            self._reject(db, challenge, "not_a_guardian", message_id)
            # Deliberately vague and deliberately final. It tells the person in front of us
            # nothing about which numbers *are* registered, and it points them at the one
            # channel that can actually fix it.
            self._say(
                wa_id,
                "This number is not registered with the school. Please contact the school "
                "office to be added.",
            )
            return "not_a_guardian"

        code = _new_code()
        challenge.guardian_phone = phone
        challenge.guardian_external_id = guardian.public_id
        challenge.display_name = guardian.display_name
        challenge.preferred_language = guardian.preferred_language
        challenge.code_hash = _hash(code)
        challenge.wa_message_id = message_id
        challenge.status = STATUS_CODE_SENT
        db.commit()

        try:
            self._gateway.send_text(wa_id, f"Your school verification code is {code}")
        except wa.WhatsAppUnavailable as error:
            # The code was stored but never delivered, so the challenge is unusable. Marked
            # rejected rather than left pending: a parent staring at a code entry box that
            # can never be satisfied is worse than one told plainly to start again.
            logger.warning("Could not deliver a verification code: %s", error)
            self._reject(db, challenge, "send_failed", message_id)
            return "send_failed"

        return "code_sent"

    # -- what the browser polls and then submits ----------------------------

    def status(self, db: Session, *, poll_secret: str) -> VerificationChallenge:
        """The challenge behind this browser's secret. Raises when there is none."""
        challenge = (
            db.query(VerificationChallenge)
            .filter(VerificationChallenge.poll_secret_hash == _hash(poll_secret))
            .first()
        )
        if challenge is None:
            raise VerificationError("not_found", "No such verification is in progress.")
        return challenge

    def verify(self, db: Session, *, poll_secret: str, code: str) -> VerificationChallenge:
        """Check the code and consume the challenge. Returns it with its guardian binding.

        Consuming here rather than in the caller means a challenge cannot mint two tokens
        even if the route is called twice concurrently — the second call finds it consumed.
        """
        now = self._clock()
        challenge = self.status(db, poll_secret=poll_secret)

        if challenge.consumed_at is not None:
            raise VerificationError("already_used", "That verification has already been used.")
        if challenge.status != STATUS_CODE_SENT:
            raise VerificationError(
                "not_ready", "No code has been sent for this verification yet."
            )
        if _aware(challenge.expires_at) <= now:
            self._reject(db, challenge, "expired", challenge.wa_message_id)
            raise VerificationError("expired", "That verification has expired.")
        if challenge.attempts >= _MAX_ATTEMPTS:
            self._reject(db, challenge, "too_many_attempts", challenge.wa_message_id)
            raise VerificationError("too_many_attempts", "Too many incorrect codes.")

        # Counted before the comparison, so a crash between the two cannot hand an attacker
        # a free guess.
        challenge.attempts += 1
        db.commit()

        if not hmac.compare_digest(challenge.code_hash, _hash(str(code).strip())):
            raise VerificationError("bad_code", "That code is not correct.")

        challenge.status = STATUS_VERIFIED
        challenge.consumed_at = now
        db.commit()
        return challenge

    # -- internals ----------------------------------------------------------

    def _reject(
        self, db: Session, challenge: VerificationChallenge, reason: str, message_id: str
    ) -> None:
        challenge.status = STATUS_REJECTED
        challenge.reason = reason
        if message_id:
            challenge.wa_message_id = message_id
        db.commit()

    def _say(self, wa_id: str, text: str) -> None:
        """Reply to the parent, and never let a failed reply fail the webhook.

        A raised exception here would leave Meta retrying the delivery for days over a
        message we merely could not answer.
        """
        try:
            self._gateway.send_text(wa_id, text)
        except wa.WhatsAppUnavailable as error:
            logger.warning("Could not reply over WhatsApp: %s", error)

    def _too_many_recently(self, db: Session, *, wa_id: str, now: datetime) -> bool:
        """Has this number claimed too many challenges lately?

        Keyed on the number rather than on an account, because at this point in the flow
        there may be no account — which is exactly why `auth.register_failure` cannot serve
        here.
        """
        since = now - timedelta(minutes=_RATE_WINDOW_MINUTES)
        recent = (
            db.query(VerificationChallenge)
            .filter(
                VerificationChallenge.guardian_phone == _e164(wa_id),
                VerificationChallenge.created_at >= since,
            )
            .count()
        )
        return recent >= _MAX_PER_PHONE


def _e164(wa_id: str) -> str:
    """WhatsApp's `wa_id` as the rest of the estate writes a number.

    A `wa_id` is already international — it is E.164 with the plus removed — so this is a
    prepend and never a guess. Parsing a national spelling is `sis/`'s job and needs a
    default country this service deliberately does not hold an opinion about.
    """
    digits = wa.wa_id_of(wa_id)
    return f"+{digits}" if digits else ""
