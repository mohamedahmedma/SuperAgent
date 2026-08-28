"""Proving that a browser and a WhatsApp number belong to the same person.

The rules of the flow, with no HTTP client, no SQLAlchemy and no FastAPI in sight. The
repositories and the two gateways arrive as ports, so every rule below is testable with
plain classes and a dict.

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
Neither alone is enough and neither is inferred from the browser: the sender's number
comes from Meta's payload, never from anything the caller said. A number the school does
not hold is refused — this flow never creates a guardian, because whose parent somebody
is, is the registrar's fact and not a claim to be self-asserted.

## What is deliberately not here

No enumeration surface. `start` takes no phone number at all, so there is nothing to
probe: a caller learns only about the number they can actually send a message from.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

from identity.application.dto import ChallengeStatus, SchoolChannel, StartedChallenge
from identity.application.ports.repositories import ChallengeRepository
from identity.domain import challenges as rules
from identity.domain.accounts import as_aware
from identity.domain.errors import (
    BadCode,
    GuardianDirectoryUnavailable,
    NotConfigured,
    TooManyAttempts,
    VerificationAlreadyUsed,
    VerificationExpired,
    VerificationNotFound,
    VerificationNotReady,
    WhatsAppUnavailable,
)
from identity.domain.phone import click_to_chat_link, to_e164

logger = logging.getLogger(__name__)

#: Outcomes `claim` reports to its caller for the log. Strings rather than an enum because
#: they are written to a log line and read by a human, never branched on.
OUTCOME_DUPLICATE = "duplicate"
OUTCOME_NO_NONCE = "no_nonce"
OUTCOME_UNKNOWN_NONCE = "unknown_nonce"
OUTCOME_ALREADY_CLAIMED = "already_claimed"
OUTCOME_WRONG_SCHOOL = "wrong_school"
OUTCOME_UNKNOWN_SCHOOL = "unknown_school"
OUTCOME_EXPIRED = "expired"
OUTCOME_RATE_LIMITED = "rate_limited"
OUTCOME_DIRECTORY_UNAVAILABLE = "directory_unavailable"
OUTCOME_NOT_A_GUARDIAN = "not_a_guardian"
OUTCOME_SEND_FAILED = "send_failed"
OUTCOME_CODE_SENT = "code_sent"


class WhatsAppLoginService:
    """The whole flow, over a challenge repository, a channel resolver and a clock."""

    def __init__(
        self,
        *,
        challenges: ChallengeRepository,
        channel_for: Callable[[str | None], SchoolChannel],
        ttl_minutes: int = 10,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        """`channel_for` resolves the school, per call, from what the request proves.

        A resolver rather than a fixed channel, so the school is chosen from the login
        page for `start` and from the WhatsApp number the message arrived on for `claim` —
        instead of being fixed when this object is built. In a single-school deployment it
        returns the one channel for `None` and raises for anything else, which is what
        makes "a named school arriving at a service configured for one" a refusal rather
        than a cross-school read.
        """
        self._challenges = challenges
        self._channel_for = channel_for
        self._ttl = timedelta(minutes=ttl_minutes)
        self._clock = clock

    def channel(self, school_code: str | None = None) -> SchoolChannel:
        """The channel for one school. Raises rather than guessing at an unknown one."""
        return self._channel_for(school_code)

    # -- the browser's first step -------------------------------------------

    def start(self, *, school_code: str | None = None) -> StartedChallenge:
        """Mint a challenge. Takes no phone number, which is what makes it unprobeable.

        Refuses outright when the school's own number is not configured. The alternative
        is a link to `wa.me/` with no number, which opens WhatsApp's contact picker and
        asks the parent to choose who to send a verification code to — a question they
        cannot answer, from a screen where everything looks like it worked. Refusing costs
        one error message; not refusing costs every parent, silently.

        The school is stamped onto the row here, where it is known for free: the login page
        the parent is standing on belongs to one school. That recorded value is what the
        webhook checks the inbound number against, so the two halves of the flow have to
        agree about which school this is before any code is issued.
        """
        channel = self._channel_for(school_code)
        if not (channel.business_number or "").strip():
            raise NotConfigured(
                "Parent sign-in is not available: the school's WhatsApp number is not "
                "configured on this server."
            )

        nonce = rules.new_nonce()
        poll_secret = rules.new_poll_secret()
        challenge = self._challenges.create(
            nonce=nonce,
            poll_secret_hash=rules.hash_secret(poll_secret),
            school_code=channel.code or "",
            expires_at=self._clock() + self._ttl,
        )

        message = rules.prefilled_message(nonce)
        return StartedChallenge(
            nonce=nonce,
            poll_secret=poll_secret,
            link=click_to_chat_link(channel.business_number, message),
            message=message,
            business_number=channel.business_number,
            expires_at=challenge.expires_at,
        )

    # -- what the webhook calls ---------------------------------------------

    def claim(
        self,
        *,
        wa_id: str,
        body: str,
        message_id: str,
        school_code: str | None = None,
    ) -> str:
        """Handle one inbound WhatsApp message. Returns a short outcome for the log.

        **Never raises for a bad message.** Meta retries any webhook it does not see
        acknowledged, for up to seven days, so an exception here becomes a delivery that is
        replayed indefinitely. Every outcome — no nonce, unknown nonce, expired, not a
        parent, wrong school — is a recorded string and a 200.

        `school_code` is the school that owns the WhatsApp number this message arrived on,
        which the caller reads from the delivery's `phone_number_id`. It decides which
        school's directory the sender's number is looked up in, and it is checked against
        the school recorded when the challenge was started.
        """
        now = self._clock()

        if message_id and self._challenges.message_already_handled(message_id):
            # Meta's retries are guaranteed, not hypothetical. Without this one parent tap
            # sends several conflicting codes and burns several challenges.
            return OUTCOME_DUPLICATE

        nonce = rules.extract_nonce(body)
        if nonce is None:
            # Somebody messaged the school's number without a code in it — a parent saying
            # hello, or a wrong number. Not our conversation; say nothing at all rather
            # than reply to strangers.
            return OUTCOME_NO_NONCE

        challenge = self._challenges.by_nonce(nonce)
        if challenge is None:
            return OUTCOME_UNKNOWN_NONCE
        if challenge.status != rules.STATUS_PENDING:
            return OUTCOME_ALREADY_CLAIMED

        # Two independent facts have to agree about which school this is: the one the
        # browser was on when the challenge was minted, and the one that owns the number
        # the parent actually messaged. A mismatch means the nonce reached the wrong
        # school — the multi-school shape of "a parent tricked into sending an attacker's
        # nonce" — and resolving it would look this parent up in a database their children
        # are not in. Rejected rather than ignored, so the nonce is spent either way.
        if (challenge.school_code or "") != (school_code or ""):
            self._challenges.mark_rejected(
                challenge, reason=OUTCOME_WRONG_SCHOOL, message_id=message_id
            )
            return OUTCOME_WRONG_SCHOOL

        try:
            channel = self._channel_for(school_code)
        except Exception:  # noqa: BLE001 - an unknown school must not retry for a week
            # Only reachable when a school was removed from the configuration between the
            # challenge being started and the parent replying.
            self._challenges.mark_rejected(
                challenge, reason=OUTCOME_UNKNOWN_SCHOOL, message_id=message_id
            )
            return OUTCOME_UNKNOWN_SCHOOL

        if as_aware(challenge.expires_at) <= now:
            self._challenges.mark_rejected(
                challenge, reason=OUTCOME_EXPIRED, message_id=message_id
            )
            self._say(
                channel, wa_id, "That link has expired. Please start again from the school app."
            )
            return OUTCOME_EXPIRED

        phone = to_e164(wa_id)
        since = now - timedelta(minutes=rules.RATE_WINDOW_MINUTES)
        if self._challenges.count_recent_for_phone(phone, since=since) >= rules.MAX_PER_PHONE:
            self._challenges.mark_rejected(
                challenge, reason=OUTCOME_RATE_LIMITED, message_id=message_id
            )
            return OUTCOME_RATE_LIMITED

        try:
            guardian = channel.directory.resolve(phone, school_code=channel.code)
        except GuardianDirectoryUnavailable as error:
            # Left pending on purpose: the school's records were unreachable, which is our
            # problem and not the parent's, and the challenge is still good if they try
            # again inside its window.
            logger.warning("Guardian lookup failed during verification: %s", error)
            self._say(
                channel,
                wa_id,
                "We could not reach the school's records just now. Please try again in a "
                "few minutes.",
            )
            return OUTCOME_DIRECTORY_UNAVAILABLE

        if guardian is None:
            self._challenges.mark_rejected(
                challenge, reason=OUTCOME_NOT_A_GUARDIAN, message_id=message_id
            )
            # Deliberately vague and deliberately final. It tells the person in front of us
            # nothing about which numbers *are* registered, and it points them at the one
            # channel that can actually fix it.
            self._say(
                channel,
                wa_id,
                "This number is not registered with the school. Please contact the school "
                "office to be added.",
            )
            return OUTCOME_NOT_A_GUARDIAN

        code = rules.new_code()
        self._challenges.mark_code_sent(
            challenge,
            guardian_phone=phone,
            guardian_external_id=guardian.public_id,
            display_name=guardian.display_name,
            preferred_language=guardian.preferred_language,
            code_hash=rules.hash_secret(code),
            message_id=message_id,
        )

        try:
            # Sent through *this school's* credentials. Knowing which number the parent
            # used is not enough: a code sent out through another school's number arrives
            # in a different conversation, and the parent never sees it.
            channel.gateway.send_text(wa_id, f"Your school verification code is {code}")
        except WhatsAppUnavailable as error:
            # The code was stored but never delivered, so the challenge is unusable. Marked
            # rejected rather than left pending: a parent staring at a code entry box that
            # can never be satisfied is worse than one told plainly to start again.
            logger.warning("Could not deliver a verification code: %s", error)
            self._challenges.mark_rejected(
                challenge, reason=OUTCOME_SEND_FAILED, message_id=message_id
            )
            return OUTCOME_SEND_FAILED

        return OUTCOME_CODE_SENT

    # -- what the browser polls and then submits ----------------------------

    def status(self, *, poll_secret: str) -> ChallengeStatus:
        """Where has this verification got to? Polled while the parent goes to tap send."""
        challenge = self._by_poll_secret(poll_secret)
        return ChallengeStatus(
            status=challenge.status,
            display_name=challenge.display_name,
            expires_at=challenge.expires_at,
        )

    def verify(self, *, poll_secret: str, code: str):
        """Check the code and consume the challenge. Returns it with its guardian binding.

        Consuming here rather than in the caller means a challenge cannot mint two tokens
        even if the route is called twice concurrently — the second call finds it consumed.
        """
        now = self._clock()
        challenge = self._by_poll_secret(poll_secret)

        if challenge.consumed_at is not None:
            raise VerificationAlreadyUsed()
        if challenge.status != rules.STATUS_CODE_SENT:
            raise VerificationNotReady()
        if as_aware(challenge.expires_at) <= now:
            self._challenges.mark_rejected(challenge, reason=OUTCOME_EXPIRED)
            raise VerificationExpired()
        if challenge.attempts >= rules.MAX_ATTEMPTS:
            self._challenges.mark_rejected(challenge, reason="too_many_attempts")
            raise TooManyAttempts()

        # Counted before the comparison, so a crash between the two cannot hand an attacker
        # a free guess.
        self._challenges.count_attempt(challenge)

        if not rules.code_matches(challenge.code_hash, code):
            raise BadCode()

        self._challenges.mark_verified(challenge, at=now)
        return challenge

    # -- internals ----------------------------------------------------------

    def _by_poll_secret(self, poll_secret: str):
        challenge = self._challenges.by_poll_secret_hash(rules.hash_secret(poll_secret))
        if challenge is None:
            raise VerificationNotFound()
        return challenge

    def _say(self, channel: SchoolChannel, wa_id: str, text: str) -> None:
        """Reply to the parent, and never let a failed reply fail the webhook.

        A raised exception here would leave Meta retrying the delivery for days over a
        message we merely could not answer.

        The channel is passed rather than read off the service because the reply has to go
        back out through the number the parent messaged. Sent through any other school's
        credentials it would land in a different conversation, and the parent would be left
        watching a chat where nothing ever arrives.
        """
        try:
            channel.gateway.send_text(wa_id, text)
        except WhatsAppUnavailable as error:
            logger.warning("Could not reply over WhatsApp: %s", error)


__all__ = ["WhatsAppLoginService"]
