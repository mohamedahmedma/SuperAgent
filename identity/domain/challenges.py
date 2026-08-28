"""The rules of the WhatsApp verification flow, with nothing that does I/O.

Nonce and code generation, the alphabet they are drawn from, the message a parent sends,
the scan that finds a nonce inside whatever they actually typed, and the state machine a
challenge moves through. Every one of these is testable by calling it with a string.

The flow itself — which database row, which gateway, which school — is
`application/services/whatsapp_login.py`. What lives here is the part that would be
identical if the transport were SMS.

## Two secrets, and why

`nonce` goes out in a URL and comes back over WhatsApp. `poll_secret` never leaves the
browser that asked. Holding one without the other is worth nothing:

  * A nonce lifted from a screenshot and sent from the attacker's own phone delivers the
    code to *the attacker's* WhatsApp — but they hold no poll secret, so they cannot
    finish, and the parent whose nonce it was is merely blocked.
  * A parent tricked into sending an attacker's nonce delivers the code to *the parent's*
    WhatsApp, which the attacker cannot read.
"""
from __future__ import annotations

import hashlib
import secrets
from typing import Final

#: Crockford-ish: no I, O, U, or digits that look like letters at a glance. The parent may
#: have to read this off a screen and type it into WhatsApp by hand when an in-app browser
#: swallows the link, and `0` against `O` is the pair that costs a support call.
NONCE_ALPHABET: Final[str] = "ABCDEFGHJKLMNPQRSTVWXYZ23456789"
NONCE_LENGTH: Final[int] = 8

#: Six digits, because it is typed on a phone keypad by somebody holding a second phone.
#: Short enough to be usable, and useless without `MAX_ATTEMPTS` and the TTL together.
CODE_DIGITS: Final[int] = 6

#: Guesses allowed against one challenge before it dies. Six digits is a million
#: possibilities; five guesses makes brute force pointless, and a parent who fat-fingers
#: the code twice still gets in.
MAX_ATTEMPTS: Final[int] = 5

#: How many challenges one number may claim in the window below. A parent who taps the
#: link repeatedly is normal; a script walking nonces is not, and WhatsApp itself throttles
#: replies to one user to roughly one every six seconds.
MAX_PER_PHONE: Final[int] = 5
RATE_WINDOW_MINUTES: Final[int] = 15

# -- the state machine ------------------------------------------------------
#
# Strings rather than an enum column, for the reason the audit `reason` is one: they are
# read far more often than branched on, and a new state must not need a migration in a
# service whose schema is built by `create_all`.

STATUS_PENDING: Final[str] = "pending"
STATUS_CODE_SENT: Final[str] = "code_sent"
STATUS_VERIFIED: Final[str] = "verified"
STATUS_REJECTED: Final[str] = "rejected"


def hash_secret(raw: str) -> str:
    """The stored form of every bearer value in this flow.

    SHA-256 rather than PBKDF2, which is right here and wrong for a password: these are
    high-entropy machine-generated values with no dictionary behind them, and stretching
    would add latency to a request a parent is waiting on without adding strength.
    """
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def new_nonce() -> str:
    """`secrets`, never `random` — this value is what an attacker would guess."""
    return "".join(secrets.choice(NONCE_ALPHABET) for _ in range(NONCE_LENGTH))


def new_code() -> str:
    """A zero-padded six-digit code. `secrets`, never `random`."""
    return f"{secrets.randbelow(10 ** CODE_DIGITS):0{CODE_DIGITS}d}"


def new_poll_secret() -> str:
    """The browser's half of the proof. Shown once, stored only as a hash."""
    return secrets.token_urlsafe(32)


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
    have dropped the link entirely so the parent typed the code off a screen. The nonce
    alphabet excludes lookalike characters precisely so this scan can be strict about the
    token while forgiving about its surroundings.

    A run *longer* than the nonce is a word that happens to be spelled from the alphabet,
    not a nonce — only an exact-length run counts.
    """
    if not body:
        return None
    candidates: list[str] = []
    current: list[str] = []
    for character in body.upper():
        if character in NONCE_ALPHABET:
            current.append(character)
            continue
        if len(current) >= NONCE_LENGTH:
            candidates.append("".join(current))
        current = []
    if len(current) >= NONCE_LENGTH:
        candidates.append("".join(current))
    for candidate in candidates:
        if len(candidate) == NONCE_LENGTH:
            return candidate
    return None


def code_matches(code_hash: str, submitted: str) -> bool:
    """Constant-time comparison of a submitted code against the stored hash."""
    import hmac

    return hmac.compare_digest(code_hash, hash_secret(str(submitted).strip()))


__all__ = [
    "CODE_DIGITS",
    "MAX_ATTEMPTS",
    "MAX_PER_PHONE",
    "NONCE_ALPHABET",
    "NONCE_LENGTH",
    "RATE_WINDOW_MINUTES",
    "STATUS_CODE_SENT",
    "STATUS_PENDING",
    "STATUS_REJECTED",
    "STATUS_VERIFIED",
    "code_matches",
    "extract_nonce",
    "hash_secret",
    "new_code",
    "new_nonce",
    "new_poll_secret",
    "prefilled_message",
]
