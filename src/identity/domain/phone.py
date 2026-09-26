"""How this service writes a phone number, and the one conversion it is entitled to make.

A `wa_id` is already international — it is E.164 with the `+` removed — so turning one
into the form the rest of the estate stores is a prepend and never a guess. Parsing a
*national* spelling is `sis/`'s job: it needs a default country code, and this service
deliberately holds no opinion about which country a school is in.

That restraint is the whole content of this module. A number this service invented a
country for would resolve against the wrong family, or against nobody, and the failure
would look exactly like a parent who is not registered.
"""
from __future__ import annotations

from typing import Final

from identity.domain.errors import NotConfigured

#: Shorter than this is not a dialable international number in any country, and is in
#: practice a national spelling with its country code missing.
_MIN_DIGITS: Final[int] = 8


def wa_id_of(phone: str) -> str:
    """WhatsApp's form of a number: digits only, no `+`, no spaces.

    The inverse of what the rest of the estate stores. `sis/` keeps E.164 with the plus
    because that is the unambiguous written form; Meta's `wa_id` drops it. Converting in
    one named place keeps the difference from being rediscovered at each call site.
    """
    return "".join(character for character in str(phone) if character.isdigit())


def to_e164(wa_id: str) -> str:
    """A `wa_id` as the rest of the estate writes a number. `""` for an empty input.

    A prepend, never a guess — see the module docstring.
    """
    digits = wa_id_of(wa_id)
    return f"+{digits}" if digits else ""


def e164_or_raise(number: str, *, setting: str) -> str:
    """Accept a number only in explicit international form, and say so loudly otherwise.

    This exists because of a failure with no symptom. Meta requires the school's number in
    full international form with no leading zero, and `wa.me` does not validate: configure
    the Egyptian national spelling `01288339613` and every parent gets a link to
    `wa.me/01288339613`, which is a different number that does not exist. The link opens,
    WhatsApp shows an empty chat, no message ever reaches the webhook, and nothing anywhere
    logs an error. Refusing at startup turns a silent estate-wide outage into a deploy that
    does not come up.

    National-form input is refused rather than converted, because converting requires a
    default country this service has no business holding an opinion about — `sis/` owns
    that rule, and two places deciding it is how the two disagree.
    """
    cleaned = str(number).strip()
    digits = wa_id_of(cleaned)
    if not cleaned.startswith("+") or len(digits) < _MIN_DIGITS:
        raise NotConfigured(
            f"{setting} must be a phone number in international form beginning with '+', "
            f"for example '+201288339613'. Got {number!r}. WhatsApp reads a national "
            f"spelling such as '01288339613' as an entirely different number, and the "
            f"resulting links fail silently."
        )
    return "+" + digits


def click_to_chat_link(business_number: str, message: str) -> str:
    """The `wa.me` link that opens WhatsApp with `message` already typed.

    **It is never sent automatically.** WhatsApp deliberately requires the parent to tap
    send, and any wording that implies otherwise produces a queue of parents who clicked
    and then waited. The screen that shows this link has to say so.

    `business_number` must already be E.164 — see `e164_or_raise`, which the composition
    root applies once at startup so this cannot be reached with a national spelling.

    `quote` with `safe=""` rather than `urlencode`, because every reserved character has to
    be escaped: an unescaped `&` or `#` truncates the message at exactly the point where
    the nonce would have been, and the resulting message arrives looking almost right.

    **An empty number is refused rather than rendered.** `https://wa.me/?text=...` is a
    valid URL that WhatsApp answers by opening the contact picker — the parent is asked to
    choose who to send the school's verification code to, which in production they cannot
    possibly know. Every visible sign says the feature works: the button opens WhatsApp,
    the message is prefilled, nothing logs. Only the chat is with nobody.

    Raising here rather than checking at the call site, because this is the function whose
    output is the bug, and a guard anywhere else leaves the broken string one new caller
    away.
    """
    from urllib.parse import quote

    if not (business_number or "").strip():
        raise ValueError(
            "click_to_chat_link needs the school's number. Without it the link opens "
            "WhatsApp's contact picker instead of a chat, and no parent can complete "
            "sign-in. Set IDENTITY_WHATSAPP_NUMBER."
        )
    return f"https://wa.me/{wa_id_of(business_number)}?text={quote(message, safe='')}"


__all__ = ["click_to_chat_link", "e164_or_raise", "to_e164", "wa_id_of"]
