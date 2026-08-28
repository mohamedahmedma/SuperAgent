"""Sending one message to one parent.

Deliberately the whole interface. This service sends exactly one kind of message — a short
verification code, inside a window the parent opened by messaging first — so a seam
offering templates, media or read receipts would be surface nobody implements correctly
and nobody tests.

Narrow enough, too, that the transport is replaceable: nothing in `application/` mentions
WhatsApp except by the name of this port, and an SMS gateway satisfying it would need no
change to any use case.
"""
from __future__ import annotations

from typing import Protocol


class WhatsAppGateway(Protocol):
    """One text message, out."""

    def send_text(self, to_wa_id: str, body: str) -> None:
        """Send `body` to `to_wa_id`. Raises `WhatsAppUnavailable` on any failure.

        One exception for transport failure, timeout, a refused token, a closed customer
        service window and an unexpected status alike. A caller that could tell "Meta is
        down" from "our token is wrong" is a caller that can probe this service's
        configuration from the outside, and neither answer changes what it does next: the
        parent is told the code could not be sent, and nothing is bound.

        `to_wa_id` is WhatsApp's own form of the number — E.164 digits with **no** leading
        `+`, exactly as it arrives in a webhook payload. Passing a `+`-prefixed number is
        the single most likely integration mistake, so implementations normalise rather
        than reject: a code that fails to reach a parent because of a punctuation
        difference is a support call nobody can diagnose from the outside.
        """


__all__ = ["WhatsAppGateway"]
