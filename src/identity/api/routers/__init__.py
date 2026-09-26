"""The HTTP surface, one module per credential and per feature.

Four routers, separated by **what authenticates the caller** rather than only by prefix:

    wellknown   nothing — a public key is not a secret
    auth        a password, or a refresh token
    whatsapp    a nonce over WhatsApp, and a poll secret in the browser
    admin       a shared admin key, which a browser never holds

That separation is structural. A parent's token cannot reach the admin routes at all,
because those routes do not accept a bearer token as a credential in the first place.
"""
from identity.api.routers import admin, auth, health, wellknown, whatsapp

__all__ = ["admin", "auth", "health", "wellknown", "whatsapp"]
