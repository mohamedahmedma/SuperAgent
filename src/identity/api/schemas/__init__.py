"""The wire format, one module per router.

Split out of a single `schemas.py` so that a change to the WhatsApp flow touches the
WhatsApp schema and nothing else, and so the admin shapes — the ones that can bind a
guardian — sit apart from the public ones.
"""
from identity.api.schemas.admin import AccountIn, AccountOut, GuardianBindingIn
from identity.api.schemas.auth import (
    AccessTokenOut,
    LoginIn,
    MeOut,
    RefreshIn,
    TokenOut,
)
from identity.api.schemas.common import ErrorOut
from identity.api.schemas.whatsapp import (
    WhatsAppStartOut,
    WhatsAppStatusIn,
    WhatsAppStatusOut,
    WhatsAppVerifyIn,
)

__all__ = [
    "AccessTokenOut",
    "AccountIn",
    "AccountOut",
    "ErrorOut",
    "GuardianBindingIn",
    "LoginIn",
    "MeOut",
    "RefreshIn",
    "TokenOut",
    "WhatsAppStartOut",
    "WhatsAppStatusIn",
    "WhatsAppStatusOut",
    "WhatsAppVerifyIn",
]
