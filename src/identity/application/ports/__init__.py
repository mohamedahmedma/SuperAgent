"""What the use cases need from the outside world, stated as Protocols.

Every one of these is declared by the layer that *uses* it and implemented by the layer
underneath. That inversion is what keeps `application/services/` free of SQLAlchemy,
`httpx` and FastAPI, and it is what makes each service constructible in a test out of
plain classes.
"""
from identity.application.ports.directory import GuardianDirectory
from identity.application.ports.messaging import WhatsAppGateway
from identity.application.ports.repositories import (
    Account,
    AccountRepository,
    AuditSink,
    ChallengeRepository,
    RefreshTokenRepository,
)
from identity.application.ports.security import PasswordHasher, TokenIssuer

__all__ = [
    "Account",
    "AccountRepository",
    "AuditSink",
    "ChallengeRepository",
    "GuardianDirectory",
    "PasswordHasher",
    "RefreshTokenRepository",
    "TokenIssuer",
    "WhatsAppGateway",
]
