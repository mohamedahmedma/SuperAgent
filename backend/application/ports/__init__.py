"""Ports: the persistence interfaces the backend's services depend on."""
from backend.application.ports.repositories import (
    ConversationRepository,
    DocumentPairRecord,
    DocumentPairRepository,
    MessageHead,
    NewMessage,
    SessionSummary,
    StoredMessage,
    StoredSession,
)
from backend.application.ports.unit_of_work import UnitOfWork, UnitOfWorkFactory

__all__ = [
    "ConversationRepository",
    "DocumentPairRecord",
    "DocumentPairRepository",
    "MessageHead",
    "NewMessage",
    "SessionSummary",
    "StoredMessage",
    "StoredSession",
    "UnitOfWork",
    "UnitOfWorkFactory",
]
