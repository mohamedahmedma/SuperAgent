"""Ports: the persistence interfaces the backend's services depend on."""
from backend.application.ports.repositories import (
    ConversationRepository,
    CorpusDigestRepository,
    DigestRecord,
    DocumentPairRecord,
    DocumentPairRepository,
    MessageHead,
    NewMessage,
    ParentChunkRecord,
    ParentChunkRepository,
    SectionSummaryRepository,
    SessionSummary,
    StoredMessage,
    StoredSession,
)
from backend.application.ports.unit_of_work import UnitOfWork, UnitOfWorkFactory

__all__ = [
    "ConversationRepository",
    "CorpusDigestRepository",
    "DigestRecord",
    "DocumentPairRecord",
    "DocumentPairRepository",
    "MessageHead",
    "NewMessage",
    "ParentChunkRecord",
    "ParentChunkRepository",
    "SectionSummaryRepository",
    "SessionSummary",
    "StoredMessage",
    "StoredSession",
    "UnitOfWork",
    "UnitOfWorkFactory",
]
