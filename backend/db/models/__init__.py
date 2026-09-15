"""The chat backend's ORM models, one module per aggregate.

Import from the package — `from backend.db.models import ChatSession` — not from the
module a model lives in. The package is the stable address: models can move between
modules without their callers changing, and importing it registers every table on
`Base.metadata`, which Alembic's autogenerate and relationship lookups by name rely on.

The schema is Alembic's (backend/alembic.ini). A change to a model here is half a
change: the revision that makes the database match it is the other half, and the
backend refuses to start against a database that has not applied it.
"""
from backend.db.models.assets import AssetExtraction, DocumentAsset, EntityAttribute
from backend.db.models.attachments import ChatAttachment
from backend.db.models.conversations import ChatMessage, ChatSession
from backend.db.models.indexing import CorpusDigest, DocumentPair, ParentChunk, SectionSummary
from backend.db.models.jobs import IngestJob
from backend.db.models.users import User

__all__ = [
    "AssetExtraction",
    "ChatAttachment",
    "ChatMessage",
    "ChatSession",
    "CorpusDigest",
    "DocumentAsset",
    "DocumentPair",
    "EntityAttribute",
    "IngestJob",
    "ParentChunk",
    "SectionSummary",
    "User",
]
