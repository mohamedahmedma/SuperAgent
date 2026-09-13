"""SQLAlchemy implementations of the ports in `backend.application.ports.repositories`."""
from backend.infra.repositories.asset_repository import (
    SqlAlchemyAssetExtractionRepository,
    SqlAlchemyDocumentAssetRepository,
)
from backend.infra.repositories.conversation_repository import SqlAlchemyConversationRepository
from backend.infra.repositories.document_pair_repository import SqlAlchemyDocumentPairRepository
from backend.infra.repositories.entity_attribute_repository import SqlAlchemyEntityAttributeRepository
from backend.infra.repositories.parent_chunk_repository import SqlAlchemyParentChunkRepository
from backend.infra.repositories.section_catalogue_repository import (
    SqlAlchemyCorpusDigestRepository,
    SqlAlchemySectionSummaryRepository,
)

__all__ = [
    "SqlAlchemyAssetExtractionRepository",
    "SqlAlchemyConversationRepository",
    "SqlAlchemyCorpusDigestRepository",
    "SqlAlchemyDocumentAssetRepository",
    "SqlAlchemyDocumentPairRepository",
    "SqlAlchemyEntityAttributeRepository",
    "SqlAlchemyParentChunkRepository",
    "SqlAlchemySectionSummaryRepository",
]
