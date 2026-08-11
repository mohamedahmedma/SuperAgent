"""Entity retrieval: context first, then attributes.

    query text  --semantic/BM25 recall-->  candidate entities   (~candidate_pool)
                --scalar attribute filter-->  results            (no model call)

The ordering matters and is deliberate. Semantic recall does the *finding*, because a
shopper's phrasing rarely maps cleanly onto a facet — "something smart for a wedding"
has no attribute. Filters then do the *narrowing*, deterministically, over a set small
enough that the scan is free.

Doing it the other way round (filter the whole catalogue, then rank) throws away the
query's meaning and returns everything red whether or not it is a shoe.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.assets.attributes import AttributeSchema

logger = logging.getLogger(__name__)


@dataclass
class EntityHit:
    asset_id: str
    score: float = 0.0
    caption: str = ""
    summary: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    filename: str = ""
    page_number: int = 0

    def as_dict(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "score": self.score,
            "caption": self.caption,
            "summary": self.summary,
            "attributes": dict(self.attributes),
            "filename": self.filename,
            "page_number": self.page_number,
        }


@dataclass
class EntitySearchResult:
    hits: List[EntityHit] = field(default_factory=list)
    recalled: int = 0
    after_filter: int = 0
    filters_applied: Dict[str, Any] = field(default_factory=dict)
    rejected_filters: List[str] = field(default_factory=list)
    # True when filters removed every candidate — a genuinely different situation from
    # "nothing matched the query", and one the agent should relay differently.
    filtered_to_empty: bool = False

    def as_dict(self) -> dict:
        return {
            "hits": [hit.as_dict() for hit in self.hits],
            "recalled": self.recalled,
            "after_filter": self.after_filter,
            "filters_applied": {key: str(value) for key, value in self.filters_applied.items()},
            "rejected_filters": list(self.rejected_filters),
            "filtered_to_empty": self.filtered_to_empty,
        }


class EntityRetriever:
    def __init__(self, profile=None, asset_store=None, entity_index=None, recall=None):
        self._profile = profile
        self._asset_store = asset_store
        self._entity_index = entity_index
        self._recall = recall
        self._schema: Optional[AttributeSchema] = None

    @property
    def profile(self):
        if self._profile is None:
            from backend.profiles import get_profile

            self._profile = get_profile()
        return self._profile

    @property
    def schema(self) -> AttributeSchema:
        if self._schema is None:
            from backend.assets.attributes import build_attribute_schema

            self._schema = build_attribute_schema(self.profile.assets.entities)
        return self._schema

    @property
    def asset_store(self):
        if self._asset_store is None:
            from backend.assets.store import get_asset_store

            self._asset_store = get_asset_store()
        return self._asset_store

    @property
    def entity_index(self):
        if self._entity_index is None:
            from backend.assets.entity_store import get_entity_index

            self._entity_index = get_entity_index()
        return self._entity_index

    def _recall_candidates(self, query: str, top_k: int) -> List[dict]:
        """Semantic + BM25 recall over the normal chunk index.

        Entity surrogates are ordinary chunks, so this reuses the existing hybrid
        retriever rather than a parallel path — the attribute text the extractor wrote
        into the surrogate is exactly what makes "red shoes" hit before any filter.
        """
        if self._recall is not None:
            return self._recall(query, top_k)
        from backend.rag.utils import retrieve_documents

        return retrieve_documents(query, top_k=top_k).get("docs", [])

    def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> EntitySearchResult:
        entities_config = self.profile.assets.entities
        limit = limit or entities_config.max_results

        usable, rejected = self.schema.validate_filters(filters)
        result = EntitySearchResult(filters_applied=usable, rejected_filters=rejected)

        docs = self._recall_candidates(query, entities_config.candidate_pool)
        # Retrieval order is the ranking; dict insertion order preserves it.
        ranked: Dict[str, float] = {}
        for doc in docs:
            for asset_id in doc.get("asset_ids") or []:
                if asset_id and asset_id not in ranked:
                    ranked[asset_id] = float(doc.get("rerank_score") or doc.get("score") or 0.0)
        result.recalled = len(ranked)
        if not ranked:
            return result

        candidates = list(ranked)
        if usable:
            try:
                candidates = self.entity_index.narrow(
                    candidates, usable, self.schema, profile=self.profile.name
                )
            except Exception:
                # A filter failure must not silently widen the result set into
                # something that looks filtered but is not — report and return nothing.
                logger.exception("Attribute filtering failed; returning no entity results")
                return result
            result.filtered_to_empty = not candidates
        result.after_filter = len(candidates)

        for dossier in self.asset_store.get_many(candidates[:limit]):
            text = dossier.extraction.text if dossier.extraction else None
            result.hits.append(EntityHit(
                asset_id=dossier.asset_id,
                score=ranked.get(dossier.asset_id, 0.0),
                caption=text.caption if text else "",
                summary=text.description if text else "",
                attributes=dict(dossier.extraction.structured.attributes) if dossier.extraction else {},
                filename=dossier.source.filename,
                page_number=dossier.source.page_number,
            ))
        return result


_retriever: Optional[EntityRetriever] = None


def get_entity_retriever() -> EntityRetriever:
    global _retriever
    if _retriever is None:
        _retriever = EntityRetriever()
    return _retriever


def set_entity_retriever(retriever: Optional[EntityRetriever]) -> None:
    global _retriever
    _retriever = retriever
