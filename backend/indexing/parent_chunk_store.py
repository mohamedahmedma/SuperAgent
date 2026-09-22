"""Parent chunks for the auto-merging retriever: Postgres through a unit of work, Redis in front."""
from __future__ import annotations

from typing import List

from backend.application.ports.repositories import ParentChunkRecord
from backend.application.ports.unit_of_work import UnitOfWorkFactory
from backend.infra.cache import RedisCache
from backend.infra.unit_of_work import SqlAlchemyUnitOfWork


class ParentChunkStore:
    """Parent chunks by id: read through the cache, written through a unit of work."""

    def __init__(
        self,
        unit_of_work: UnitOfWorkFactory = SqlAlchemyUnitOfWork,
        cache: RedisCache | None = None,
    ) -> None:
        self._unit_of_work = unit_of_work
        if cache is None:
            from backend.composition import default_services

            cache = default_services().cache
        self._cache = cache

    @staticmethod
    def _cache_key(chunk_id: str) -> str:
        return f"parent_chunk:{chunk_id}"

    @staticmethod
    def _to_dict(chunk: ParentChunkRecord) -> dict:
        return {
            "text": chunk.text,
            "filename": chunk.filename,
            "file_type": chunk.file_type,
            "file_path": chunk.file_path,
            "page_number": chunk.page_number,
            "chunk_id": chunk.chunk_id,
            "parent_chunk_id": chunk.parent_chunk_id,
            "root_chunk_id": chunk.root_chunk_id,
            "chunk_level": chunk.chunk_level,
            "chunk_idx": chunk.chunk_idx,
            "modality": chunk.modality or "text",
            "asset_ids": list(chunk.asset_ids),
        }

    @staticmethod
    def _from_document(doc: dict) -> ParentChunkRecord | None:
        chunk_id = (doc.get("chunk_id") or "").strip()
        if not chunk_id:
            return None
        return ParentChunkRecord(
            chunk_id=chunk_id,
            text=doc.get("text", ""),
            filename=doc.get("filename", ""),
            file_type=doc.get("file_type", ""),
            file_path=doc.get("file_path", ""),
            page_number=int(doc.get("page_number", 0) or 0),
            parent_chunk_id=doc.get("parent_chunk_id", ""),
            root_chunk_id=doc.get("root_chunk_id", ""),
            chunk_level=int(doc.get("chunk_level", 0) or 0),
            chunk_idx=int(doc.get("chunk_idx", 0) or 0),
            modality=doc.get("modality", "text") or "text",
            asset_ids=tuple(doc.get("asset_ids") or ()),
        )

    def upsert_documents(self, docs: List[dict]) -> int:
        """Inserts/updates parent chunks, returning the number of records written."""
        chunks = [chunk for chunk in map(self._from_document, docs or []) if chunk is not None]
        if not chunks:
            return 0
        with self._unit_of_work() as uow:
            uow.parent_chunks.upsert_many(chunks)
            uow.commit()
        # After the commit, so the cache never serves a chunk the database rolled back.
        for chunk in chunks:
            self._cache.set_json(self._cache_key(chunk.chunk_id), self._to_dict(chunk))
        return len(chunks)

    def get_documents_by_ids(self, chunk_ids: List[str]) -> List[dict]:
        if not chunk_ids:
            return []

        found: dict[str, dict] = {}
        missing: list[str] = []
        for chunk_id in chunk_ids:
            key = (chunk_id or "").strip()
            if not key:
                continue
            cached = self._cache.get_json(self._cache_key(key))
            if cached:
                found[key] = cached
            else:
                missing.append(key)

        if missing:
            with self._unit_of_work() as uow:
                chunks = uow.parent_chunks.get_many(missing)
            for chunk in chunks:
                payload = self._to_dict(chunk)
                found[chunk.chunk_id] = payload
                self._cache.set_json(self._cache_key(chunk.chunk_id), payload)

        return [found[item] for item in chunk_ids if item in found]

    def delete_by_filename(self, filename: str) -> int:
        """Deletes a document's parent chunks, returning how many were deleted."""
        if not filename:
            return 0
        with self._unit_of_work() as uow:
            removed = uow.parent_chunks.delete_by_filename(filename)
            uow.commit()
        for chunk_id in removed:
            self._cache.delete(self._cache_key(chunk_id))
        return len(removed)

    def sections(self, level: int) -> List[ParentChunkRecord]:
        """Every chunk at `level`, in file and position order. Uncached: only a catalogue
        build reads it, and it wants the database's view rather than the cache's."""
        with self._unit_of_work() as uow:
            return list(uow.parent_chunks.sections(level))

    def documents_by_filename(self, filename: str) -> List[dict]:
        """One document's parent chunks, as documents, in level and position order.

        For the admin inspector, which needs the levels Milvus does not hold: only leaves
        are vectorised, so a tree drawn from Milvus alone is every leaf orphaned from the
        L1 and L2 it belongs under. Uncached for the same reason `sections` is — an
        inspector exists to show what the database actually contains.
        """
        if not filename:
            return []
        with self._unit_of_work() as uow:
            return [self._to_dict(chunk) for chunk in uow.parent_chunks.by_filename(filename)]
