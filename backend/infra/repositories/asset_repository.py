"""`DocumentAssetRepository` and `AssetExtractionRepository` over SQLAlchemy.

Occurrences are stored as a dossier JSON document beside the scalar columns the system
filters and joins on; the repository writes both from the dossier and reads the dossier
back. Writes are Postgres upserts: the store this replaced read every row before writing
it, and two ingests of the same document could race between the read and the insert.

`AssetDossier` is imported where a dossier is built, not at the top of the module, so
opening a unit of work does not load the assets package.
"""
from __future__ import annotations

from collections.abc import Collection, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import Text, cast, delete, distinct, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from backend.application.ports.repositories import StoredAssetRef, StoredExtraction
from backend.db.models import AssetExtraction, DocumentAsset

if TYPE_CHECKING:
    from backend.assets.dossier import AssetDossier

#: The value `status` holds for an occurrence whose extraction is usable.
_EXTRACTED = "extracted"


def _upsert_occurrence():
    statement = insert(DocumentAsset)
    return statement.on_conflict_do_update(
        index_elements=[DocumentAsset.asset_id],
        set_={
            column.name: statement.excluded[column.name]
            for column in DocumentAsset.__table__.columns
            if column.name not in ("asset_id", "created_at")
        },
    )


#: Insert an occurrence, or overwrite everything but its id and creation time.
_UPSERT_OCCURRENCE = _upsert_occurrence()

#: The dossier as JSON text rather than parsed by the driver; see `_dossier`.
_DOSSIER_JSON = cast(DocumentAsset.dossier, Text)


class SqlAlchemyDocumentAssetRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def upsert_many(self, dossiers: Sequence[AssetDossier]) -> None:
        # One row per id, the last occurrence winning: ON CONFLICT cannot update the same
        # row twice within a single statement.
        latest = {dossier.asset_id: dossier for dossier in dossiers}
        if not latest:
            return
        now = datetime.now(UTC)
        rows = [_occurrence_values(dossier, now) for dossier in latest.values()]
        # Rows go as parameters, not baked into the statement: SQLAlchemy compiles the
        # one-row upsert once, caches it, and batches the rows into multi-row VALUES on
        # the wire, splitting where Postgres's bind-parameter limit requires.
        self._session.execute(_UPSERT_OCCURRENCE, rows)

    # Reads select the dossier column alone, as JSON text. It is the record; the scalar
    # columns beside it exist to be filtered on, not to be loaded and discarded. As text,
    # pydantic parses and validates it in one pass instead of json.loads building dicts
    # for pydantic to walk a second time.

    def get(self, asset_id: str) -> AssetDossier | None:
        payload = self._session.scalar(
            select(_DOSSIER_JSON).where(DocumentAsset.asset_id == asset_id)
        )
        return None if payload is None else _dossier(payload)

    def get_many(self, asset_ids: Sequence[str]) -> Sequence[AssetDossier]:
        if not asset_ids:
            return []
        payloads = self._session.scalars(
            select(_DOSSIER_JSON).where(DocumentAsset.asset_id.in_(list(asset_ids)))
        )
        return [_dossier(payload) for payload in payloads]

    def list_by_filename(self, filename: str, *, extracted_only: bool) -> Sequence[AssetDossier]:
        statement = select(_DOSSIER_JSON).where(DocumentAsset.filename == filename)
        if extracted_only:
            statement = statement.where(DocumentAsset.status == _EXTRACTED)
        payloads = self._session.scalars(statement.order_by(DocumentAsset.page_number, DocumentAsset.asset_id))
        return [_dossier(payload) for payload in payloads]

    def displayable_hashes(self, filenames: Sequence[str]) -> Sequence[tuple[str, str]]:
        if not filenames:
            return []
        rows = self._session.execute(
            select(DocumentAsset.filename, DocumentAsset.sha256).where(
                DocumentAsset.filename.in_(list(filenames)), DocumentAsset.storage_uri != ""
            )
        ).all()
        return [(filename, sha256) for filename, sha256 in rows]

    def delete_by_filename(self, filename: str) -> Sequence[StoredAssetRef]:
        rows = self._session.execute(
            delete(DocumentAsset)
            .where(DocumentAsset.filename == filename)
            .returning(DocumentAsset.asset_id, DocumentAsset.sha256, DocumentAsset.storage_uri)
            .execution_options(synchronize_session=False)
        ).all()
        return [StoredAssetRef(asset_id, sha256, storage_uri) for asset_id, sha256, storage_uri in rows]

    def referenced_digests(self, digests: Collection[str]) -> set[str]:
        if not digests:
            return set()
        return set(
            self._session.scalars(
                select(DocumentAsset.sha256).where(DocumentAsset.sha256.in_(list(digests))).distinct()
            )
        )

    def older_than(self, dossier_version: int, *, after_asset_id: str, limit: int) -> Sequence[AssetDossier]:
        payloads = self._session.scalars(
            select(_DOSSIER_JSON)
            .where(DocumentAsset.dossier_version < dossier_version, DocumentAsset.asset_id > after_asset_id)
            .order_by(DocumentAsset.asset_id)
            .limit(limit)
        )
        return [_dossier(payload) for payload in payloads]

    def status_counts(self) -> dict[str, int]:
        rows = self._session.execute(
            select(DocumentAsset.status, func.count(DocumentAsset.asset_id)).group_by(DocumentAsset.status)
        ).all()
        return {status: int(count) for status, count in rows}

    def occurrence_count(self) -> int:
        return int(self._session.scalar(select(func.count(DocumentAsset.asset_id))) or 0)

    def distinct_image_count(self) -> int:
        return int(self._session.scalar(select(func.count(distinct(DocumentAsset.sha256)))) or 0)


class SqlAlchemyAssetExtractionRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def find_many(
        self, digests: Collection[str], profile: str, dossier_version: int
    ) -> Sequence[StoredExtraction]:
        keys = [digest for digest in set(digests) if digest]
        if not keys:
            return []
        rows = self._session.scalars(
            select(AssetExtraction).where(
                AssetExtraction.sha256.in_(keys),
                AssetExtraction.profile == profile,
                AssetExtraction.dossier_version == dossier_version,
            )
        )
        return [StoredExtraction(row.sha256, row.payload) for row in rows]

    def save(
        self,
        sha256: str,
        profile: str,
        dossier_version: int,
        payload: dict,
        *,
        model_used: str,
        confidence: float,
        needs_review: bool,
    ) -> None:
        now = datetime.now(UTC)
        values = {
            "sha256": sha256,
            "profile": profile,
            "dossier_version": dossier_version,
            "payload": payload,
            "model_used": model_used,
            "confidence": confidence,
            "needs_review": needs_review,
            "created_at": now,
            "updated_at": now,
        }
        statement = insert(AssetExtraction).values(values)
        self._session.execute(
            statement.on_conflict_do_update(
                index_elements=[AssetExtraction.sha256, AssetExtraction.profile, AssetExtraction.dossier_version],
                set_={
                    name: statement.excluded[name]
                    for name in ("payload", "model_used", "confidence", "needs_review", "updated_at")
                },
            )
        )

    def count(self) -> int:
        return int(self._session.scalar(select(func.count()).select_from(AssetExtraction)) or 0)


def _occurrence_values(dossier: AssetDossier, now: datetime) -> dict:
    return {
        "asset_id": dossier.asset_id,
        "sha256": dossier.sha256,
        "profile": dossier.profile,
        "dossier_version": dossier.dossier_version,
        "filename": dossier.source.filename,
        "page_number": int(dossier.source.page_number or 0),
        "role": dossier.role.value,
        "tier": dossier.tier.value,
        "status": dossier.status.value,
        "storage_uri": dossier.blob.uri,
        "content_type": dossier.blob.content_type,
        "byte_size": int(dossier.blob.byte_size or 0),
        "width": int(dossier.blob.width or 0),
        "height": int(dossier.blob.height or 0),
        "dossier": dossier.model_dump(mode="json"),
        "created_at": now,
        "updated_at": now,
    }


def _dossier(document: str) -> AssetDossier:
    from backend.assets.dossier import AssetDossier

    return AssetDossier.model_validate_json(document)


__all__ = ["SqlAlchemyAssetExtractionRepository", "SqlAlchemyDocumentAssetRepository"]
