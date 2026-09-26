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

    def clear_needs_review(self, sha256: str) -> int:
        """Clear the review flag on EVERY occurrence of these bytes. Returns how many.

        An occurrence keeps its own copy of the whole dossier — the JSON is the source
        of truth that `get` hydrates from, and the scalar columns beside it are the
        duplicate. Clearing the flag on the shared extraction alone is therefore
        invisible: the extraction is keyed by digest and the read path never consults
        it, so the asset comes back still asking to be reviewed.

        By digest rather than by asset id, because the flag belongs to the extraction
        and one extraction is shared by every occurrence of the same bytes. Clearing
        one and leaving its twins is a state the storage cannot mean.
        """
        if not sha256:
            return 0
        # Imported here rather than at the top, for the reason the module docstring
        # gives: opening a unit of work must not load the assets package.
        from backend.assets.dossier import AssetDossier

        rows = list(self._session.scalars(
            select(DocumentAsset).where(DocumentAsset.sha256 == sha256)
        ))
        now = datetime.now(UTC)
        changed = 0
        for row in rows:
            # The column is JSONB, so this is a dict rather than the text the read path
            # casts it to. Written back the same way `_occurrence_values` writes it.
            dossier = AssetDossier.model_validate(row.dossier)
            if dossier.extraction is None or not dossier.extraction.provenance.needs_review:
                continue
            dossier.extraction.provenance.needs_review = False
            row.dossier = dossier.model_dump(mode="json")
            row.updated_at = now
            changed += 1
        return changed

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

    def clear_needs_review(self, sha256: str, profile: str, dossier_version: int) -> bool:
        """Mark one extraction as reviewed. False when there was no such row.

        BOTH copies are cleared. `needs_review` is a scalar column so it can be filtered
        and joined on, and it is also inside `payload`, which the dossier is rebuilt
        from — updating one would leave the API reading a flag the queue no longer
        agrees with. The comment on `DocumentAsset` calls the columns a duplicate of the
        JSON, and a duplicate only stays true if both are written together.
        """
        row = self._session.get(AssetExtraction, (sha256, profile, dossier_version))
        if row is None:
            return False

        payload = dict(row.payload or {})
        provenance = dict(payload.get("provenance") or {})
        provenance["needs_review"] = False
        payload["provenance"] = provenance

        row.payload = payload
        row.needs_review = False
        row.updated_at = datetime.now(UTC)
        # JSONB is mutable-tracked by identity, and the dict above is a new object, so
        # the assignment is what marks it dirty. Reassigning is deliberate, not a style.
        return True

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
