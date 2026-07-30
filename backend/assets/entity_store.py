"""Attribute index for entity assets.

Two query shapes, both LLM-free:

    narrow(candidates, filters)  post-recall filtering — the "context first, then
                                 attributes" flow, over a set semantic search produced
    find(filters)                catalogue-wide filtering, for browse and facets

Everything is a scalar comparison in the database. Nothing here calls a model, loads
an embedding, or looks at a pixel, which is the whole point: extraction happened once
at ingest, and query time only narrows.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from backend.assets.attributes import AttributeSchema, AttributeSpec, AttributeType, NumberRange

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _value_key(value: Any) -> str:
    return str(value).strip().lower()[:255]


class EntityAttributeIndex:
    """Repository over the entity_attributes table."""

    def __init__(self, session_factory: Optional[Callable] = None):
        self._session_factory = session_factory

    @property
    def session_factory(self) -> Callable:
        if self._session_factory is None:
            from backend.infra.database import SessionLocal

            self._session_factory = SessionLocal
        return self._session_factory

    @staticmethod
    def _model():
        from backend.db.models import EntityAttribute

        return EntityAttribute

    # -- writing ----------------------------------------------------------------

    @staticmethod
    def _rows_for(asset_id: str, profile: str, attributes: Dict[str, Any], schema: AttributeSchema):
        """Flatten an attribute dict into index rows — one per value, so a
        multi-valued attribute is genuinely queryable on each of its values."""
        rows: List[dict] = []
        for name, value in (attributes or {}).items():
            spec = schema.get(name)
            if spec is None or value is None:
                continue
            values = value if isinstance(value, list) else [value]
            for item in values:
                if item is None:
                    continue
                row = {
                    "asset_id": asset_id,
                    "profile": profile,
                    "name": name,
                    "value_key": _value_key(item),
                    "value_text": None,
                    "value_number": None,
                    "value_bool": None,
                }
                if spec.type is AttributeType.NUMBER:
                    row["value_number"] = float(item)
                elif spec.type is AttributeType.BOOLEAN:
                    row["value_bool"] = bool(item)
                else:
                    row["value_text"] = str(item)[:255]
                rows.append(row)
        return rows

    def index_asset(
        self,
        asset_id: str,
        profile: str,
        attributes: Dict[str, Any],
        schema: AttributeSchema,
    ) -> int:
        """Replace an asset's indexed attributes. Delete-then-insert rather than
        upsert, so an attribute that disappeared on re-extraction disappears from the
        index too instead of lingering as a stale facet."""
        if not asset_id:
            return 0
        EntityAttribute = self._model()
        rows = self._rows_for(asset_id, profile, attributes, schema)

        session = self.session_factory()
        try:
            session.query(EntityAttribute).filter(EntityAttribute.asset_id == asset_id).delete(
                synchronize_session=False
            )
            for row in rows:
                session.add(EntityAttribute(updated_at=_utcnow(), **row))
            session.commit()
        finally:
            session.close()
        return len(rows)

    def index_many(self, items: Sequence[tuple], schema: AttributeSchema) -> int:
        """items: sequence of (asset_id, profile, attributes)."""
        return sum(
            self.index_asset(asset_id, profile, attributes, schema)
            for asset_id, profile, attributes in items
        )

    def delete_assets(self, asset_ids: Iterable[str]) -> int:
        ids = [item for item in asset_ids if item]
        if not ids:
            return 0
        EntityAttribute = self._model()
        session = self.session_factory()
        try:
            deleted = (
                session.query(EntityAttribute)
                .filter(EntityAttribute.asset_id.in_(ids))
                .delete(synchronize_session=False)
            )
            session.commit()
            return int(deleted or 0)
        finally:
            session.close()

    # -- querying ---------------------------------------------------------------

    def _matching_ids(
        self,
        spec: AttributeSpec,
        condition: Any,
        profile: Optional[str],
        restrict_to: Optional[Sequence[str]],
    ) -> set:
        EntityAttribute = self._model()
        session = self.session_factory()
        try:
            query = session.query(EntityAttribute.asset_id).filter(EntityAttribute.name == spec.name)
            if profile:
                query = query.filter(EntityAttribute.profile == profile)
            if restrict_to is not None:
                query = query.filter(EntityAttribute.asset_id.in_(list(restrict_to)))

            if spec.type is AttributeType.NUMBER:
                bounds = condition if isinstance(condition, NumberRange) else NumberRange.model_validate(condition)
                if bounds.min is not None:
                    query = query.filter(EntityAttribute.value_number >= bounds.min)
                if bounds.max is not None:
                    query = query.filter(EntityAttribute.value_number <= bounds.max)
            elif spec.type is AttributeType.BOOLEAN:
                query = query.filter(EntityAttribute.value_bool == bool(condition))
            else:
                wanted = condition if isinstance(condition, (list, tuple, set)) else [condition]
                keys = [_value_key(item) for item in wanted if item is not None]
                if not keys:
                    return set()
                query = query.filter(EntityAttribute.value_key.in_(keys))

            return {row[0] for row in query.distinct().all()}
        finally:
            session.close()

    def find(
        self,
        filters: Dict[str, Any],
        schema: AttributeSchema,
        profile: Optional[str] = None,
        restrict_to: Optional[Sequence[str]] = None,
    ) -> Optional[set]:
        """Asset ids satisfying every filter (AND across attributes).

        None means "no filters applied", which the caller must distinguish from an
        empty set: one means everything qualifies, the other that nothing does.
        """
        active = {key: value for key, value in (filters or {}).items() if value is not None}
        if not active:
            return None

        result: Optional[set] = None
        for name, condition in active.items():
            spec = schema.get(name)
            if spec is None:
                continue
            matched = self._matching_ids(spec, condition, profile, restrict_to)
            # Intersection, evaluated attribute by attribute: an empty result short-
            # circuits the rest rather than querying for a set that cannot grow.
            result = matched if result is None else (result & matched)
            if not result:
                return set()
        return result

    def narrow(
        self,
        candidates: Sequence[str],
        filters: Dict[str, Any],
        schema: AttributeSchema,
        profile: Optional[str] = None,
    ) -> List[str]:
        """The `context first, then attributes` path: filter a recalled candidate set,
        preserving retrieval rank."""
        if not candidates:
            return []
        matched = self.find(filters, schema, profile=profile, restrict_to=candidates)
        if matched is None:
            return list(candidates)
        return [asset_id for asset_id in candidates if asset_id in matched]

    def facets(
        self,
        name: str,
        schema: AttributeSchema,
        profile: Optional[str] = None,
        restrict_to: Optional[Sequence[str]] = None,
        limit: int = 50,
    ) -> List[tuple]:
        """(value, count) for one attribute — what a UI needs to draw filter chips,
        computed without a model call."""
        spec = schema.get(name)
        if spec is None:
            return []
        from sqlalchemy import func

        EntityAttribute = self._model()
        column = {
            AttributeType.NUMBER: EntityAttribute.value_number,
            AttributeType.BOOLEAN: EntityAttribute.value_bool,
        }.get(spec.type, EntityAttribute.value_text)

        session = self.session_factory()
        try:
            query = (
                session.query(column, func.count(EntityAttribute.asset_id))
                .filter(EntityAttribute.name == name)
                .group_by(column)
                .order_by(func.count(EntityAttribute.asset_id).desc())
            )
            if profile:
                query = query.filter(EntityAttribute.profile == profile)
            if restrict_to is not None:
                query = query.filter(EntityAttribute.asset_id.in_(list(restrict_to)))
            return [(value, int(count)) for value, count in query.limit(limit).all() if value is not None]
        finally:
            session.close()

    def stats(self) -> dict:
        from sqlalchemy import func

        EntityAttribute = self._model()
        session = self.session_factory()
        try:
            total = session.query(EntityAttribute).count()
            assets = session.query(EntityAttribute.asset_id).distinct().count()
            by_name = {
                name: int(count)
                for name, count in session.query(
                    EntityAttribute.name, func.count(EntityAttribute.id)
                ).group_by(EntityAttribute.name).all()
            }
        finally:
            session.close()
        return {"rows": total, "indexed_assets": assets, "by_attribute": by_name}


_index: Optional[EntityAttributeIndex] = None


def get_entity_index() -> EntityAttributeIndex:
    global _index
    if _index is None:
        _index = EntityAttributeIndex()
    return _index


def set_entity_index(index: Optional[EntityAttributeIndex]) -> None:
    global _index
    _index = index
