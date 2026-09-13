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
from typing import Any, Dict, Iterable, List, Optional, Sequence

from backend.application.ports.repositories import EntityAttributeRecord
from backend.application.ports.unit_of_work import UnitOfWorkFactory
from backend.assets.attributes import AttributeSchema, AttributeSpec, AttributeType, NumberRange
from backend.infra.unit_of_work import SqlAlchemyUnitOfWork

logger = logging.getLogger(__name__)

#: Which typed column an attribute type's values live in.
_VALUE_KIND = {AttributeType.NUMBER: "number", AttributeType.BOOLEAN: "boolean"}


def _value_key(value: Any) -> str:
    return str(value).strip().lower()[:255]


class EntityAttributeIndex:
    """The attribute index over entity assets, reached through a unit of work."""

    def __init__(self, unit_of_work: UnitOfWorkFactory = SqlAlchemyUnitOfWork):
        self._unit_of_work = unit_of_work

    # -- writing ----------------------------------------------------------------

    @staticmethod
    def _rows_for(
        asset_id: str, profile: str, attributes: Dict[str, Any], schema: AttributeSchema
    ) -> List[EntityAttributeRecord]:
        """Flatten an attribute dict into index rows — one per value, so a
        multi-valued attribute is genuinely queryable on each of its values."""
        rows: List[EntityAttributeRecord] = []
        for name, value in (attributes or {}).items():
            spec = schema.get(name)
            if spec is None or value is None:
                continue
            values = value if isinstance(value, list) else [value]
            for item in values:
                if item is None:
                    continue
                typed: Dict[str, Any] = {}
                if spec.type is AttributeType.NUMBER:
                    typed["value_number"] = float(item)
                elif spec.type is AttributeType.BOOLEAN:
                    typed["value_bool"] = bool(item)
                else:
                    typed["value_text"] = str(item)[:255]
                rows.append(
                    EntityAttributeRecord(
                        asset_id=asset_id, profile=profile, name=name, value_key=_value_key(item), **typed
                    )
                )
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
        rows = self._rows_for(asset_id, profile, attributes, schema)
        with self._unit_of_work() as uow:
            uow.entity_attributes.replace_for_asset(asset_id, rows)
            uow.commit()
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
        with self._unit_of_work() as uow:
            deleted = uow.entity_attributes.delete_for_assets(ids)
            uow.commit()
        return deleted

    # -- querying ---------------------------------------------------------------

    def _matching_ids(
        self,
        spec: AttributeSpec,
        condition: Any,
        profile: Optional[str],
        restrict_to: Optional[Sequence[str]],
    ) -> set:
        criteria: Dict[str, Any] = {}
        if spec.type is AttributeType.NUMBER:
            bounds = condition if isinstance(condition, NumberRange) else NumberRange.model_validate(condition)
            criteria.update(minimum=bounds.min, maximum=bounds.max)
        elif spec.type is AttributeType.BOOLEAN:
            criteria["boolean"] = bool(condition)
        else:
            wanted = condition if isinstance(condition, (list, tuple, set)) else [condition]
            keys = [_value_key(item) for item in wanted if item is not None]
            if not keys:
                return set()
            criteria["keys"] = keys

        with self._unit_of_work() as uow:
            return uow.entity_attributes.matching_asset_ids(
                spec.name, profile=profile, restrict_to=restrict_to, **criteria
            )

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
        with self._unit_of_work() as uow:
            return uow.entity_attributes.facet_counts(
                name,
                kind=_VALUE_KIND.get(spec.type, "text"),
                profile=profile,
                restrict_to=restrict_to,
                limit=limit,
            )

    def stats(self) -> dict:
        with self._unit_of_work() as uow:
            return uow.entity_attributes.stats()


_index: Optional[EntityAttributeIndex] = None


def get_entity_index() -> EntityAttributeIndex:
    global _index
    if _index is None:
        _index = EntityAttributeIndex()
    return _index


def set_entity_index(index: Optional[EntityAttributeIndex]) -> None:
    global _index
    _index = index
