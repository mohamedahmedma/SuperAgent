"""`EntityAttributeRepository` over SQLAlchemy.

Scalar comparisons only. Which criterion an attribute's declared type calls for is the
index's decision (`backend/assets/entity_store.py`); this runs the query it asks for.
"""
from __future__ import annotations

from collections.abc import Collection, Sequence
from datetime import UTC, datetime

from sqlalchemy import delete, distinct, func, select
from sqlalchemy.orm import Session

from backend.application.ports.repositories import EntityAttributeRecord
from backend.db.models import EntityAttribute

_VALUE_COLUMNS = {
    "number": EntityAttribute.value_number,
    "boolean": EntityAttribute.value_bool,
    "text": EntityAttribute.value_text,
}


class SqlAlchemyEntityAttributeRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def replace_for_asset(self, asset_id: str, rows: Sequence[EntityAttributeRecord]) -> None:
        self._session.execute(delete(EntityAttribute).where(EntityAttribute.asset_id == asset_id))
        now = datetime.now(UTC)
        self._session.add_all(
            EntityAttribute(
                asset_id=row.asset_id,
                profile=row.profile,
                name=row.name,
                value_key=row.value_key,
                value_text=row.value_text,
                value_number=row.value_number,
                value_bool=row.value_bool,
                updated_at=now,
            )
            for row in rows
        )
        self._session.flush()

    def delete_for_assets(self, asset_ids: Sequence[str]) -> int:
        if not asset_ids:
            return 0
        result = self._session.execute(
            delete(EntityAttribute)
            .where(EntityAttribute.asset_id.in_(list(asset_ids)))
            .execution_options(synchronize_session=False)
        )
        return int(result.rowcount or 0)

    def matching_asset_ids(
        self,
        name: str,
        *,
        profile: str | None,
        restrict_to: Collection[str] | None,
        minimum: float | None = None,
        maximum: float | None = None,
        boolean: bool | None = None,
        keys: Collection[str] | None = None,
    ) -> set[str]:
        statement = self._scoped(select(EntityAttribute.asset_id).distinct(), name, profile, restrict_to)
        if minimum is not None:
            statement = statement.where(EntityAttribute.value_number >= minimum)
        if maximum is not None:
            statement = statement.where(EntityAttribute.value_number <= maximum)
        if boolean is not None:
            statement = statement.where(EntityAttribute.value_bool == boolean)
        if keys is not None:
            statement = statement.where(EntityAttribute.value_key.in_(list(keys)))
        return set(self._session.scalars(statement))

    def facet_counts(
        self,
        name: str,
        *,
        kind: str,
        profile: str | None,
        restrict_to: Collection[str] | None,
        limit: int,
    ) -> list[tuple[object, int]]:
        column = _VALUE_COLUMNS[kind]
        count = func.count(EntityAttribute.asset_id)
        statement = self._scoped(select(column, count), name, profile, restrict_to)
        rows = self._session.execute(
            statement.group_by(column).order_by(count.desc()).limit(limit)
        ).all()
        return [(value, int(total)) for value, total in rows if value is not None]

    def stats(self) -> dict:
        total = self._session.scalar(select(func.count(EntityAttribute.id)))
        assets = self._session.scalar(select(func.count(distinct(EntityAttribute.asset_id))))
        by_name = self._session.execute(
            select(EntityAttribute.name, func.count(EntityAttribute.id)).group_by(EntityAttribute.name)
        ).all()
        return {
            "rows": int(total or 0),
            "indexed_assets": int(assets or 0),
            "by_attribute": {attribute: int(count) for attribute, count in by_name},
        }

    @staticmethod
    def _scoped(statement, name: str, profile: str | None, restrict_to: Collection[str] | None):
        statement = statement.where(EntityAttribute.name == name)
        if profile:
            statement = statement.where(EntityAttribute.profile == profile)
        if restrict_to is not None:
            statement = statement.where(EntityAttribute.asset_id.in_(list(restrict_to)))
        return statement


__all__ = ["SqlAlchemyEntityAttributeRepository"]
