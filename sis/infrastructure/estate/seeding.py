"""The one `schools` row that makes a freshly migrated database a school.

Separate from the provisioners because it is the only part of creating a school that goes
through the domain: a migrated database is a schema with nothing in it, and every read
path in the service resolves a school code against this row.

Written through the repository rather than a raw INSERT. The statement this replaces
hard-coded `is_active` and `created_at` and named its columns literally, so the day a
migration added one the seeded row and every other row in the service would have differed
-- and nothing would have reported it, because the row is valid, just not the same.
"""
from __future__ import annotations

from sis.domain.structure import School
from sis.infrastructure.db.unit_of_work import SqlAlchemyUnitOfWork


def seed_school_row(code: str, *, name_en: str = "", name_ar: str = "") -> None:
    """Write the school's own row into its own database.

    The caller must already have made `code` routable -- the registry read at process
    start does not know about a school provisioned since. `sis.schools.cmd_provision` and
    the admin route both refresh the registry before calling this, and the failure when
    they do not is an `UnknownSchool` naming the school that was just created.
    """
    with SqlAlchemyUnitOfWork(school_code=code) as uow:
        uow.schools.upsert_many(
            [School(code=code, name_en=name_en or code, name_ar=name_ar or "")]
        )
        uow.commit()


__all__ = ["seed_school_row"]
