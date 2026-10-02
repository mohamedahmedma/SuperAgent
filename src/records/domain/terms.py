"""When a term runs.

A value object built at the boundary, which is what lets it never be naive: the ORM row
this replaced came back without a timezone from SQLite and every caller had to remember
to re-attach one before comparing.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True, slots=True)
class SchoolTerm:
    """One term, in the shape the parent-facing contract renders.

    Carries `starts_on`/`ends_on` as timezone-aware UTC. The ORM row this replaces came
    back naive from SQLite and every caller had to remember to re-attach a timezone before
    comparing; a value object built at the boundary can simply never be naive.
    """

    code: str
    name_ar: str = ""
    name_en: str = ""
    academic_year: str = ""
    starts_on: datetime | None = None
    ends_on: datetime | None = None
    is_closed: bool = False

    @property
    def is_current(self) -> bool:
        """Does today fall inside it? `False` when either end is unknown."""
        if self.starts_on is None or self.ends_on is None:
            return False
        now = datetime.now(timezone.utc)
        return self.starts_on <= now <= self.ends_on


def as_datetime(raw: object) -> datetime | None:
    """An ISO date or datetime from the wire, as timezone-aware UTC.

    SIS reports a term's bounds as plain dates, which have no time and no zone. Read as
    midnight UTC: a term boundary is a school's decision about a day, and inventing a
    local time for it would move the boundary by hours depending on where this runs.
    """
    if not raw:
        return None
    text = str(raw)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


__all__ = ["SchoolTerm", "as_datetime"]
