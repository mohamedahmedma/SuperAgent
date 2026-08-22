"""When a term runs — asked of the school, not remembered here.

The last of this service's own data to move. `terms` was a `records/` table because
Moodle has no notion of a term at all: somebody had to say that "2026-T1" runs from
September to December, and this service was the only place that could.

That is no longer true. SIS holds the academic calendar the registrar actually maintains,
and keeping a second copy here means a term whose dates were corrected in one place keeps
answering from the other — which shows up as a parent being handed the wrong term's report
card, with nothing anywhere reporting an error.

Same seam as `records/guardian_directory.py` and `records/lms.py`: a `Protocol`, a class
per backend, a deterministic fake, and a module-level slot chosen once at startup. Kept
apart from the guardian directory because the two answer unrelated questions — one is who
a parent's children are, the other is what today's term is called — and a module that did
both would be a module with two reasons to change.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final, Protocol
from urllib.parse import quote

logger = logging.getLogger(__name__)


class CalendarUnavailable(RuntimeError):
    """The calendar could not be read. Distinct from "no such term", which is `None`."""


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


class SchoolCalendar(Protocol):
    """The academic calendar, as far as this service needs it."""

    def term(self, code: str) -> SchoolTerm | None:
        """That named term, or `None` when the school has no such term."""

    def current_term(self) -> SchoolTerm | None:
        """The term we are in, or the most recent one when today falls in no term.

        The fallback is what a parent means. Asked in August, "how is she doing" is a
        question about the term that just ended, not an error — so a gap between years
        answers with the last term rather than refusing.
        """


def _as_datetime(raw: object) -> datetime | None:
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


class SisSchoolCalendar:
    """The calendar as SIS reports it.

    Cached briefly, unlike the guardian directory, and the difference is deliberate: a
    guardian link can be revoked the minute a court order arrives and must take effect on
    the next question, whereas a term's dates change when a registrar edits the school
    year. Ten minutes of staleness there costs nothing; a lookup on every parent question
    costs two requests each.
    """

    DEFAULT_TIMEOUT_SECONDS: Final[float] = 5.0
    CACHE_SECONDS: Final[float] = 600.0

    def __init__(
        self, *, base_url: str, api_key: str = "", timeout_seconds: float | None = None
    ) -> None:
        if not base_url:
            raise RuntimeError("The SIS calendar needs SIS_BASE_URL.")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout_seconds or self.DEFAULT_TIMEOUT_SECONDS
        self._client = None
        self._lock = threading.Lock()
        self._terms: list[SchoolTerm] = []
        self._loaded_at: float = 0.0

    def _http(self):
        import httpx

        with self._lock:
            if self._client is None:
                self._client = httpx.Client(
                    base_url=self._base_url,
                    timeout=httpx.Timeout(self._timeout),
                    transport=httpx.HTTPTransport(retries=0),
                    follow_redirects=False,
                )
            return self._client

    def _get(self, path: str, params: dict | None = None):
        import httpx

        headers = {"X-API-Key": self._api_key} if self._api_key else {}
        try:
            response = self._http().get(path, headers=headers, params=params or {})
        except httpx.HTTPError as error:
            raise CalendarUnavailable(
                f"The school calendar could not be read: {error}"
            ) from error
        if response.status_code >= 400:
            raise CalendarUnavailable(
                f"The school calendar answered {response.status_code}."
            )
        try:
            return response.json()
        except ValueError as error:
            raise CalendarUnavailable(
                "The school calendar returned a body this service could not read."
            ) from error

    def _load(self) -> list[SchoolTerm]:
        import time

        now = time.monotonic()
        with self._lock:
            if self._terms and (now - self._loaded_at) < self.CACHE_SECONDS:
                return list(self._terms)

        years = self._get("/v1/structure/years") or {}
        year_code = years.get("current_academic_year_code") or ""
        if not year_code:
            return []

        payload = self._get(
            "/v1/terms", {"academic_year": quote(year_code, safe="")}
        )
        rows = payload if isinstance(payload, list) else (payload or {}).get("terms") or []

        terms = [
            SchoolTerm(
                code=str(row.get("code") or ""),
                name_ar=str(row.get("name_ar") or ""),
                name_en=str(row.get("name_en") or ""),
                academic_year=year_code,
                starts_on=_as_datetime(row.get("starts_on")),
                ends_on=_as_datetime(row.get("ends_on")),
                is_closed=bool(row.get("is_closed")),
            )
            for row in rows
            if row.get("code")
        ]

        with self._lock:
            self._terms = terms
            self._loaded_at = now
        return list(terms)

    def term(self, code: str) -> SchoolTerm | None:
        return next((t for t in self._load() if t.code == str(code)), None)

    def current_term(self) -> SchoolTerm | None:
        terms = self._load()
        if not terms:
            return None
        current = [t for t in terms if t.is_current]
        if current:
            # The latest-starting of any that overlap today, matching what the local
            # implementation did: overlapping terms are a data problem, and picking the
            # one that started most recently is the least surprising resolution.
            return sorted(current, key=lambda t: t.starts_on or datetime.min.replace(tzinfo=timezone.utc))[-1]
        dated = [t for t in terms if t.starts_on is not None]
        if dated:
            return sorted(dated, key=lambda t: t.starts_on)[-1]
        return terms[-1]


class FakeSchoolCalendar:
    """A calendar in a list. The default when no SIS is configured."""

    def __init__(self, terms: list[SchoolTerm] | None = None, *, unavailable: bool = False) -> None:
        self.terms = list(terms or [])
        self.unavailable = unavailable

    def _all(self) -> list[SchoolTerm]:
        if self.unavailable:
            raise CalendarUnavailable("The fake calendar is switched off.")
        return self.terms

    def term(self, code: str) -> SchoolTerm | None:
        return next((t for t in self._all() if t.code == str(code)), None)

    def current_term(self) -> SchoolTerm | None:
        terms = self._all()
        if not terms:
            return None
        current = [t for t in terms if t.is_current]
        return current[-1] if current else terms[-1]


_calendar: SchoolCalendar = FakeSchoolCalendar()


def set_calendar(calendar: SchoolCalendar) -> None:
    global _calendar
    _calendar = calendar


def get_calendar() -> SchoolCalendar:
    return _calendar
