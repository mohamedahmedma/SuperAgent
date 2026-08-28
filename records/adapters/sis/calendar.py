"""The academic calendar as SIS reports it.

Cached briefly, unlike the guardian directory, and the difference is deliberate: a
guardian link can be revoked the minute a court order arrives and must take effect on the
next question, whereas a term's dates change when a registrar edits the school year. Ten
minutes of staleness there costs nothing; a lookup on every parent question costs two SIS
requests each.
"""
from __future__ import annotations

import logging
import threading
from typing import Final
from urllib.parse import quote

from records.adapters.sis.http import PooledClient
from records.config import settings
from records.domain.errors import CalendarUnavailable
from records.domain.terms import SchoolTerm, as_datetime

logger = logging.getLogger(__name__)


class SisSchoolCalendar:
    """The calendar as SIS reports it.

    Cached briefly, unlike the guardian directory, and the difference is deliberate: a
    guardian link can be revoked the minute a court order arrives and must take effect on
    the next question, whereas a term's dates change when a registrar edits the school
    year. Ten minutes of staleness there costs nothing; a lookup on every parent question
    costs two requests each.
    """

    #: Both now come from `records.config`, so a deployment tunes them without editing
    #: a class attribute. Kept as names here because the docstring above refers to them.

    def __init__(
        self, *, base_url: str, api_key: str = "", timeout_seconds: float | None = None
    ) -> None:
        if not base_url:
            raise RuntimeError("The SIS calendar needs SIS_BASE_URL.")
        self._api_key = api_key
        self._timeout = timeout_seconds or settings().lookup_timeout_seconds
        self._pool = PooledClient(base_url=base_url, timeout_seconds=self._timeout)
        self._cache_seconds = settings().calendar_cache_seconds
        self._lock = threading.Lock()
        self._terms: list[SchoolTerm] = []
        self._loaded_at: float = 0.0
        #: Set while one thread is refreshing, so the others wait rather than each
        #: making the same two SIS calls. See `_load`.
        self._refreshing: threading.Event | None = None


    def _get(self, path: str, params: dict | None = None):
        import httpx

        headers = {"X-API-Key": self._api_key} if self._api_key else {}
        try:
            response = self._pool.get().get(path, headers=headers, params=params or {})
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
        """The school's terms, from cache when fresh and from SIS otherwise.

        ## One refresh at a time

        The cache check released the lock before fetching, so every thread that arrived
        during a lapsed TTL made the same **two** SIS calls. At 40 workers that is eighty
        requests to answer one question the school changes twice a year, all issued in the
        same millisecond, and the burst lands precisely when the service is busiest —
        cache expiry correlates with nothing except elapsed time.

        Now one thread refreshes and the rest wait for it. A waiter that times out falls
        back to the stale list rather than fetching its own: term dates that are ten
        minutes and one second old are not worth a second stampede.

        The fetch stays outside the lock, so a slow SIS delays the refresh rather than
        blocking every reader of an already-good cache.
        """
        import time

        now = time.monotonic()
        cached = self._terms
        if cached and (now - self._loaded_at) < self._cache_seconds:
            return list(cached)

        with self._lock:
            if self._terms and (time.monotonic() - self._loaded_at) < self._cache_seconds:
                return list(self._terms)  # somebody refreshed while we waited
            in_flight = self._refreshing
            if in_flight is None:
                in_flight = self._refreshing = threading.Event()
                leader = True
            else:
                leader = False

        if not leader:
            in_flight.wait(timeout=self._timeout * 2 + 1.0)
            return list(self._terms)

        try:
            return self._refresh()
        finally:
            with self._lock:
                self._refreshing = None
            in_flight.set()

    def _refresh(self) -> list[SchoolTerm]:
        """The two SIS calls, run by one thread at a time and outside the lock."""
        import time

        now = time.monotonic()
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
                starts_on=as_datetime(row.get("starts_on")),
                ends_on=as_datetime(row.get("ends_on")),
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

    def close(self) -> None:
        """Release the pooled client. Called from the app's shutdown hook."""
        self._pool.close()


__all__ = ["SisSchoolCalendar"]
