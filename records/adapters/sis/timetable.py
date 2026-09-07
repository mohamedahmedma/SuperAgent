"""The weekly plan, read from the school's own SIS on :8300.

One route, one call: `GET /v1/guardians/by-id/{public_id}/students/{n}/timetable?term=`.
The fourth SIS client in this package and the thinnest, because SIS does all the work this
adapter would otherwise have to do itself.

**The class is resolved on the other side of the wire, and that is the point.** A timetable
is a statement about a room; a child reaches one through the placement she holds for the
term, and placements are time-bounded and change mid-year. SIS owns them, resolves the room
for the term asked about and reads that room's week in one transaction. This adapter
therefore never sees, holds or caches a class code — which is what keeps a parent-facing
service from eventually asking for the week of a room a child has left.

**No cache.** A timetable changes when a teacher leaves, and a school that has just fixed
Tuesday should not be answering yesterday's grid to a parent already reading it. This is one
call per parent question, which is the same trade the guardian directory makes and for the
same reason.

Failures normalise to `TimetableUnavailable`, which is an `UpstreamUnavailable` and so
`lms_unavailable` on the wire — the code the agent is written against. A child with no class
this term is **not** a failure: SIS answers 200 with a null class, and that arrives here as
`TimetableStatus.NO_CLASS`.
"""
from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote

from records.adapters.sis.http import PooledClient, REDIRECT_STATUSES, error_code
from records.domain.errors import TimetableUnavailable
from records.domain.timetable import (
    StudentTimetable,
    TimetableLesson,
    TimetablePeriodSlot,
    TimetableStatus,
)

logger = logging.getLogger(__name__)

#: The SIS error code for "this code names nothing" — an unknown student, an unknown term,
#: or a child this guardian may not be told about. The only 404 this adapter reads as
#: "nothing on file"; a bare one from a wrong path means the opposite and must not be
#: reported to every family in the school as an empty week.
_UNKNOWN_REFERENCE = "unknown_reference"


def _clock_time(value: Any) -> str:
    """`"08:00:00"` as `"08:00"`; anything absent as `""`.

    Trimmed here rather than by each consumer, because the seconds on a bell schedule are
    always zero and a parent reads a time rather than a timestamp. `""` is "the school has
    not fixed this boundary" and is never filled in with a plausible hour — see
    `TimetablePeriodSlot.starts_at`.
    """
    text = str(value or "").strip()
    if not text:
        return ""
    parts = text.split(":")
    if len(parts) >= 2:
        return f"{parts[0]}:{parts[1]}"
    return text


class SisTimetableAdapter:
    """Reads one student's week from the Student Information Service.

    Holds the same pooled client, the same refusal to retry and the same refusal to follow
    a redirect as its three siblings — see [http.py](http.py). The guardian handle travels
    in the path, so SIS re-checks the link from the registrar's own data on this request:
    the decision is made twice, independently, and a fully compromised facade reaches one
    family rather than the school.
    """

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        api_key: str = "",
    ) -> None:
        """`timeout_seconds` is required, and this file reads no configuration.

        Its three siblings each reach for `records.config` for a default, and each therefore
        holds a second copy of a number that lives in one place — which is how two clients
        to one service end up with two answers to "how long do we wait". Handed in instead,
        by the composition root that already reads the rest of this adapter's settings, so
        there is nothing here to drift. The layering suite whitelists the files allowed to
        read configuration precisely so a new one has to make this choice deliberately.

        The value the root passes is the shorter LOOKUP budget rather than the marks one:
        this asks a smaller question than a gradebook read, and a slow one should fail fast
        rather than spend a chat turn's whole patience.
        """
        if not base_url:
            raise RuntimeError("The SIS timetable adapter needs SIS_BASE_URL.")
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        headers = {"Accept": "application/json"}
        if api_key:
            headers["X-API-Key"] = api_key
        self._pool = PooledClient(
            base_url=base_url, timeout_seconds=self._timeout, headers=headers
        )

    def get_timetable(
        self, *, student_ref: str, term: str, guardian_ref: str = ""
    ) -> StudentTimetable:
        # Every segment quoted with no safe characters: a stray "/" in a school's numbering
        # — or in a handle — would otherwise rewrite the path and ask a different question.
        path = (
            f"/v1/guardians/by-id/{quote(guardian_ref, safe='')}"
            f"/students/{quote(student_ref, safe='')}/timetable"
        )
        payload = self._get(path, {"term": term})
        if payload is None:
            # No such student, not this guardian's, or no such term. Indistinguishable by
            # design — `records/` deliberately collapses those three, and a difference here
            # is the one signal an outsider would need to enumerate the school roll.
            return StudentTimetable(status=TimetableStatus.NO_CLASS)
        return self._to_timetable(payload)

    # -- transport ----------------------------------------------------------

    def _get(self, path: str, params: dict[str, str]) -> dict | None:
        """One SIS call. Returns the body, or `None` for "nothing on file".

        Every other outcome — transport failure, a refused key, a body that is not the JSON
        object promised — becomes `TimetableUnavailable`. Including refusals: a revoked key
        and a switched-off SIS are indistinguishable from the parent's side, and
        distinguishing them in the response would let a caller probe SIS's configuration
        through this service.
        """
        import httpx

        try:
            # Relative to the client's base_url, so a SIS mounted under a path prefix joins
            # correctly instead of having its prefix truncated by concatenation.
            response = self._pool.get().get(path.lstrip("/"), params=params)
        except httpx.HTTPError as exc:
            raise TimetableUnavailable(
                f"{path}: transport failure — {exc}"
            ) from exc

        if response.status_code in REDIRECT_STATUSES:
            raise TimetableUnavailable(
                f"{path}: SIS redirected to {response.headers.get('location')!r}. "
                "SIS_BASE_URL must name the service's own origin."
            )

        if response.status_code == 200:
            try:
                body = response.json()
            except ValueError as exc:
                raise TimetableUnavailable(f"{path}: response was not JSON") from exc
            if not isinstance(body, dict):
                raise TimetableUnavailable(f"{path}: response was not a JSON object")
            return body

        code = error_code(response)
        if response.status_code == 404 and code == _UNKNOWN_REFERENCE:
            return None

        logger.warning(
            "SIS refused %s: HTTP %s (%s)", path, response.status_code, code or "no code"
        )
        raise TimetableUnavailable(
            f"{path}: HTTP {response.status_code} ({code or 'no code'})"
        )

    # -- reshaping ----------------------------------------------------------

    @staticmethod
    def _to_timetable(payload: dict) -> StudentTimetable:
        """SIS's `StudentWeekOut` as this contract's own shape.

        The status is decided here and nowhere else. SIS states a null class for a child no
        placement covered, and an empty lesson list for a class nobody has timetabled yet;
        those are two different answers and this is the one place that turns them into a
        word a consumer can branch on rather than an emptiness it has to interpret.
        """
        class_code = str(payload.get("class_code") or "")
        if not class_code:
            return StudentTimetable(status=TimetableStatus.NO_CLASS)

        lessons = tuple(
            TimetableLesson(
                day_of_week=str(row.get("day_of_week") or ""),
                period_number=int(row.get("period_number") or 0),
                subject_code=str(row.get("subject_code") or ""),
                subject_name_ar=str(row.get("subject_name_ar") or ""),
                subject_name_en=str(row.get("subject_name_en") or ""),
            )
            for row in payload.get("lessons") or []
        )
        periods = tuple(
            TimetablePeriodSlot(
                period_number=int(row.get("period_number") or 0),
                name_ar=str(row.get("name_ar") or ""),
                name_en=str(row.get("name_en") or ""),
                starts_at=_clock_time(row.get("starts_at")),
                ends_at=_clock_time(row.get("ends_at")),
                # Read from the wire rather than defaulted true: a break that arrived as a
                # teaching period would be counted as a lesson the child does not have.
                is_teaching=bool(row.get("is_teaching", True)),
            )
            for row in payload.get("periods") or []
        )

        return StudentTimetable(
            # She has a room; whether it has a week yet is the other question.
            status=TimetableStatus.OK if lessons else TimetableStatus.NO_TIMETABLE,
            class_code=class_code,
            class_name_ar=str(payload.get("class_name_ar") or ""),
            class_name_en=str(payload.get("class_name_en") or ""),
            # Never sorted: the school stated its own week order and only it knows whether
            # that week begins on Saturday or Sunday.
            days=tuple(str(day) for day in payload.get("days") or []),
            periods=periods,
            lessons=lessons,
            teaching_slots=int(payload.get("teaching_slots") or 0),
        )

    def close(self) -> None:
        """Release the pooled client. Called from the app's shutdown hook."""
        self._pool.close()


__all__ = ["SisTimetableAdapter"]
