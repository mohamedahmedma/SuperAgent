"""The class, its subjects and its staff, read from the school's own SIS on :8300.

One route, one call: `GET /v1/guardians/by-id/{public_id}/students/{n}/classroom?term=`.
The fifth SIS client in this package and, like the timetable, thin — SIS does the work.

**The room is resolved on the other side of the wire, and three answers come back
together.** Which class, which subjects, which teachers all depend on the placement a child
holds for the term, and SIS resolves it once inside one transaction. This adapter therefore
never sees, holds or caches a class code, and cannot be told about two different rooms in
one answer.

**No cache.** A registrar reassigns a teacher the morning a parent asks, and a curriculum
board is edited in September. This is one call per parent question — the same trade the
guardian directory makes, for the same reason.

Failures normalise to `ClassroomUnavailable`, which is an `UpstreamUnavailable` and so
`lms_unavailable` on the wire — the code the agent is written against. A child with no class
this term is **not** a failure: SIS answers 200 with a null class, which arrives here as
`ClassroomStatus.NO_CLASS`.
"""
from __future__ import annotations

import logging
from urllib.parse import quote

from records.adapters.sis.http import PooledClient, REDIRECT_STATUSES, error_code
from records.domain.classroom import (
    ClassTeacher,
    ClassroomStatus,
    StudentClassroom,
    StudySubject,
)
from records.domain.errors import ClassroomUnavailable

logger = logging.getLogger(__name__)

#: The SIS error code for "this code names nothing" — an unknown student, an unknown term,
#: or a child this guardian may not be told about. The only 404 this adapter reads as
#: "nothing on file"; a bare one from a wrong path means the opposite and must not be
#: reported to every family in the school as a child with no class.
_UNKNOWN_REFERENCE = "unknown_reference"


class SisClassroomAdapter:
    """Reads one student's class, subject board and teachers from the SIS.

    Holds the same pooled client, the same refusal to retry and the same refusal to follow
    a redirect as its siblings — see [http.py](http.py). The guardian handle travels in the
    path, so SIS re-checks the link from the registrar's own data on this request: the
    decision is made twice, independently, and a fully compromised facade reaches one family
    rather than the school.
    """

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        api_key: str = "",
    ) -> None:
        """`timeout_seconds` is required, and this file reads no configuration.

        Handed in by the composition root that already reads the rest of this adapter's
        settings, so there is no second copy of a number to drift — the same choice the
        timetable adapter makes, and the layering suite whitelists which files may read
        configuration precisely so a new one has to make it deliberately.
        """
        if not base_url:
            raise RuntimeError("The SIS classroom adapter needs SIS_BASE_URL.")
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        headers = {"Accept": "application/json"}
        if api_key:
            headers["X-API-Key"] = api_key
        self._pool = PooledClient(
            base_url=base_url, timeout_seconds=self._timeout, headers=headers
        )

    def get_classroom(
        self, *, student_ref: str, term: str, guardian_ref: str = ""
    ) -> StudentClassroom:
        # Both segments quoted with no safe characters: a stray "/" in a school's numbering
        # — or in a handle — would otherwise rewrite the path and ask a different question.
        path = (
            f"/v1/guardians/by-id/{quote(guardian_ref, safe='')}"
            f"/students/{quote(student_ref, safe='')}/classroom"
        )
        payload = self._get(path, {"term": term})
        if payload is None:
            # No such student, not this guardian's, or no such term. Indistinguishable by
            # design — `records/` deliberately collapses those three, and a difference here
            # is the one signal an outsider would need to enumerate the school roll.
            return StudentClassroom(status=ClassroomStatus.NO_CLASS)
        return self._to_classroom(payload)

    # -- transport ----------------------------------------------------------

    def _get(self, path: str, params: dict[str, str]) -> dict | None:
        """One SIS call. Returns the body, or `None` for "nothing on file".

        Every other outcome — transport failure, a refused key, a body that is not the JSON
        object promised — becomes `ClassroomUnavailable`. Including refusals: a revoked key
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
            raise ClassroomUnavailable(f"{path}: transport failure — {exc}") from exc

        if response.status_code in REDIRECT_STATUSES:
            raise ClassroomUnavailable(
                f"{path}: SIS redirected to {response.headers.get('location')!r}. "
                "SIS_BASE_URL must name the service's own origin."
            )

        if response.status_code == 200:
            try:
                body = response.json()
            except ValueError as exc:
                raise ClassroomUnavailable(f"{path}: response was not JSON") from exc
            if not isinstance(body, dict):
                raise ClassroomUnavailable(f"{path}: response was not a JSON object")
            return body

        code = error_code(response)
        if response.status_code == 404 and code == _UNKNOWN_REFERENCE:
            return None

        logger.warning(
            "SIS refused %s: HTTP %s (%s)", path, response.status_code, code or "no code"
        )
        raise ClassroomUnavailable(
            f"{path}: HTTP {response.status_code} ({code or 'no code'})"
        )

    # -- reshaping ----------------------------------------------------------

    @staticmethod
    def _to_classroom(payload: dict) -> StudentClassroom:
        """SIS's `StudentClassroomOut` as this contract's own shape.

        The status is decided here and nowhere else. SIS states a null class for a child no
        placement covered, and empty lists for a room whose board or staffing nobody has
        entered; those are different answers, and this is the one place that turns the first
        into a word a consumer can branch on rather than an emptiness it has to interpret.
        The other two stay as emptiness deliberately — a room can lack either, both or
        neither, which no single status could say.
        """
        class_code = str(payload.get("class_code") or "")
        if not class_code:
            return StudentClassroom(status=ClassroomStatus.NO_CLASS)

        return StudentClassroom(
            status=ClassroomStatus.OK,
            class_code=class_code,
            class_name_ar=str(payload.get("class_name_ar") or ""),
            class_name_en=str(payload.get("class_name_en") or ""),
            year_level_code=str(payload.get("year_level_code") or ""),
            year_level_name_ar=str(payload.get("year_level_name_ar") or ""),
            year_level_name_en=str(payload.get("year_level_name_en") or ""),
            # Never re-sorted: SIS sent the school's own display order, and alphabetical
            # differs between the two scripts this estate renders.
            subjects=tuple(
                StudySubject(
                    code=str(row.get("code") or ""),
                    name_ar=str(row.get("name_ar") or ""),
                    name_en=str(row.get("name_en") or ""),
                )
                for row in payload.get("subjects") or []
            ),
            teachers=tuple(
                ClassTeacher(
                    full_name_ar=str(row.get("full_name_ar") or ""),
                    full_name_en=str(row.get("full_name_en") or ""),
                    subject_code=str(row.get("subject_code") or ""),
                    subject_name_ar=str(row.get("subject_name_ar") or ""),
                    subject_name_en=str(row.get("subject_name_en") or ""),
                )
                for row in payload.get("teachers") or []
            ),
        )

    def close(self) -> None:
        """Release the pooled client. Called from the app's shutdown hook."""
        self._pool.close()


__all__ = ["SisClassroomAdapter"]
