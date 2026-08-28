"""The guardian links as SIS reports them.

**No cache, deliberately.** A guardian link can be revoked by a registrar the minute a
court order arrives, and a cached "yes" would keep letting somebody in for as long as the
entry lived. This is asked once per parent question, not once per page view.
"""
from __future__ import annotations

import logging
import threading
from typing import Final
from urllib.parse import quote

from records.adapters.sis.http import PooledClient, error_code
from records.config import settings
from records.domain.errors import GuardianDirectoryUnavailable
from records.domain.people import PermittedStudent

logger = logging.getLogger(__name__)


class SisGuardianDirectory:
    """The school's SIS, asked by the opaque handle the parent's token carries.

    Uses SIS's by-handle route rather than its by-phone one, so a parent's phone number
    never reaches this service, its logs, or the chat backend beyond it.

    No cache. A registrar revoking access the minute a court order arrives must take effect
    on the next question, and a cached "yes" would keep answering for as long as the entry
    lived. This is one request per parent question, not per page view.
    """

    #: Now from `records.config`; kept named for the docstring above.

    def __init__(
        self, *, base_url: str, api_key: str = "", timeout_seconds: float | None = None
    ) -> None:
        if not base_url:
            raise RuntimeError("The SIS guardian directory needs SIS_BASE_URL.")
        self._api_key = api_key
        self._timeout = timeout_seconds or settings().lookup_timeout_seconds
        self._pool = PooledClient(base_url=base_url, timeout_seconds=self._timeout)


    def children_of(
        self, guardian_id: str, *, school_code: str | None = None
    ) -> list[PermittedStudent]:
        import httpx

        if not guardian_id:
            return []
        path = f"/v1/guardians/by-id/{quote(guardian_id, safe='')}/students"
        headers = {"X-API-Key": self._api_key} if self._api_key else {}
        if school_code:
            # Schools are separated physically: this header picks the database SIS
            # answers from. Without it a split SIS refuses, which is the right failure —
            # far better than an unscoped read that silently reaches the wrong branch.
            headers["X-School-Code"] = school_code
        try:
            response = self._pool.get().get(path, headers=headers)
        except httpx.HTTPError as error:
            raise GuardianDirectoryUnavailable(
                f"The school's system of record could not be reached: {error}"
            ) from error

        if response.is_redirect:
            raise GuardianDirectoryUnavailable(
                f"The guardian lookup was redirected to "
                f"{response.headers.get('location')!r}. SIS_BASE_URL must name the SIS "
                f"service's own origin."
            )
        if response.status_code == 404 and error_code(response) == _UNKNOWN_REFERENCE:
            # No such guardian. Indistinguishable, to the caller, from a guardian with no
            # visible children — see `children_of` on the Protocol.
            return []
        if response.status_code >= 400:
            logger.warning(
                "guardian lookup refused: status=%s code=%s",
                response.status_code,
                error_code(response),
            )
            raise GuardianDirectoryUnavailable(
                f"The guardian lookup failed with status {response.status_code}."
            )

        try:
            rows = (response.json() or {}).get("students") or []
        except Exception as error:  # noqa: BLE001 - any unreadable body is one outcome
            raise GuardianDirectoryUnavailable(
                "The guardian lookup returned a body this service could not read."
            ) from error

        return [
            PermittedStudent(
                student_id=str(row.get("student_number") or ""),
                full_name_ar=str(row.get("full_name_ar") or ""),
                full_name_en=str(row.get("full_name_en") or ""),
                # Defaulted rather than required, so a SIS that predates the column — or
                # one that never fills it in — keeps answering this route instead of
                # failing it over a field nothing depends on.
                gender=str(row.get("gender") or "unspecified"),
                # SIS calls it `year_level`; this contract has always called it
                # `grade_level`, and the Moodle path fills the same field from its own
                # course binding. One name on the wire, whichever system answered.
                grade_level=str(row.get("year_level") or "").strip(),
            )
            for row in rows
            if row.get("student_number")
        ]

    def permits(self, guardian_id: str, student_id: str) -> PermittedStudent | None:
        """Answered from the list rather than by a second call.

        One request instead of two, and — more importantly — one source of truth for the
        turn: a caller cannot be told "yes, that child" by a lookup that disagrees with the
        list it was given a moment earlier.
        """
        wanted = str(student_id)
        return next(
            (child for child in self.children_of(guardian_id) if child.student_id == wanted),
            None,
        )


    def close(self) -> None:
        """Release the pooled client. Called from the app's shutdown hook."""
        self._pool.close()


__all__ = ["SisGuardianDirectory"]
