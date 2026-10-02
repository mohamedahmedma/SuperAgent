"""Which children a guardian may be told about, and which one this request is about.

The first of **two** checks, and no longer the authority. `sis/` makes the same decision
again, from the same data, before it answers — the guardian handle travels with the read.
This one still earns its place: it fails the request before a second service is troubled,
and it fetches the child's name and year group, which the response needs and which the
marks call does not return.

**The answer is asked for, never remembered.** This service holds no tables. A registrar
revoking access the minute a court order arrives takes effect on the next question rather
than whenever something here was next synchronised.

Lifted out of `records/auth.py`, where it sat beside the credential checks and could only
be exercised through an HTTP request. It raises domain errors rather than
`HTTPException`; `api/errors.py` decides the status once.
"""
from __future__ import annotations

import logging

from records.domain.errors import GuardianDirectoryUnavailable, StudentNotFound
from records.domain.people import PermittedStudent
from records.ports.directory import GuardianDirectory

logger = logging.getLogger(__name__)


class AccessService:
    """The link check, over whichever directory the deployment wired in."""

    def __init__(self, directory: GuardianDirectory) -> None:
        self._directory = directory

    def permitted_students(
        self, *, guardian_external_id: str, school_code: str | None = None
    ) -> list[PermittedStudent]:
        """Every child this guardian may ask about. Restricted links are already excluded.

        The filtering happens in `sis/` rather than here — it returns only links carrying
        `can_view_records` — so a barred parent arrives holding no children at all rather
        than holding children this service would have to remember to hide.

        An unreachable directory returns empty rather than raising, because the only
        caller is the "list my children" route and an empty list there renders as "no
        children on file", which is recoverable. `resolve` below raises properly.
        """
        try:
            return list(
                self._directory.children_of(
                    guardian_external_id, school_code=school_code
                )
            )
        except GuardianDirectoryUnavailable as error:
            logger.error(
                "Guardian directory unavailable while listing children: %s", error
            )
            return []

    def resolve(
        self,
        *,
        guardian_external_id: str,
        student_external_id: str,
        school_code: str | None = None,
    ) -> PermittedStudent:
        """The child, when the school says this guardian may be told about her.

        Note what stays deliberately indistinguishable from the caller's side: an unknown
        student, a student who exists but is not this guardian's, and a student whose
        records are restricted all raise the same `StudentNotFound` with the same message.
        **`sis/` records which one actually happened**; the response does not, because a
        caller who could tell them apart could enumerate the student body and detect
        custody restrictions by their error code alone.

        A directory that cannot be reached is the one case that is *not* a not-found.
        Telling a parent "no such child" because another service was briefly down is a lie
        about their own family, so it propagates as an outage and says so.
        """
        children = self._directory.children_of(
            guardian_external_id, school_code=school_code
        )
        wanted = str(student_external_id)
        for child in children:
            if child.student_id == wanted:
                return child
        raise StudentNotFound()


__all__ = ["AccessService"]
