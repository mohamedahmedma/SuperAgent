"""A guardian directory held in a dict. The default when no SIS is configured.

Being the default matters: an unconfigured deployment tells every parent they have no
children on file, rather than authorising them against nothing.
"""
from __future__ import annotations

from records.domain.errors import GuardianDirectoryUnavailable
from records.domain.people import PermittedStudent


class FakeGuardianDirectory:
    """A directory in a dict, and the default when no SIS is configured.

    Being the default is the safe direction: an unconfigured deployment tells every parent
    they have no children on file, which is a support call. The alternative default — one
    that answered from stale local tables — would be a deployment quietly serving records
    from data nobody is maintaining.
    """

    def __init__(
        self,
        children: dict[str, list[PermittedStudent]] | None = None,
        *,
        unavailable: bool = False,
    ) -> None:
        self.children = dict(children or {})
        self.unavailable = unavailable

    def children_of(
        self, guardian_id: str, *, school_code: str | None = None
    ) -> list[PermittedStudent]:
        if self.unavailable:
            raise GuardianDirectoryUnavailable("The fake directory is switched off.")
        return list(self.children.get(guardian_id, ()))

    def permits(self, guardian_id: str, student_id: str) -> PermittedStudent | None:
        return next(
            (c for c in self.children_of(guardian_id) if c.student_id == str(student_id)),
            None,
        )


__all__ = ["FakeGuardianDirectory"]
