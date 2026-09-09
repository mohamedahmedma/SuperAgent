"""Data transfer objects: the shapes that cross a layer boundary.

Dataclasses, never pydantic. A pydantic model here would mean the services are written
against request-validation types, and the first consequence is a unit test that has to
build a valid API payload to call a use case. The second is that the wire format and the
internal shape become the same object, so adding a field for the UI silently changes what
a service accepts.

Re-exported here so callers import from `sis.application.dto` and the module layout
underneath stays free to change.
"""
from sis.application.dto.common import (
    ImportCommitResult,
    ImportPreviewResult,
    Page,
    PageRequest,
    RowCode,
    RowOutcome,
    tally_by_code,
)
from sis.application.dto.grades import (
    GradeCommitCommand,
    GradePreviewCommand,
    ParsedGradeRow,
)
from sis.application.dto.guardians import (
    GuardianCommitCommand,
    GuardianPreviewCommand,
    ParsedGuardianRow,
)
from sis.application.dto.parsing import ParseResult
from sis.application.dto.roster import (
    ParsedRosterRow,
    RosterCommitCommand,
    RosterPreviewCommand,
)
from sis.application.dto.structure import (
    TERM_LABELS,
    GeneratedItem,
    GenerateStructureCommand,
    GenerateStructureResult,
    TermPlan,
    term_code_for,
)

__all__ = [
    "TERM_LABELS",
    "GenerateStructureCommand",
    "GenerateStructureResult",
    "GeneratedItem",
    "GradeCommitCommand",
    "GradePreviewCommand",
    "GuardianCommitCommand",
    "GuardianPreviewCommand",
    "ImportCommitResult",
    "ImportPreviewResult",
    "Page",
    "PageRequest",
    "ParseResult",
    "ParsedGradeRow",
    "ParsedGuardianRow",
    "ParsedRosterRow",
    "RosterCommitCommand",
    "RosterPreviewCommand",
    "RowCode",
    "RowOutcome",
    "TermPlan",
    "tally_by_code",
    "term_code_for",
]
