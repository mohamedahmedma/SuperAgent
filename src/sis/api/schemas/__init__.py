"""The wire contract. Pydantic lives here and in no other package of this service.

Everything below mirrors an `application/dto` shape without being one. The DTOs are what
the services are unit-tested against with fake repositories and no database; these are what
a browser and the `records/` adapter are promised. Keeping them apart is what lets a use
case gain an argument without it appearing in OpenAPI, and what keeps a rename of a wire
field from silently rewriting a service signature.

Re-exported here so routes import from `sis.api.schemas` and the module split underneath
stays free to change.
"""
from sis.api.schemas.common import (
    CodeStr,
    ErrorDetail,
    ErrorResponse,
    ImportCommitResponse,
    ImportPreviewResponse,
    ImportRowsPage,
    NamedOut,
    PageParams,
    PageResponse,
    RequestModel,
    ResponseModel,
    RowCode,
    RowOutcomeOut,
)
from sis.api.schemas.grades import (
    GradeCommitRequest,
    GradePreviewRequest,
    StudentGradesResponse,
    SubjectGradeOut,
)
from sis.api.schemas.roster import (
    ClassEnrolmentOut,
    ClassRosterEntryOut,
    ClassRosterPage,
    ClassRosterResponse,
    RosterCommitRequest,
    RosterPreviewRequest,
    StudentOut,
)
from sis.api.schemas.structure import (
    AcademicYearOut,
    ClassSectionOut,
    GenerateStructureRequest,
    GenerateStructureResponse,
    GeneratedItemOut,
    SubjectOut,
    TermOut,
    YearLevelOut,
)

__all__ = [
    "AcademicYearOut",
    "ClassEnrolmentOut",
    "ClassRosterEntryOut",
    "ClassRosterPage",
    "ClassRosterResponse",
    "ClassSectionOut",
    "CodeStr",
    "ErrorDetail",
    "ErrorResponse",
    "GenerateStructureRequest",
    "GenerateStructureResponse",
    "GeneratedItemOut",
    "GradeCommitRequest",
    "GradePreviewRequest",
    "ImportCommitResponse",
    "ImportPreviewResponse",
    "ImportRowsPage",
    "NamedOut",
    "PageParams",
    "PageResponse",
    "RequestModel",
    "ResponseModel",
    "RosterCommitRequest",
    "RosterPreviewRequest",
    "RowCode",
    "RowOutcomeOut",
    "StudentGradesResponse",
    "StudentOut",
    "SubjectGradeOut",
    "SubjectOut",
    "TermOut",
    "YearLevelOut",
]
