"""Concrete repositories: the only place ORM rows and domain entities both appear.

Everything here implements a Protocol from `sis.application.ports.repositories` by
structural typing alone — nothing inherits from it, and nothing in `application/` imports
this package. That is the dependency inversion the architecture is built on: a service
takes the port, a request wires in one of these, and a unit test wires in a fake with no
session, no engine and no migration.

Each repository takes a `Session` and never commits. The transaction boundary belongs to
whoever composes the request, because a repository that commits on its own turns a failed
import commit into "half the roster landed, then a row failed".

Every class is named `SqlAlchemy<Port>`, and the prefix is load-bearing rather than
decorative. A concrete class named `AcademicYearRepository` — as these were — is spelled
identically to the Protocol it implements, so `from ... import AcademicYearRepository`
resolves to whichever module was imported last: a service annotated with the port and a
request wiring the implementation read the same in source and differ in meaning. The name
is the only thing distinguishing "the interface `application/` may depend on" from "the
SQLAlchemy class it must never see".

`bulk_upsert` is re-exported because the idempotent insert-or-update it performs is the
same shape for students, enrolments and grades, and a second hand-rolled copy is how one
of them ends up looping per row.
"""
from sis.infrastructure.repositories.access_audit_repository import (
    SqlAlchemyAccessAuditRepository,
)
from sis.infrastructure.repositories.api_key_repository import SqlAlchemyApiKeyRepository
from sis.infrastructure.repositories.grade_repository import SqlAlchemyGradeRepository
from sis.infrastructure.repositories.guardian_repository import (
    SqlAlchemyGuardianRepository,
    SqlAlchemyStudentGuardianRepository,
)
from sis.infrastructure.repositories.import_repository import (
    SqlAlchemyImportBatchRepository,
)
from sis.infrastructure.repositories.people_repository import (
    SqlAlchemyEnrolmentRepository,
    SqlAlchemyStudentRepository,
)
from sis.infrastructure.repositories.attendance_repository import (
    SqlAlchemyAttendanceRepository,
)
from sis.infrastructure.repositories.structure_repository import (
    SqlAlchemyAcademicYearRepository,
    SqlAlchemyClassSectionRepository,
    SqlAlchemySchoolRepository,
    SqlAlchemySubjectRepository,
    SqlAlchemyTimetableRepository,
    SqlAlchemyTermRepository,
    SqlAlchemyYearLevelRepository,
    bulk_upsert,
)
from sis.infrastructure.repositories.staff_repository import SqlAlchemyTeacherRepository

__all__ = [
    "SqlAlchemyAcademicYearRepository",
    "SqlAlchemyAccessAuditRepository",
    "SqlAlchemyAttendanceRepository",
    "SqlAlchemyApiKeyRepository",
    "SqlAlchemyClassSectionRepository",
    "SqlAlchemySchoolRepository",
    "SqlAlchemyEnrolmentRepository",
    "SqlAlchemyGradeRepository",
    "SqlAlchemyGuardianRepository",
    "SqlAlchemyImportBatchRepository",
    "SqlAlchemyStudentGuardianRepository",
    "SqlAlchemyStudentRepository",
    "SqlAlchemySubjectRepository",
    "SqlAlchemyTimetableRepository",
    "SqlAlchemyTeacherRepository",
    "SqlAlchemyTermRepository",
    "SqlAlchemyYearLevelRepository",
    "bulk_upsert",
]
