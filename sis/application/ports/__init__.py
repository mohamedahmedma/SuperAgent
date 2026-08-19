"""Ports: the interfaces the use cases declare and the infrastructure fits.

Re-exported here so a service writes `from sis.application.ports import UnitOfWork` and
the module layout underneath stays free to change. The package had no `__init__.py` at
all, which on 3.12 leaves it an implicit namespace package — importable by luck rather
than by declaration, and silently unimportable once anything ships as a zip or a wheel.

Every name here is a `Protocol`. Nothing in this package may import sqlalchemy, fastapi,
pydantic or `sis.config`: these types are what a service depends on, and a fake in a unit
test satisfies them structurally without importing this module at all.
"""
from sis.application.ports.parsers import (
    GradeFileParser,
    GuardianFileParser,
    RosterFileParser,
)
from sis.application.ports.repositories import (
    AcademicYearRepository,
    ApiKeyRepository,
    ClassSectionKey,
    ClassSectionRepository,
    EnrolmentKey,
    EnrolmentRepository,
    GradeKey,
    GradeRepository,
    GuardianRepository,
    ImportBatchRepository,
    StudentGuardianKey,
    StudentGuardianRepository,
    StudentRepository,
    SubjectRepository,
    TermRepository,
    YearLevelRepository,
)
from sis.application.ports.unit_of_work import UnitOfWork

__all__ = [
    "AcademicYearRepository",
    "ApiKeyRepository",
    "ClassSectionKey",
    "ClassSectionRepository",
    "EnrolmentKey",
    "EnrolmentRepository",
    "GradeFileParser",
    "GradeKey",
    "GradeRepository",
    "GuardianFileParser",
    "GuardianRepository",
    "ImportBatchRepository",
    "RosterFileParser",
    "StudentGuardianKey",
    "StudentGuardianRepository",
    "StudentRepository",
    "SubjectRepository",
    "TermRepository",
    "UnitOfWork",
    "YearLevelRepository",
]
