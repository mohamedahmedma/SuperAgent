"""Use cases: the rules of the school, written against ports and nothing else.

Every service here takes a `UnitOfWork` and calls Protocols. Nothing in this package
imports sqlalchemy, fastapi, pydantic or `sis.config`, and that constraint is the point
rather than a tidiness preference: it is what lets "re-running generation adds only the
missing rung" and "one bad row does not discard the good ones" be tested with dicts in
a millisecond. A single concrete import here would make every one of those tests need an
engine, a migration and a fixture, and the tests that assert rules would start asserting
SQL instead.

Services own the transaction boundary -- they enter the unit of work and commit once --
because they are what composes a request. Repositories never commit on their own.
"""
from sis.application.services.grade_import import GradeImportService
from sis.application.services.guardian_import import GuardianImportService
from sis.application.services.queries import (
    GuardianLink,
    QueryService,
    resolve_section_for_term,
    resolve_sections_for_term,
)
from sis.application.services.roster_import import RosterImportService
from sis.application.services.structure import StructureGenerationService

__all__ = [
    "GradeImportService",
    "GuardianImportService",
    "GuardianLink",
    "QueryService",
    "RosterImportService",
    "StructureGenerationService",
    "resolve_section_for_term",
    "resolve_sections_for_term",
]
