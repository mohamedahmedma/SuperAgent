"""End-of-year student promotion, previewed before one atomic commit."""
from __future__ import annotations

from collections import Counter
from datetime import date
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from sis.api.deps import Principal, UowFactoryDep, require_permission
from sis.domain.people import ClassEnrolment
from sis.domain.rbac import Permission
from sis.domain.structure import AcademicYearCode, ClassCode, SchoolCode, YearCode


router = APIRouter(prefix="/v1/promotions", tags=["student promotions"])
Manager = Annotated[Principal, Depends(require_permission(Permission.STUDENTS_WRITE))]
PromotionAction = Literal["promote", "repeat", "graduate", "exclude"]


class PromotionPreviewIn(BaseModel):
    source_year_code: str
    target_year_code: str


class PromotionRowOut(BaseModel):
    student_number: str
    full_name_en: str
    full_name_ar: str
    source_class_code: str
    source_year_level_code: str
    action: PromotionAction
    target_year_level_code: str | None = None
    target_class_code: str | None = None


class PromotionPreviewOut(BaseModel):
    source_year_code: str
    target_year_code: str
    target_starts_on: date
    rows: list[PromotionRowOut]
    target_classes: dict[str, list[dict[str, str]]]
    counts: dict[str, int]


class PromotionCommitRow(BaseModel):
    student_number: str
    action: PromotionAction
    target_class_code: str | None = None


class PromotionCommitIn(PromotionPreviewIn):
    rows: list[PromotionCommitRow] = Field(min_length=1)
    make_target_current: bool = True


class PromotionCommitOut(BaseModel):
    source_year_code: str
    target_year_code: str
    promoted: int = 0
    repeated: int = 0
    graduated: int = 0
    excluded: int = 0


def _refuse(code: str, message: str, field: str | None = None, status_code: int = 422):
    raise HTTPException(status_code=status_code, detail={"code": code, "message": message, "field": field})


def _context(uow, body, caller):  # noqa: ANN001
    source = uow.academic_years.get(AcademicYearCode(body.source_year_code))
    target = uow.academic_years.get(AcademicYearCode(body.target_year_code))
    if source is None:
        _refuse("unknown_reference", "Source academic year was not found.", "source_year_code", 404)
    if target is None:
        _refuse("unknown_reference", "Target academic year was not found.", "target_year_code", 404)
    if str(source.school_code) != str(target.school_code):
        _refuse("invalid_value", "Both academic years must belong to the same school.", "target_year_code")
    if target.starts_on <= source.ends_on:
        _refuse("invalid_value", "The target year must start after the source year ends.", "target_year_code")
    caller.narrow(Permission.STUDENTS_WRITE, lambda scopes: scopes.for_school(str(source.school_code)))
    levels = list(uow.year_levels.list_for_school(SchoolCode(source.school_code)))
    source_sections = list(uow.class_sections.list_for_year(AcademicYearCode(source.code)))
    target_sections = list(uow.class_sections.list_for_year(AcademicYearCode(target.code)))
    if not source_sections:
        _refuse("invalid_value", "The source year has no classes.", "source_year_code")
    if not target_sections:
        _refuse("invalid_value", "Create the target year's classes before promotion.", "target_year_code")
    return source, target, levels, source_sections, target_sections


def _target_map(target_sections):  # noqa: ANN001
    grouped: dict[str, list] = {}
    for section in target_sections:
        grouped.setdefault(str(section.year_level_code), []).append(section)
    for rows in grouped.values():
        rows.sort(key=lambda row: str(row.code))
    return grouped


@router.post("/preview", response_model=PromotionPreviewOut)
def preview_promotions(body: PromotionPreviewIn, caller: Manager, uow_factory: UowFactoryDep) -> PromotionPreviewOut:
    with uow_factory() as uow:
        source, target, levels, source_sections, target_sections = _context(uow, body, caller)
        level_codes = [str(level.code) for level in levels]
        target_by_level = _target_map(target_sections)
        source_by_level: dict[str, list] = {}
        for section in source_sections:
            source_by_level.setdefault(str(section.year_level_code), []).append(section)
        for rows in source_by_level.values():
            rows.sort(key=lambda row: str(row.code))

        enrolments = []
        source_for_student = {}
        for section in source_sections:
            rows = uow.enrolments.roster_on(
                AcademicYearCode(source.code), ClassCode(section.code), source.ends_on
            )
            enrolments.extend(rows)
            for row in rows:
                source_for_student[str(row.student_number)] = section
        students = uow.students.get_many([row.student_number for row in enrolments])

        result = []
        for enrolment in sorted(enrolments, key=lambda row: str(row.student_number)):
            number = str(enrolment.student_number)
            section = source_for_student[number]
            level_code = str(section.year_level_code)
            index = level_codes.index(level_code) if level_code in level_codes else len(level_codes) - 1
            next_level = level_codes[index + 1] if index + 1 < len(level_codes) else None
            action: PromotionAction = "promote" if next_level and target_by_level.get(next_level) else "graduate"
            targets = target_by_level.get(next_level or "", [])
            target_class = None
            if targets:
                source_peers = source_by_level.get(level_code, [])
                source_index = next((i for i, row in enumerate(source_peers) if str(row.code) == str(section.code)), 0)
                target_class = str(targets[min(source_index, len(targets) - 1)].code)
            student = students.get(number)
            result.append(PromotionRowOut(
                student_number=number,
                full_name_en="" if student is None else student.full_name_en,
                full_name_ar="" if student is None else student.full_name_ar,
                source_class_code=str(section.code),
                source_year_level_code=level_code,
                action=action,
                target_year_level_code=next_level if action == "promote" else None,
                target_class_code=target_class,
            ))
        counts = Counter(row.action for row in result)
        return PromotionPreviewOut(
            source_year_code=str(source.code), target_year_code=str(target.code),
            target_starts_on=target.starts_on, rows=result,
            target_classes={code: [{"code": str(row.code), "name_en": row.name_en, "name_ar": row.name_ar} for row in rows]
                            for code, rows in target_by_level.items()},
            counts={key: counts.get(key, 0) for key in ("promote", "repeat", "graduate", "exclude")},
        )


@router.post("/commit", response_model=PromotionCommitOut)
def commit_promotions(body: PromotionCommitIn, caller: Manager, uow_factory: UowFactoryDep) -> PromotionCommitOut:
    with uow_factory() as uow:
        source, target, levels, _source_sections, target_sections = _context(uow, body, caller)
        level_codes = [str(level.code) for level in levels]
        targets = {(str(row.code)): row for row in target_sections}
        counts = Counter()
        new_enrolments = []
        seen = set()
        for row in body.rows:
            if row.student_number in seen:
                _refuse("invalid_value", "A student may appear only once.", "rows")
            seen.add(row.student_number)
            if row.action == "exclude":
                counts["excluded"] += 1
                continue
            current = uow.enrolments.open_enrolment(row.student_number)
            if current is None or str(current.academic_year_code) != str(source.code):
                _refuse("changed_since_preview", f"Student {row.student_number} is no longer in the source year.", "rows", 409)
            source_section = uow.class_sections.get(AcademicYearCode(source.code), ClassCode(current.class_code))
            if source_section is None:
                _refuse("unknown_reference", f"The current class for {row.student_number} no longer exists.", "rows", 409)
            source_level = str(source_section.year_level_code)
            source_index = level_codes.index(source_level)
            if row.action == "graduate":
                if source_index != len(level_codes) - 1:
                    _refuse("invalid_value", f"{row.student_number} is not in the final grade.", "rows")
                uow.enrolments.close_open_enrolment(row.student_number, ends_on=source.ends_on)
                counts["graduated"] += 1
                continue
            target_section = targets.get(row.target_class_code or "")
            if target_section is None:
                _refuse("unknown_reference", f"Choose a valid target class for {row.student_number}.", "rows")
            expected_level = source_level if row.action == "repeat" else (
                level_codes[source_index + 1] if source_index + 1 < len(level_codes) else None
            )
            if expected_level is None or str(target_section.year_level_code) != expected_level:
                _refuse("invalid_value", f"The target class is not valid for {row.student_number}'s decision.", "rows")
            uow.enrolments.close_open_enrolment(row.student_number, ends_on=source.ends_on)
            new_enrolments.append(ClassEnrolment(
                student_number=row.student_number,
                academic_year_code=str(target.code),
                class_code=str(target_section.code),
                starts_on=target.starts_on,
            ))
            counts["promoted" if row.action == "promote" else "repeated"] += 1
        if new_enrolments:
            uow.enrolments.upsert_many(new_enrolments)
        if body.make_target_current:
            uow.academic_years.set_current(AcademicYearCode(target.code))
        uow.commit()
        return PromotionCommitOut(
            source_year_code=str(source.code), target_year_code=str(target.code),
            promoted=counts["promoted"], repeated=counts["repeated"],
            graduated=counts["graduated"], excluded=counts["excluded"],
        )
