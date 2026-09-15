"""Private and automatically-scoped conversations for signed-in school staff.

Group membership is deliberately not stored.  It is derived from the active account,
role scopes and teaching assignments on every request, so removing a teacher or changing
their subject/class assignment removes access immediately without a second sync job.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import mimetypes
import os
from pathlib import Path
import re
from typing import Annotated, Literal
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select, update
from sqlalchemy.exc import IntegrityError

from sis.api.deps import UowFactoryDep, require_user_permission
from sis.api.errors import error_detail
from sis.domain.arabic import SEARCH_FOLDING, compact_for_search
from sis.domain.rbac import AccessProfile, Permission, RoleCode
from sis.infrastructure.db import models as m

router = APIRouter(prefix="/v1/schools/{school_code}/chat", tags=["staff chat"])
ChatReader = Annotated[AccessProfile, Depends(require_user_permission(Permission.CHAT_READ))]
ChatWriter = Annotated[AccessProfile, Depends(require_user_permission(Permission.CHAT_WRITE))]

MAX_MESSAGE_LENGTH = 4000
MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024
MAX_MESSAGE_ATTACHMENTS = 5
MAX_MESSAGE_UPLOAD_BYTES = 50 * 1024 * 1024
ONLINE_WINDOW_SECONDS = 45
TYPING_WINDOW_SECONDS = 6
ALLOWED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".txt", ".csv", ".zip",
    ".webm", ".ogg", ".mp3", ".m4a", ".wav", ".mp4",
}
SCHOOL_LEADER_ROLES = {
    RoleCode.SYSTEM_ADMIN.value,
    RoleCode.SCHOOL_OWNER.value,
    RoleCode.SCHOOL_MANAGER.value,
}
SCOPED_SUPERVISOR_ROLES = {
    RoleCode.FLOOR_SUPERVISOR.value,
    RoleCode.ATTENDANCE_SUPERVISOR.value,
}


def _refuse(code: str, message: str, status_code: int = 403) -> HTTPException:
    return HTTPException(status_code=status_code, detail=error_detail(code, message))


def _as_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class GroupSpec:
    key: str
    title_en: str
    title_ar: str
    category: Literal["school", "leadership", "floor", "subject", "class"]
    scope_code: str | None = None
    scope_name_en: str | None = None
    scope_name_ar: str | None = None


class NamedAssignmentOut(BaseModel):
    code: str
    name_en: str
    name_ar: str


class PersonOut(BaseModel):
    user_id: int
    username: str
    full_name_en: str
    full_name_ar: str
    roles: list[str]
    role_caption_en: str
    role_caption_ar: str
    subjects: list[NamedAssignmentOut] = Field(default_factory=list)
    grades: list[NamedAssignmentOut] = Field(default_factory=list)
    classes: list[NamedAssignmentOut] = Field(default_factory=list)
    online: bool = False
    last_seen_at: datetime | None = None


class PresenceMemberOut(PersonOut):
    typing: bool = False


class PresenceIn(BaseModel):
    conversation_id: int | None = Field(default=None, gt=0)
    typing: bool | None = None


class PresenceOut(BaseModel):
    online: bool = True
    delivered_messages: int = 0


class LastMessageOut(BaseModel):
    body: str
    attachment_kind: Literal["image", "file", "audio"] | None = None
    deleted: bool = False
    edited: bool = False
    sender_name_en: str
    sender_name_ar: str
    created_at: datetime


class ConversationOut(BaseModel):
    id: int
    kind: Literal["group", "direct"]
    category: Literal["school", "leadership", "floor", "subject", "class", "direct"]
    title_en: str
    title_ar: str
    scope_code: str | None = None
    scope_name_en: str | None = None
    scope_name_ar: str | None = None
    peer_user_id: int | None = None
    observer_view: bool = False
    subtitle_en: str | None = None
    subtitle_ar: str | None = None
    online: bool = False
    unread_count: int = 0
    is_muted: bool = False
    muted_until: datetime | None = None
    last_message: LastMessageOut | None = None
    updated_at: datetime


class MessageIn(BaseModel):
    body: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)


class ConversationMuteIn(BaseModel):
    duration: Literal["off", "eight_hours", "one_week", "always"]


class ConversationMuteOut(BaseModel):
    is_muted: bool
    muted_until: datetime | None = None


class AttachmentOut(BaseModel):
    id: str
    kind: Literal["image", "file", "audio"]
    original_filename: str
    mime_type: str
    size_bytes: int
    duration_seconds: int | None = None


class ReceiptSummaryOut(BaseModel):
    total: int = 0
    delivered: int = 0
    read: int = 0


class MessageOut(BaseModel):
    id: int
    conversation_id: int
    sender_user_id: int
    sender_name_en: str
    sender_name_ar: str
    body: str
    created_at: datetime
    deleted: bool = False
    deleted_at: datetime | None = None
    edited: bool = False
    edited_at: datetime | None = None
    original_body: str | None = None
    attachments: list[AttachmentOut] = Field(default_factory=list)
    receipts: ReceiptSummaryOut | None = None


class ReceiptPersonOut(BaseModel):
    user_id: int
    full_name_en: str
    full_name_ar: str
    delivered_at: datetime | None = None
    read_at: datetime | None = None


class DirectConversationIn(BaseModel):
    user_id: int = Field(gt=0)


def _school(session, school_code: str, profile: AccessProfile) -> m.School:  # noqa: ANN001
    school = session.scalar(select(m.School).where(m.School.code == school_code))
    if school is None or not school.is_active:
        raise _refuse("unknown_reference", "No active school with that code exists.", 404)
    if not profile.is_system_admin and profile.school_id != school.id:
        raise _refuse("not_authorized", "This account belongs to another school.")
    return school


def _roles_for(session, user_id: int) -> set[str]:  # noqa: ANN001
    return set(session.scalars(
        select(m.Role.code)
        .join(m.UserRole, m.UserRole.role_id == m.Role.id)
        .where(m.UserRole.user_id == user_id)
    ).all())


def _admin_user_ids():
    """A composable query for accounts that must be invisible to ordinary chat users."""
    return (
        select(m.UserRole.user_id)
        .join(m.Role, m.Role.id == m.UserRole.role_id)
        .where(m.Role.code == RoleCode.SYSTEM_ADMIN.value)
    )


def _is_admin_user(session, user_id: int | None) -> bool:  # noqa: ANN001
    if user_id is None:
        return False
    return session.scalar(
        select(m.UserRole.user_id)
        .join(m.Role, m.Role.id == m.UserRole.role_id)
        .where(
            m.UserRole.user_id == user_id,
            m.Role.code == RoleCode.SYSTEM_ADMIN.value,
        )
        .limit(1)
    ) is not None


def _presence_online(presence: m.ChatPresence | None, now: datetime | None = None) -> bool:
    if presence is None:
        return False
    current = now or datetime.now(UTC)
    return _timestamp(presence.last_seen_at) >= _timestamp(
        current - timedelta(seconds=ONLINE_WINDOW_SECONDS)
    )


def _named(row) -> NamedAssignmentOut:  # noqa: ANN001
    return NamedAssignmentOut(code=row.code, name_en=row.name_en or row.code, name_ar=row.name_ar or row.name_en or row.code)


def _staff_profile(session, school: m.School, user: m.User) -> PersonOut:  # noqa: ANN001
    roles = sorted(_roles_for(session, user.id))
    current_year = session.scalar(select(m.AcademicYear).where(
        m.AcademicYear.school_id == school.id,
        m.AcademicYear.is_current.is_(True),
    ).order_by(m.AcademicYear.starts_on.desc()).limit(1))
    teacher = session.scalar(select(m.Teacher).where(
        m.Teacher.school_id == school.id,
        m.Teacher.user_id == user.id,
        m.Teacher.is_active.is_(True),
    ).limit(1))

    subjects: list[m.Subject] = []
    grades: list[m.YearLevel] = []
    classes: list[m.ClassSection] = []
    if teacher is not None and current_year is not None:
        subjects = list(session.scalars(
            select(m.Subject).distinct()
            .join(m.TeacherSubject, m.TeacherSubject.subject_id == m.Subject.id)
            .where(
                m.TeacherSubject.teacher_id == teacher.id,
                m.Subject.academic_year_id == current_year.id,
            ).order_by(m.Subject.display_order, m.Subject.code)
        ).all())
        grades = list(session.scalars(
            select(m.YearLevel).distinct()
            .join(m.TeacherYearLevel, m.TeacherYearLevel.year_level_id == m.YearLevel.id)
            .join(m.Subject, m.Subject.id == m.TeacherYearLevel.subject_id)
            .where(
                m.TeacherYearLevel.teacher_id == teacher.id,
                m.Subject.academic_year_id == current_year.id,
            ).order_by(m.YearLevel.display_order, m.YearLevel.code)
        ).all())
        classes = list(session.scalars(
            select(m.ClassSection).distinct()
            .join(m.TeacherClassSection, m.TeacherClassSection.class_section_id == m.ClassSection.id)
            .where(
                m.TeacherClassSection.teacher_id == teacher.id,
                m.ClassSection.academic_year_id == current_year.id,
                m.ClassSection.is_active.is_(True),
            ).order_by(m.ClassSection.code)
        ).all())

    if RoleCode.SCHOOL_MANAGER.value in roles:
        caption_en, caption_ar = "School manager", "مدير المدرسة"
    elif RoleCode.SCHOOL_OWNER.value in roles:
        caption_en, caption_ar = "School owner", "مالك المدرسة"
    elif RoleCode.FLOOR_SUPERVISOR.value in roles:
        if not grades:
            grades = list(session.scalars(
                select(m.YearLevel).distinct()
                .join(m.UserRole, m.UserRole.scope_id == m.YearLevel.id)
                .join(m.Role, m.Role.id == m.UserRole.role_id)
                .where(
                    m.UserRole.user_id == user.id,
                    m.UserRole.scope_type == "year_level",
                    m.Role.code == RoleCode.FLOOR_SUPERVISOR.value,
                ).order_by(m.YearLevel.display_order, m.YearLevel.code)
            ).all())
        caption_en = "Grade supervisor" + (f" · {', '.join(row.name_en or row.code for row in grades)}" if grades else "")
        caption_ar = "مشرف صفوف" + (f" · {'، '.join(row.name_ar or row.name_en or row.code for row in grades)}" if grades else "")
    elif RoleCode.ATTENDANCE_SUPERVISOR.value in roles:
        caption_en, caption_ar = "Attendance supervisor", "مشرف الحضور والغياب"
    elif teacher is not None:
        subject_en = ", ".join(row.name_en or row.code for row in subjects)
        subject_ar = "، ".join(row.name_ar or row.name_en or row.code for row in subjects)
        grade_en = ", ".join(row.name_en or row.code for row in grades)
        grade_ar = "، ".join(row.name_ar or row.name_en or row.code for row in grades)
        caption_en = (f"{subject_en} teacher" if subject_en else "Teacher") + (f" · {grade_en}" if grade_en else "")
        caption_ar = (f"مدرس {subject_ar}" if subject_ar else "مدرس") + (f" · {grade_ar}" if grade_ar else "")
    else:
        caption_en, caption_ar = "School staff", "هيئة المدرسة"

    presence = session.get(m.ChatPresence, user.id)
    return PersonOut(
        user_id=user.id,
        username=user.username,
        full_name_en=_name(user, "en"),
        full_name_ar=_name(user, "ar"),
        roles=roles,
        role_caption_en=caption_en,
        role_caption_ar=caption_ar,
        subjects=[_named(row) for row in subjects],
        grades=[_named(row) for row in grades],
        classes=[_named(row) for row in classes],
        online=_presence_online(presence),
        last_seen_at=presence.last_seen_at if presence is not None else None,
    )


def _name(row: object, language: str) -> str:
    primary = getattr(row, f"full_name_{language}", "") or ""
    secondary = getattr(row, "full_name_ar" if language == "en" else "full_name_en", "") or ""
    return str(primary or secondary or getattr(row, "username", ""))


def _timestamp(value: datetime) -> float:
    """Sort SQLite's naive datetimes and Postgres' aware datetimes identically as UTC."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.timestamp()


def _compact_column(column):  # noqa: ANN001
    expression = column
    for written, matched in SEARCH_FOLDING:
        expression = func.replace(expression, written, matched)
    for spacing in (" ", "\t", "\r", "\n", "\u00a0"):
        expression = func.replace(expression, spacing, "")
    return expression


def _like_term(value: str) -> str:
    escaped = compact_for_search(value).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _storage_root() -> Path:
    return Path(os.getenv("SIS_CHAT_ATTACHMENT_STORAGE", "/app/data/chat_attachments"))


def _safe_filename(value: str) -> str:
    name = Path(value or "attachment").name
    name = re.sub(r'[\x00-\x1f<>:"/\\|?*]+', "_", name).strip(" .")
    return name[:255] or "attachment"


def _attachment_kind(extension: str, content_type: str) -> Literal["image", "file", "audio"]:
    if extension in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
        return "image"
    if extension in {".webm", ".ogg", ".mp3", ".m4a", ".wav", ".mp4"} and content_type.startswith("audio/"):
        return "audio"
    return "file"


def _group_specs(session, school: m.School, profile: AccessProfile) -> dict[str, GroupSpec]:  # noqa: ANN001
    specs = {
        "school": GroupSpec("school", "All school staff", "كل هيئة المدرسة", "school")
    }
    roles = _roles_for(session, profile.user_id)
    is_school_leader = bool(roles & SCHOOL_LEADER_ROLES)
    is_floor_supervisor = RoleCode.FLOOR_SUPERVISOR.value in roles
    if is_school_leader or is_floor_supervisor:
        specs["leadership:floor-supervisors"] = GroupSpec(
            "leadership:floor-supervisors",
            "School manager and floor supervisors",
            "مدير المدرسة ومشرفو الصفوف",
            "leadership",
        )

    current_year = session.scalar(
        select(m.AcademicYear)
        .where(m.AcademicYear.school_id == school.id, m.AcademicYear.is_current.is_(True))
        .order_by(m.AcademicYear.starts_on.desc())
        .limit(1)
    )
    if current_year is None:
        return specs

    teacher = session.scalar(
        select(m.Teacher).where(
            m.Teacher.user_id == profile.user_id,
            m.Teacher.school_id == school.id,
            m.Teacher.is_active.is_(True),
        ).limit(1)
    )

    school_levels = list(session.scalars(
        select(m.YearLevel).where(m.YearLevel.school_id == school.id)
    ).all())
    levels_by_id = {level.id: level for level in school_levels}
    level_ids: set[int] = set()
    managed_level_ids: set[int] = set()
    class_ids: set[int] = set()
    subject_memberships: dict[tuple[int, int], tuple[m.Subject, m.YearLevel]] = {}

    if teacher is not None:
        teaching_rows = session.execute(
            select(m.Subject, m.YearLevel)
            .join(m.TeacherYearLevel, m.TeacherYearLevel.subject_id == m.Subject.id)
            .join(m.YearLevel, m.YearLevel.id == m.TeacherYearLevel.year_level_id)
            .where(
                m.TeacherYearLevel.teacher_id == teacher.id,
                m.Subject.academic_year_id == current_year.id,
                m.YearLevel.school_id == school.id,
            )
        ).all()
        for subject, level in teaching_rows:
            level_ids.add(level.id)
            subject_memberships[(subject.id, level.id)] = (subject, level)

        class_ids.update(session.scalars(
            select(m.TeacherClassSection.class_section_id)
            .join(m.ClassSection, m.ClassSection.id == m.TeacherClassSection.class_section_id)
            .where(
                m.TeacherClassSection.teacher_id == teacher.id,
                m.ClassSection.academic_year_id == current_year.id,
                m.ClassSection.is_active.is_(True),
            )
        ).all())

    # Scoped supervisors may not also have a Teacher row. Attendance supervisors retain
    # their own floor/class membership, while only floor supervisors inherit every group
    # beneath the floor they manage.
    scoped_grants = session.execute(
        select(m.Role.code, m.UserRole.scope_type, m.UserRole.scope_id)
        .join(m.Role, m.UserRole.role_id == m.Role.id)
        .where(
            m.UserRole.user_id == profile.user_id,
            m.Role.code.in_(SCOPED_SUPERVISOR_ROLES),
        )
    ).all()
    for role_code, scope_type, scope_id in scoped_grants:
        if scope_type in {"global", "school"}:
            if role_code == RoleCode.FLOOR_SUPERVISOR.value:
                managed_level_ids.update(levels_by_id)
            continue
        if scope_type == "year_level":
            level = levels_by_id.get(scope_id)
            if level is not None:
                level_ids.add(level.id)
                if role_code == RoleCode.FLOOR_SUPERVISOR.value:
                    managed_level_ids.add(level.id)
        elif scope_type == "class_section":
            section = session.get(m.ClassSection, scope_id)
            if section is not None and section.academic_year_id == current_year.id:
                class_ids.add(section.id)
                level_ids.add(section.year_level_id)
                if role_code == RoleCode.FLOOR_SUPERVISOR.value:
                    managed_level_ids.add(section.year_level_id)

    if is_school_leader:
        managed_level_ids.update(levels_by_id)

    # Leadership sees every active class and subject group within its managed floors.
    # These are the same group keys the assigned teachers receive, so there is one shared
    # conversation rather than a supervisor copy of each conversation.
    if managed_level_ids:
        level_ids.update(managed_level_ids)
        class_ids.update(session.scalars(
            select(m.ClassSection.id).where(
                m.ClassSection.academic_year_id == current_year.id,
                m.ClassSection.year_level_id.in_(managed_level_ids),
                m.ClassSection.is_active.is_(True),
            )
        ).all())
        managed_subjects = session.execute(
            select(m.Subject, m.YearLevel)
            .join(m.SubjectYearLevel, m.SubjectYearLevel.subject_id == m.Subject.id)
            .join(m.YearLevel, m.YearLevel.id == m.SubjectYearLevel.year_level_id)
            .where(
                m.Subject.academic_year_id == current_year.id,
                m.Subject.is_active.is_(True),
                m.YearLevel.id.in_(managed_level_ids),
            )
        ).all()
        for subject, level in managed_subjects:
            subject_memberships[(subject.id, level.id)] = (subject, level)

    if class_ids:
        sections = session.scalars(
            select(m.ClassSection).where(m.ClassSection.id.in_(class_ids))
        ).all()
        for section in sections:
            # A class assignment is enough to establish the teacher's floor even if
            # older imported data is missing the redundant teacher-year-level row.
            level_ids.add(section.year_level_id)
            level = levels_by_id[section.year_level_id]
            specs[f"class:{section.id}"] = GroupSpec(
                f"class:{section.id}",
                f"Class {section.name_en or section.code} · {current_year.code}",
                f"فصل {section.name_ar or section.name_en or section.code} · {current_year.code}",
                "class",
                level.code,
                level.name_en or level.code,
                level.name_ar or level.name_en or level.code,
            )

    if level_ids:
        levels = session.scalars(select(m.YearLevel).where(m.YearLevel.id.in_(level_ids))).all()
        for level in levels:
            specs[f"floor:{current_year.id}:{level.id}"] = GroupSpec(
                f"floor:{current_year.id}:{level.id}",
                f"{level.name_en or level.code} staff · {current_year.code}",
                f"فريق {level.name_ar or level.name_en or level.code} · {current_year.code}",
                "floor",
                level.code,
                level.name_en or level.code,
                level.name_ar or level.name_en or level.code,
            )

    for subject, level in subject_memberships.values():
        key = f"subject:{subject.id}:level:{level.id}"
        specs[key] = GroupSpec(
            key,
            f"{subject.name_en or subject.code} teachers · {level.name_en or level.code}",
            f"مدرسو {subject.name_ar or subject.name_en or subject.code} · {level.name_ar or level.name_en or level.code}",
            "subject",
            level.code,
            level.name_en or level.code,
            level.name_ar or level.name_en or level.code,
        )
    return specs


def _ensure_group_conversations(session, school: m.School, specs: dict[str, GroupSpec]):  # noqa: ANN001
    existing = {
        row.group_key: row
        for row in session.scalars(select(m.ChatConversation).where(
            m.ChatConversation.school_id == school.id,
            m.ChatConversation.kind == "group",
            m.ChatConversation.group_key.in_(specs),
        )).all()
    }
    for key, spec in specs.items():
        conversation = existing.get(key)
        if conversation is None:
            candidate = m.ChatConversation(
                school_id=school.id, kind="group", group_key=key,
                title_en=spec.title_en, title_ar=spec.title_ar,
            )
            try:
                # Two staff opening chat for the first time can discover the same
                # automatic group together. The savepoint makes the unique-key loser
                # reread the winner instead of returning a server error.
                with session.begin_nested():
                    session.add(candidate)
                    session.flush([candidate])
                conversation = candidate
            except IntegrityError:
                conversation = session.scalar(select(m.ChatConversation).where(
                    m.ChatConversation.school_id == school.id,
                    m.ChatConversation.kind == "group",
                    m.ChatConversation.group_key == key,
                ))
                if conversation is None:  # pragma: no cover - defensive database failure
                    raise
            existing[key] = conversation
        else:
            conversation.title_en = spec.title_en
            conversation.title_ar = spec.title_ar
    session.flush()
    return list(existing.values())


def _authorised_conversation(
    session, school: m.School, profile: AccessProfile, conversation_id: int  # noqa: ANN001
) -> tuple[m.ChatConversation, dict[str, GroupSpec]]:
    conversation = session.scalar(select(m.ChatConversation).where(
        m.ChatConversation.id == conversation_id,
        m.ChatConversation.school_id == school.id,
    ))
    if conversation is None:
        raise _refuse("unknown_reference", "No conversation with that id exists.", 404)
    specs = _group_specs(session, school, profile)
    if conversation.kind == "group":
        if conversation.group_key not in specs:
            raise _refuse("not_authorized", "You are no longer a member of this group.")
    else:
        if (
            not profile.is_system_admin
            and profile.user_id not in {
                conversation.direct_user_one_id,
                conversation.direct_user_two_id,
            }
        ):
            raise _refuse("not_authorized", "This private conversation belongs to other staff.")
        peer_id = None
        if profile.user_id in {
            conversation.direct_user_one_id,
            conversation.direct_user_two_id,
        }:
            peer_id = (
                conversation.direct_user_two_id
                if conversation.direct_user_one_id == profile.user_id
                else conversation.direct_user_one_id
            )
        if not profile.is_system_admin and _is_admin_user(session, peer_id):
            # Return the same answer as a missing conversation so the protected account's
            # existence cannot be inferred by guessing a conversation id.
            raise _refuse("unknown_reference", "No conversation with that id exists.", 404)
    return conversation, specs


def _recipient_ids(
    session, school: m.School, conversation: m.ChatConversation, sender_user_id: int  # noqa: ANN001
) -> list[int]:
    if conversation.kind == "direct":
        return [user_id for user_id in (
            conversation.direct_user_one_id, conversation.direct_user_two_id
        ) if user_id is not None and user_id != sender_user_id]

    # Group membership remains derived, not copied into a membership table. At send
    # time we snapshot the current recipients so later staff changes do not rewrite
    # the historical meaning of delivered/read counts.
    eligible = session.scalars(
        select(m.User).distinct()
        .join(m.UserRole, m.UserRole.user_id == m.User.id)
        .join(m.RolePermission, m.RolePermission.role_id == m.UserRole.role_id)
        .join(m.PermissionRow, m.PermissionRow.id == m.RolePermission.permission_id)
        .where(
            m.User.school_id == school.id,
            m.User.is_active.is_(True),
            m.User.id != sender_user_id,
            ~m.User.id.in_(_admin_user_ids()),
            m.PermissionRow.code == Permission.CHAT_READ.value,
            ~m.User.id.in_(select(m.Teacher.user_id).where(
                m.Teacher.user_id.is_not(None), m.Teacher.is_active.is_(False)
            )),
        )
    ).all()
    if conversation.group_key == "school":
        return [user.id for user in eligible]
    if conversation.group_key == "leadership:floor-supervisors":
        leadership_ids = set(session.scalars(
            select(m.UserRole.user_id)
            .join(m.Role, m.Role.id == m.UserRole.role_id)
            .where(
                m.UserRole.user_id.in_([user.id for user in eligible]),
                m.Role.code.in_((*SCHOOL_LEADER_ROLES, RoleCode.FLOOR_SUPERVISOR.value)),
            )
        ).all())
        return [user.id for user in eligible if user.id in leadership_ids]
    return [
        user.id for user in eligible
        if conversation.group_key in _group_specs(
            session, school, AccessProfile(user_id=user.id, username=user.username, school_id=school.id)
        )
    ]


def _add_receipts(
    session, school: m.School, conversation: m.ChatConversation, message: m.ChatMessage  # noqa: ANN001
) -> None:
    # Admin activity is intentionally invisible to school accounts. Do not create a
    # delivery trail that could surface through presence counters or read receipts.
    if _is_admin_user(session, message.sender_user_id):
        return
    now = datetime.now(UTC)
    online_ids = set(session.scalars(
        select(m.ChatPresence.user_id).where(
            m.ChatPresence.school_id == school.id,
            m.ChatPresence.last_seen_at >= now - timedelta(seconds=ONLINE_WINDOW_SECONDS),
        )
    ).all())
    session.add_all([
        m.ChatReceipt(
            message_id=message.id,
            user_id=user_id,
            delivered_at=now if user_id in online_ids else None,
        )
        for user_id in _recipient_ids(session, school, conversation, message.sender_user_id)
    ])


def _message_out(
    session, message: m.ChatMessage, sender: m.User, viewer: AccessProfile  # noqa: ANN001
) -> MessageOut:
    reveal_deleted = message.deleted_at is None or viewer.is_system_admin
    attachments = []
    if reveal_deleted:
        attachments = session.scalars(
            select(m.ChatAttachment).where(m.ChatAttachment.message_id == message.id)
            .order_by(m.ChatAttachment.created_at, m.ChatAttachment.id)
        ).all()
    summary = None
    if message.sender_user_id == viewer.user_id:
        receipt_rows = session.scalars(
            select(m.ChatReceipt).where(m.ChatReceipt.message_id == message.id)
        ).all()
        summary = ReceiptSummaryOut(
            total=len(receipt_rows),
            delivered=sum(row.delivered_at is not None for row in receipt_rows),
            read=sum(row.read_at is not None for row in receipt_rows),
        )
    return MessageOut(
        id=message.id,
        conversation_id=message.conversation_id,
        sender_user_id=message.sender_user_id,
        sender_name_en=_name(sender, "en"),
        sender_name_ar=_name(sender, "ar"),
        body=message.body if reveal_deleted else "",
        created_at=_as_utc(message.created_at),
        deleted=message.deleted_at is not None,
        deleted_at=_as_utc(message.deleted_at),
        edited=message.edited_at is not None,
        edited_at=_as_utc(message.edited_at),
        original_body=message.original_body if viewer.is_system_admin else None,
        attachments=[AttachmentOut.model_validate(row, from_attributes=True) for row in attachments],
        receipts=summary,
    )


def _conversation_out(
    session, conversation: m.ChatConversation, profile: AccessProfile, specs: dict[str, GroupSpec]  # noqa: ANN001
) -> ConversationOut:
    category = "direct"
    title_en, title_ar = conversation.title_en, conversation.title_ar
    scope_code = scope_name_en = scope_name_ar = None
    peer_user_id = None
    subtitle_en = subtitle_ar = None
    online = False
    observer_view = False
    if conversation.kind == "group":
        spec = specs[conversation.group_key]
        category, title_en, title_ar = spec.category, spec.title_en, spec.title_ar
        scope_code = spec.scope_code
        scope_name_en = spec.scope_name_en
        scope_name_ar = spec.scope_name_ar
    else:
        participant_ids = {
            conversation.direct_user_one_id,
            conversation.direct_user_two_id,
        }
        observer_view = profile.is_system_admin and profile.user_id not in participant_ids
        if observer_view:
            participants = {
                user.id: user for user in session.scalars(
                    select(m.User).where(m.User.id.in_(participant_ids))
                ).all()
            }
            first = participants.get(conversation.direct_user_one_id)
            second = participants.get(conversation.direct_user_two_id)
            if first is not None and second is not None:
                title_en = f"{_name(first, 'en')} / {_name(second, 'en')}"
                title_ar = f"{_name(first, 'ar')} / {_name(second, 'ar')}"
            subtitle_en = "Private conversation"
            subtitle_ar = "\u0645\u062d\u0627\u062f\u062b\u0629 \u062e\u0627\u0635\u0629"
        else:
            other_id = (
                conversation.direct_user_two_id
                if conversation.direct_user_one_id == profile.user_id
                else conversation.direct_user_one_id
            )
            other = session.get(m.User, other_id)
            if other is not None:
                title_en, title_ar = _name(other, "en"), _name(other, "ar")
                conversation_school = session.get(m.School, conversation.school_id)
                if conversation_school is not None:
                    peer = _staff_profile(session, conversation_school, other)
                    peer_user_id = other.id
                    subtitle_en, subtitle_ar = peer.role_caption_en, peer.role_caption_ar
                    online = peer.online

    last_statement = (
        select(m.ChatMessage, m.User)
        .join(m.User, m.User.id == m.ChatMessage.sender_user_id)
        .where(m.ChatMessage.conversation_id == conversation.id)
    )
    if not profile.is_system_admin:
        last_statement = last_statement.where(~m.ChatMessage.sender_user_id.in_(_admin_user_ids()))
    last = session.execute(last_statement.order_by(m.ChatMessage.id.desc()).limit(1)).first()
    read_id = session.scalar(select(m.ChatRead.last_read_message_id).where(
        m.ChatRead.conversation_id == conversation.id,
        m.ChatRead.user_id == profile.user_id,
    )) or 0
    unread_statement = select(func.count(m.ChatMessage.id)).where(
        m.ChatMessage.conversation_id == conversation.id,
        m.ChatMessage.id > read_id,
        m.ChatMessage.sender_user_id != profile.user_id,
    )
    if not profile.is_system_admin:
        unread_statement = unread_statement.where(
            ~m.ChatMessage.sender_user_id.in_(_admin_user_ids())
        )
    unread = session.scalar(unread_statement) or 0
    last_message, last_sender = last if last else (None, None)
    last_attachment_kind = None
    if last_message is not None:
        if last_message.deleted_at is None or profile.is_system_admin:
            last_attachment_kind = session.scalar(
                select(m.ChatAttachment.kind).where(m.ChatAttachment.message_id == last_message.id).limit(1)
            )
    preference = session.get(m.ChatConversationPreference, (conversation.id, profile.user_id))
    now = datetime.now(UTC)
    is_muted = bool(preference and preference.muted and (
        preference.muted_until is None
        or _timestamp(preference.muted_until) > _timestamp(now)
    ))
    return ConversationOut(
        id=conversation.id,
        kind=conversation.kind,
        category=category,
        title_en=title_en,
        title_ar=title_ar,
        scope_code=scope_code,
        scope_name_en=scope_name_en,
        scope_name_ar=scope_name_ar,
        peer_user_id=peer_user_id,
        observer_view=observer_view,
        subtitle_en=subtitle_en,
        subtitle_ar=subtitle_ar,
        online=online,
        unread_count=unread,
        is_muted=is_muted,
        muted_until=(_as_utc(preference.muted_until) if is_muted and preference else None),
        last_message=(LastMessageOut(
            body=(
                last_message.body
                if last_message.deleted_at is None or profile.is_system_admin
                else ""
            ),
            attachment_kind=last_attachment_kind,
            deleted=last_message.deleted_at is not None,
            edited=last_message.edited_at is not None,
            sender_name_en=_name(last_sender, "en"),
            sender_name_ar=_name(last_sender, "ar"),
            created_at=_as_utc(last_message.created_at),
        ) if last_message is not None else None),
        updated_at=_as_utc(conversation.updated_at),
    )


@router.post("/presence", response_model=PresenceOut)
def heartbeat_presence(
    school_code: str,
    body: PresenceIn,
    profile: ChatReader,
    uow_factory: UowFactoryDep,
) -> PresenceOut:
    """Keep availability current and acknowledge every message delivered to this browser."""
    with uow_factory() as uow:
        session = uow._session
        school = _school(session, school_code, profile)
        if body.typing is True and body.conversation_id is None:
            raise _refuse("invalid_value", "Typing requires a conversation id.", 422)
        if body.conversation_id is not None:
            _authorised_conversation(session, school, profile, body.conversation_id)

        now = datetime.now(UTC)
        presence = session.get(m.ChatPresence, profile.user_id)
        if presence is None:
            candidate = m.ChatPresence(
                user_id=profile.user_id,
                school_id=school.id,
                last_seen_at=now,
            )
            try:
                with session.begin_nested():
                    session.add(candidate)
                    session.flush([candidate])
                presence = candidate
            except IntegrityError:
                presence = session.get(m.ChatPresence, profile.user_id)
                if presence is None:  # pragma: no cover - defensive database failure
                    raise
        presence.school_id = school.id
        presence.last_seen_at = now
        if body.typing is not None:
            if body.typing:
                presence.typing_conversation_id = body.conversation_id
                presence.typing_until = now + timedelta(seconds=TYPING_WINDOW_SECONDS)
            elif body.conversation_id is None or presence.typing_conversation_id == body.conversation_id:
                presence.typing_conversation_id = None
                presence.typing_until = None

        delivery_conditions = [
            m.ChatReceipt.user_id == profile.user_id,
            m.ChatReceipt.delivered_at.is_(None),
        ]
        if not profile.is_system_admin:
            visible_message_ids = select(m.ChatMessage.id).where(
                ~m.ChatMessage.sender_user_id.in_(_admin_user_ids())
            )
            delivery_conditions.append(m.ChatReceipt.message_id.in_(visible_message_ids))
        delivered = session.execute(
            update(m.ChatReceipt)
            .where(*delivery_conditions)
            .values(delivered_at=now)
        ).rowcount or 0
        uow.commit()
        return PresenceOut(delivered_messages=delivered)


@router.get("/conversations/{conversation_id}/members", response_model=list[PresenceMemberOut])
def conversation_members(
    school_code: str,
    conversation_id: int,
    profile: ChatReader,
    uow_factory: UowFactoryDep,
) -> list[PresenceMemberOut]:
    with uow_factory() as uow:
        session = uow._session
        school = _school(session, school_code, profile)
        conversation, _ = _authorised_conversation(session, school, profile, conversation_id)
        member_ids = [
            user_id for user_id in _recipient_ids(session, school, conversation, -1)
            if user_id != profile.user_id
        ]
        users = session.scalars(
            select(m.User).where(m.User.id.in_(member_ids)).order_by(m.User.full_name_en, m.User.username)
        ).all() if member_ids else []
        now = datetime.now(UTC)
        result: list[PresenceMemberOut] = []
        for user in users:
            person = _staff_profile(session, school, user)
            presence = session.get(m.ChatPresence, user.id)
            result.append(PresenceMemberOut(
                **person.model_dump(),
                typing=bool(
                    presence is not None
                    and presence.typing_conversation_id == conversation.id
                    and presence.typing_until is not None
                    and _timestamp(presence.typing_until) > _timestamp(now)
                ),
            ))
        return result


@router.get("/conversations", response_model=list[ConversationOut])
def list_conversations(
    school_code: str, profile: ChatReader, uow_factory: UowFactoryDep
) -> list[ConversationOut]:
    with uow_factory() as uow:
        school = _school(uow._session, school_code, profile)
        specs = _group_specs(uow._session, school, profile)
        groups = _ensure_group_conversations(uow._session, school, specs)
        direct_statement = select(m.ChatConversation).where(
            m.ChatConversation.school_id == school.id,
            m.ChatConversation.kind == "direct",
        )
        if not profile.is_system_admin:
            direct_statement = direct_statement.where(
                or_(
                    m.ChatConversation.direct_user_one_id == profile.user_id,
                    m.ChatConversation.direct_user_two_id == profile.user_id,
                ),
                ~m.ChatConversation.direct_user_one_id.in_(_admin_user_ids()),
                ~m.ChatConversation.direct_user_two_id.in_(_admin_user_ids()),
            )
        # Admin deliberately receives the school's complete private-chat index without
        # being added as a participant or creating delivery/read receipts.
        directs = uow._session.scalars(direct_statement).all()
        output = [_conversation_out(uow._session, row, profile, specs) for row in [*groups, *directs]]
        uow.commit()
    return sorted(
        output,
        key=lambda item: (item.last_message is not None, _timestamp(item.updated_at)),
        reverse=True,
    )


@router.get("/people", response_model=list[PersonOut])
def search_people(
    school_code: str,
    profile: ChatReader,
    uow_factory: UowFactoryDep,
    q: Annotated[str, Query(min_length=1, max_length=100)],
) -> list[PersonOut]:
    with uow_factory() as uow:
        school = _school(uow._session, school_code, profile)
        needle = _like_term(q)
        statement = select(m.User).where(
                m.User.school_id == school.id,
                m.User.is_active.is_(True),
                m.User.id != profile.user_id,
                or_(
                    _compact_column(m.User.username).ilike(needle, escape="\\"),
                    _compact_column(m.User.full_name_en).ilike(needle, escape="\\"),
                    _compact_column(m.User.full_name_ar).ilike(needle, escape="\\"),
                ),
                ~m.User.id.in_(select(m.Teacher.user_id).where(
                    m.Teacher.user_id.is_not(None), m.Teacher.is_active.is_(False)
                )),
            )
        if not profile.is_system_admin:
            statement = statement.where(~m.User.id.in_(_admin_user_ids()))
        users = uow._session.scalars(
            statement.order_by(m.User.full_name_en, m.User.username).limit(30)
        ).all()
        result = [_staff_profile(uow._session, school, user) for user in users]
    return result


@router.post("/direct", response_model=ConversationOut, status_code=status.HTTP_201_CREATED)
def open_direct_conversation(
    school_code: str, body: DirectConversationIn, profile: ChatWriter, uow_factory: UowFactoryDep
) -> ConversationOut:
    if body.user_id == profile.user_id:
        raise _refuse("invalid_value", "Choose another staff member.", 422)
    with uow_factory() as uow:
        school = _school(uow._session, school_code, profile)
        other = uow._session.get(m.User, body.user_id)
        if other is None or not other.is_active or other.school_id != school.id:
            raise _refuse("unknown_reference", "No active staff account with that id exists.", 404)
        if not profile.is_system_admin and _is_admin_user(uow._session, other.id):
            raise _refuse("unknown_reference", "No active staff account with that id exists.", 404)
        inactive_teacher = uow._session.scalar(select(m.Teacher.id).where(
            m.Teacher.user_id == other.id, m.Teacher.is_active.is_(False)
        ))
        if inactive_teacher is not None:
            raise _refuse("unknown_reference", "That staff account is no longer active.", 404)
        first, second = sorted((profile.user_id, other.id))
        conversation = uow._session.scalar(select(m.ChatConversation).where(
            m.ChatConversation.school_id == school.id,
            m.ChatConversation.kind == "direct",
            m.ChatConversation.direct_user_one_id == first,
            m.ChatConversation.direct_user_two_id == second,
        ))
        if conversation is None:
            candidate = m.ChatConversation(
                school_id=school.id, kind="direct",
                direct_user_one_id=first, direct_user_two_id=second,
            )
            try:
                with uow._session.begin_nested():
                    uow._session.add(candidate)
                    uow._session.flush([candidate])
                conversation = candidate
            except IntegrityError:
                conversation = uow._session.scalar(select(m.ChatConversation).where(
                    m.ChatConversation.school_id == school.id,
                    m.ChatConversation.kind == "direct",
                    m.ChatConversation.direct_user_one_id == first,
                    m.ChatConversation.direct_user_two_id == second,
                ))
                if conversation is None:  # pragma: no cover - defensive database failure
                    raise
        specs = _group_specs(uow._session, school, profile)
        result = _conversation_out(uow._session, conversation, profile, specs)
        uow.commit()
    return result


@router.get("/conversations/{conversation_id}/messages", response_model=list[MessageOut])
def list_messages(
    school_code: str,
    conversation_id: int,
    profile: ChatReader,
    uow_factory: UowFactoryDep,
    before_id: Annotated[int | None, Query(gt=0)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> list[MessageOut]:
    with uow_factory() as uow:
        school = _school(uow._session, school_code, profile)
        _authorised_conversation(uow._session, school, profile, conversation_id)
        statement = (
            select(m.ChatMessage, m.User)
            .join(m.User, m.User.id == m.ChatMessage.sender_user_id)
            .where(m.ChatMessage.conversation_id == conversation_id)
        )
        if not profile.is_system_admin:
            statement = statement.where(~m.ChatMessage.sender_user_id.in_(_admin_user_ids()))
        if before_id is not None:
            statement = statement.where(m.ChatMessage.id < before_id)
        rows = uow._session.execute(statement.order_by(m.ChatMessage.id.desc()).limit(limit)).all()
        message_ids = [message.id for message, _ in rows]
        if message_ids:
            now = datetime.now(UTC)
            uow._session.execute(
                update(m.ChatReceipt)
                .where(
                    m.ChatReceipt.message_id.in_(message_ids),
                    m.ChatReceipt.user_id == profile.user_id,
                    m.ChatReceipt.delivered_at.is_(None),
                )
                .values(delivered_at=now)
            )
            uow._session.flush()
        # Materialise the response while the ORM rows are still attached. The unit of
        # work closes its session on exit and SQLAlchemy expires loaded attributes there.
        result = [
            _message_out(uow._session, message, sender, profile)
            for message, sender in reversed(rows)
        ]
        uow.commit()
    return result


@router.post(
    "/conversations/{conversation_id}/messages",
    response_model=MessageOut,
    status_code=status.HTTP_201_CREATED,
)
def send_message(
    school_code: str,
    conversation_id: int,
    body: MessageIn,
    profile: ChatWriter,
    uow_factory: UowFactoryDep,
) -> MessageOut:
    text = body.body.strip()
    if not text:
        raise _refuse("invalid_value", "A message cannot be blank.", 422)
    with uow_factory() as uow:
        school = _school(uow._session, school_code, profile)
        conversation, _ = _authorised_conversation(
            uow._session, school, profile, conversation_id
        )
        sender = uow._session.get(m.User, profile.user_id)
        if sender is None or not sender.is_active:
            raise _refuse("not_authorized", "This account is no longer active.")
        message = m.ChatMessage(
            conversation_id=conversation.id,
            sender_user_id=profile.user_id,
            body=text,
        )
        now = datetime.now(UTC)
        conversation.updated_at = now
        uow._session.add(message)
        uow._session.flush()
        _add_receipts(uow._session, school, conversation, message)
        uow._session.flush()
        result = _message_out(uow._session, message, sender, profile)
        uow.commit()
    return result


@router.delete(
    "/conversations/{conversation_id}/messages/{message_id}",
    response_model=MessageOut,
)
def delete_message(
    school_code: str,
    conversation_id: int,
    message_id: int,
    profile: ChatWriter,
    uow_factory: UowFactoryDep,
) -> MessageOut:
    """Hide a sender's own message while retaining its original audit record for Admin."""
    with uow_factory() as uow:
        school = _school(uow._session, school_code, profile)
        conversation, _ = _authorised_conversation(
            uow._session, school, profile, conversation_id
        )
        row = uow._session.execute(
            select(m.ChatMessage, m.User)
            .join(m.User, m.User.id == m.ChatMessage.sender_user_id)
            .where(
                m.ChatMessage.id == message_id,
                m.ChatMessage.conversation_id == conversation.id,
            )
        ).first()
        if row is None:
            raise _refuse("unknown_reference", "No message with that id exists.", 404)
        message, sender = row
        if message.sender_user_id != profile.user_id:
            raise _refuse("not_authorized", "Only the sender may delete this message.")
        if message.deleted_at is None:
            now = datetime.now(UTC)
            message.deleted_at = now
            conversation.updated_at = now
            uow._session.flush()
        result = _message_out(uow._session, message, sender, profile)
        uow.commit()
    return result


EDIT_WINDOW_SECONDS = 3600  # one hour


@router.put(
    "/conversations/{conversation_id}/messages/{message_id}",
    response_model=MessageOut,
)
def edit_message(
    school_code: str,
    conversation_id: int,
    message_id: int,
    body: MessageIn,
    profile: ChatWriter,
    uow_factory: UowFactoryDep,
) -> MessageOut:
    """Replace the text of a sender's own message within the edit window."""
    new_text = body.body.strip()
    if not new_text:
        raise _refuse("invalid_value", "A message cannot be blank.", 422)
    with uow_factory() as uow:
        school = _school(uow._session, school_code, profile)
        conversation, _ = _authorised_conversation(
            uow._session, school, profile, conversation_id
        )
        row = uow._session.execute(
            select(m.ChatMessage, m.User)
            .join(m.User, m.User.id == m.ChatMessage.sender_user_id)
            .where(
                m.ChatMessage.id == message_id,
                m.ChatMessage.conversation_id == conversation.id,
            )
        ).first()
        if row is None:
            raise _refuse("unknown_reference", "No message with that id exists.", 404)
        message, sender = row
        if message.sender_user_id != profile.user_id:
            raise _refuse("not_authorized", "Only the sender may edit this message.")
        if message.deleted_at is not None:
            raise _refuse("not_authorized", "A deleted message cannot be edited.")
        now = datetime.now(UTC)
        created_at = (
            message.created_at
            if message.created_at.tzinfo is not None
            else message.created_at.replace(tzinfo=UTC)
        )
        created_at = _as_utc(message.created_at)
        age = (now - created_at).total_seconds()
        if age > EDIT_WINDOW_SECONDS:
            raise _refuse(
                "edit_window_expired",
                "Messages can only be edited within one hour of sending.",
            )
        # Preserve the original body on the first edit for admin audit.
        if message.original_body is None:
            message.original_body = message.body
        message.body = new_text
        message.edited_at = now
        conversation.updated_at = now
        uow._session.flush()
        result = _message_out(uow._session, message, sender, profile)
        uow.commit()
    return result


@router.post(
    "/conversations/{conversation_id}/attachments",
    response_model=MessageOut,
    status_code=status.HTTP_201_CREATED,
)
async def send_attachments(
    school_code: str,
    conversation_id: int,
    profile: ChatWriter,
    uow_factory: UowFactoryDep,
    files: Annotated[list[UploadFile], File()],
    body: Annotated[str, Form(max_length=MAX_MESSAGE_LENGTH)] = "",
    duration_seconds: Annotated[int | None, Form(ge=1, le=60)] = None,
) -> MessageOut:
    if not files or len(files) > MAX_MESSAGE_ATTACHMENTS:
        raise _refuse("invalid_value", f"Attach between 1 and {MAX_MESSAGE_ATTACHMENTS} files.", 422)

    prepared: list[tuple[str, str, str, bytes, str]] = []
    total = 0
    for upload in files:
        original = _safe_filename(upload.filename or "attachment")
        extension = Path(original).suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            raise _refuse("unsupported_file", f"{original}: this file type is not allowed.", 415)
        blob = await upload.read(MAX_ATTACHMENT_BYTES + 1)
        if not blob:
            raise _refuse("empty_file", f"{original}: the selected file is empty.", 422)
        if len(blob) > MAX_ATTACHMENT_BYTES:
            raise _refuse("upload_too_large", f"{original}: each attachment must be 20 MB or smaller.", 413)
        total += len(blob)
        if total > MAX_MESSAGE_UPLOAD_BYTES:
            raise _refuse("upload_too_large", "Attachments in one message must total 50 MB or less.", 413)
        claimed = (upload.content_type or "").lower()
        guessed = mimetypes.guess_type(original)[0] or "application/octet-stream"
        kind = _attachment_kind(extension, claimed)
        mime_type = claimed if kind == "audio" and claimed.startswith("audio/") else guessed
        stored = uuid.uuid4().hex + extension
        prepared.append((original, stored, mime_type, blob, kind))

    text = body.strip()
    root = _storage_root()
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    try:
        with uow_factory() as uow:
            school = _school(uow._session, school_code, profile)
            conversation, _ = _authorised_conversation(
                uow._session, school, profile, conversation_id
            )
            sender = uow._session.get(m.User, profile.user_id)
            if sender is None or not sender.is_active:
                raise _refuse("not_authorized", "This account is no longer active.")
            message = m.ChatMessage(
                conversation_id=conversation.id,
                sender_user_id=profile.user_id,
                body=text,
            )
            conversation.updated_at = datetime.now(UTC)
            uow._session.add(message)
            uow._session.flush()
            for original, stored, mime_type, blob, kind in prepared:
                temporary = root / f".{stored}"
                destination = root / stored
                temporary.write_bytes(blob)
                temporary.replace(destination)
                written.append(destination)
                uow._session.add(m.ChatAttachment(
                    id=str(uuid.uuid4()), message_id=message.id, kind=kind,
                    file_key=stored, original_filename=original, mime_type=mime_type,
                    size_bytes=len(blob),
                    duration_seconds=duration_seconds if kind == "audio" else None,
                ))
            _add_receipts(uow._session, school, conversation, message)
            uow._session.flush()
            result = _message_out(uow._session, message, sender, profile)
            uow.commit()
        return result
    except Exception:
        for path in written:
            path.unlink(missing_ok=True)
        raise


@router.get("/attachments/{attachment_id}/file")
def download_attachment(
    school_code: str,
    attachment_id: str,
    profile: ChatReader,
    uow_factory: UowFactoryDep,
):
    with uow_factory() as uow:
        school = _school(uow._session, school_code, profile)
        row = uow._session.execute(
            select(m.ChatAttachment, m.ChatMessage)
            .join(m.ChatMessage, m.ChatMessage.id == m.ChatAttachment.message_id)
            .where(m.ChatAttachment.id == attachment_id)
        ).first()
        if row is None:
            raise _refuse("unknown_reference", "No attachment with that id exists.", 404)
        attachment, message = row
        _authorised_conversation(uow._session, school, profile, message.conversation_id)
        if not profile.is_system_admin and _is_admin_user(uow._session, message.sender_user_id):
            raise _refuse("unknown_reference", "No attachment with that id exists.", 404)
        if message.deleted_at is not None and not profile.is_system_admin:
            raise _refuse("unknown_reference", "No attachment with that id exists.", 404)
        path = _storage_root() / attachment.file_key
        if not path.is_file():
            raise _refuse("file_missing", "The stored attachment is unavailable.", 404)
        disposition = "inline" if attachment.kind in {"image", "audio"} else "attachment"
        return FileResponse(
            path,
            media_type=attachment.mime_type,
            filename=attachment.original_filename,
            content_disposition_type=disposition,
            headers={"X-Content-Type-Options": "nosniff"},
        )


@router.get("/messages/{message_id}/receipts", response_model=list[ReceiptPersonOut])
def message_receipts(
    school_code: str,
    message_id: int,
    profile: ChatReader,
    uow_factory: UowFactoryDep,
) -> list[ReceiptPersonOut]:
    with uow_factory() as uow:
        school = _school(uow._session, school_code, profile)
        message = uow._session.get(m.ChatMessage, message_id)
        if message is None:
            raise _refuse("unknown_reference", "No message with that id exists.", 404)
        _authorised_conversation(uow._session, school, profile, message.conversation_id)
        if message.sender_user_id != profile.user_id:
            raise _refuse("not_authorized", "Only the sender can view message receipts.")
        receipt_statement = (
            select(m.ChatReceipt, m.User)
            .join(m.User, m.User.id == m.ChatReceipt.user_id)
            .where(m.ChatReceipt.message_id == message.id)
        )
        if not profile.is_system_admin:
            receipt_statement = receipt_statement.where(~m.User.id.in_(_admin_user_ids()))
        rows = uow._session.execute(
            receipt_statement.order_by(m.User.full_name_en, m.User.username)
        ).all()
        return [ReceiptPersonOut(
            user_id=user.id,
            full_name_en=_name(user, "en"),
            full_name_ar=_name(user, "ar"),
            delivered_at=receipt.delivered_at,
            read_at=receipt.read_at,
        ) for receipt, user in rows]


@router.post("/conversations/{conversation_id}/read", status_code=status.HTTP_204_NO_CONTENT)
def mark_read(
    school_code: str,
    conversation_id: int,
    profile: ChatReader,
    uow_factory: UowFactoryDep,
) -> None:
    with uow_factory() as uow:
        school = _school(uow._session, school_code, profile)
        _authorised_conversation(uow._session, school, profile, conversation_id)
        latest = uow._session.scalar(select(func.max(m.ChatMessage.id)).where(
            m.ChatMessage.conversation_id == conversation_id
        ))
        marker = uow._session.get(m.ChatRead, (conversation_id, profile.user_id))
        if marker is None:
            candidate = m.ChatRead(conversation_id=conversation_id, user_id=profile.user_id)
            try:
                with uow._session.begin_nested():
                    uow._session.add(candidate)
                    uow._session.flush([candidate])
                marker = candidate
            except IntegrityError:
                marker = uow._session.get(m.ChatRead, (conversation_id, profile.user_id))
                if marker is None:  # pragma: no cover - defensive database failure
                    raise
        marker.last_read_message_id = latest
        now = datetime.now(UTC)
        marker.read_at = now
        if latest is not None:
            message_ids = select(m.ChatMessage.id).where(
                m.ChatMessage.conversation_id == conversation_id,
                m.ChatMessage.id <= latest,
            )
            uow._session.execute(
                update(m.ChatReceipt)
                .where(
                    m.ChatReceipt.message_id.in_(message_ids),
                    m.ChatReceipt.user_id == profile.user_id,
                    m.ChatReceipt.read_at.is_(None),
                )
                .values(delivered_at=func.coalesce(m.ChatReceipt.delivered_at, now), read_at=now)
            )
        uow.commit()


@router.put(
    "/conversations/{conversation_id}/mute",
    response_model=ConversationMuteOut,
)
def set_conversation_mute(
    school_code: str,
    conversation_id: int,
    body: ConversationMuteIn,
    profile: ChatReader,
    uow_factory: UowFactoryDep,
) -> ConversationMuteOut:
    """Mute notifications for this user only; unread state remains untouched."""
    with uow_factory() as uow:
        school = _school(uow._session, school_code, profile)
        _authorised_conversation(uow._session, school, profile, conversation_id)
        preference = uow._session.get(
            m.ChatConversationPreference, (conversation_id, profile.user_id)
        )
        if preference is None:
            preference = m.ChatConversationPreference(
                conversation_id=conversation_id,
                user_id=profile.user_id,
            )
            uow._session.add(preference)

        now = datetime.now(UTC)
        preference.muted = body.duration != "off"
        preference.muted_until = {
            "eight_hours": now + timedelta(hours=8),
            "one_week": now + timedelta(days=7),
        }.get(body.duration)
        preference.updated_at = now
        result = ConversationMuteOut(
            is_muted=preference.muted,
            muted_until=preference.muted_until,
        )
        uow.commit()
    return result
