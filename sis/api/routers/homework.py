from __future__ import annotations

import hashlib
import io
import mimetypes
import os
import re
import uuid
import zipfile
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Annotated
from xml.etree import ElementTree as ET

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import text

from sis.api.deps import Principal, SchoolCodeDep, get_teaching_service, require_permission
from sis.application.services.teaching import TeachingService
from sis.domain.rbac import Permission
from sis.infrastructure.db.unit_of_work import SqlAlchemyUnitOfWork

router = APIRouter(prefix="/v1/homework", tags=["homework"])

TeacherReader = Annotated[Principal, Depends(require_permission(Permission.GRADES_READ))]
TeacherWriter = Annotated[Principal, Depends(require_permission(Permission.GRADES_WRITE))]
HomeworkRestorer = Annotated[Principal, Depends(require_permission(Permission.SYSTEM_MANAGE))]
Teaching = Annotated[TeachingService, Depends(get_teaching_service)]

DATA_ROOT = Path(os.getenv("SIS_HOMEWORK_STORAGE", "/app/data/homework_files"))
MAX_BYTES = 20 * 1024 * 1024
ALLOWED = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
}
WORD_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


class HomeworkOut(BaseModel):
    id: str
    academic_year_code: str
    class_code: str
    subject_code: str
    title: str
    details: str
    original_filename: str | None = None
    mime_type: str | None = None
    size_bytes: int | None = None
    uploaded_on: date
    uploaded_at: datetime
    uploaded_by: str
    extracted_text: str = ""
    download_url: str | None = None


class HomeworkListOut(BaseModel):
    assignments: list[HomeworkOut]


class StudentHomeworkOut(BaseModel):
    student_number: str
    on_date: date
    class_codes: list[str]
    assignments: list[HomeworkOut]


def _teacher_only(caller: Principal) -> None:
    if caller.profile is None or not caller.profile.has_role("teacher"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "teacher_only", "message": "Homework publishing is available to teachers only."},
        )


def _safe_name(name: str) -> str:
    raw = Path(name or "attachment").name
    raw = re.sub(r"[^A-Za-z0-9._ -]+", "_", raw).strip(" .")
    return raw[:180] or "attachment"


def _extract_docx(blob: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            xml = zf.read("word/document.xml")
        root = ET.fromstring(xml)
        chunks = [node.text or "" for node in root.findall(".//w:t", WORD_NS)]
        return " ".join(x.strip() for x in chunks if x.strip())[:120000]
    except Exception:
        return ""


def _extract_pdf(blob: bytes) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(blob))
        parts = []
        for page in reader.pages[:100]:
            value = page.extract_text() or ""
            if value.strip():
                parts.append(value.strip())
        return "\n\n".join(parts)[:120000]
    except Exception:
        return ""


def _extract(ext: str, blob: bytes) -> str:
    if ext == ".pdf":
        return _extract_pdf(blob)
    if ext == ".docx":
        return _extract_docx(blob)
    return ""


def _row(row) -> HomeworkOut:
    m = row._mapping
    return HomeworkOut(
        id=m["id"],
        academic_year_code=m["academic_year_code"],
        class_code=m["class_code"],
        subject_code=m["subject_code"],
        title=m["title"],
        details=m["details"] or "",
        original_filename=m["original_filename"],
        mime_type=m["mime_type"],
        size_bytes=int(m["size_bytes"]) if m["size_bytes"] is not None else None,
        uploaded_on=date.fromisoformat(str(m["uploaded_on"])),
        uploaded_at=datetime.fromisoformat(str(m["uploaded_at"]).replace("Z", "+00:00")),
        uploaded_by=m["uploaded_by"],
        extracted_text=m["extracted_text"] or "",
        download_url=f"/v1/homework/{m['id']}/file" if m["stored_filename"] else None,
    )


@router.get("", response_model=HomeworkListOut)
def list_teacher_homework(
    caller: TeacherReader,
    school_code: SchoolCodeDep,
    academic_year: Annotated[str, Query()],
) -> HomeworkListOut:
    _teacher_only(caller)
    with SqlAlchemyUnitOfWork(school_code=school_code) as uow:
        rows = uow._session.execute(
            text(
                "SELECT * FROM teacher_homework "
                "WHERE academic_year_code=:y AND uploaded_by=:u AND deleted_at IS NULL "
                "ORDER BY uploaded_at DESC, id DESC"
            ),
            {"y": academic_year, "u": caller.username},
        ).all()
    return HomeworkListOut(assignments=[_row(r) for r in rows])


@router.post("", response_model=HomeworkOut, status_code=201)
async def upload_homework(
    caller: TeacherWriter,
    teaching: Teaching,
    school_code: SchoolCodeDep,
    academic_year: Annotated[str, Form()],
    class_code: Annotated[str, Form()],
    subject_code: Annotated[str, Form()],
    title: Annotated[str, Form()],
    details: Annotated[str, Form()] = "",
    attachment: UploadFile | None = File(None),
) -> HomeworkOut:
    _teacher_only(caller)
    title = title.strip()
    details = details.strip()
    if not title:
        raise HTTPException(422, detail={"code": "required", "field": "title", "message": "Homework title is required."})

    caller.narrow(
        Permission.GRADES_WRITE,
        lambda scopes: scopes.for_class(academic_year_code=academic_year, class_code=class_code),
    )
    if caller.profile is None or not teaching.may_record_by_code(
        caller.profile.user_id,
        academic_year_code=academic_year,
        class_code=class_code,
        subject_code=subject_code,
    ):
        raise HTTPException(
            403,
            detail={"code": "not_authorized", "message": "You may publish homework only for your own subject and class."},
        )

    file_id = str(uuid.uuid4())

    original = None
    stored = None
    mime = None
    size_bytes = None
    sha256 = None
    extracted = ""

    # An attachment is optional. A teacher may publish a text-only homework item.
    if attachment is not None and attachment.filename:
        original = _safe_name(attachment.filename)
        ext = Path(original).suffix.lower()
        if ext not in ALLOWED:
            raise HTTPException(
                415,
                detail={"code": "unsupported_file", "message": "Supported files: PDF, PNG, JPG, WEBP, DOCX and DOC."},
            )

        blob = await attachment.read(MAX_BYTES + 1)
        if len(blob) > MAX_BYTES:
            raise HTTPException(
                413,
                detail={"code": "too_large", "message": "Homework attachment must be 20 MB or smaller."},
            )
        if not blob:
            raise HTTPException(
                422,
                detail={"code": "empty_file", "message": "The selected file is empty."},
            )

        size_bytes = len(blob)
        sha256 = hashlib.sha256(blob).hexdigest()
        mime = attachment.content_type or ALLOWED[ext] or mimetypes.guess_type(original)[0] or "application/octet-stream"
        extracted = _extract(ext, blob)

        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        stored = DATA_ROOT / f"{file_id}{ext}"
        temp = DATA_ROOT / f".{file_id}.tmp"
        temp.write_bytes(blob)
        temp.replace(stored)

    now = datetime.now(ZoneInfo("Africa/Cairo"))
    try:
        with SqlAlchemyUnitOfWork(school_code=school_code) as uow:
            uow._session.execute(
                text(
                    "INSERT INTO teacher_homework("
                    "id,academic_year_code,class_code,subject_code,title,details,"
                    "original_filename,stored_filename,mime_type,size_bytes,sha256,"
                    "uploaded_on,uploaded_at,uploaded_by,extracted_text"
                    ") VALUES("
                    ":id,:y,:c,:s,:t,:d,:o,:stored,:m,:size,:sha,:day,:at,:by,:x)"
                ),
                {
                    "id": file_id, "y": academic_year, "c": class_code, "s": subject_code,
                    "t": title, "d": details, "o": original,
                    "stored": stored.name if stored is not None else None,
                    "m": mime, "size": size_bytes, "sha": sha256,
                    "day": now.date().isoformat(), "at": now.isoformat(), "by": caller.username,
                    "x": extracted,
                },
            )
            uow.commit()
            row = uow._session.execute(
                text("SELECT * FROM teacher_homework WHERE id=:id"), {"id": file_id}
            ).one()
    except Exception:
        if stored is not None:
            stored.unlink(missing_ok=True)
        raise
    return _row(row)


@router.delete("/{homework_id}", status_code=204)
def delete_homework(
    homework_id: str,
    caller: TeacherWriter,
    school_code: SchoolCodeDep,
):
    _teacher_only(caller)
    with SqlAlchemyUnitOfWork(school_code=school_code) as uow:
        row = uow._session.execute(
            text("SELECT * FROM teacher_homework WHERE id=:id AND deleted_at IS NULL"), {"id": homework_id}
        ).first()
        if row is None:
            raise HTTPException(404, detail={"code": "not_found", "message": "Homework item not found."})
        if row._mapping["uploaded_by"] != caller.username:
            raise HTTPException(403, detail={"code": "not_authorized", "message": "You may delete only homework you uploaded."})
        uow._session.execute(
            text("UPDATE teacher_homework SET deleted_at=:now WHERE id=:id"),
            {"id": homework_id, "now": datetime.now(ZoneInfo("Africa/Cairo")).isoformat()},
        )
        uow.commit()
    return None


@router.post("/{homework_id}/restore", response_model=HomeworkOut)
def restore_homework(
    homework_id: str,
    caller: HomeworkRestorer,
    school_code: SchoolCodeDep,
) -> HomeworkOut:
    """Restore an archived homework record without changing its original evidence."""
    with SqlAlchemyUnitOfWork(school_code=school_code) as uow:
        row = uow._session.execute(
            text("SELECT * FROM teacher_homework WHERE id=:id AND deleted_at IS NOT NULL"),
            {"id": homework_id},
        ).first()
        if row is None:
            raise HTTPException(404, detail={"code": "not_found", "message": "Archived homework item not found."})
        uow._session.execute(
            text("UPDATE teacher_homework SET deleted_at=NULL WHERE id=:id"),
            {"id": homework_id},
        )
        uow.commit()
        restored = uow._session.execute(
            text("SELECT * FROM teacher_homework WHERE id=:id"), {"id": homework_id}
        ).one()
    return _row(restored)


@router.get("/student/{student_number}", response_model=StudentHomeworkOut)
def homework_for_student(
    student_number: str,
    school_code: SchoolCodeDep,
    on_date: Annotated[date, Query()],
) -> StudentHomeworkOut:
    # Integration endpoint intentionally uses the SIS integration door.
    # The parent-facing service resolves the guardian/student relationship before asking SIS.
    with SqlAlchemyUnitOfWork(school_code=school_code) as uow:
        session = uow._session
        classes = session.execute(
            text(
                "SELECT DISTINCT cs.code "
                "FROM students s "
                "JOIN class_enrolments e ON e.student_id=s.id "
                "JOIN class_sections cs ON cs.id=e.class_section_id "
                "WHERE s.student_number=:n "
                "AND date(e.starts_on)<=date(:d) "
                "AND (e.ends_on IS NULL OR date(e.ends_on)>=date(:d))"
            ),
            {"n": student_number, "d": on_date.isoformat()},
        ).scalars().all()

        rows = []
        if classes:
            holders = ",".join(f":c{i}" for i in range(len(classes)))
            params = {"d": on_date.isoformat()}
            params.update({f"c{i}": code for i, code in enumerate(classes)})
            rows = session.execute(
                text(
                    "SELECT * FROM teacher_homework "
                    f"WHERE class_code IN ({holders}) AND uploaded_on=:d AND deleted_at IS NULL "
                    "ORDER BY subject_code,title,uploaded_at"
                ),
                params,
            ).all()

    return StudentHomeworkOut(
        student_number=student_number,
        on_date=on_date,
        class_codes=list(classes),
        assignments=[_row(r) for r in rows],
    )

@router.get("/{homework_id}/file")
def download_homework_file(
    homework_id: str,
    caller: TeacherReader,
    school_code: SchoolCodeDep,
):
    with SqlAlchemyUnitOfWork(school_code=school_code) as uow:
        row = uow._session.execute(
            text("SELECT * FROM teacher_homework WHERE id=:id AND deleted_at IS NULL"), {"id": homework_id}
        ).first()
    if row is None:
        raise HTTPException(404, detail={"code": "not_found", "message": "Homework item not found."})
    m = row._mapping
    # Holding grades.read somewhere does not make every classroom's documents visible.
    # The same scope boundary used for a marksheet also protects its attachment.
    caller.narrow(
        Permission.GRADES_READ,
        lambda scopes: scopes.for_class(
            academic_year_code=m["academic_year_code"], class_code=m["class_code"]
        ),
    )
    stored_filename = m["stored_filename"]
    if not stored_filename:
        raise HTTPException(
            404,
            detail={"code": "no_attachment", "message": "This homework item has no attachment."},
        )
    path = DATA_ROOT / stored_filename
    if not path.is_file():
        raise HTTPException(404, detail={"code": "file_missing", "message": "Homework file is no longer available."})
    return FileResponse(path, media_type=m["mime_type"], filename=m["original_filename"])
