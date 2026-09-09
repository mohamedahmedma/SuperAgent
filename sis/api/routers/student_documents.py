"""Authenticated, soft-deletable files attached to one student record."""
from __future__ import annotations
import os, re, uuid
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Annotated
from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import select
from sis.api.deps import Principal, UowFactoryDep, require_permission
from sis.domain.rbac import Permission
from sis.infrastructure.db import models as m

router = APIRouter(prefix="/v1/students/{student_number}/documents", tags=["student documents"])
Reader = Annotated[Principal, Depends(require_permission(Permission.DOCUMENTS_READ))]
Writer = Annotated[Principal, Depends(require_permission(Permission.DOCUMENTS_WRITE))]
ROOT = Path(os.getenv("SIS_STUDENT_DOCUMENT_STORAGE", "/app/data/student_documents"))
MAX_BYTES = 20 * 1024 * 1024
ALLOWED = {".pdf": "application/pdf", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}

class DocumentOut(BaseModel):
    id: str; document_type: str; original_filename: str; mime_type: str; size_bytes: int
    uploaded_by: str; uploaded_at: datetime; expiry_date: date | None; status: str
    created_at: datetime; updated_at: datetime; deleted_at: datetime | None = None
class DocumentMetadataIn(BaseModel):
    document_type: str | None = None
    expiry_date: date | None = None
    status: str | None = None

def _student(session, number: str):
    row = session.scalar(select(m.Student).where(m.Student.student_number == number))
    if row is None: raise HTTPException(404, detail={"code":"unknown_reference", "message":"No such student."})
    return row
def _row(row): return DocumentOut.model_validate(row, from_attributes=True)
def _name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._ -]+", "_", Path(value or "document").name).strip(" .")[:255] or "document"

@router.get("", response_model=list[DocumentOut])
def list_documents(student_number: str, caller: Reader, uow_factory: UowFactoryDep):
    with uow_factory() as uow:
        student = _student(uow._session, student_number)
        rows = uow._session.scalars(select(m.StudentDocument).where(m.StudentDocument.student_id == student.id, m.StudentDocument.deleted_at.is_(None)).order_by(m.StudentDocument.uploaded_at.desc())).all()
    return [_row(row) for row in rows]

@router.post("", response_model=DocumentOut, status_code=201)
async def upload_document(student_number: str, caller: Writer, uow_factory: UowFactoryDep, document_type: Annotated[str, Form(min_length=1, max_length=64)], file: UploadFile = File(...), expiry_date: Annotated[date | None, Form()] = None):
    original = _name(file.filename or "")
    extension = Path(original).suffix.lower()
    if extension not in ALLOWED: raise HTTPException(415, detail={"code":"unsupported_file", "message":"Only PDF, PNG, and JPEG files are allowed."})
    blob = await file.read(MAX_BYTES + 1)
    if not blob: raise HTTPException(422, detail={"code":"empty_file", "message":"The selected file is empty."})
    if len(blob) > MAX_BYTES: raise HTTPException(413, detail={"code":"upload_too_large", "message":"Student documents must be 20 MB or smaller."})
    file_id, file_key = str(uuid.uuid4()), uuid.uuid4().hex + extension
    ROOT.mkdir(parents=True, exist_ok=True); temporary = ROOT / ("." + file_key); destination = ROOT / file_key
    temporary.write_bytes(blob); temporary.replace(destination)
    try:
        with uow_factory() as uow:
            student = _student(uow._session, student_number)
            now = datetime.now(UTC)
            row = m.StudentDocument(id=file_id, student_id=student.id, document_type=document_type.strip(), file_key=file_key, original_filename=original, mime_type=ALLOWED[extension], size_bytes=len(blob), uploaded_by=caller.username, uploaded_at=now, expiry_date=expiry_date, status="unverified")
            uow._session.add(row); uow.commit(); return _row(row)
    except Exception:
        destination.unlink(missing_ok=True); raise

@router.delete("/{document_id}", status_code=204)
def delete_document(student_number: str, document_id: str, caller: Writer, uow_factory: UowFactoryDep):
    with uow_factory() as uow:
        student = _student(uow._session, student_number)
        row = uow._session.scalar(select(m.StudentDocument).where(m.StudentDocument.id == document_id, m.StudentDocument.student_id == student.id, m.StudentDocument.deleted_at.is_(None)))
        if row is None: raise HTTPException(404, detail={"code":"unknown_reference", "message":"No active document found."})
        row.deleted_at = datetime.now(UTC); uow.commit()

@router.patch("/{document_id}", response_model=DocumentOut)
def update_document_metadata(student_number: str, document_id: str, body: DocumentMetadataIn, caller: Writer, uow_factory: UowFactoryDep):
    with uow_factory() as uow:
        student = _student(uow._session, student_number)
        row = uow._session.scalar(select(m.StudentDocument).where(m.StudentDocument.id == document_id, m.StudentDocument.student_id == student.id, m.StudentDocument.deleted_at.is_(None)))
        if row is None: raise HTTPException(404, detail={"code":"unknown_reference", "message":"No active document found."})
        if body.document_type is not None: row.document_type = body.document_type.strip()[:64]
        if body.expiry_date is not None: row.expiry_date = body.expiry_date
        if body.status is not None: row.status = body.status.strip()[:24]
        uow.commit(); return _row(row)

@router.get("/{document_id}/file")
def download_document(student_number: str, document_id: str, caller: Reader, uow_factory: UowFactoryDep):
    with uow_factory() as uow:
        student = _student(uow._session, student_number)
        row = uow._session.scalar(select(m.StudentDocument).where(m.StudentDocument.id == document_id, m.StudentDocument.student_id == student.id, m.StudentDocument.deleted_at.is_(None)))
        if row is None: raise HTTPException(404, detail={"code":"unknown_reference", "message":"No active document found."})
        path = ROOT / row.file_key
        if not path.is_file(): raise HTTPException(404, detail={"code":"file_missing", "message":"Stored document is unavailable."})
        return FileResponse(path, media_type=row.mime_type, filename=row.original_filename)
