"""`IngestJobRepository` over SQLAlchemy.

Timestamps are written as the record carries them rather than stamped here: the tracker
owns the clock, which is what lets a test say "an hour later" without waiting one.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from backend.application.ports.repositories import IngestJobRecord, JobStep
from backend.db.models import IngestJob


class SqlAlchemyIngestJobRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, job: IngestJobRecord) -> None:
        row = IngestJob(job_id=job.job_id, kind=job.kind, created_at=job.created_at)
        _apply(row, job)
        self._session.add(row)
        self._session.flush()

    def get(self, kind: str, job_id: str, *, for_update: bool = False) -> IngestJobRecord | None:
        statement = select(IngestJob).where(IngestJob.job_id == job_id, IngestJob.kind == kind)
        if for_update:
            statement = statement.with_for_update()
        row = self._session.scalars(statement).first()
        return None if row is None else _record(row)

    def save(self, job: IngestJobRecord) -> None:
        row = self._session.get(IngestJob, job.job_id)
        if row is None:
            raise LookupError(f"ingest job {job.job_id} does not exist")
        _apply(row, job)
        self._session.flush()

    def recent(self, kind: str, limit: int) -> Sequence[IngestJobRecord]:
        rows = self._session.scalars(
            select(IngestJob)
            .where(IngestJob.kind == kind)
            .order_by(IngestJob.created_at.desc(), IngestJob.job_id)
            .limit(limit)
        )
        return [_record(row) for row in rows]

    def delete_older_than(self, kind: str, cutoff: datetime) -> int:
        result = self._session.execute(
            delete(IngestJob)
            .where(IngestJob.kind == kind, IngestJob.created_at < cutoff)
            .execution_options(synchronize_session=False)
        )
        return int(result.rowcount or 0)


def _apply(row: IngestJob, job: IngestJobRecord) -> None:
    row.filename = job.filename
    row.status = job.status
    row.current_step = job.current_step
    row.completion_step = job.completion_step
    row.message = job.message
    row.error = job.error
    row.total_chunks = job.total_chunks
    row.processed_chunks = job.processed_chunks
    row.steps = [asdict(step) for step in job.steps]
    row.updated_at = job.updated_at


def _record(row: IngestJob) -> IngestJobRecord:
    return IngestJobRecord(
        job_id=row.job_id,
        kind=row.kind,
        filename=row.filename,
        status=row.status,
        current_step=row.current_step,
        completion_step=row.completion_step,
        message=row.message,
        steps=tuple(JobStep(**step) for step in (row.steps or [])),
        created_at=row.created_at,
        updated_at=row.updated_at,
        error=row.error,
        total_chunks=row.total_chunks,
        processed_chunks=row.processed_chunks,
    )


__all__ = ["SqlAlchemyIngestJobRepository"]
