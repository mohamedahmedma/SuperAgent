"""Progress of document uploads and deletes, kept in Postgres.

An upload or a delete runs as a background task while the admin UI polls for its
progress. That progress used to live in the memory of the process that started the job,
which held only while that process answered every poll: any other worker answered 404,
and a restart turned every job, finished or not, into "does not exist or has expired".
In the database, any worker can answer, and a job outlives the process that ran it.

A process that dies cannot report that it died, so a job still pending or running that
has reported nothing for `STALLED_AFTER` is presented as failed when read. The stored row
is left as it is: the job may be alive on another worker and merely slow, and if it
reports again its next update is simply shown.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from typing import Literal
from uuid import uuid4

from backend.application.ports.repositories import IngestJobRecord, JobStep
from backend.application.ports.unit_of_work import UnitOfWorkFactory
from backend.infra.unit_of_work import SqlAlchemyUnitOfWork

StepStatus = Literal["pending", "running", "completed", "failed"]
JobStatus = Literal["pending", "running", "completed", "failed"]


DEFAULT_STEPS = [
    ("upload", "Document upload"),
    ("cleanup", "Clean up old version"),
    ("parse", "Parsing and chunking"),
    ("parent_store", "Storing parent chunks"),
    ("vector_store", "Vectorizing and storing"),
]

DELETE_STEPS = [
    ("prepare", "Preparing deletion"),
    ("bm25", "Syncing BM25 statistics"),
    ("milvus", "Deleting vector data"),
    ("parent_store", "Deleting parent chunks"),
]

#: How long a job may report nothing before it is presumed dead. Generous on purpose:
#: parsing a large illustrated document through a vision model reports nothing until it
#: finishes, and calling that failed would be a false alarm.
STALLED_AFTER = timedelta(hours=1)
#: How long jobs are kept. Creating a job prunes its kind's jobs older than this.
RETAINED_FOR = timedelta(days=30)
#: How many jobs the list endpoint returns.
LIST_LIMIT = 100

_UNFINISHED = frozenset({"pending", "running"})


def _utc_now() -> datetime:
    return datetime.now(UTC)


class IngestJobTracker:
    """The progress of one kind of ingest job ("upload" or "delete"), as the UI reads it.

    Every method returns the job as the API serialises it — a plain dict — or None when
    the job does not exist. Every change is made under a row lock: the vectoriser's
    progress callback and the job's own step transitions run on different threads, and
    without the lock one overwrites the other's step.
    """

    def __init__(
        self,
        kind: str,
        unit_of_work: UnitOfWorkFactory = SqlAlchemyUnitOfWork,
        *,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._kind = kind
        self._unit_of_work = unit_of_work
        self._clock = clock

    def create_job(
        self,
        filename: str,
        *,
        steps: list[tuple[str, str]] | None = None,
        current_step: str = "upload",
        message: str = "Waiting to upload",
        completion_step: str = "vector_store",
    ) -> dict:
        now = self._clock()
        job = IngestJobRecord(
            job_id=uuid4().hex,
            kind=self._kind,
            filename=filename,
            status="pending",
            current_step=current_step,
            # Distinguishes upload from delete jobs, so complete_job needs no hardcoded
            # final step.
            completion_step=completion_step,
            message=message,
            steps=tuple(JobStep(key=key, label=label) for key, label in (steps or DEFAULT_STEPS)),
            created_at=now,
            updated_at=now,
        )
        with self._unit_of_work() as uow:
            uow.ingest_jobs.delete_older_than(self._kind, now - RETAINED_FOR)
            uow.ingest_jobs.add(job)
            uow.commit()
        return self._present(job, now)

    def get_job(self, job_id: str) -> dict | None:
        with self._unit_of_work() as uow:
            job = uow.ingest_jobs.get(self._kind, job_id)
        return None if job is None else self._present(job, self._clock())

    def list_jobs(self) -> list[dict]:
        now = self._clock()
        with self._unit_of_work() as uow:
            jobs = uow.ingest_jobs.recent(self._kind, LIST_LIMIT)
        return [self._present(job, now) for job in jobs]

    def update_step(
        self,
        job_id: str,
        step_key: str,
        percent: int,
        status: StepStatus = "running",
        message: str = "",
        *,
        total_chunks: int | None = None,
        processed_chunks: int | None = None,
        sub_label: str | None = None,
        sub_done: int | None = None,
        sub_total: int | None = None,
    ) -> dict | None:
        """Move a step, and optionally the nested bar inside it.

        The sub-stage fields are None-means-leave-alone rather than defaulting to zero:
        an ordinary `update_step` during a sub-stage must not erase the nested bar that
        another caller is driving.
        """
        percent = max(0, min(100, int(percent)))
        sub = {
            name: value
            for name, value in (
                ("sub_label", sub_label), ("sub_done", sub_done), ("sub_total", sub_total)
            )
            if value is not None
        }

        def change(job: IngestJobRecord) -> IngestJobRecord | None:
            if not any(step.key == step_key for step in job.steps):
                return None
            return replace(
                job,
                steps=tuple(
                    replace(step, percent=percent, status=status, message=message, **sub)
                    if step.key == step_key
                    else step
                    for step in job.steps
                ),
                status="failed" if status == "failed" else "running",
                current_step=step_key,
                message=message,
                total_chunks=job.total_chunks if total_chunks is None else int(total_chunks),
                processed_chunks=job.processed_chunks if processed_chunks is None else int(processed_chunks),
            )

        return self._change(job_id, change)

    def complete_step(self, job_id: str, step_key: str, message: str = "") -> dict | None:
        return self.update_step(job_id, step_key, 100, "completed", message)

    def complete_job(self, job_id: str, message: str = "Document ingestion complete") -> dict | None:
        def change(job: IngestJobRecord) -> IngestJobRecord:
            return replace(
                job,
                steps=tuple(
                    step if step.status == "failed" else replace(step, percent=100, status="completed")
                    for step in job.steps
                ),
                status="completed",
                current_step=job.completion_step or job.current_step,
                message=message,
                error=None,
            )

        return self._change(job_id, change)

    def fail_job(self, job_id: str, step_key: str, error: str) -> dict | None:
        def change(job: IngestJobRecord) -> IngestJobRecord:
            return replace(
                job,
                steps=tuple(
                    replace(step, status="failed", message=error) if step.key == step_key else step
                    for step in job.steps
                ),
                status="failed",
                current_step=step_key,
                message=error,
                error=error,
            )

        return self._change(job_id, change)

    def _change(
        self, job_id: str, change: Callable[[IngestJobRecord], IngestJobRecord | None]
    ) -> dict | None:
        now = self._clock()
        with self._unit_of_work() as uow:
            job = uow.ingest_jobs.get(self._kind, job_id, for_update=True)
            if job is None:
                return None
            changed = change(job)
            if changed is None:
                return None
            changed = replace(changed, updated_at=now)
            uow.ingest_jobs.save(changed)
            uow.commit()
        return self._present(changed, now)

    @staticmethod
    def _present(job: IngestJobRecord, now: datetime) -> dict:
        status, message, error = job.status, job.message, job.error
        if status in _UNFINISHED and now - job.updated_at > STALLED_AFTER:
            status = "failed"
            message = error = (
                f"No progress for over {int(STALLED_AFTER.total_seconds() // 60)} minutes; the "
                "process running this job has most likely stopped. Start it again."
            )
        return {
            "job_id": job.job_id,
            "filename": job.filename,
            "status": status,
            "current_step": job.current_step,
            "message": message,
            "completion_step": job.completion_step,
            "total_chunks": job.total_chunks,
            "processed_chunks": job.processed_chunks,
            "error": error,
            "created_at": job.created_at.isoformat(),
            "updated_at": job.updated_at.isoformat(),
            "steps": [asdict(step) for step in job.steps],
        }

