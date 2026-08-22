"""The records facade application.

Runs on its own port, against its own database, with its own dependency set. It does
not import `backend` and `backend` does not import it — if this service is down, chat
still works and the agent says so; if chat is down, teachers and registrars are
unaffected.

    uvicorn records.app:app --port 8100

`RECORDS_LMS` selects the adapter — `fake`, `moodle`, or `sis` for the school's own
Student Information Service on :8300. It defaults to `fake`, which is the right default
for a service whose real adapter is still a skeleton: an explicit fixture backend is
honest, whereas defaulting to a half-built Moodle client would fail at the first
parent question instead of at startup.

Each backend's credentials are demanded here, at startup, rather than discovered on the
first parent question. A misconfigured deployment should refuse to start; the failure
mode it replaces is a service that looks healthy until someone asks about a child.
"""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from records import calendar as school_calendar
from records import guardian_directory, lms
from records.db import SessionLocal, init_db
from records.routes import admin_router, agent_router
from records.sis_adapter import SisAdapter

logger = logging.getLogger(__name__)


def _configure_guardian_directory() -> None:
    """Choose where "which children are this parent's" is answered.

    Left as the empty in-memory fake when `SIS_BASE_URL` is unset, which makes an
    unconfigured deployment tell every parent they have no children on file. That is the
    safe direction: a support call, rather than a service quietly answering from local
    tables nobody maintains any more.

    The same `SIS_BASE_URL` the LMS adapter uses, because it is the same service — this
    reads guardians from it while `SisAdapter` reads marks.
    """
    base_url = os.getenv("SIS_BASE_URL", "")
    if not base_url:
        logger.warning(
            "SIS_BASE_URL is not set; guardian links resolve against an empty directory "
            "and every parent will be told they have no children on file."
        )
        return
    api_key = os.getenv("SIS_API_KEY", "")
    guardian_directory.set_directory(
        guardian_directory.SisGuardianDirectory(base_url=base_url, api_key=api_key)
    )
    # The academic calendar comes from the same service, for the same reason: a term
    # whose dates a registrar corrected must not keep answering from a stale copy here.
    school_calendar.set_calendar(
        school_calendar.SisSchoolCalendar(base_url=base_url, api_key=api_key)
    )


def _configure_adapter() -> None:
    backend = (os.getenv("RECORDS_LMS") or "fake").strip().lower()

    if backend == "moodle":
        base_url = os.getenv("MOODLE_BASE_URL", "")
        token = os.getenv("MOODLE_TOKEN", "")
        if not base_url or not token:
            raise RuntimeError("RECORDS_LMS=moodle requires MOODLE_BASE_URL and MOODLE_TOKEN.")
        lms.set_adapter(lms.MoodleAdapter(base_url=base_url, token=token))
        return

    if backend == "sis":
        base_url = os.getenv("SIS_BASE_URL", "")
        # A `reader`-scoped key. A registrar key would also read grades, and handing the
        # school's write credential to the process that answers parents is how a
        # read-only integration becomes the blast radius of a leak.
        api_key = os.getenv("SIS_API_KEY", "")
        if not base_url or not api_key:
            raise RuntimeError("RECORDS_LMS=sis requires SIS_BASE_URL and SIS_API_KEY.")
        # The same calendar the routes read, handed to the adapter rather than letting it
        # build a second one: attendance is addressed by dates, and two components
        # resolving one term separately is how they come to disagree about it.
        lms.set_adapter(
            SisAdapter(
                base_url=base_url,
                api_key=api_key,
                calendar=school_calendar.get_calendar(),
            )
        )
        return

    lms.set_adapter(lms.FakeLms())


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()

    from records.auth import bootstrap_admin_key

    db = SessionLocal()
    try:
        bootstrap_admin_key(db)
    finally:
        db.close()

    # Order matters: the LMS adapter is handed the calendar, so the calendar is
    # chosen first.
    _configure_guardian_directory()
    _configure_adapter()
    yield


app = FastAPI(
    title="School Academic Records Facade",
    version="0.1.0",
    description=(
        "Read-only academic records for the school assistant agent, plus the guardian "
        "authorisation and audit trail that govern them.\n\n"
        "**Two independent credentials are required for every parent-facing read.** The "
        "`X-API-Key` header proves which *system* is calling. An `Authorization: Bearer` "
        "identity token, signed by the identity service, proves *which parent* it asks on "
        "behalf of — and its `guardian_id` claim must match the `guardian_id` in the path. "
        "Neither alone grants anything: a leaked API key cannot choose a guardian, and a "
        "parent's token cannot be presented without a valid system key. The permitted "
        "student set is then resolved server-side from the link table on every request, "
        "and is never supplied by the caller.\n\n"
        "**On failure, do not improvise.** A 503 with `code: lms_unavailable` means the "
        "system of record could not be reached. The correct response to a parent is that "
        "records are temporarily unavailable, never a remembered or inferred figure."
    ),
    lifespan=lifespan,
)

app.include_router(agent_router)
app.include_router(admin_router)


@app.get("/health", tags=["ops"])
def health() -> dict:
    """Liveness only.

    Deliberately does not probe the LMS. A health check that fails when Moodle is
    down would take this service out of rotation exactly when it is still perfectly
    able to serve report card snapshots and to tell the agent, honestly, that live
    grades are unavailable.
    """
    return {"status": "ok", "service": "records-facade"}
