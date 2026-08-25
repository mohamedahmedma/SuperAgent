"""The records facade application.

Runs on its own port, against its own database, with its own dependency set. It does
not import `backend` and `backend` does not import it — if this service is down, chat
still works and the agent says so; if chat is down, teachers and registrars are
unaffected.

    uvicorn records.app:app --port 8100

`RECORDS_LMS` selects the adapter — `fake`, or `sis` for the school's own Student
Information Service on :8300. It defaults to `fake`, so a deployment that has not said
which system of record it runs against gets an explicit fixture backend rather than a
half-configured live one that would fail at the first parent question instead of at
startup.

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
from records.env import load_env

# Before anything below reads the environment. This service is deployed as its own
# process, so nothing else has loaded the project's `.env` for it.
load_env()

from records.db import init_db, new_session
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

    if backend == "sis":
        base_url = os.getenv("SIS_BASE_URL", "")
        # A `reader`-scoped key, when SIS is checking one. It is sent on every request and
        # is deliberately not a registrar key: a registrar key also WRITES, and handing the
        # school's write credential to the process that answers parents is how a read-only
        # integration becomes the blast radius of a leak.
        #
        # Optional, because SIS currently admits every caller as a registrar and verifies
        # no key at all (`sis/api/deps.py`: "No API key is required"). Demanding one here
        # meant demanding a credential nothing checks — a deployment either invented a
        # value to get past this line, or the service refused to boot over a setting that
        # could not affect anything. The header is still sent when a key is configured, so
        # the day SIS starts enforcing again, setting it is the whole change.
        api_key = os.getenv("SIS_API_KEY", "")
        if not base_url:
            raise RuntimeError("RECORDS_LMS=sis requires SIS_BASE_URL.")
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

    db = new_session()
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

    Deliberately does not probe the LMS. A health check that fails when the system of
    record is down would take this service out of rotation exactly when it is still
    perfectly able to serve report card snapshots and to tell the agent, honestly, that
    live grades are unavailable.
    """
    return {"status": "ok", "service": "records-facade"}
