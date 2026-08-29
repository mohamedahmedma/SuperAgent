"""The records facade application.

Runs on its own port, with its own dependency set. It does not import `backend` and
`backend` does not import it — if this service is down, chat still works and the agent
says so; if chat is down, teachers and registrars are unaffected.

    uvicorn records.app:app --port 8100

`RECORDS_LMS` selects the adapter — `fake`, or `sis` for the school's own Student
Information Service on :8300. It defaults to `fake`, so a deployment that has not said
which system of record it runs against gets an explicit fixture backend rather than a
half-configured live one that would fail at the first parent question instead of at
startup.

## This file is the composition root

Every adapter, every pooled HTTP client and every configuration read happens here, once,
at startup, and lands on `app.state` for `api/deps.py` to hand to a request. Nothing under
`domain/`, `ports/` or `application/` imports `records.config`, FastAPI or `httpx`.

Each backend's credentials are demanded here rather than discovered on the first parent
question. A misconfigured deployment should refuse to start; the failure mode it replaces
is a service that looks healthy until someone asks about a child.

**Order matters in one place**: the marks adapter is handed the calendar, so the calendar
is built first. Two components resolving a term separately is how they come to disagree
about it.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from records.config import load_env, settings

# Before anything below reads the environment. This service is deployed as its own
# process, so nothing else has loaded the project's `.env` for it.
load_env()

from records.adapters.fake.calendar import FakeSchoolCalendar  # noqa: E402
from records.adapters.fake.directory import FakeGuardianDirectory  # noqa: E402
from records.adapters.fake.lms import FakeLms  # noqa: E402
from records.adapters.sis.calendar import SisSchoolCalendar  # noqa: E402
from records.adapters.sis.directory import SisGuardianDirectory  # noqa: E402
from records.adapters.sis.grades import SisAdapter  # noqa: E402
from records.api import errors  # noqa: E402
from records.api.routers import admin, records  # noqa: E402
from records.domain.grading import GradingPolicy  # noqa: E402

logger = logging.getLogger(__name__)


def _sis_api_key(resolved) -> str:
    """The credential this service presents to SIS. **No longer required.**

    SIS stopped authenticating its callers — see `sis/api/deps.py` — so `SIS_API_KEY` is
    now a value SIS ignores rather than one it checks. It is still read and still sent,
    because it costs nothing and it is what the adapters go back to presenting the day
    SIS asks for a credential again; an unset one is a warning here and not a refusal to
    start, so a fresh deployment is not blocked on minting a key nothing verifies.

    Read in one place so the guardian directory, the calendar and the marks adapter cannot
    end up disagreeing about which key they hold — they are three questions asked of one
    service, and a deployment that keyed two of them would fail on the third at the first
    parent question rather than at boot.
    """
    if not resolved.sis_api_key:
        logger.warning(
            "SIS_BASE_URL is set without SIS_API_KEY. That is survivable only because SIS "
            "currently authenticates nobody; set it again when SIS has sign-in."
        )
    return resolved.sis_api_key


def _build_calendar(resolved):
    """Where "when does this term run" is answered.

    Left as the in-memory fake when `SIS_BASE_URL` is unset, which is what keeps a laptop
    and the test suite working with no second service running.
    """
    if not resolved.sis_base_url:
        return FakeSchoolCalendar()
    return SisSchoolCalendar(
        base_url=resolved.sis_base_url, api_key=_sis_api_key(resolved)
    )


def _build_directory(resolved):
    """Where "which children are this parent's" is answered.

    Left as the empty in-memory fake when `SIS_BASE_URL` is unset, which means an
    unconfigured deployment tells every parent they have no children on file rather than
    authorising them against nothing. That is the safe direction.
    """
    if not resolved.sis_base_url:
        logger.warning(
            "SIS_BASE_URL is not set; guardian links resolve against an empty directory "
            "and every parent will be told they have no children on file."
        )
        return FakeGuardianDirectory()
    return SisGuardianDirectory(
        base_url=resolved.sis_base_url, api_key=_sis_api_key(resolved)
    )


def _build_lms(resolved, calendar):
    """Which system of record holds the marks."""
    if not resolved.uses_sis:
        return FakeLms()
    if not resolved.sis_base_url:
        raise RuntimeError("RECORDS_LMS=sis requires SIS_BASE_URL.")
    # A `reader`-scoped key, and required. It is sent on every request and is deliberately
    # not a registrar key: a registrar key also WRITES, and handing the school's write
    # credential to the process that answers parents is how a read-only integration
    # becomes the blast radius of a leak.
    return SisAdapter(
        base_url=resolved.sis_base_url,
        api_key=_sis_api_key(resolved),
        # The same calendar the reads use, handed in rather than built a second time:
        # attendance is addressed by dates, and two components resolving one term
        # separately is how they come to disagree about it.
        calendar=calendar,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Choose what this process talks to, and close it cleanly.

    There is no schema to create and no first row to seed: the facade holds no guardian
    links, no API keys and no audit rows. What remains is composition.
    """
    resolved = settings()
    app.state.settings = resolved
    app.state.policy = GradingPolicy(primary_figure=resolved.primary_figure)

    app.state.calendar = _build_calendar(resolved)
    app.state.directory = _build_directory(resolved)
    app.state.lms = _build_lms(resolved, app.state.calendar)

    try:
        yield
    finally:
        # Release the pooled clients. Without this, `uvicorn --reload` leaks a connection
        # pool per reload until the process runs out of sockets.
        for adapter in (app.state.lms, app.state.directory, app.state.calendar):
            closer = getattr(adapter, "close", None)
            if callable(closer):
                closer()


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
        "student set is then resolved from the school's own records on every request, "
        "never supplied by the caller and never cached — and the guardian travels with the "
        "read, so the system of record checks the link again before it answers.\n\n"
        "**On failure, do not improvise.** A 503 with `code: lms_unavailable` means the "
        "system of record could not be reached. The correct response to a parent is that "
        "records are temporarily unavailable, never a remembered or inferred figure."
    ),
    lifespan=lifespan,
)

# One envelope for every refusal, and one place that decides the status.
errors.install(app)

app.include_router(records.router)
app.include_router(admin.router)


@app.get("/health", tags=["ops"])
def health() -> dict:
    """Liveness only.

    Deliberately does not probe the SIS. A health check that fails when the system of
    record is down would take this service out of rotation exactly when it is still able
    to do the one useful thing left — tell the agent, honestly, that live grades are
    unavailable, so it says that to a parent instead of inventing a figure.
    """
    return {"status": "ok", "service": "records-facade"}
