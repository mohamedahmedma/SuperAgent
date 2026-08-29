"""The SIS application: composition root, and the only place the wiring lives.

    alembic -c sis/alembic.ini upgrade head
    uvicorn sis.app:app --port 8300

Runs on its own port, against its own database, with its own dependency set. It does
not import `backend`, `records` or `identity`, and none of them import it — a registrar
importing next term's roster is unaffected by a chat outage, and vice versa.

Everything below is assembly. No route, no schema and no rule is defined in this file;
the layers underneath it are what get unit-tested, and they are testable with fake
repositories because nothing in them knows this file exists.

**The startup checks live in the lifespan, not at import time.** `export_openapi.py`
imports `app` to render the contract, and a test client builds it to exercise a route
against a fixture database. Neither should need a migrated Postgres to be reachable,
and an import-time check would mean the OpenAPI file could only be regenerated on a
machine with production credentials.
"""
from __future__ import annotations

import importlib
import logging
import os
import pathlib
import pkgutil
import re
from collections.abc import Iterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles

from sis.env import load_env

# Before anything reads the environment. `sis.config` memoises its settings on
# first use, so a `.env` loaded after that point would be read too late — see
# sis/env.py.
load_env()

log = logging.getLogger("sis.app")

_HERE = pathlib.Path(__file__).resolve().parent
_ALEMBIC_INI = _HERE / "alembic.ini"
_WEB_ROOT = _HERE / "web"

VERSION = "0.1.0"

DESCRIPTION = (
    "Student Information Service — the registrar's system of record for school "
    "structure, class placement and stated grades.\n\n"
    "**A blank grade is `null`, never `0`.** Zero is a mark a child can earn; the "
    "absence of a mark is not. Every grade field in this API distinguishes the two, "
    "and a client that renders `null` as `0%` is reporting a failure that did not "
    "happen.\n\n"
    "**Class placement is a dated membership, not a property of the student.** A child "
    "who moves from 3A to 3B in March is in 3A for Term 1 and 3B for Term 2, and both "
    "remain true forever. Ask for a placement *as of* a date or a term; there is no "
    "'current class' field to read.\n\n"
    "**Imports are preview-then-commit.** An upload is parsed and validated into a "
    "batch with a per-row outcome, and nothing is written until that batch is "
    "committed. One malformed row is reported as one rejected row — it never discards "
    "the rows around it.\n\n"
    "**Codes are immutable; labels are not.** A class is identified by its code for "
    "its whole life, so renaming '3A' to 'Falcons' changes a label and detaches no "
    "student and no grade. Every nameable thing carries both an Arabic and an English "
    "name.\n\n"
    "Grades here are stated figures, reported exactly as the school recorded them. "
    "This service does not average, weight, or drop the lowest."
)


# ---------------------------------------------------------------------------
# Startup gate: the schema is alembic's, and the service refuses to guess.
# ---------------------------------------------------------------------------

def _sanitised_url(url: object) -> str:
    """The database URL with the password removed, for putting in a log line."""
    try:
        return url.render_as_string(hide_password=True)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 -- a URL we cannot render must not mask the real error
        return "<unprintable url>"


def _expected_heads() -> set[str]:
    """The revisions this build's migration directory ends at."""
    from alembic.config import Config as AlembicConfig
    from alembic.script import ScriptDirectory

    return set(ScriptDirectory.from_config(AlembicConfig(str(_ALEMBIC_INI))).get_heads())


def verify_database_is_migrated() -> None:
    """Fail startup unless `alembic_version` matches this build's head revision.

    There is no `create_all` in this service (invariant 8), so an unmigrated database
    does not fail at startup on its own — it fails hours later, on the first request
    that touches a missing table, as a 500 in front of a registrar mid-import. Worse
    is the *partially* migrated case: the tables from month one exist, so the service
    starts and serves reads happily, and only the column added in month two is absent.
    Comparing the applied revision against the head this code was written for catches
    both, before the process accepts a single request.

    Deliberately not `alembic upgrade head` on boot. Two replicas starting together
    would race on the same DDL, and a schema change is a decision an operator makes
    with a backup in hand — not a side effect of a restart.
    """
    if os.getenv("SIS_SKIP_MIGRATION_CHECK", "").strip().lower() in {"1", "true", "yes"}:
        # For tests that build the schema themselves. Setting this in a deployment
        # trades a clear startup failure for an unclear one later; don't.
        log.warning("SIS_SKIP_MIGRATION_CHECK is set — starting without a schema check")
        return

    from sis.tenancy import get_registry

    registry = get_registry()
    if not registry.is_multi_school:
        _check_one_database(None, "SIS_DATABASE_URL")
        return

    # Every school, not just the first. Physical separation means a migration can succeed
    # for four schools and fail for the fifth, leaving the estate on two schema versions at
    # once — and the school left behind is the one whose registrar meets the missing column
    # mid-import. Checking them all turns that into a startup failure naming the school.
    #
    # All of them are checked before any is reported, so an operator fixes every stale
    # school in one pass instead of restarting once per school to discover the next.
    failures: list[str] = []
    for tenant in registry.tenants:
        try:
            _check_one_database(
                tenant.code, f"SIS_DATABASE_URL_{tenant.code.replace('.', '_').replace('-', '_')}"
            )
        except RuntimeError as exc:
            failures.append(str(exc))
    if failures:
        raise RuntimeError(
            f"{len(failures)} of {len(registry.tenants)} school database(s) are not ready:"
            "\n\n" + "\n\n".join(failures)
        )


def _check_one_database(school_code: str | None, setting: str) -> None:
    """The schema gate for one database. `school_code` is `None` for single-school mode."""
    from alembic.runtime.migration import MigrationContext

    from sis.infrastructure.db.session import get_engine

    engine = get_engine(school_code)
    where = _sanitised_url(engine.url)
    named = f"school {school_code}: " if school_code else ""
    upgrade = (
        "python -m sis.schools migrate"
        if school_code
        else f'alembic -c "{_ALEMBIC_INI}" upgrade head'
    )

    try:
        with engine.connect() as connection:
            applied = set(MigrationContext.configure(connection).get_current_heads())
    except Exception as exc:  # noqa: BLE001 -- reported with the URL, which the driver omits
        raise RuntimeError(
            f"{named}cannot reach the database at {where} "
            f"({type(exc).__name__}: {exc}).\n"
            f"Check {setting} and that the database is accepting connections."
        ) from exc

    expected = _expected_heads()
    if applied == expected:
        return

    state = ", ".join(sorted(applied)) if applied else "<none — never migrated>"
    raise RuntimeError(
        f"{named}the database schema does not match this build.\n"
        f"  database: {where}\n"
        f"  applied:  {state}\n"
        f"  expected: {', '.join(sorted(expected)) or '<no migrations found>'}\n"
        "Alembic owns this schema; the service will not create or alter it. Run:\n"
        f"  {upgrade}\n"
        "then start the service again."
    )


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

# The API layer owns error translation; this file only has to know that it does. The
# name is looked up rather than imported directly because a rename in `api/errors.py`
# should surface as one clear startup message, not as an ImportError in the traceback
# of whatever imported this module first.
_HANDLER_INSTALLERS = (
    "register_error_handlers",
    "register_exception_handlers",
    "install_error_handlers",
    "add_error_handlers",
)


def _register_error_handlers(app: FastAPI) -> None:
    """Install the domain-error -> HTTP translation from `sis/api/errors.py`."""
    module = importlib.import_module("sis.api.errors")

    for name in _HANDLER_INSTALLERS:
        installer = getattr(module, name, None)
        if callable(installer):
            installer(app)
            return

    handlers = getattr(module, "EXCEPTION_HANDLERS", None)
    if handlers:
        for exc_type, handler in dict(handlers).items():
            app.add_exception_handler(exc_type, handler)
        return

    raise RuntimeError(
        "sis.api.errors exposes no way to install its handlers. Expected a callable "
        f"named one of {_HANDLER_INSTALLERS} taking the app, or an EXCEPTION_HANDLERS "
        "mapping of {exception type: handler}. Without one, every domain error would "
        "reach the client as a bare 500."
    )


def _iter_routers() -> Iterator[tuple[str, APIRouter]]:
    """Yield every router defined under `sis/api/routers/`, in module-name order.

    Discovered rather than listed, because the failure a hand-written import list
    produces is silent: a router that exists, is tested and passes review, and is
    simply absent from the deployed app because nobody added the line here. Order is
    sorted so the generated `openapi.json` diffs cleanly instead of reshuffling with
    the filesystem.
    """
    package = importlib.import_module("sis.api.routers")

    # `all_routers()` is the package's stated contract and wins when it exists: it is the
    # one case where ordering, and omission, are deliberate decisions rather than
    # oversights. Discovery below is the fallback, not the plan.
    listing = getattr(package, "all_routers", None)
    declared = listing() if callable(listing) else getattr(package, "ROUTERS", None)
    if declared is not None:
        for index, router in enumerate(declared):
            yield f"all_routers()[{index}]", router
        return

    seen: set[int] = set()
    for info in sorted(pkgutil.iter_modules(package.__path__), key=lambda m: m.name):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{package.__name__}.{info.name}")
        for attr, value in vars(module).items():
            # id() dedupes a router imported into a sibling for composition; including
            # it twice would duplicate every one of its paths in the OpenAPI contract.
            if attr.startswith("_") or not isinstance(value, APIRouter) or id(value) in seen:
                continue
            seen.add(id(value))
            yield f"{info.name}.{attr}", value


def _include_routers(app: FastAPI) -> None:
    included: list[str] = []
    for name, router in _iter_routers():
        app.include_router(router)
        included.append(name)

    if not included:
        raise RuntimeError(
            "No routers found in sis.api.routers — the service would start and serve "
            "nothing but /health. Check that the router modules import cleanly."
        )
    log.info("mounted %d router(s): %s", len(included), ", ".join(included))


class _ConsoleStaticFiles(StaticFiles):
    """`StaticFiles`, plus the one header it does not send: `Cache-Control`.

    Starlette sends `ETag` and `Last-Modified` and stops there. With no `Cache-Control` a
    browser is entitled to guess how long the file stays fresh — the heuristic in every
    major engine is a fraction of the file's age — and it serves the cached copy **without
    asking the server**. The symptom is the one that actually happened: a stylesheet was
    edited on disk, the page was refreshed, and the old design came back, because the
    request never left the browser. It is the worst kind of caching bug, because everything
    is working correctly and the only evidence is that nothing changed.

    The console is a Vite build, so there are two populations of file here and they want
    opposite policies:

    **`assets/*` are content-hashed** — `index-DFBO-eQN.js` changes its name whenever its
    bytes change. A URL that can never mean anything else is safe to cache for a year, and
    `immutable` additionally tells the browser not to revalidate even on a manual reload.
    These are the megabyte or so of every page load, and this is what makes the second visit
    free.

    **`index.html` is not hashed**, because it is the file that names the hashed ones. It
    gets `no-cache`, which is not `no-store`: cache it, but revalidate before every use. So
    the browser still asks with its ETag and still gets a 304 with an empty body when nothing
    has changed. One conditional request buys the guarantee that a deploy is visible on
    refresh — and since the entry point names new hashed assets, revalidating it is enough to
    pull the whole new build through.

    Getting these two the wrong way round is not a symmetric mistake. An immutable
    `index.html` is a console that cannot be updated without renaming the deployment; a
    `no-cache` bundle is merely slower.
    """

    #: Where Vite writes its content-hashed output. Everything under it is immutable.
    _HASHED = "assets"

    #: One year, the maximum any cache is expected to honour.
    _FOREVER = "public, max-age=31536000, immutable"

    _REVALIDATE = "no-cache, must-revalidate"

    def file_response(self, full_path, stat_result, scope, *args, **kwargs):  # noqa: ANN001, ANN201
        response = super().file_response(full_path, stat_result, scope, *args, **kwargs)
        path = scope.get("path", "") if isinstance(scope, dict) else ""
        hashed = f"/{self._HASHED}/" in path and _looks_hashed(path)
        response.headers.setdefault(
            "Cache-Control", self._FOREVER if hashed else self._REVALIDATE
        )
        return response


def _looks_hashed(path: str) -> bool:
    """True when the filename carries a content hash, so the URL can never be reused.

    Checked rather than assumed: `assets/` is Vite's convention, but a file copied in by
    hand — a logo, a font, a favicon — lands in the same directory under its own plain name,
    and telling a browser to keep that for a year is how a logo becomes unchangeable. The
    shape is Vite's: `name-HASH.ext`, where HASH is exactly eight characters of the base64url
    alphabet. Note that the hash itself may contain a dash — `index-DFBO-eQN.js` is one file,
    not two — so this reads the last eight characters rather than splitting on the separator.
    """
    stem = path.rsplit("/", 1)[-1].split(".", 1)[0]
    if len(stem) < 10 or stem[-9] != "-":
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9_-]{8}", stem[-8:]))


def _mount_ui(app: FastAPI) -> None:
    """Serve the registrar UI from `/ui`, if it has been built into `sis/web/`.

    Missing directory is a warning, not a crash: the HTTP API is the contract other
    systems depend on, and it must not be taken down by the absence of static files.
    """
    if not _WEB_ROOT.is_dir():
        log.warning("no static UI at %s — /ui will not be served", _WEB_ROOT)
        return
    # html=True so `/ui/` resolves index.html rather than 404ing on a directory.
    app.mount(
        "/ui",
        _ConsoleStaticFiles(directory=str(_WEB_ROOT), html=True),
        name="ui",
    )


def _configure_compression(app: FastAPI) -> None:
    """Compress text responses, which is most of what this service sends.

    The console is deliberately unbundled and unminified — the files under `sis/web/` are
    the files the browser gets, so a technician can read them on the machine that is
    misbehaving. Uncompressed that is about 250 KB of stylesheet and script on a first
    load; gzipped it is closer to 65 KB, and the school wifi it arrives over is the
    reason that difference is worth a middleware.

    It earns its place on the API side too: a class register of four hundred children is
    tens of kilobytes of JSON with a very small alphabet, and compresses by roughly five
    to one.

    `minimum_size` keeps it off the small stuff. Compressing a 200-byte health check
    costs CPU on both ends and makes the response bigger.
    """
    app.add_middleware(GZipMiddleware, minimum_size=1024)


def _configure_cors(app: FastAPI) -> None:
    """Cross-origin access, off unless `SIS_CORS_ORIGINS` names who may call.

    Read here rather than in `sis/config.py` because it is a property of the HTTP edge
    and of nothing else — no use case, repository or parser is entitled to know it.

    Closed by default on purpose. This service answers with named children's marks; a
    permissive default would mean any page the registrar happens to have open can read
    them the moment a browser session exists, and nobody would notice because the
    service works exactly as well either way.
    """
    raw = (os.getenv("SIS_CORS_ORIGINS") or "").strip()
    if not raw:
        return

    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    wildcard = "*" in origins

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if wildcard else origins,
        # Credentials are forced off against a wildcard: browsers reject that pairing
        # outright, so honouring the request would silently break every call rather
        # than loosen anything.
        allow_credentials=not wildcard
        and (os.getenv("SIS_CORS_ALLOW_CREDENTIALS", "").strip().lower() in {"1", "true", "yes"}),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    log.info("CORS enabled for: %s", ", ".join(origins))


def warn_that_authentication_is_disabled() -> None:
    """Say so, on every start: this service refuses nobody.

    API-key authentication was removed from `sis/api/deps.py`, so every route — the
    imports that rewrite a term's marks included — answers whoever reaches the port.
    That is a deliberate choice while sign-in is being built and a dangerous one to
    forget, and a log line at boot is the only thing that will keep saying so after the
    person who made the change has moved on.
    """
    log.warning(
        "SIS is running WITHOUT authentication: no API key is required and every "
        "request is accepted, including writes. Keep this service off the public "
        "internet until sign-in is in place."
    )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    verify_database_is_migrated()
    warn_that_authentication_is_disabled()
    yield


def create_app() -> FastAPI:
    """Build the application. A function, so a test can build a second one."""
    app = FastAPI(
        title="Student Information Service",
        version=VERSION,
        description=DESCRIPTION,
        lifespan=lifespan,
    )

    _register_error_handlers(app)
    _include_routers(app)
    _mount_ui(app)
    _configure_compression(app)
    _configure_cors(app)
    return app


app = create_app()

# `/health` is deliberately absent from this file. It lives in `sis/api/routers/health.py`
# with every other route, and a second declaration here would register a duplicate path:
# the router's wins the match because it is mounted first, so the one written beside the
# app — the one a reader of this file would edit — is dead code that still shows up twice
# in the generated OpenAPI contract.

__all__ = [
    "app",
    "create_app",
    "lifespan",
    "verify_database_is_migrated",
    "warn_that_authentication_is_disabled",
]
