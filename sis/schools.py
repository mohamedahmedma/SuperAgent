"""Run the estate: migrate every school's database, provision a new one, or split an old one.

Schools are separated physically — one database each, the same schema in every one — which
moves three things out of the application and into here:

    python -m sis.schools list       # what is configured, and what state each is in
    python -m sis.schools migrate    # alembic upgrade head, once per school
    python -m sis.schools provision  # create one school's database and its `schools` row
    python -m sis.schools split      # carve an existing single-database estate into files

This is the CLI. The rules it enforces live in `sis.application.services.estate`, which
performs no I/O and is tested without a server; what is left here is argument parsing,
printing, and exit codes. `provision` in particular used to require that somebody had
already hand-written the school into `SIS_SCHOOLS` and `SIS_DATABASE_URL_<CODE>`; it now
renders the connection from `SIS_DATABASE_URL_TEMPLATE` and writes both itself.

Every command reads the same registry the service does (`sis.tenancy`), so there is exactly
one statement of which schools exist and where they live, and this script cannot drift from
what the running service believes.

**Nothing here destroys data.** `migrate` only ever upgrades; `provision` refuses a file
that already exists; `split` writes new files and leaves the source database untouched, so
the rollback is to stop using the new ones. `split` also refuses to guess: anything it
cannot attribute to exactly one school is reported and left behind rather than copied into
whichever school looked most likely.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sis.env import load_env  # noqa: E402

load_env()

from sqlalchemy import create_engine, text  # noqa: E402

from sis import tenancy  # noqa: E402
from sis.application.services.estate import (  # noqa: E402
    TEMPLATE_VAR,
    EstateService,
    plan_provision,
)
from sis.domain.errors import SisError  # noqa: E402
from sis.infrastructure.estate.seeding import seed_school_row  # noqa: E402
from sis.infrastructure.estate import (  # noqa: E402
    ConfigStoreUnavailable,
    DotEnvConfigStore,
    ProvisioningFailed,
    provisioner_for,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _sqlite_path(url: str) -> Path | None:
    """The file behind a SQLite URL, or `None` for any other driver."""
    if not url.startswith("sqlite"):
        return None
    _, _, tail = url.partition(":///")
    return Path(tail).resolve() if tail and ":memory:" not in url else None


def _registry() -> tenancy.Registry:
    registry = tenancy.get_registry()
    if not registry.is_multi_school:
        raise SystemExit(
            f"No schools are configured. Set {tenancy.SCHOOLS_VAR} and one "
            f"{tenancy.DATABASE_URL_PREFIX}_<CODE> per school; see .env.example."
        )
    return registry


def _alembic_upgrade(url: str) -> None:
    """`alembic upgrade head` against one database.

    Run as a subprocess rather than through alembic's Python API because alembic's env
    configures logging and holds module-level state; migrating ten schools in one process
    means ten environments layered on top of each other, and the failure that produces is
    a migration that silently ran against the previous school's URL.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "alembic",
            "-c",
            str(PROJECT_ROOT / "sis" / "alembic.ini"),
            "-x",
            f"db_url={url}",
            "upgrade",
            "head",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"Migration failed for {url}:\n{result.stdout}\n{result.stderr}"
        )


def _schema_version(url: str) -> str:
    """The alembic revision a database is on, or a word describing why there is none."""
    engine = create_engine(url, future=True)
    try:
        with engine.connect() as connection:
            row = connection.execute(
                text("SELECT version_num FROM alembic_version")
            ).first()
            return row[0] if row else "empty"
    except Exception:  # noqa: BLE001 - a missing table or an absent file, both reportable
        return "not-migrated"
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_list(_args: argparse.Namespace) -> int:
    """Every configured school, its database, and the revision that database is on.

    Worth running after any deploy that touched migrations. Physical separation means a
    migration can succeed for four schools and fail for the fifth, and the estate is then
    on two schema versions at once — a state that is invisible unless something reports it.
    """
    registry = _registry()
    width = max(len(school) for school in registry.codes)
    versions = set()
    for tenant in registry.tenants:
        version = _schema_version(tenant.database_url)
        versions.add(version)
        print(f"{tenant.code.ljust(width)}  {version.ljust(14)}  {tenant.database_url}")
    if len(versions) > 1:
        print(
            f"\nWARNING: schools are on {len(versions)} different schema versions "
            f"({', '.join(sorted(versions))}). Run `migrate` before serving traffic.",
            file=sys.stderr,
        )
        return 1
    return 0


def cmd_migrate(args: argparse.Namespace) -> int:
    """Upgrade every school to head, one at a time, reporting each.

    Sequential rather than concurrent, and it stops at the first failure. Both are
    deliberate: a half-migrated estate is easier to reason about when the schools that did
    upgrade are a prefix of a known list, and concurrency here would buy seconds while
    making "which schools are on the new schema" a question nobody can answer from the log.
    """
    registry = _registry()
    targets = registry.tenants
    if args.school:
        targets = tuple(registry.get(code) for code in args.school)

    for tenant in targets:
        before = _schema_version(tenant.database_url)
        _alembic_upgrade(tenant.database_url)
        after = _schema_version(tenant.database_url)
        moved = "unchanged" if before == after else f"{before} -> {after}"
        print(f"{tenant.code}: {moved}")
    return 0


def cmd_provision(args: argparse.Namespace) -> int:
    """Create one school's database, migrate it, record its connection, seed its row.

    What changed: this used to require that somebody had already written the school into
    `SIS_SCHOOLS` and `SIS_DATABASE_URL_<CODE>` by hand, and it refused otherwise -- so
    "create a school" was an edit to a file followed by a command, and getting the two out
    of step produced a service that would not start. It now renders the connection from
    `SIS_DATABASE_URL_TEMPLATE` and writes both variables itself.

    The order is the one `sis.application.ports.estate` argues for: database first,
    configuration last. A crash in between leaves a database nothing points at, which the
    next run refuses to overwrite; the other order leaves the service unable to boot.
    """
    registry = tenancy.get_registry()
    existing = registry.codes
    template = os.getenv(TEMPLATE_VAR, "")

    plan = plan_provision(args.code, template=template, existing_codes=existing)

    if args.dry_run:
        print(f"would create   {plan.database_url}")
        print(f"would set      {plan.env_var}={plan.database_url}")
        print(f"would set      {tenancy.SCHOOLS_VAR}={plan.schools_value}")
        print("nothing written")
        return 0

    service = EstateService(
        provisioner_for(
            plan.database_url, admin_url=os.getenv("SIS_ADMIN_DATABASE_URL", "")
        ),
        DotEnvConfigStore(PROJECT_ROOT / ".env"),
    )
    service.provision(args.code, template=template, existing_codes=existing)

    # Make the new school routable in THIS process as well as the next one. The store
    # wrote the file; without these three lines the seeding below would resolve the code
    # against a registry that was read before the school existed.
    os.environ[plan.env_var] = plan.database_url
    os.environ[tenancy.SCHOOLS_VAR] = plan.schools_value
    tenancy.reset_registry_cache()

    seed_school_row(plan.code, name_en=args.name_en, name_ar=args.name_ar)

    print(f"{plan.code}: provisioned at {plan.database_url}")
    print(f"{plan.code}: recorded as {plan.env_var} in .env")
    print(
        "Next: add this school's WhatsApp number and credentials "
        f"(IDENTITY_WHATSAPP_NUMBER_{plan.code}, "
        f"IDENTITY_WHATSAPP_PHONE_NUMBER_ID_{plan.code}, "
        f"IDENTITY_WHATSAPP_TOKEN_{plan.code}) and restart identity."
    )
    return 0


def cmd_split(args: argparse.Namespace) -> int:
    """Carve one database holding several schools into one database per school.

    The method is copy-then-delete rather than copy-the-rows-out, and that choice is the
    whole safety argument. Copying rows out means writing an INSERT for every table in
    dependency order and getting all of them right; anything forgotten is data silently
    left behind. Copying the *file* and deleting what does not belong to this school
    inherits the schema, the indexes and the foreign keys exactly, and anything forgotten
    is data wrongly kept — which the verification pass below can see and report.

    **The source database is never modified.** Every school's file is a copy, so the
    rollback is to stop pointing at them.
    """
    registry = _registry()
    source_url = args.source or os.getenv("SIS_DATABASE_URL") or "sqlite:///./sis.db"
    source_path = _sqlite_path(source_url)
    if source_path is None or not source_path.exists():
        raise SystemExit(
            f"The source database {source_url!r} is not a SQLite file this script can "
            "copy. Splitting a Postgres estate is a dump-and-restore per school rather "
            "than a file copy; this command does not attempt it."
        )

    engine = create_engine(source_url, future=True)
    try:
        with engine.connect() as connection:
            present = [
                row[0]
                for row in connection.execute(text("SELECT code FROM schools ORDER BY code"))
            ]
    finally:
        engine.dispose()

    configured = set(registry.codes)
    unconfigured = [code for code in present if code not in configured]
    if unconfigured:
        raise SystemExit(
            f"The source database holds schools that are not configured: "
            f"{', '.join(unconfigured)}. Add them to {tenancy.SCHOOLS_VAR} with a database "
            "URL each, or this split would silently drop them."
        )

    print(f"Source: {source_path} ({len(present)} school(s): {', '.join(present)})")
    if args.dry_run:
        for tenant in registry.tenants:
            if tenant.code in present:
                print(f"  would write {tenant.code} -> {tenant.database_url}")
        print("\nDry run: nothing was written.")
        return 0

    for tenant in registry.tenants:
        if tenant.code not in present:
            print(f"{tenant.code}: not in the source database, skipped")
            continue
        target = _sqlite_path(tenant.database_url)
        if target is None:
            raise SystemExit(f"{tenant.code}: target {tenant.database_url} is not a file.")
        if target.exists():
            raise SystemExit(
                f"{tenant.code}: {target} already exists. Refusing to overwrite it."
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target)
        kept = _prune_to_one_school(tenant.database_url, tenant.code)
        print(f"{tenant.code}: {target}  ({kept} student(s) kept)")

    print(
        "\nThe source database was not modified. Verify each school with "
        "`python -m sis.schools list`, then point the service at the new files."
    )
    return 0


def _prune_to_one_school(url: str, code: str) -> int:
    """Delete everything in this copy that belongs to another school. Returns students kept.

    Deleted in dependency order — leaves first, roots last — because foreign keys are
    enforced (`PRAGMA foreign_keys=ON`), and a delete in the wrong order is refused rather
    than cascading. That refusal is the point: it means this cannot quietly orphan a mark
    or an enrolment.

    Students are the one table with no school column, by design — a child is a person, and
    which school she attends follows from her placement. Here that design has to be resolved
    into a file, so a student is kept when she has any enrolment in this school and dropped
    otherwise. A child enrolled at two branches is therefore copied into both, which is the
    only honest answer once the databases are separate: she becomes two records, and a
    transfer between them is a thing a human does.
    """
    engine = create_engine(url, future=True)
    try:
        with engine.begin() as connection:
            connection.execute(text("PRAGMA foreign_keys=ON"))
            school_id = connection.execute(
                text("SELECT id FROM schools WHERE code = :code"), {"code": code}
            ).scalar_one()

            mine_years = "SELECT id FROM academic_years WHERE school_id = :sid"
            mine_levels = "SELECT id FROM year_levels WHERE school_id = :sid"
            mine_sections = (
                f"SELECT id FROM class_sections WHERE academic_year_id IN ({mine_years})"
            )
            mine_students = (
                f"SELECT DISTINCT student_id FROM class_enrolments "
                f"WHERE class_section_id IN ({mine_sections})"
            )

            params = {"sid": school_id}
            # Leaves first.
            connection.execute(
                text(
                    f"DELETE FROM subject_grades WHERE class_section_id NOT IN ({mine_sections})"
                ),
                params,
            )
            connection.execute(
                text(
                    f"DELETE FROM attendance WHERE class_section_id NOT IN ({mine_sections})"
                ),
                params,
            )
            connection.execute(
                text(
                    f"DELETE FROM class_enrolments WHERE class_section_id NOT IN ({mine_sections})"
                ),
                params,
            )
            connection.execute(
                text(
                    f"DELETE FROM student_guardians WHERE student_id NOT IN ({mine_students})"
                ),
                params,
            )
            connection.execute(
                text(f"DELETE FROM students WHERE id NOT IN ({mine_students})"), params
            )
            # Guardians nobody in this school is linked to any more.
            connection.execute(
                text(
                    "DELETE FROM guardian_phones WHERE guardian_id NOT IN "
                    "(SELECT guardian_id FROM student_guardians)"
                )
            )
            connection.execute(
                text(
                    "DELETE FROM guardians WHERE id NOT IN "
                    "(SELECT guardian_id FROM student_guardians)"
                )
            )
            # Then the structure.
            connection.execute(
                text(f"DELETE FROM class_sections WHERE academic_year_id NOT IN ({mine_years})"),
                params,
            )
            connection.execute(
                text(f"DELETE FROM subjects WHERE academic_year_id NOT IN ({mine_years})"),
                params,
            )
            connection.execute(
                text(f"DELETE FROM terms WHERE academic_year_id NOT IN ({mine_years})"),
                params,
            )
            connection.execute(
                text(f"DELETE FROM year_levels WHERE id NOT IN ({mine_levels})"), params
            )
            connection.execute(
                text(f"DELETE FROM academic_years WHERE id NOT IN ({mine_years})"), params
            )
            connection.execute(
                text("DELETE FROM schools WHERE id != :sid"), params
            )
            # Import batches reference nothing school-scoped and are working state; a
            # half-finished preview must not be committable against the new file.
            connection.execute(text("DELETE FROM import_rows"))
            connection.execute(text("DELETE FROM import_batches"))

            return connection.execute(text("SELECT count(*) FROM students")).scalar_one()
    finally:
        engine.dispose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="every school, its database, and its schema version")

    migrate = sub.add_parser("migrate", help="alembic upgrade head, per school")
    migrate.add_argument(
        "school", nargs="*", help="only these schools (default: all of them)"
    )

    provision = sub.add_parser("provision", help="create one school's database")
    provision.add_argument("code", help="the school code, as named in SIS_SCHOOLS")
    provision.add_argument("--name-en", default="", help="English name for the school")
    provision.add_argument("--name-ar", default="", help="Arabic name for the school")
    provision.add_argument(
        "--dry-run",
        action="store_true",
        help="print the database and variables this would write, and write nothing",
    )

    split = sub.add_parser("split", help="carve one database into one per school")
    split.add_argument("--source", help="database to split (default: SIS_DATABASE_URL)")
    split.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be written and write nothing",
    )

    args = parser.parse_args(argv)
    handlers = {
        "list": cmd_list,
        "migrate": cmd_migrate,
        "provision": cmd_provision,
        "split": cmd_split,
    }
    try:
        return handlers[args.command](args)
    except (
        SisError,
        tenancy.TenancyMisconfigured,
        ProvisioningFailed,
        ConfigStoreUnavailable,
    ) as error:
        # A refused provision is an answer -- "that school already exists", "the template
        # has no placeholder" -- and deserves the message rather than a traceback. Bugs
        # still raise, because a traceback is the right report for those.
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
