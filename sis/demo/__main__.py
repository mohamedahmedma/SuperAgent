"""The demo-data command line.

    python -m sis.demo load
    python -m sis.demo sync
    python -m sis.demo reset
    python -m sis.demo status
    python -m sis.demo accounts
    python -m sis.demo classes

`--school CODE` picks a tenant in multi-school mode, exactly as `sis/schools.py` does; in
single-school mode it is unnecessary and ignored by the session layer.

Every writing command runs inside one transaction and commits once at the end, so a run
that fails halfway leaves the database as it found it. That matters more here than it
looks: a half-loaded demo is worse than none, because the missing half is invisible until
somebody tries the screen that needed it.
"""
from __future__ import annotations

import argparse
import logging
import sys

from sis.demo import blueprint as bp
from sis.demo import seeder
from sis.env import load_env

# Before `sis.config` memoises anything — the same rule `sis/app.py` follows, and the
# reason the seeder can be pointed at a different database with SIS_DATABASE_URL.
load_env()


def _configure_output() -> None:
    """UTF-8 on stdout and stderr, whatever the console claims its codepage is.

    Windows still hands a Python process `cp1252` unless told otherwise, and every one of
    these commands prints Arabic — so without this, `classes` and `accounts` do not
    produce mangled output, they raise `UnicodeEncodeError` and print nothing at all. The
    reconfigure is guarded because a redirected or wrapped stream may not support it, and
    a tool that cannot change its encoding should still run.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def cmd_load(args: argparse.Namespace) -> int:
    seeder.guard_environment(allow_remote=args.allow_remote)
    with seeder.open_session(args.school) as session:
        if seeder.demo_school(session) is not None and not args.force:
            print(
                f"The demo school {bp.SCHOOL_CODE} is already in this database.\n"
                "Use `reset` to replace it, or `load --force` to add to it (which will "
                "collide on the school code and fail).",
                file=sys.stderr,
            )
            return 1
        roles, permissions = seeder.sync_roles(session)
        counts = seeder.load(session)
        session.commit()

    print(f"Reference data: {roles} roles, {permissions} permissions.")
    print("Demo data written:")
    for line in counts.as_lines():
        print(line)
    _print_password_warning()
    return 0


def cmd_portfolio(args: argparse.Namespace) -> int:
    """Write the three-school local sales showcase into an empty database."""
    seeder.guard_environment(allow_remote=args.allow_remote)
    with seeder.open_session(args.school) as session:
        if session.query(seeder.m.School).count():
            print("The showcase loader requires an empty database.", file=sys.stderr)
            return 1
        roles, permissions = seeder.sync_roles(session)
        counts = seeder.load_showcase_portfolio(session)
        session.commit()
    print(f"Reference data: {roles} roles, {permissions} permissions.")
    print("Three-school showcase written:")
    for line in counts.as_lines():
        print(line)
    _print_password_warning()
    return 0


def cmd_reset(args: argparse.Namespace) -> int:
    seeder.guard_environment(allow_remote=args.allow_remote)
    with seeder.open_session(args.school) as session:
        removed = seeder.remove(session)
        roles, permissions = seeder.sync_roles(session)
        counts = seeder.load(session)
        session.commit()

    print(f"Removed {removed} row(s) belonging to the demo school.")
    print(f"Reference data: {roles} roles, {permissions} permissions.")
    print("Demo data written:")
    for line in counts.as_lines():
        print(line)
    _print_password_warning()
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    """Refresh mutable demo labels without removing students or related records."""
    seeder.guard_environment(allow_remote=args.allow_remote)
    with seeder.open_session(args.school) as session:
        roles, permissions = seeder.sync_roles(session)
        touched = seeder.sync_demo(session)
        session.commit()
    print(f"Updated {touched} demo label row(s) without deleting any demo data.")
    print(f"Reference data: {roles} roles, {permissions} permissions.")
    return 0


def cmd_drop(args: argparse.Namespace) -> int:
    """Remove the demo and put nothing back. The other half of `reset`."""
    seeder.guard_environment(allow_remote=args.allow_remote)
    with seeder.open_session(args.school) as session:
        removed = seeder.remove(session)
        session.commit()
    print(f"Removed {removed} row(s). Roles and permissions were left in place.")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    with seeder.open_session(args.school) as session:
        for line in seeder.status(session):
            print(line)
    return 0


def cmd_classes(args: argparse.Namespace) -> int:
    with seeder.open_session(args.school) as session:
        lines = list(seeder.describe_classes(session))
    if not lines:
        print("No demo classes in this database. Run `python -m sis.demo load` first.")
        return 1
    print(f"  {'code':<14} {'title (English)':<38} title (Arabic)")
    for line in lines:
        print(line)
    return 0


def cmd_accounts(_args: argparse.Namespace) -> int:
    """The credentials table, generated from the blueprint so it cannot go stale."""
    print(f"Every demo account uses the password: {bp.DEMO_PASSWORD}\n")
    width = max(len(person.username) for person in bp.STAFF) + 2
    for person in bp.STAFF:
        roles = ", ".join(
            f"{grant.role.value}"
            + ("" if grant.scope_ref is None else f"@{grant.scope_ref}")
            for grant in person.roles
        )
        print(f"{person.username:<{width}} {person.full_name_en}")
        print(f"{'':<{width}} roles: {roles or '(no login roles — teaching record only)'}")
        if person.subject:
            print(
                f"{'':<{width}} teaches: {person.subject} on "
                f"{', '.join(person.rungs) or '(no rungs)'}"
            )
            print(
                f"{'':<{width}} rooms:   "
                f"{', '.join(person.rooms) or '(not yet placed by a supervisor)'}"
            )
        print(f"{'':<{width}} {person.purpose}")
        print()
    return 0


def _print_password_warning() -> None:
    print()
    print(f"Every demo account signs in with the password {bp.DEMO_PASSWORD!r}.")
    print(
        "That is a shared, published, deliberately weak credential for a development "
        "database.\nIt must never exist anywhere a real child's record does."
    )


def main(argv: list[str] | None = None) -> int:
    _configure_output()
    parser = argparse.ArgumentParser(
        prog="python -m sis.demo", description=__doc__.splitlines()[0]
    )
    parser.add_argument(
        "--school",
        default=None,
        help="school code, in multi-school mode. Ignored when there is one database.",
    )
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help=(
            "confirm that a non-SQLite SIS_DATABASE_URL really is a development database. "
            "Never use this against anything holding real records."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    load = sub.add_parser("load", help="write the demo school")
    load.add_argument(
        "--force",
        action="store_true",
        help="write even if a demo school is already present (will usually fail)",
    )
    load.set_defaults(handler=cmd_load)

    sub.add_parser(
        "portfolio", help="write the three populated sales-demo schools into an empty database"
    ).set_defaults(handler=cmd_portfolio)

    sub.add_parser(
        "sync", help="refresh mutable labels without deleting existing demo data"
    ).set_defaults(handler=cmd_sync)

    sub.add_parser("reset", help="remove the demo school, then write it again").set_defaults(
        handler=cmd_reset
    )
    sub.add_parser("drop", help="remove the demo school and stop").set_defaults(
        handler=cmd_drop
    )
    sub.add_parser("status", help="what is in this database").set_defaults(handler=cmd_status)
    sub.add_parser("classes", help="every class and its generated title").set_defaults(
        handler=cmd_classes
    )
    sub.add_parser("accounts", help="the demo credentials table").set_defaults(
        handler=cmd_accounts
    )

    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except seeder.DemoRefused as refusal:
        print(f"Refused: {refusal}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
