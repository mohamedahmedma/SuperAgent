"""Copy accounts from the old chat backend's `users` table into this service.

    python -m identity.import_legacy_accounts --source postgresql+psycopg2://.../langchain_app
    python -m identity.import_legacy_accounts --source ... --dry-run

**Password hashes are copied verbatim, not reset.** The old backend wrote PBKDF2 in
the same format this service reads, and older rows may hold passlib bcrypt — which
`identity.auth.verify_password` also accepts, and upgrades in place on the user's next
successful login. So the migration is invisible: nobody is asked to choose a new
password, and the bcrypt rows drain away on their own as people sign in.

The source is read with **raw SQL, not the backend's ORM**. This service must not
import `backend`, and a migration script is exactly where that boundary would
otherwise be broken for convenience — after which the two are coupled forever.

Idempotent: an existing username is skipped, never overwritten. Re-running after a
partial import is safe, and a second run cannot clobber a password someone has since
changed here.

It deliberately does NOT set `guardian_external_id` on anything. The old system had no
concept of a guardian, so there is nothing to carry over, and inventing a binding
during a bulk import is precisely the mistake that would hand one family's records to
another. Bindings are a separate, deliberate, admin-only act.
"""
import argparse
import sys

from sqlalchemy import create_engine, text

from identity.auth import ASSIGNABLE_ROLES
from identity.db import init_db, new_session
from identity.models import Account


def import_accounts(source_url: str, *, dry_run: bool = False) -> dict:
    source = create_engine(source_url)
    with source.connect() as connection:
        rows = connection.execute(
            text("SELECT username, password_hash, role FROM users ORDER BY id")
        ).fetchall()

    created = 0
    skipped = 0
    db = new_session()
    try:
        for row in rows:
            username = (row.username or "").strip()
            if not username:
                skipped += 1
                continue

            if db.query(Account).filter(Account.username == username).first():
                skipped += 1
                continue

            # The old vocabulary was {user, admin}; both survive unchanged. Anything
            # unrecognised lands on "user" rather than being guessed upward.
            role = (row.role or "user").strip().lower()
            if role not in ASSIGNABLE_ROLES:
                role = "user"

            if not dry_run:
                db.add(
                    Account(
                        username=username,
                        password_hash=row.password_hash or "",
                        role=role,
                        display_name=username,
                    )
                )
            created += 1

        if not dry_run:
            db.commit()
    finally:
        db.close()

    return {"source_rows": len(rows), "created": created, "skipped": skipped}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="SQLAlchemy URL of the old backend database.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would happen, write nothing.")
    args = parser.parse_args()

    init_db()
    result = import_accounts(args.source, dry_run=args.dry_run)

    print(
        f"{'[dry run] ' if args.dry_run else ''}"
        f"read {result['source_rows']}, "
        f"{'would create' if args.dry_run else 'created'} {result['created']}, "
        f"skipped {result['skipped']} (already present)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
