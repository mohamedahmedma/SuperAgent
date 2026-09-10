"""Seed the machine credentials the integrations authenticate with.

Stage 9 gave this service an API-key door again (`sis/api/deps.py::_require_scopes`) and
left no way to make the first key. `SIS_BOOTSTRAP_REGISTRAR_KEY` is read by nothing,
`ApiKeyRepository.has_any` — documented as "the guard on one-time bootstrap" — has no
callers, and minting a key over HTTP needs a registrar key to authenticate with. The
result was a closed loop: a freshly migrated database refuses every caller, and there is
no first request that could create the credential to unlock it.

That is not theoretical. It took production down. `identity/` asks "which guardian does
this number reach" before it will send an OTP, got a 401, reported it to the parent as
"we could not reach the school's records", and WhatsApp login stopped for everyone.

## One key per role, and that is forced

`Scope.permits` is exact equality — a `registrar` key does not satisfy a `reader` check —
but the route gate softens that for reads: `require_permission` admits either scope when
the permission is a read, and only `registrar` for a write. So the narrowest scope that
still works is `reader` for the two read-only integrations, and `registrar` only where
something writes.

`docker-compose.yml` hands one `LOCAL_SERVICE_KEY` to five roles. That still works —
see the note on ordering below — but one secret per caller is what makes a compromised
reader unable to rewrite a term's marks, and it is why these are three rows and not one.

Each caller gets its own row, with the narrowest scope that lets it work:

    identity/  -> reader     resolving a guardian, reading a child
    records/   -> reader     every read the parent-facing facade makes
    registrar  -> registrar  writes: imports, structure, minting further keys

## Read from the environment, never written down

The secrets come from env vars, the same way `IDENTITY_BOOTSTRAP_ADMIN_USER` and
`..._PASSWORD` already seed the first administrator. A literal in this file would be a
credential in git history, readable by anyone who ever clones the repository, and
unrotatable without a code change.

Unset means skip. A test fixture, a CI run and a developer's laptop all migrate to head
without inventing credentials, and only a deployment that actually supplies them gets
rows. Nothing here fails a migration: a database that cannot be seeded is a database an
operator must provision, which is a clearer failure than a half-migrated schema.

Idempotent by prefix. Re-running it, or deploying over a database that already has the
keys, changes nothing — and a ROTATED secret is inserted as a new row beside the old one
rather than overwriting it, so a rollout where old and new containers overlap keeps
working with both.
"""
from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timezone

import sqlalchemy as sa
from alembic import op

revision = "0033"
down_revision = "0032"
branch_labels = None
depends_on = None

log = logging.getLogger("alembic.runtime.migration")

#: Mirrors `sis.domain.auth.PREFIX_LENGTH`. Duplicated rather than imported: a migration
#: is a historical record and has to keep meaning what it meant, so it must not change
#: behaviour because application code moved a constant.
_PREFIX_LENGTH = 12

#: `(env var, label, scope)`. The env names are the ones `docker-compose.yml` already
#: passes to the services, so a deployment that works today needs no new variables — and
#: the fallback chain lets a deployment split the shared key one caller at a time.
#: REGISTRAR IS FIRST, and the order is load-bearing.
#:
#: A read route admits either scope (`require_permission` allows `(REGISTRAR, READER)`
#: when the permission is a read); a write route admits only `registrar`. So the widest
#: scope is the one that works everywhere.
#:
#: That matters because of the fallback chain. A deployment that has not yet split its
#: credentials resolves all three entries to the same `LOCAL_SERVICE_KEY`, and only the
#: first can be inserted — the rest collide on the prefix and are skipped. Seeded as
#: `reader` first, that one shared key would authenticate every read and be refused by
#: every import, roster upload and structure change, which is a worse failure than the
#: one this migration exists to fix: it would look fixed.
#:
#: Seeded as `registrar` first, a shared key works everywhere, and a deployment that
#: gives each caller its own secret still gets three narrow rows.
_CREDENTIALS: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("SIS_BOOTSTRAP_REGISTRAR_KEY", "LOCAL_SERVICE_KEY"),
     "bootstrap registrar", "registrar"),
    (("SIS_IDENTITY_API_KEY", "IDENTITY_SIS_API_KEY", "LOCAL_SERVICE_KEY"),
     "identity service", "reader"),
    (("SIS_RECORDS_API_KEY", "RECORDS_API_KEY", "LOCAL_SERVICE_KEY"),
     "records facade", "reader"),
)


def _first_set(names: tuple[str, ...]) -> str:
    for name in names:
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    return ""


def _hash(raw: str) -> str:
    """SHA-256 hex, matching `sis.api.deps.hash_api_key`.

    Inlined for the reason `_PREFIX_LENGTH` is: this file must hash the same way in five
    years as it does today. An encoding difference here would store a verifier nothing
    can ever satisfy, and the only symptom would be every integration failing at once.
    """
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def upgrade() -> None:
    bind = op.get_bind()
    keys = sa.table(
        "api_keys",
        sa.column("prefix", sa.String),
        sa.column("key_hash", sa.String),
        sa.column("label", sa.String),
        sa.column("scope", sa.String),
        sa.column("is_active", sa.Boolean),
        sa.column("created_at", sa.DateTime),
    )
    now = datetime.now(timezone.utc)

    for names, label, scope in _CREDENTIALS:
        secret = _first_set(names)
        if not secret:
            log.info("no secret supplied for the %s key; skipping", label)
            continue

        prefix = secret[:_PREFIX_LENGTH]
        # By prefix, because that is what authentication looks a key up by and what the
        # table makes unique. A second row with the same prefix would make the verify
        # step decide identity by whichever the query returned first.
        exists = bind.execute(
            sa.text("SELECT 1 FROM api_keys WHERE prefix = :prefix"), {"prefix": prefix}
        ).first()
        if exists is not None:
            log.info("the %s key (%s…) is already present; leaving it alone", label, prefix)
            continue

        op.bulk_insert(
            keys,
            [{
                "prefix": prefix,
                "key_hash": _hash(secret),
                "label": label,
                "scope": scope,
                "is_active": True,
                "created_at": now,
            }],
        )
        log.info("seeded the %s key (%s…) with scope %s", label, prefix, scope)


def downgrade() -> None:
    """Remove only the rows this migration could have written.

    By prefix and label together: a key an operator minted over the API must survive a
    downgrade, and matching on the label alone would delete a hand-made row that happened
    to be named the same.
    """
    for names, label, _scope in _CREDENTIALS:
        secret = _first_set(names)
        if not secret:
            continue
        op.get_bind().execute(
            sa.text("DELETE FROM api_keys WHERE prefix = :prefix AND label = :label"),
            {"prefix": secret[:_PREFIX_LENGTH], "label": label},
        )
