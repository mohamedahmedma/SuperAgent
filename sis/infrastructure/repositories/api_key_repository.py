"""`ApiKeyRepository` over SQLAlchemy: find a credential by the public half of it.

Every authenticated request enters here, by `prefix` — the leading characters that
*name* a key without authorising it. The hash is loaded so the caller can verify the
presented secret; it is never compared in SQL, because a `WHERE key_hash = ?` is a
non-constant-time comparison performed by a database that logs slow queries.

`get_by_prefix` deliberately does not filter on `is_active`. A revoked key must be
*found* and then refused, so the caller can say "this key was revoked" instead of "no
such key" — the second sends an operator hunting for a typo in a config file that is
correct. Usability is `ApiKey.is_usable_at`, evaluated against a `now` the caller owns.

Timestamps are reattached to UTC on read: `DateTime(timezone=True)` keeps no offset
under SQLite, and `ApiKey.__post_init__` refuses naive datetimes because an aware `now`
compared against a naive `expires_at` raises `TypeError` — a 500 on every request rather
than a refusal. Everything written here is UTC, so reattaching is a restoration and not
a guess. Nothing commits; the request's session boundary decides.
"""
import logging
from collections.abc import Sequence
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from sis.domain.auth import ApiKey, Scope
from sis.infrastructure.db import models

logger = logging.getLogger(__name__)


class SqlAlchemyApiKeyRepository:
    """Stored credentials, addressed by prefix."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get_by_prefix(self, prefix: str) -> ApiKey | None:
        record = self._find(prefix)
        return None if record is None else _to_domain(record)

    def list_all(self) -> Sequence[ApiKey]:
        """Newest first — the order the key screen is read in."""
        stmt = select(models.ApiKey).order_by(
            models.ApiKey.created_at.desc(), models.ApiKey.id.desc()
        )
        return [_to_domain(record) for record in self._session.scalars(stmt)]

    def add(self, key: ApiKey) -> ApiKey:
        """Store a newly minted key. Only the hash is written; the secret never arrives here."""
        self._session.add(
            models.ApiKey(
                prefix=key.prefix,
                key_hash=key.key_hash,
                label=key.label,
                scope=key.scope.value,
                is_active=key.is_active,
                expires_at=key.expires_at,
                created_at=key.created_at,
                last_used_at=key.last_used_at,
            )
        )
        self._session.flush()
        return key

    def revoke(self, prefix: str) -> ApiKey | None:
        """Deactivate, never delete: an audit line naming a deleted key names nothing."""
        record = self._find(prefix)
        if record is None:
            return None
        record.is_active = False
        self._session.flush()
        return _to_domain(record)

    def touch(self, prefix: str, *, at: datetime) -> None:
        """Record last use. Best effort, and the swallowed exception is the point.

        This column exists so an operator can see which keys are dead weight. Letting it
        fail a request would take the school's data offline because a bookkeeping write
        was contended — and the request it fails is an already-authenticated one. The
        error is logged rather than raised, and no rollback is issued here: the session
        belongs to the caller, whose own error handling decides what to abandon.
        """
        try:
            self._session.execute(
                update(models.ApiKey)
                .where(models.ApiKey.prefix == prefix)
                .values(last_used_at=at)
            )
        except SQLAlchemyError:
            logger.warning("could not record last use of api key %s", prefix, exc_info=True)

    def has_any(self) -> bool:
        """Whether any key exists at all — the guard on one-time bootstrap.

        Asks for one id rather than a `COUNT(*)`: the answer is "is the school
        configured yet", and counting every key to learn that scans the whole table.
        """
        return self._session.scalar(select(models.ApiKey.id).limit(1)) is not None

    def _find(self, prefix: str) -> models.ApiKey | None:
        return self._session.scalars(
            select(models.ApiKey).where(models.ApiKey.prefix == prefix)
        ).first()


def _as_utc(moment: datetime) -> datetime:
    """Reattach UTC to a timestamp SQLite handed back without one. See the module docstring."""
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)


def _to_domain(record: models.ApiKey) -> ApiKey:
    return ApiKey(
        prefix=record.prefix,
        key_hash=record.key_hash,
        label=record.label,
        scope=Scope(record.scope),
        is_active=record.is_active,
        expires_at=None if record.expires_at is None else _as_utc(record.expires_at),
        created_at=_as_utc(record.created_at),
        last_used_at=None if record.last_used_at is None else _as_utc(record.last_used_at),
    )


__all__ = ["SqlAlchemyApiKeyRepository"]
