"""Authentication, authorisation, and the audit trail they write.

The rule this module exists to enforce, stated once:

    An API key proves *which system* is calling. It never proves *which parent* is
    asking. Those are two separate facts and both are required before a single grade
    is returned.

A leaked agent key must therefore be worth nothing on its own. It buys the ability to
ask on behalf of a guardian; the guardian's own permitted-student set is resolved here,
from the database, on every request. There is no code path in which the caller supplies
the answer to "which students may I see".

The corollary, which matters because the caller is a language model: the permitted set
is applied *before* the LMS is queried, not as a filter over results. Nothing the model
says — no clever phrasing, no injected instruction inside a chat message — reaches this
decision, because the decision is made from the URL's guardian id and the link table,
and the LMS is never asked about a student that check excluded.
"""
import hashlib
import hmac
import logging
import os
import secrets
from datetime import datetime, timezone

from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy.orm import Session

from records import identity
from records.db import get_db
from records.models import AccessAudit, ApiKey, Guardian, GuardianStudent, Student

logger = logging.getLogger(__name__)

KEY_PREFIX_LENGTH = 8
_KEY_BYTES = 32


def generate_api_key() -> tuple[str, str, str]:
    """Return `(full_key, prefix, key_hash)`.

    SHA-256 rather than bcrypt/PBKDF2 is the right call *here* and nowhere near
    passwords: the input is 32 bytes of CSPRNG output, so there is no dictionary to
    attack and key stretching would only add latency to every request. Stretching
    protects low-entropy secrets; this is not one.
    """
    raw = secrets.token_urlsafe(_KEY_BYTES)
    prefix = raw[:KEY_PREFIX_LENGTH]
    return raw, prefix, _hash_key(raw)


def _hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def write_audit(
    db: Session,
    *,
    endpoint: str,
    allowed: bool,
    reason: str,
    guardian_external_id: str = "",
    student_external_id: str = "",
    api_key_prefix: str = "",
    request_id: str = "",
) -> None:
    """Append one access-attempt row, and commit it on its own.

    Committed separately from whatever the request goes on to do, because the audit
    must survive the request failing. An audit that is rolled back alongside a denial
    records only the accesses that succeeded — which is exactly backwards, since the
    denials are the interesting ones.
    """
    db.add(
        AccessAudit(
            guardian_external_id=guardian_external_id or "",
            student_external_id=student_external_id or "",
            api_key_prefix=api_key_prefix or "",
            endpoint=endpoint[:160],
            allowed=allowed,
            reason=reason[:40],
            request_id=request_id or "",
        )
    )
    db.commit()


class Caller:
    """The authenticated *system* behind a request. Never a person."""

    def __init__(self, prefix: str, scope: str, request_id: str = ""):
        self.prefix = prefix
        self.scope = scope
        self.request_id = request_id


def _require_scope(required: str):
    def dependency(
        request: Request,
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
        x_request_id: str | None = Header(default=None, alias="X-Request-Id"),
        db: Session = Depends(get_db),
    ) -> Caller:
        if not x_api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"code": "not_authorized", "message": "Missing X-API-Key."},
            )

        prefix = x_api_key[:KEY_PREFIX_LENGTH]
        record = db.query(ApiKey).filter(ApiKey.prefix == prefix, ApiKey.is_active.is_(True)).first()

        # Hash unconditionally, even when no row matched, so a wrong prefix and a
        # wrong secret take the same time. Skipping the work on a miss turns key
        # enumeration into a timing measurement.
        candidate = _hash_key(x_api_key)
        expected = record.key_hash if record else _hash_key("")
        matched = hmac.compare_digest(candidate, expected) and record is not None

        if not matched:
            write_audit(
                db,
                endpoint=str(request.url.path),
                allowed=False,
                reason="not_authorized",
                api_key_prefix=prefix,
                request_id=x_request_id or "",
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"code": "not_authorized", "message": "Invalid API key."},
            )

        if record.expires_at is not None and record.expires_at < _now():
            write_audit(
                db,
                endpoint=str(request.url.path),
                allowed=False,
                reason="key_expired",
                api_key_prefix=prefix,
                request_id=x_request_id or "",
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"code": "not_authorized", "message": "Expired API key."},
            )

        # Scopes do not nest. An admin key manages links and keys; it cannot read a
        # child's grades through the parent-facing routes. Making admin a superset
        # would mean the most widely-shared credential in the school is also the one
        # that reads every record.
        if record.scope != required:
            write_audit(
                db,
                endpoint=str(request.url.path),
                allowed=False,
                reason="wrong_scope",
                api_key_prefix=prefix,
                request_id=x_request_id or "",
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"code": "not_authorized", "message": f"Key scope '{record.scope}' cannot call this endpoint."},
            )

        record.last_used_at = _now()
        db.commit()
        return Caller(prefix=record.prefix, scope=record.scope, request_id=x_request_id or "")

    return dependency


require_agent_key = _require_scope("agent")
require_admin_key = _require_scope("admin")


class ParentSubject:
    """A verified (system, parent) pair. Both halves proved, neither assumed."""

    def __init__(self, caller: Caller, guardian_id: str):
        self.caller = caller
        self.guardian_id = guardian_id


def require_parent_subject(
    guardian_id: str,
    request: Request,
    authorization: str | None = Header(default=None),
    caller: Caller = Depends(require_agent_key),
    db: Session = Depends(get_db),
) -> ParentSubject:
    """Both credentials, checked together. Every parent-facing route depends on this.

    The API key has already proved which system is calling. This adds the second,
    independent proof: a token signed by the identity service naming the guardian.
    The `guardian_id` in the path must equal the `guardian_id` in the signed claim.

    That equality check is the point. It means the calling system cannot choose whose
    records it reads — it can only relay a parent's own identity, because it has no
    way to produce a signature for a different one. A fully compromised chat backend
    still cannot read a family it does not hold a token for.

    FastAPI supplies `guardian_id` from the path, so a route that declares this
    dependency without a `{guardian_id}` segment fails at startup rather than
    silently skipping the comparison.
    """
    token = ""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()

    try:
        claims = identity.verify_token(token)
        claimed_guardian = identity.guardian_id_from_claims(claims)
    except identity.IdentityNotConfigured as exc:
        # Fail closed. No verification material means no reads, not unverified reads.
        logger.error("Identity verification is not configured: %s", exc)
        write_audit(
            db,
            endpoint=str(request.url.path),
            allowed=False,
            reason="identity_not_configured",
            guardian_external_id=guardian_id,
            api_key_prefix=caller.prefix,
            request_id=caller.request_id,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "not_configured", "message": "Identity verification unavailable."},
        )
    except identity.IdentityError:
        write_audit(
            db,
            endpoint=str(request.url.path),
            allowed=False,
            reason="invalid_identity",
            guardian_external_id=guardian_id,
            api_key_prefix=caller.prefix,
            request_id=caller.request_id,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "not_authorized", "message": "Missing or invalid identity token."},
        )

    if claimed_guardian != guardian_id:
        # The signature was valid but named someone else. This is the signal that a
        # caller is relaying one parent's token while asking about another, and it is
        # audited under its own reason so it can be alerted on.
        write_audit(
            db,
            endpoint=str(request.url.path),
            allowed=False,
            reason="guardian_mismatch",
            guardian_external_id=guardian_id,
            api_key_prefix=caller.prefix,
            request_id=caller.request_id,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "not_authorized", "message": "Token does not authorise this guardian."},
        )

    return ParentSubject(caller=caller, guardian_id=claimed_guardian)


def resolve_permitted_student(
    db: Session,
    *,
    guardian_external_id: str,
    student_external_id: str,
    caller: Caller,
    endpoint: str,
) -> Student:
    """The chokepoint. Every parent-facing read passes through here first.

    Returns the student only when an active link exists *and* carries
    `can_view_records`. Anything else raises, and every outcome — including the
    successful one — is written to the audit before this function returns.

    Note what is deliberately indistinguishable from the caller's side: an unknown
    student, a student who exists but is not this guardian's, and a student whose
    records are restricted all produce the same 404 and the same message. The audit
    records which one actually happened; the response does not, because a caller who
    can tell those apart can enumerate the student body and detect custody
    restrictions by their error code alone.
    """
    guardian = (
        db.query(Guardian)
        .filter(Guardian.external_id == guardian_external_id, Guardian.is_active.is_(True))
        .first()
    )
    student = db.query(Student).filter(Student.external_id == student_external_id).first()

    link = None
    if guardian is not None and student is not None:
        link = (
            db.query(GuardianStudent)
            .filter(
                GuardianStudent.guardian_id == guardian.id,
                GuardianStudent.student_id == student.id,
            )
            .first()
        )

    if guardian is None:
        reason = "unknown_guardian"
    elif student is None:
        reason = "unknown_student"
    elif link is None:
        reason = "no_link"
    elif not link.can_view_records:
        reason = "records_restricted"
    else:
        reason = "ok"

    write_audit(
        db,
        endpoint=endpoint,
        allowed=reason == "ok",
        reason=reason,
        guardian_external_id=guardian_external_id,
        student_external_id=student_external_id,
        api_key_prefix=caller.prefix,
        request_id=caller.request_id,
    )

    if reason != "ok":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "not_found", "message": "No such student record for this guardian."},
        )

    return student


def permitted_students(db: Session, *, guardian_external_id: str) -> list[Student]:
    """Every student this guardian may ask about. Restricted links are excluded."""
    guardian = (
        db.query(Guardian)
        .filter(Guardian.external_id == guardian_external_id, Guardian.is_active.is_(True))
        .first()
    )
    if guardian is None:
        return []

    rows = (
        db.query(Student)
        .join(GuardianStudent, GuardianStudent.student_id == Student.id)
        .filter(
            GuardianStudent.guardian_id == guardian.id,
            GuardianStudent.can_view_records.is_(True),
            Student.is_active.is_(True),
        )
        .all()
    )
    return list(rows)


def bootstrap_admin_key(db: Session) -> str | None:
    """Create the first admin key from `RECORDS_BOOTSTRAP_ADMIN_KEY`, once.

    Solves the chicken-and-egg of needing an admin key to create an admin key. It is
    a no-op when any admin key already exists, so leaving the variable set in an
    environment file cannot silently mint a second one.
    """
    raw = os.getenv("RECORDS_BOOTSTRAP_ADMIN_KEY")
    if not raw:
        return None
    if db.query(ApiKey).filter(ApiKey.scope == "admin").first() is not None:
        return None

    db.add(
        ApiKey(
            prefix=raw[:KEY_PREFIX_LENGTH],
            key_hash=_hash_key(raw),
            label="bootstrap admin",
            scope="admin",
        )
    )
    db.commit()
    return raw[:KEY_PREFIX_LENGTH]
