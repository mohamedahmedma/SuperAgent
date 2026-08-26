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
is applied *before* the system of record is queried, not as a filter over results.
Nothing the model says — no clever phrasing, no injected instruction inside a chat
message — reaches this decision, because the decision is made from the URL's guardian id
and the school's own link, and the system of record is never asked about a student that
check excluded.

**And it is no longer the only check.** The guardian handle now travels with the read, so
`sis/` re-checks the link itself before answering — see `records/sis_adapter.py`. Two
refusals made independently from the same registrar data, rather than one made here and
trusted downstream. Nothing below is redundant because of it: this check is what stops a
request early, writes the audit row, and keeps the failure modes distinguishable to an
operator while staying indistinguishable to a caller. What the second one buys is that a
fully compromised facade reaches one family instead of the school.
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
from records.guardian_directory import (
    GuardianDirectoryUnavailable,
    PermittedStudent,
    get_directory,
)
from records.models import AccessAudit, ApiKey

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

    def __init__(self, caller: Caller, guardian_id: str, school_code: str | None = None):
        self.caller = caller
        self.guardian_id = guardian_id
        #: Which school's database answers for this parent, off the token's `school`
        #: claim. `None` in a single-school estate. Carried here rather than looked up,
        #: because it was settled at sign-in from the WhatsApp number the parent messaged
        #: and nothing since has been in a position to know better.
        self.school_code = school_code


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

    return ParentSubject(
        caller=caller,
        guardian_id=claimed_guardian,
        school_code=identity.school_from_claims(claims),
    )


def resolve_permitted_student(
    db: Session,
    *,
    guardian_external_id: str,
    student_external_id: str,
    caller: Caller,
    endpoint: str,
    school_code: str | None = None,
) -> PermittedStudent:
    """The first of two checks. Every parent-facing read passes through here.

    Returns the child only when the school's system of record says this guardian may be
    told about her. Anything else raises, and every outcome — including the successful
    one — is written to the audit before this function returns.

    It stopped being *the* chokepoint when the guardian handle started travelling with the
    read: `sis/` makes the same decision again, from the same data, before answering. This
    one still earns its place — it fails the request before a second service is troubled,
    it is where the audit is written, and it is where the reason behind a refusal is
    recorded. It is no longer the only thing standing between a compromised caller and
    another family's child.

    **The answer is asked for, never remembered.** This service used to hold guardian links
    in its own tables; it now puts the question to SIS on every request, so a registrar
    revoking access the minute a court order arrives takes effect on the next question
    rather than whenever something here was next synchronised.

    Note what stays deliberately indistinguishable from the caller's side: an unknown
    student, a student who exists but is not this guardian's, and a student whose records
    are restricted all produce the same 404 and the same message. The audit records which
    one actually happened; the response does not, because a caller who could tell them
    apart could enumerate the student body and detect custody restrictions by their error
    code alone.

    A directory that cannot be reached is the one case that is *not* a 404. Telling a
    parent "no such child" because another service was briefly down is a lie about their
    own family, so it is a 503 and says so.
    """
    try:
        # The full list rather than a single lookup, so the audit can still say *why* a
        # denial happened. It costs nothing extra — `permits` reads the same list — and
        # the reason is the part that matters later: a run of `no_children` against one
        # guardian is somebody probing with a handle that reaches nobody, while a run of
        # `no_link` is somebody walking student numbers against a real parent's handle.
        children = get_directory().children_of(
            guardian_external_id, school_code=school_code
        )
        student = next(
            (child for child in children if child.student_id == str(student_external_id)),
            None,
        )
    except GuardianDirectoryUnavailable as error:
        logger.error("Guardian directory unavailable: %s", error)
        write_audit(
            db,
            endpoint=endpoint,
            allowed=False,
            reason="directory_unavailable",
            guardian_external_id=guardian_external_id,
            student_external_id=student_external_id,
            api_key_prefix=caller.prefix,
            request_id=caller.request_id,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "not_configured",
                "message": "The school's records are temporarily unavailable.",
            },
        ) from error

    if student is not None:
        reason = "ok"
    elif not children:
        # The handle reaches nobody the school will talk about: an unknown guardian, or
        # one every link of whose is restricted. SIS filters restricted links out before
        # answering, so those two arrive here identical — deliberately, since a caller who
        # could tell them apart could detect a custody restriction from the outside.
        reason = "no_children"
    else:
        reason = "no_link"

    write_audit(
        db,
        endpoint=endpoint,
        allowed=student is not None,
        reason=reason,
        guardian_external_id=guardian_external_id,
        student_external_id=student_external_id,
        api_key_prefix=caller.prefix,
        request_id=caller.request_id,
    )

    if student is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "not_found", "message": "No such student record for this guardian."},
        )

    return student


def permitted_students(
    db: Session, *, guardian_external_id: str, school_code: str | None = None
) -> list[PermittedStudent]:
    """Every child this guardian may ask about. Restricted links are already excluded.

    The filtering happens in SIS rather than here — it returns only links carrying
    `can_view_records` — so a barred parent arrives holding no children at all rather than
    holding children this service would have to remember to hide.

    An unreachable directory returns empty rather than raising, because the only caller is
    the "list my children" route and an empty list there renders as "no children on file",
    which is recoverable. The read routes below it raise properly.
    """
    try:
        return list(
            get_directory().children_of(guardian_external_id, school_code=school_code)
        )
    except GuardianDirectoryUnavailable as error:
        logger.error("Guardian directory unavailable while listing children: %s", error)
        return []


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
