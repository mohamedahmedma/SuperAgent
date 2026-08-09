"""Authentication for the chat backend — now a verifier, not an implementation.

This module used to hash passwords, mint JWTs, and own the login and registration
routes. All of that moved to the identity service. What is left is the half a backend
should have: check a signature, read the claims, hand the route a subject.

The exported names are unchanged on purpose. `get_current_user`, `require_admin`,
`get_db` and `oauth2_scheme` mean what they always meant, so `chat.py`, `sessions.py`,
`assets.py` and `documents.py` did not have to change at all. The dependency was
already the right seam; only what sits behind it moved.

What is deliberately gone, and where it went:

  * password hashing, verification, bcrypt migration  -> identity/auth.py
  * token minting, expiry, the signing secret         -> identity/tokens.py, keys.py
  * login, registration, the admin invite code        -> identity/routes.py
  * the users table as a credential store             -> identity/models.py

`backend.db.models.User` survives as a **local projection**, not a credential store.
`chat_sessions.user_id` is a foreign key to it, so deleting the table would mean
migrating every stored conversation; keeping a row per username costs nothing and lets
session ownership work unchanged. Its `password_hash` column now holds a sentinel that
matches no hash format, and nothing in this codebase reads it any more.
"""
import logging
import os

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from backend.db.models import User
from backend.infra.database import SessionLocal
from backend.infra.identity import IdentityError, IdentityNotConfigured, verify_token

logger = logging.getLogger(__name__)

# Points at the identity service so the OpenAPI "Authorize" button still works. It is
# an absolute URL because that service is a different origin — the old relative
# "/auth/login" would now point at a route this app no longer serves.
IDENTITY_BASE_URL = os.getenv("IDENTITY_BASE_URL", "http://localhost:8200").rstrip("/")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{IDENTITY_BASE_URL}/v1/auth/login")

# Written into the projection row's NOT NULL password_hash. Matches no hash format any
# verifier here accepts, and there is no longer any code that would check it — but a
# value that reads as an obvious sentinel beats an empty string when someone opens the
# table in six months and wonders whether these accounts have blank passwords.
EXTERNAL_CREDENTIAL_SENTINEL = "external:identity-service"


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class AuthenticatedUser:
    """The subject of a request, read from a verified token.

    Duck-compatible with the old `User` ORM object at the only two attributes any
    route ever touched — `.username` and `.role` — which is why no call site changed.
    `.guardian_id` is new and present only for parents.

    Not an ORM object on purpose: it is derived from a signature, not from a row, and
    giving routes something they could mutate and commit would reintroduce exactly the
    coupling this refactor removed.
    """

    __slots__ = ("username", "role", "guardian_id", "display_name")

    def __init__(self, username: str, role: str, guardian_id: str = "", display_name: str = ""):
        self.username = username
        self.role = role
        self.guardian_id = guardian_id
        self.display_name = display_name

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"AuthenticatedUser(username={self.username!r}, role={self.role!r})"


def _ensure_projection_row(db: Session, username: str) -> None:
    """Make sure a `users` row exists for this username.

    Session ownership hangs off `chat_sessions.user_id`, so a user who has never been
    seen here before needs a row before they can save a conversation. Auto-provisioned
    rather than synced, because a sync job between two services is one more thing to
    run, monitor and be woken up by — and the only fact needed is one the token
    already carries.

    A concurrent duplicate insert is caught and ignored: the unique constraint on
    username is the real guard, and losing a race means the row exists, which is the
    outcome wanted anyway.
    """
    if db.query(User.id).filter(User.username == username).first() is not None:
        return

    db.add(User(username=username, password_hash=EXTERNAL_CREDENTIAL_SENTINEL, role="user"))
    try:
        db.commit()
    except Exception:
        db.rollback()
        logger.debug("Projection row for %s already existed", username)


def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
) -> AuthenticatedUser:
    """Verify the bearer token and return its subject.

    The role comes from the **token**, not from the local projection row. That matters:
    the projection is created with role "user" and never updated, so reading authority
    from it would mean an administrator silently losing their privileges here. The
    identity service is the authority on who someone is; this table only records that
    they exist.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired authentication token",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        claims = verify_token(token)
    except IdentityNotConfigured as exc:
        # Fail closed. A backend that cannot verify identities must reject them, not
        # wave them through.
        logger.error("Identity verification is not configured: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is temporarily unavailable",
        )
    except IdentityError:
        raise credentials_exception

    username = claims.get("sub")
    if not username:
        raise credentials_exception

    _ensure_projection_row(db, username)

    return AuthenticatedUser(
        username=username,
        role=claims.get("role") or "user",
        guardian_id=claims.get("guardian_id") or "",
        display_name=claims.get("name") or "",
    )


def require_admin(current_user: AuthenticatedUser = Depends(get_current_user)) -> AuthenticatedUser:
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient administrator privileges",
        )
    return current_user
