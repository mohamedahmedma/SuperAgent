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

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from backend.db.models import User
from backend.infra.database import SessionLocal
from backend.infra.identity import IdentityError, IdentityNotConfigured, verify_token

logger = logging.getLogger(__name__)

# This service verifies bearer tokens. It does not issue them and cannot take part in any
# grant, and `HTTPBearer` is the scheme that says so: /docs then offers one "paste your
# token" box instead of a sign-in form.
#
# It replaced `OAuth2PasswordBearer(tokenUrl=f"{IDENTITY_BASE_URL}/v1/auth/login")`, which
# was wrong three separate ways:
#
#   * It advertised the OAuth2 password grant, so Swagger POSTed form-encoded
#     username/password/client_id/client_secret — while identity's `/v1/auth/login` reads
#     a JSON body (`LoginIn`). The Authorize button could not have worked against it at
#     any URL; it was a 422 waiting to happen.
#   * `tokenUrl` is copied verbatim into the public `openapi.json`, so a deployment that
#     pointed IDENTITY_BASE_URL at an in-cluster name published `http://identity:8200` —
#     a host no browser resolves, and a piece of internal topology no reader needed.
#   * It made this service's documentation depend on another service's address for no
#     runtime benefit whatsoever: nothing here has ever called that URL. Verification is
#     offline against IDENTITY_JWKS_URL (see `backend/infra/identity.py`).
#
# Runtime behaviour is deliberately unchanged. FastAPI builds the same refusal for both
# schemes — 401, detail "Not authenticated", `WWW-Authenticate: Bearer` — so a caller that
# omits the header sees exactly what it saw before.
bearer_scheme = HTTPBearer(
    description=(
        "An access token from the identity service. Obtain one with "
        "POST /v1/auth/login there, then paste it here."
    ),
)


def bearer_token(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> str:
    """The token out of the `Authorization` header, as a plain string.

    One line of adapter, and it earns its place: `HTTPBearer` yields the parsed header
    object where `OAuth2PasswordBearer` yielded the bare parameter. Unwrapping it here
    means `get_current_user` keeps the signature it has always had — `token: str` — so no
    call site and no test had to be touched to change what /docs advertises.
    """
    return credentials.credentials


#: The name this module has always exported for its security dependency, and the module
#: docstring names it as part of the contract. Nothing outside this file imports it today,
#: so it is an alias rather than a deprecation — and it is aliased to `bearer_token`, not
#: to the scheme, because what callers depended on was "the dependency that yields the
#: token", which is what this still is.
oauth2_scheme = bearer_token

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

    __slots__ = (
        "username", "role", "guardian_id", "display_name", "access_token", "children",
    )

    def __init__(
        self,
        username: str,
        role: str,
        guardian_id: str = "",
        children: tuple = (),
        display_name: str = "",
        access_token: str = "",
    ):
        self.username = username
        self.role = role
        self.guardian_id = guardian_id
        # What identity knew about this parent's family when the token was signed.
        # A HINT and never permission — see `mint_access_token`. Held as a tuple so a
        # caller cannot mutate one request's view of a family.
        self.children = tuple(children or ())
        self.display_name = display_name
        # The verified token, kept so it can be relayed to the records facade on this
        # user's behalf. The chat backend cannot forge one and cannot alter whose
        # records it authorises — it only passes along what the caller already proved.
        self.access_token = access_token

    def __repr__(self) -> str:
        """Redacted. `access_token` is a live bearer credential.

        Reprs reach log lines, tracebacks and captured test output, none of which are
        audited for secrets.
        """
        return (
            f"AuthenticatedUser(username={self.username!r}, role={self.role!r}, "
            f"guardian_id={self.guardian_id!r})"
        )


def _children_from(claims: dict) -> tuple:
    """Read the `children` claim, distrusting its shape.

    The token is signed, so this is not about tampering — a bad shape here means identity
    and this service disagree about the contract, and the honest response is to carry
    nothing rather than half a family. The roster is fetched from the records facade
    anyway; this claim only ever saves that call or covers for it being down.
    """
    raw = claims.get("children")
    if not isinstance(raw, list):
        return ()
    found = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        student_id = str(item.get("id") or "").strip()
        if not student_id:
            continue
        found.append(
            {
                "student_id": student_id,
                "full_name_ar": str(item.get("ar") or "").strip(),
                "full_name_en": str(item.get("en") or "").strip(),
                "year_level": str(item.get("yr") or "").strip(),
                "gender": str(item.get("g") or "unspecified").strip() or "unspecified",
            }
        )
    return tuple(found)



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
    token: str = Depends(bearer_token), db: Session = Depends(get_db)
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
        children=_children_from(claims),
        display_name=claims.get("name") or "",
        access_token=token,
    )


def require_admin(current_user: AuthenticatedUser = Depends(get_current_user)) -> AuthenticatedUser:
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient administrator privileges",
        )
    return current_user
