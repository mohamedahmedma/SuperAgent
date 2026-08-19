"""Request-scoped wiring for the HTML surface: who is at the keyboard, and what they get.

The companion to `sis/api/deps.py`, and deliberately thin beside it. Everything about a
key — what a valid one looks like, which scope it carries, whether it is the bootstrap
value or a stored row, when its last use is recorded — is decided *there* and reused
here by calling `require_registrar` directly. There is no second implementation of key
checking in this file, because two of them would drift and the drift would be silent:
one surface would keep honouring a revoked key that the other had already stopped
accepting, and nothing in either log would say so.

**A browser cannot send `X-API-Key`.** An HTML form has no way to set a header, so the
credential has to travel as a cookie. That is the whole reason this module exists, and
the cookie is deliberately narrower than the `sessionStorage` it replaces:

* `HttpOnly` — JavaScript cannot read it *at all*. `sessionStorage` is readable by any
  script that gets onto the page, so one injected `<script>` exfiltrated the registrar's
  key and, with it, write access to every child's record. This closes that entirely.
* `SameSite=Strict` — **this is the CSRF defence for every form in this UI.** A POST
  originating from another site does not carry the cookie, so it authenticates as
  nobody and is redirected to the login page instead of committing an import. Do not
  "simplify" this to `Lax`: `Lax` sends the cookie on top-level cross-site *GET*
  navigations, and the moment somebody adds a link-shaped destructive action, or a
  handler that accepts either verb, the hole is open again. If a genuine cross-site
  entry point is ever needed, add per-form tokens first — do not weaken this.
* `Path=/ui` — not sent to `/v1/...`, so the JSON API keeps proving identity the way it
  always has and a stray cookie can never stand in for a header there.
* `Secure` whenever the request arrived over HTTPS, so a deployment behind TLS never
  emits the key over a plaintext hop.

The key is never put in a URL, a hidden field, a log line or a template. A URL is
written to the access log, the browser history and the `Referer` header of the next
request; a hidden field is readable by any script and lands in a page cache. The value
exists in one place — the cookie — and reaches Python as one function argument.

Flashes ("37 rows committed") are the other browser-shaped problem: a redirect discards
whatever the POST handler wanted to say. They travel in their own short-lived signed
cookie rather than a server-side session store, because a store would be shared mutable
state between workers, would need eviction, and would put a registrar's messages on a
different replica from the one serving her next request.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import logging
import secrets
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Final

from fastapi import Depends, HTTPException, Request, status
from starlette.responses import RedirectResponse, Response

from sis.api import deps as api_deps
from sis.api.deps import (
    ApiKeyMinter,
    ApiKeyMinterDep,
    Caller,
    GradeImportServiceDep,
    ImportReports,
    ImportReportsDep,
    MaxUploadBytesDep,
    QueryServiceDep,
    RosterImportServiceDep,
    StructureCatalogue,
    StructureCatalogueDep,
    StructureServiceDep,
    UnitOfWorkFactoryDep,
)
from sis.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The shape of the UI's URL space. Every module in this package uses these names
# rather than repeating the strings, so the prefix is movable in one edit.
# ---------------------------------------------------------------------------

UI_PREFIX: Final[str] = "/ui"
"""Mount point of the whole HTML surface. `/v1/...` is untouched by anything here."""

LOGIN_PATH: Final[str] = f"{UI_PREFIX}/login"
LOGOUT_PATH: Final[str] = f"{UI_PREFIX}/logout"
DASHBOARD_PATH: Final[str] = f"{UI_PREFIX}/"
STATIC_PREFIX: Final[str] = f"{UI_PREFIX}/static"

SESSION_COOKIE: Final[str] = "sis_session"
FLASH_COOKIE: Final[str] = "sis_flash"

COOKIE_PATH: Final[str] = UI_PREFIX
"""Scopes both cookies to the UI. The JSON API must never see either of them."""

SEE_OTHER: Final[int] = status.HTTP_303_SEE_OTHER
"""The redirect of POST/Redirect/GET: 303 turns the follow-up into a GET, unlike 302."""


# ---------------------------------------------------------------------------
# The session cookie
# ---------------------------------------------------------------------------


def is_secure_request(request: Request) -> bool:
    """Whether this request reached us over TLS, honouring one reverse proxy hop.

    `request.url.scheme` alone says `http` for every request behind nginx or a load
    balancer that terminates TLS, which would mean the `Secure` flag is never set in
    exactly the deployments that have HTTPS. `X-Forwarded-Proto` is trusted only for
    this one decision, and the worst a forged header can do is add a flag that makes the
    cookie *stricter*.
    """
    forwarded = (request.headers.get("x-forwarded-proto") or "").split(",")[0].strip()
    return (forwarded or request.url.scheme).lower() == "https"


def set_session(response: Response, key: str, *, request: Request | None = None) -> None:
    """Log a registrar in by storing her key in the session cookie.

    Called from exactly one place — the login handler, after `authenticate` has already
    accepted the key. Storing an unverified value would hand the next request a cookie
    that fails on every page and looks to the user like the site is broken rather than
    like the password was wrong.

    No `Max-Age`: this is a browser-session cookie, so closing the browser ends the
    session. On the shared machine in a school office that is the difference between the
    next person seeing a login page and the next person being the registrar.

    Pass `request` so the `Secure` flag can follow the scheme. Omitting it deliberately
    leaves the flag off rather than guessing `True`, because a `Secure` cookie set over
    plain HTTP is dropped by the browser without a word: the login form would appear to
    succeed and every page after it would bounce straight back to login, with nothing in
    any log to explain it.
    """
    response.set_cookie(
        SESSION_COOKIE,
        key,
        path=COOKIE_PATH,
        httponly=True,
        # SameSite=Strict is this UI's CSRF defence. See the module docstring before
        # changing it; `lax` re-opens cross-site GET navigation.
        samesite="strict",
        secure=is_secure_request(request) if request is not None else False,
    )


def clear_session(response: Response) -> None:
    """Log out: delete the cookie with the same attributes it was set with.

    Path must match. A `delete_cookie` on the default `/` leaves the `/ui`-scoped cookie
    exactly where it was, and the user stays logged in after clicking "Sign out" — the
    single most misleading outcome this file can produce.
    """
    response.delete_cookie(
        SESSION_COOKIE, path=COOKIE_PATH, httponly=True, samesite="strict"
    )


# ---------------------------------------------------------------------------
# Authentication — borrowed wholesale from the API layer
# ---------------------------------------------------------------------------


def authenticate(key: str) -> Caller | None:
    """Verify a presented key, or return `None`. Registrar scope required.

    Delegates to `sis.api.deps.require_registrar`, called as a plain function rather
    than as a FastAPI dependency. That one call carries the whole of the API's key
    handling: the constant-time bootstrap comparison, the stored-key lookup and hash
    check, the expiry and revocation tests, the exact-equality scope check, and the
    best-effort recording of last use. Re-implementing any of it here — even the "is it
    the bootstrap value" line, which is four lines long — would create a second
    definition of who may write to the register, and the two would diverge on the first
    change nobody thought to make twice.

    The `HTTPException` it raises for a bad key is swallowed into `None` on purpose. A
    browser is not an API client: it needs a login page, not a 401 envelope, and the
    caller of this function is the one that knows which of those to produce. Nothing is
    logged here beyond what the API layer already logs, and the key itself never is.
    """
    presented = (key or "").strip()
    if not presented:
        return None
    try:
        return api_deps.require_registrar(x_api_key=presented)
    except HTTPException:
        # 401 (absent, unknown, wrong, revoked, expired) and 403 (wrong scope) are
        # answered identically, exactly as the API answers them: telling a browser which
        # of those it was is the same enumeration hint in a friendlier font.
        return None


class LoginRequired(HTTPException):
    """Raised to bounce an unauthenticated browser to the login page.

    An `HTTPException` subclass rather than a bespoke exception with its own handler,
    because `sis/api/errors.py` already installs a handler for every `HTTPException` and
    that handler passes `headers` through untouched. So this redirects correctly with no
    registration in `sis/app.py` at all — and, just as important, it cannot be broken by
    an agent editing the app's handler list without knowing this file exists.

    303 rather than 302: the interrupted request may have been a POST, and 302 lets a
    browser replay it as a POST at the login page.
    """

    def __init__(self, next_path: str | None = None) -> None:
        target = LOGIN_PATH
        safe = safe_next(next_path)
        if safe:
            # Only ever a path this service owns — see `safe_next`.
            target = f"{LOGIN_PATH}?next={_quote(safe)}"
        super().__init__(
            status_code=SEE_OTHER,
            detail={"code": "not_authorized", "message": "Sign in to continue."},
            headers={"Location": target, "Cache-Control": "no-store"},
        )


def safe_next(raw: str | None) -> str | None:
    """Sanitise a post-login destination, or return `None`.

    An open redirect is the classic form of this bug: `?next=https://evil.example` sends
    a registrar who has just typed her key to a page that looks like this one and asks
    her to type it again. Only a path *inside this UI* is ever accepted, so the value can
    do nothing but move the user around the service she just signed in to.

    `//host` and `/\\host` are rejected explicitly. Both are protocol-relative URLs that
    begin with a slash and therefore pass the naive "starts with /" test, and browsers
    read the backslash form as the forward-slash form.
    """
    if not raw:
        return None
    candidate = raw.strip()
    if not candidate.startswith(UI_PREFIX):
        return None
    rest = candidate[len(UI_PREFIX) :]
    # Guards `/ui` itself (rest == "") and `/ui/...`, while rejecting `/uixyz`.
    if rest and not rest.startswith("/"):
        return None
    if candidate.startswith(("//", "/\\")) or "\\" in candidate or "\n" in candidate:
        return None
    return candidate


def _quote(value: str) -> str:
    """Percent-encode a path for use as a query value."""
    from urllib.parse import quote

    return quote(value, safe="/")


def current_caller(request: Request) -> Caller:
    """The signed-in registrar, or a redirect to the login page. The gate for every page.

    Every route under `/ui` except the login form itself declares this, and it is the
    only thing standing between an anonymous browser and a school's records. It is a
    plain dependency rather than a router-level `dependencies=[...]` argument so that a
    handler receives the `Caller` and can name it in an audit line.

    The current path is carried into the redirect so that signing in returns the user to
    the page she asked for, which is what stops a session timeout mid-import from also
    losing her place.
    """
    presented = request.cookies.get(SESSION_COOKIE)
    if not presented:
        raise LoginRequired(_path_of(request))

    caller = authenticate(presented)
    if caller is None:
        # A key that was valid this morning and has since been revoked lands here. The
        # stale cookie is not cleared on this response — the login page clears it — so
        # that this stays a pure read and can be used from any handler.
        raise LoginRequired(_path_of(request))
    return caller


def optional_caller(request: Request) -> Caller | None:
    """The signed-in registrar if there is one, without redirecting.

    For the login page alone, which needs to know whether to show a form or send an
    already-authenticated user on to the dashboard. Using `current_caller` there would
    redirect the login page to the login page.
    """
    presented = request.cookies.get(SESSION_COOKIE)
    return authenticate(presented) if presented else None


def _path_of(request: Request) -> str | None:
    """The path and query of the current request, for `?next=`. Never the full URL."""
    path = request.url.path
    if request.method != "GET":
        # Bouncing back to a POST target as a GET would render a page that only exists
        # to receive a form. The parent listing is the useful destination, and there is
        # no way to know it here, so the dashboard is the honest answer.
        return None
    query = request.url.query
    return f"{path}?{query}" if query else path


# ---------------------------------------------------------------------------
# Flash messages
# ---------------------------------------------------------------------------

FLASH_LEVELS: Final[frozenset[str]] = frozenset(
    {"success", "info", "warning", "danger"}
)
"""Bootstrap contextual names, used verbatim as `alert-{level}` in `_flash.html`."""

_DEFAULT_LEVEL: Final[str] = "info"
_FLASH_TTL_SECONDS: Final[int] = 120
"""Long enough to survive a redirect, short enough that a stale message cannot reappear."""

_MAX_FLASHES: Final[int] = 8
_MAX_FLASH_COOKIE_BYTES: Final[int] = 3072
"""Browsers drop a cookie over ~4KB, and drop it silently. Truncate before they do."""

_PENDING_ATTR: Final[str] = "_sis_pending_flashes"
_TAKEN_ATTR: Final[str] = "_sis_taken_flashes"

_FLASH_KEY_INFO: Final[bytes] = b"sis.ui.flash.v1"
_PROCESS_FLASH_KEY: Final[bytes] = secrets.token_bytes(32)


@dataclass(frozen=True, slots=True)
class Flash:
    """One message awaiting display. `level` is always one of `FLASH_LEVELS`."""

    message: str
    level: str = _DEFAULT_LEVEL


def _flash_signing_key() -> bytes:
    """The HMAC key for flash cookies.

    Derived from the bootstrap registrar key when one is configured, so that every
    worker and every replica agrees and a flash survives a redirect that lands on a
    different process. The derivation is one-way and the derived value never leaves the
    process — what travels is a tag over the message text, never the key material.

    Without a configured bootstrap key the fallback is a per-process random secret,
    which is correct but means a flash can be dropped when two workers disagree. Dropped
    is the right failure: the alternative, an unsigned cookie, lets anyone who can set a
    cookie on this host paint arbitrary text into a page the registrar trusts — "Import
    committed successfully" over an import that did nothing at all.
    """
    configured = get_settings().bootstrap_registrar_key
    if not configured:
        return _PROCESS_FLASH_KEY
    return hmac.new(
        configured.encode("utf-8"), _FLASH_KEY_INFO, hashlib.sha256
    ).digest()


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _unb64(raw: str) -> bytes:
    return base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4))


def _encode_flashes(messages: Sequence[Flash]) -> str:
    payload = json.dumps(
        [{"m": item.message, "l": item.level} for item in messages],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    tag = hmac.new(_flash_signing_key(), payload, hashlib.sha256).digest()
    # Truncated to 16 bytes: 128 bits of authentication on a message that is public
    # anyway and expires in two minutes, at half the cookie budget of a full digest.
    return f"{_b64(payload)}.{_b64(tag[:16])}"


def _decode_flashes(raw: str) -> tuple[Flash, ...]:
    """Verify and parse a flash cookie. Anything unexpected yields no messages at all.

    Every failure path is the same silent empty tuple. A tampered or corrupted cookie is
    not an error worth showing a registrar — she did not cause it and cannot fix it —
    and rendering "could not read your messages" above a page that is otherwise correct
    would send her looking for a problem that does not exist.
    """
    try:
        encoded, _, tag = raw.partition(".")
        if not encoded or not tag:
            return ()
        payload = _unb64(encoded)
        expected = hmac.new(_flash_signing_key(), payload, hashlib.sha256).digest()[:16]
        if not hmac.compare_digest(_unb64(tag), expected):
            return ()
        parsed = json.loads(payload)
        if not isinstance(parsed, list):
            return ()
    except (ValueError, TypeError, binascii.Error, UnicodeDecodeError):
        return ()

    messages: list[Flash] = []
    for item in parsed[:_MAX_FLASHES]:
        if not isinstance(item, dict):
            continue
        text = str(item.get("m", ""))
        level = str(item.get("l", _DEFAULT_LEVEL))
        if text:
            messages.append(
                Flash(text, level if level in FLASH_LEVELS else _DEFAULT_LEVEL)
            )
    return tuple(messages)


def flash(request: Request, message: str, level: str = _DEFAULT_LEVEL) -> None:
    """Queue a one-shot message for the page the user lands on next.

    Queued on the request, not written to a cookie here, because the cookie belongs to a
    response object this function has never seen. `redirect()` and
    `templates.TemplateResponse()` both flush the queue on their way out, and between
    them they cover every way a handler can finish — so a handler only ever says *what*
    to show and never has to remember to attach it.

    An unknown `level` is corrected to `info` rather than rejected: a typo'd level should
    show the registrar her message in the wrong colour, not lose it.
    """
    text = (message or "").strip()
    if not text:
        return
    pending: list[Flash] = getattr(request.state, _PENDING_ATTR, None) or []
    if len(pending) >= _MAX_FLASHES:
        return
    pending.append(Flash(text, level if level in FLASH_LEVELS else _DEFAULT_LEVEL))
    setattr(request.state, _PENDING_ATTR, pending)


def take_flashes(request: Request) -> tuple[Flash, ...]:
    """Read the messages carried into this request, consuming them.

    Idempotent within one request: the result is memoised, so a template that includes
    the flash region twice (a page header and a sticky banner, say) shows each message
    once rather than twice. The cookie is cleared by `write_flashes` on the way out,
    which is what makes them one-shot across requests.
    """
    taken = getattr(request.state, _TAKEN_ATTR, None)
    if taken is not None:
        return taken
    raw = request.cookies.get(FLASH_COOKIE)
    messages = _decode_flashes(raw) if raw else ()
    setattr(request.state, _TAKEN_ATTR, messages)
    return messages


def write_flashes(request: Request, response: Response) -> None:
    """Attach queued messages to `response`, or clear a cookie that has been shown.

    Called by `redirect()` and by the template helper; a handler should not need it.
    Setting and deleting are mutually exclusive on purpose — emitting both `Set-Cookie`
    headers for one name is undefined between browsers, and the failure mode is a
    message that either never appears or never goes away.
    """
    pending: list[Flash] = getattr(request.state, _PENDING_ATTR, None) or []
    if pending:
        encoded = _encode_flashes(pending)
        while len(encoded.encode("utf-8")) > _MAX_FLASH_COOKIE_BYTES and len(pending) > 1:
            # Drop from the front: the last thing a handler said is the outcome, and the
            # outcome is what the registrar needs to see.
            pending = pending[1:]
            encoded = _encode_flashes(pending)
        response.set_cookie(
            FLASH_COOKIE,
            encoded,
            max_age=_FLASH_TTL_SECONDS,
            path=COOKIE_PATH,
            httponly=True,
            samesite="strict",
            secure=is_secure_request(request),
        )
        setattr(request.state, _PENDING_ATTR, [])
        return

    if FLASH_COOKIE in request.cookies:
        response.delete_cookie(
            FLASH_COOKIE, path=COOKIE_PATH, httponly=True, samesite="strict"
        )


def redirect(
    request: Request,
    url: str,
    *,
    status_code: int = SEE_OTHER,
    headers: dict[str, str] | None = None,
) -> RedirectResponse:
    """The second half of POST/Redirect/GET, with any queued flashes attached.

    Every form handler in this UI ends with this call. A POST that renders its own HTML
    leaves the browser holding a form submission it will replay on refresh — and for an
    import commit, "replay on refresh" means writing a batch of grades a second time.
    303 makes the follow-up a GET, so the refresh re-reads a result page instead.

    `no-store` because the page behind these redirects lists named children's marks, and
    a shared office machine should not serve them out of the back button after a logout.
    """
    response = RedirectResponse(url, status_code=status_code)
    response.headers["Cache-Control"] = "no-store"
    for name, value in (headers or {}).items():
        response.headers[name] = value
    write_flashes(request, response)
    return response


# ---------------------------------------------------------------------------
# Dependency aliases. Page handlers depend on these names and on nothing above them.
#
# The service aliases are re-exported from `sis/api/deps.py` unchanged rather than
# rebuilt, so a page and the JSON endpoint beside it are demonstrably running the same
# use case with the same TTL and the same upload ceiling. A UI that composed its own
# services would be a second configuration surface, and the first thing to diverge would
# be the preview TTL — invisibly, until a registrar's preview expired on one screen and
# not the other.
# ---------------------------------------------------------------------------

CurrentCaller = Annotated[Caller, Depends(current_caller)]
"""Required on every `/ui` route except login. Redirects when absent or invalid."""

OptionalCaller = Annotated[Caller | None, Depends(optional_caller)]
"""For the login page only."""


__all__ = [
    "COOKIE_PATH",
    "DASHBOARD_PATH",
    "FLASH_COOKIE",
    "FLASH_LEVELS",
    "LOGIN_PATH",
    "LOGOUT_PATH",
    "SEE_OTHER",
    "SESSION_COOKIE",
    "STATIC_PREFIX",
    "UI_PREFIX",
    "ApiKeyMinter",
    "ApiKeyMinterDep",
    "Caller",
    "CurrentCaller",
    "Flash",
    "GradeImportServiceDep",
    "ImportReports",
    "ImportReportsDep",
    "LoginRequired",
    "MaxUploadBytesDep",
    "OptionalCaller",
    "QueryServiceDep",
    "RosterImportServiceDep",
    "StructureCatalogue",
    "StructureCatalogueDep",
    "StructureServiceDep",
    "UnitOfWorkFactoryDep",
    "authenticate",
    "clear_session",
    "current_caller",
    "flash",
    "is_secure_request",
    "optional_caller",
    "redirect",
    "safe_next",
    "set_session",
    "take_flashes",
    "write_flashes",
]
