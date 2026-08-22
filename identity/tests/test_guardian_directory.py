"""The SIS adapter: what it does with each answer the school's records can give.

Mocked at `httpx.Client.post` with hand-rolled responses, which is how `records/` tests its
own adapter — no network, and no extra dependency to keep current.

The distinction every test here circles is the one the seam exists to protect: **"not a
parent" is a `None` and "could not ask" is an exception.** Collapsing them either way is a
real failure. Treat unreachable as unknown and a network blip tells a mother she is not
registered; treat unknown as unreachable and a stranger is told the school is merely busy,
which invites them to keep trying.
"""
import json
from unittest.mock import patch

import httpx
import pytest

from identity.guardians import (
    GuardianDirectoryUnavailable,
    GuardianRef,
    SisGuardianDirectory,
)

PHONE = "+201001234567"


def _response(status_code: int, body: object, *, headers: dict | None = None) -> httpx.Response:
    """A real `httpx.Response`, so the adapter meets the same object it will in production."""
    return httpx.Response(
        status_code=status_code,
        content=json.dumps(body).encode("utf-8") if body is not None else b"",
        headers={"Content-Type": "application/json", **(headers or {})},
        request=httpx.Request("POST", "http://sis.test/v1/guardians/resolve"),
    )


@pytest.fixture()
def directory() -> SisGuardianDirectory:
    return SisGuardianDirectory(base_url="http://sis.test/", api_key="a-reader-key")


def test_a_known_number_becomes_a_reference(directory: SisGuardianDirectory) -> None:
    payload = {
        "public_id": "guardian-public-id-1",
        "full_name_ar": "فاطمة علي",
        "full_name_en": "Fatma Ali",
        "preferred_language": "ar",
    }
    with patch("httpx.Client.post", return_value=_response(200, payload)):
        found = directory.resolve(PHONE)

    assert found == GuardianRef(
        public_id="guardian-public-id-1",
        full_name_ar="فاطمة علي",
        full_name_en="Fatma Ali",
        preferred_language="ar",
    )
    assert found.display_name == "فاطمة علي"


def test_the_number_travels_in_a_body_never_a_url(directory: SisGuardianDirectory) -> None:
    """A phone in a path is PII in every access log, proxy and browser history it passes.

    That is the reason guardians have a `public_id` at all, and a GET here would undo it on
    the very request whose purpose is to stop holding numbers.
    """
    with patch("httpx.Client.post", return_value=_response(200, {"public_id": "g-1"})) as post:
        directory.resolve(PHONE)

    (path,), kwargs = post.call_args
    assert path == "/v1/guardians/resolve"
    assert kwargs["json"] == {"phone": PHONE}
    assert PHONE not in path


def test_an_unknown_number_is_none_rather_than_an_error(directory: SisGuardianDirectory) -> None:
    """Most numbers in the world are not this school's parents. An ordinary answer."""
    body = {"detail": {"code": "unknown_reference", "message": "no guardian", "field": "phone"}}
    with patch("httpx.Client.post", return_value=_response(404, body)):
        assert directory.resolve(PHONE) is None


def test_a_404_that_is_not_sis_speaking_is_an_outage(directory: SisGuardianDirectory) -> None:
    """A misconfigured base URL returns 404 too.

    Recognising "not found" by the bare status would read a wrong URL as "nobody in this
    school is a parent" and tell every family they are unregistered — silently, and for as
    long as the misconfiguration lasted.
    """
    with patch("httpx.Client.post", return_value=_response(404, {"detail": "Not Found"})):
        with pytest.raises(GuardianDirectoryUnavailable):
            directory.resolve(PHONE)


@pytest.mark.parametrize("status_code", [400, 401, 403, 422, 500, 503])
def test_every_refusal_collapses_to_one_outcome(
    directory: SisGuardianDirectory, status_code: int
) -> None:
    """A caller that could tell "down" from "refused" could probe our configuration.

    Neither answer changes what happens next: the parent is told to try again, and nothing
    is bound.
    """
    with patch("httpx.Client.post", return_value=_response(status_code, {"detail": {}})):
        with pytest.raises(GuardianDirectoryUnavailable):
            directory.resolve(PHONE)


def test_a_transport_failure_is_the_same_outcome(directory: SisGuardianDirectory) -> None:
    with patch("httpx.Client.post", side_effect=httpx.ConnectError("refused")):
        with pytest.raises(GuardianDirectoryUnavailable):
            directory.resolve(PHONE)


def test_a_timeout_is_the_same_outcome(directory: SisGuardianDirectory) -> None:
    """A parent is waiting. A slow answer is a failed one."""
    with patch("httpx.Client.post", side_effect=httpx.ReadTimeout("slow")):
        with pytest.raises(GuardianDirectoryUnavailable):
            directory.resolve(PHONE)


def test_a_redirect_is_refused_rather_than_followed(directory: SisGuardianDirectory) -> None:
    """Following it would re-send the number, and the API key, to whatever host it names."""
    redirect = _response(307, None, headers={"Location": "http://elsewhere.test/"})
    with patch("httpx.Client.post", return_value=redirect):
        with pytest.raises(GuardianDirectoryUnavailable) as refusal:
            directory.resolve(PHONE)
    assert "IDENTITY_SIS_BASE_URL" in str(refusal.value)


def test_an_unreadable_body_is_not_a_resolution(directory: SisGuardianDirectory) -> None:
    with patch("httpx.Client.post", return_value=_response(200, {"unexpected": "shape"})):
        with pytest.raises(GuardianDirectoryUnavailable):
            directory.resolve(PHONE)


def test_an_empty_handle_is_refused(directory: SisGuardianDirectory) -> None:
    """The one answer that must never be treated as success.

    A blank handle would bind an account to nothing while looking exactly like a
    resolution, and every token minted from it would carry an empty `guardian_id`.
    """
    with patch("httpx.Client.post", return_value=_response(200, {"public_id": ""})):
        with pytest.raises(GuardianDirectoryUnavailable):
            directory.resolve(PHONE)


def test_a_missing_base_url_fails_at_construction(directory: SisGuardianDirectory) -> None:
    """At startup, where a human is watching — never at the first parent's login."""
    with pytest.raises(RuntimeError):
        SisGuardianDirectory(base_url="")
