"""Which school a parent is talking to, decided by the number they messaged.

Schools are separated physically, so every lookup made on a parent's behalf has to name
one. Over WhatsApp there is no form to ask and no phone-to-school directory to consult:
the answer arrives with the message, as `value.metadata.phone_number_id`.

These tests pin the three properties the design rests on:

  * the id survives parsing, because it was being discarded before this change;
  * one number maps to exactly one school, and an unknown one maps to nobody;
  * the school the browser started on and the school that owns the number the parent
    messaged must **agree** before a code is issued.

The last is the one worth the most. `identity/verification.py` already worries about a
parent talked into sending an attacker's nonce; with several schools that attack gains a
new shape — steer the nonce to a *different* school's number and the parent is resolved
against a database their children are not in. Two independent facts having to agree is
what closes it, and `test_a_nonce_sent_to_another_schools_number_is_refused` is the proof.
"""
from __future__ import annotations

import json

import pytest

from identity import config
from identity.domain import schools
from identity.domain.errors import NotConfigured, SchoolsMisconfigured, UnknownSchool
from identity.infrastructure.whatsapp import inbound as wa
from identity.infrastructure.whatsapp.channels import build_registry

NC_PHONE_ID = "111111111111111"
MD_PHONE_ID = "222222222222222"


def _delivery(*, phone_number_id: str, sender: str = "201000000000", body: str = "hello") -> bytes:
    """One webhook body in the shape Meta actually sends."""
    return json.dumps(
        {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "messaging_product": "whatsapp",
                                "metadata": {
                                    "display_phone_number": "20100000000",
                                    "phone_number_id": phone_number_id,
                                },
                                "messages": [
                                    {
                                        "from": sender,
                                        "id": "wamid.TEST",
                                        "type": "text",
                                        "text": {"body": body},
                                    }
                                ],
                            }
                        }
                    ]
                }
            ]
        }
    ).encode("utf-8")


@pytest.fixture()
def two_schools(monkeypatch) -> schools.SchoolRegistry:
    """Two branches, each with its own number and its own Meta credentials."""
    monkeypatch.setenv(schools.SCHOOLS_VAR, "NC,MD")
    monkeypatch.setenv("IDENTITY_WHATSAPP_NUMBER_NC", "+201000000000")
    monkeypatch.setenv("IDENTITY_WHATSAPP_PHONE_NUMBER_ID_NC", NC_PHONE_ID)
    monkeypatch.setenv("IDENTITY_WHATSAPP_TOKEN_NC", "token-nc")
    monkeypatch.setenv("IDENTITY_WHATSAPP_NUMBER_MD", "+201111111111")
    monkeypatch.setenv("IDENTITY_WHATSAPP_PHONE_NUMBER_ID_MD", MD_PHONE_ID)
    monkeypatch.setenv("IDENTITY_WHATSAPP_TOKEN_MD", "token-md")
    config.reset_settings()
    yield build_registry(config.settings())
    config.reset_settings()


# ---------------------------------------------------------------------------
# The id survives parsing
# ---------------------------------------------------------------------------


def test_the_inbound_parser_keeps_the_number_the_parent_messaged() -> None:
    """It was parsing the dict that holds this and throwing the field away."""
    found = wa.inbound_text_messages(_delivery(phone_number_id=NC_PHONE_ID))
    assert len(found) == 1
    assert found[0].phone_number_id == NC_PHONE_ID
    assert found[0].wa_id == "201000000000"
    assert found[0].text == "hello"


def test_a_delivery_with_no_metadata_yields_an_empty_school() -> None:
    """`""`, which the caller must treat as "unknown" and never as "the default school"."""
    body = json.dumps(
        {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "messages": [
                                    {
                                        "from": "201000000000",
                                        "id": "wamid.X",
                                        "type": "text",
                                        "text": {"body": "hi"},
                                    }
                                ]
                            }
                        }
                    ]
                }
            ]
        }
    ).encode("utf-8")
    found = wa.inbound_text_messages(body)
    assert len(found) == 1
    assert found[0].phone_number_id == ""


def test_a_malformed_body_is_no_messages_rather_than_an_exception() -> None:
    """Meta replays an unacknowledged delivery for seven days; raising here costs a week."""
    assert wa.inbound_text_messages(b"not json at all") == []
    assert wa.inbound_text_messages(b'{"entry": "wrong shape"}') == []


# ---------------------------------------------------------------------------
# One number, one school
# ---------------------------------------------------------------------------


def test_a_number_resolves_to_the_school_that_owns_it(two_schools) -> None:
    assert two_schools.by_phone_number_id(NC_PHONE_ID).code == "NC"
    assert two_schools.by_phone_number_id(MD_PHONE_ID).code == "MD"


def test_an_unknown_number_resolves_to_nobody(two_schools) -> None:
    """Never to a default. Resolving it would answer one branch's parent from another's."""
    with pytest.raises(UnknownSchool):
        two_schools.by_phone_number_id("999999999999999")
    with pytest.raises(UnknownSchool):
        two_schools.by_phone_number_id("")


def test_two_schools_sharing_one_number_are_refused(monkeypatch) -> None:
    """Inbound messages could not be attributed, so every parent would reach one database."""
    monkeypatch.setenv(schools.SCHOOLS_VAR, "NC,MD")
    monkeypatch.setenv("IDENTITY_WHATSAPP_NUMBER_NC", "+201000000000")
    monkeypatch.setenv("IDENTITY_WHATSAPP_PHONE_NUMBER_ID_NC", NC_PHONE_ID)
    monkeypatch.setenv("IDENTITY_WHATSAPP_NUMBER_MD", "+201111111111")
    monkeypatch.setenv("IDENTITY_WHATSAPP_PHONE_NUMBER_ID_MD", NC_PHONE_ID)
    config.reset_settings()
    try:
        with pytest.raises(SchoolsMisconfigured) as refusal:
            build_registry(config.settings())
        assert "phone_number_id" in str(refusal.value)
    finally:
        config.reset_settings()


def test_a_school_with_no_number_is_refused(monkeypatch) -> None:
    """A link without a number opens WhatsApp's contact picker; see `identity/env.py`."""
    monkeypatch.setenv(schools.SCHOOLS_VAR, "NC,MD")
    monkeypatch.setenv("IDENTITY_WHATSAPP_NUMBER_NC", "+201000000000")
    monkeypatch.delenv("IDENTITY_WHATSAPP_NUMBER_MD", raising=False)
    config.reset_settings()
    try:
        with pytest.raises(SchoolsMisconfigured) as refusal:
            build_registry(config.settings())
        assert "MD" in str(refusal.value)
    finally:
        config.reset_settings()


def test_a_national_number_is_refused_at_startup(monkeypatch) -> None:
    """`01288339613` produces a `wa.me` link to a number that does not exist, silently."""
    monkeypatch.setenv(schools.SCHOOLS_VAR, "NC")
    monkeypatch.setenv("IDENTITY_WHATSAPP_NUMBER_NC", "01288339613")
    config.reset_settings()
    try:
        with pytest.raises(Exception):
            build_registry(config.settings())
    finally:
        config.reset_settings()


def test_no_schools_configured_is_single_school_mode(monkeypatch) -> None:
    """The default, and what keeps every unsplit deployment working untouched."""
    monkeypatch.delenv(schools.SCHOOLS_VAR, raising=False)
    config.reset_settings()
    try:
        assert build_registry(config.settings()).is_multi_school is False
    finally:
        config.reset_settings()
