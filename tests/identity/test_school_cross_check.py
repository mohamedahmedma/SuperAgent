"""A verification is only valid at the school it was started for.

`identity/verification.py` already documents the two attacks its split-secret design
defeats, and one of them is "a parent tricked into sending an attacker's nonce". Several
schools give that a new shape: steer the parent's message to a *different* school's
WhatsApp number, and the flow would look their phone up in a database their children are
not in — resolving them against strangers, or refusing them for reasons nobody can explain.

The close is that two independent facts must agree. The school the browser was on when the
challenge was minted is recorded on the row; the school that owns the number the message
arrived on is read from the delivery. A mismatch spends the nonce and issues nothing.
"""
from __future__ import annotations

import pytest

from identity import guardians, whatsapp
from identity.verification import (
    SchoolChannel,
    VerificationService,
    extract_nonce,
)

NC = "NC"
MD = "MD"


def _service(directories: dict[str | None, guardians.FakeGuardianDirectory]):
    """A service serving two schools, each with its own gateway and its own directory.

    Separate directories are the point: they stand in for the separate databases. A lookup
    that reached the wrong one would find a guardian who is not this school's, which is the
    leak in miniature.
    """
    gateways = {code: whatsapp.RecordingWhatsAppGateway() for code in directories}

    def channel_for(school_code: str | None) -> SchoolChannel:
        if school_code not in directories:
            raise LookupError(f"no school {school_code!r}")
        return SchoolChannel(
            code=school_code,
            business_number="+201288339613",
            gateway=gateways[school_code],
            directory=directories[school_code],
        )

    return VerificationService(channel_for=channel_for), gateways


@pytest.fixture()
def two_schools():
    """A parent who exists at Nasr City and is a stranger at Maadi."""
    nc = guardians.FakeGuardianDirectory(
        guardians={
            "+201000000000": guardians.GuardianRef(
                public_id="nc-guardian",
                full_name_ar="أم",
                full_name_en="Mother",
                preferred_language="ar",
            )
        }
    )
    md = guardians.FakeGuardianDirectory(guardians={})
    return {NC: nc, MD: md}


def test_a_challenge_records_the_school_it_was_started_for(db, two_schools) -> None:
    service, _ = _service(two_schools)
    started = service.start(db, school_code=NC)

    from identity.models import VerificationChallenge

    stored = (
        db.query(VerificationChallenge)
        .filter(VerificationChallenge.nonce == started.nonce)
        .one()
    )
    assert stored.school_code == NC


def test_a_nonce_sent_to_another_schools_number_is_refused(db, two_schools) -> None:
    """The whole point. Started at Nasr City, delivered on Maadi's number: nothing issued.

    Note what is *not* asserted — that the parent was not found at Maadi. That would pass
    for the wrong reason, because she genuinely is not in Maadi's directory. The assertion
    is that the flow refused before it ever asked, so the same result would hold for a
    parent who exists at both.
    """
    service, gateways = _service(two_schools)
    started = service.start(db, school_code=NC)

    outcome = service.claim(
        db,
        wa_id="+201000000000",
        body=f"SCHOOL VERIFY: {started.nonce}",
        message_id="wamid.CROSSED",
        school_code=MD,
    )

    assert outcome == "wrong_school"
    # Neither school sent anything: no code, and no explanation that would confirm to a
    # third party that this nonce or this number means something.
    assert gateways[NC].sent == []
    assert gateways[MD].sent == []
    # And the parent's number was never looked up in the wrong school's directory.
    assert two_schools[MD].asked == []


def test_the_matching_school_completes_normally(db, two_schools) -> None:
    """The control. Same nonce, same parent, delivered on the right number: a code goes out."""
    service, gateways = _service(two_schools)
    started = service.start(db, school_code=NC)

    outcome = service.claim(
        db,
        wa_id="+201000000000",
        body=f"SCHOOL VERIFY: {started.nonce}",
        message_id="wamid.CORRECT",
        school_code=NC,
    )

    assert outcome == "code_sent"
    assert len(gateways[NC].sent) == 1
    assert gateways[MD].sent == []
    # The lookup went to the school that owns the number, and to no other.
    assert two_schools[NC].asked == ["+201000000000"]
    assert two_schools[NC].asked_schools == [NC]
    assert two_schools[MD].asked == []


def test_a_spent_nonce_cannot_be_retried_at_the_right_school(db, two_schools) -> None:
    """A refused cross-school attempt consumes the challenge rather than leaving it live.

    Otherwise the refusal is only a delay: the same nonce could be steered at each school
    in turn until one of them answered.
    """
    service, _ = _service(two_schools)
    started = service.start(db, school_code=NC)

    assert service.claim(
        db,
        wa_id="+201000000000",
        body=f"SCHOOL VERIFY: {started.nonce}",
        message_id="wamid.FIRST",
        school_code=MD,
    ) == "wrong_school"

    assert service.claim(
        db,
        wa_id="+201000000000",
        body=f"SCHOOL VERIFY: {started.nonce}",
        message_id="wamid.SECOND",
        school_code=NC,
    ) == "already_claimed"


def test_a_single_school_service_still_works_with_no_school_at_all(db) -> None:
    """The unsplit path: no school anywhere, and the flow behaves exactly as before."""
    directory = guardians.FakeGuardianDirectory(
        guardians={
            "+201000000000": guardians.GuardianRef(
                public_id="g1",
                full_name_ar="أم",
                full_name_en="Mother",
                preferred_language="ar",
            )
        }
    )
    gateway = whatsapp.RecordingWhatsAppGateway()
    service = VerificationService(
        gateway=gateway, directory=directory, business_number="+201288339613"
    )

    started = service.start(db)
    outcome = service.claim(
        db,
        wa_id="+201000000000",
        body=f"SCHOOL VERIFY: {started.nonce}",
        message_id="wamid.SINGLE",
    )

    assert outcome == "code_sent"
    assert len(gateway.sent) == 1


def test_the_nonce_survives_what_parents_actually_type(db, two_schools) -> None:
    """Guarding the parser the cross-check depends on: no nonce, no school comparison."""
    service, _ = _service(two_schools)
    started = service.start(db, school_code=NC)
    assert extract_nonce(f"hi SCHOOL VERIFY: {started.nonce} thanks") == started.nonce
