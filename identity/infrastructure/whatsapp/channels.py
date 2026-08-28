"""Which number a school sends through, and which directory answers for it.

This is what used to be four module-level globals in `whatsapp.py` — `_gateway`,
`_gateways`, `_verify_token`, `_app_secret` — written by `app.py` at startup through
`set_gateway()` / `configure()` and read from anywhere by `get_gateway()`.

That shape worked and had two costs worth removing:

**A test could not scope a change.** Installing a fake gateway mutated process state, so a
suite that forgot to restore it changed the behaviour of every suite collected after it —
the same class of ordering failure `infrastructure/db/session.py` documents at length. The
existing test suites work around it by resetting globals in fixtures.

**Nothing could see the wiring.** "Which gateway will this request use" was answerable only
by tracing which module last called `set_gateway`. It is now a field on an object that one
function builds and `app.state` holds, and a test constructs its own.

The registry itself is still resolved from the environment — that has to happen somewhere —
but it happens *here*, once, at startup, and produces a value. `domain/schools.py` holds
the data and the lookups and reads nothing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from identity.application.dto import SchoolChannel
from identity.application.ports.directory import GuardianDirectory
from identity.application.ports.messaging import WhatsAppGateway
from identity.config import Settings, env_suffix, school_env
from identity.domain.errors import NotConfigured, SchoolsMisconfigured, UnknownSchool
from identity.domain.phone import e164_or_raise
from identity.domain.schools import (
    MAX_CODE_LENGTH,
    SCHOOLS_VAR,
    SchoolRegistry,
    SchoolWhatsApp,
)
from identity.infrastructure.whatsapp.gateways import (
    CloudApiWhatsAppGateway,
    RecordingWhatsAppGateway,
)

logger = logging.getLogger(__name__)

_NUMBER_PREFIX = "IDENTITY_WHATSAPP_NUMBER"
_PHONE_ID_PREFIX = "IDENTITY_WHATSAPP_PHONE_NUMBER_ID"
_TOKEN_PREFIX = "IDENTITY_WHATSAPP_TOKEN"

#: Returned when a delivery arrives on a WhatsApp number no configured school owns.
#: Deliberately not a legal school code — SIS codes are upper-case alphanumerics plus
#: '.', '-' and '_' — so it can never equal the code stored on a challenge, and the
#: delivery fails the school cross-check instead of being attributed to somebody.
UNATTRIBUTABLE_SCHOOL: str = "?unknown"


@dataclass
class WhatsAppChannels:
    """Every school's number, gateway and directory, resolved once at startup.

    Held on `app.state` and handed to the request scope by `api/deps.py`. One object
    rather than a set of globals, so "what is this process wired to" is a value you can
    print, and a test builds its own instead of mutating the module.
    """

    registry: SchoolRegistry
    directory: GuardianDirectory
    verify_token: str = ""
    app_secret: str = ""
    #: The single-school number and gateway. Empty and recording in a multi-school
    #: deployment, where there is no such thing as "the" number.
    business_number: str = ""
    default_gateway: WhatsAppGateway = field(default_factory=RecordingWhatsAppGateway)
    #: Per school. The single-school gateway deliberately never lives in here, so a lookup
    #: miss cannot silently resolve to it and send one school's code out through another
    #: school's number.
    by_school: dict[str, WhatsAppGateway] = field(default_factory=dict)

    @property
    def is_multi_school(self) -> bool:
        return self.registry.is_multi_school

    def gateway_for(self, school_code: str | None) -> WhatsAppGateway:
        """The gateway a message to this school goes out through.

        A named school with no gateway installed falls back to the default, which in that
        state is the recording gateway — codes are written to a log or discarded, never
        delivered. That is the right failure: it is the state a laptop and the test suite
        run in, and it keeps the flow exercisable end to end without a Meta account.
        """
        if school_code is None:
            return self.default_gateway
        return self.by_school.get(school_code, self.default_gateway)

    def channel_for(self, school_code: str | None) -> SchoolChannel:
        """Everything the flow needs to talk to one school: number, gateway, directory.

        In single-school mode `school_code` is `None` and this is the process-wide trio.

        In multi-school mode the number and the gateway come from that school's own
        settings, so a code goes back out through the number the parent messaged. The
        directory is the shared HTTP client either way — one SIS base URL — because the
        school travels on the request as `X-School-Code` rather than by pointing at a
        different service. That is what lets a school be moved to its own database, or
        later its own server, by changing SIS's registry and nothing here.
        """
        if school_code is None:
            if self.is_multi_school:
                # Not a fallback. An unnamed school arriving at a service that serves
                # several is a caller that half-believes the deployment is unsplit, and
                # answering from whichever database is first is the cross-school read
                # itself.
                raise NotConfigured(
                    "This server serves several schools, so a request must name one."
                )
            return SchoolChannel(
                code=None,
                business_number=self.business_number,
                gateway=self.default_gateway,
                directory=self.directory,
            )

        if not self.is_multi_school:
            # The mirror of the case above, and the reason it is not a fallback either.
            raise NotConfigured(
                f"this server serves a single school, so it cannot answer for "
                f"school {school_code!r}."
            )

        school = self.registry.by_code(school_code)
        return SchoolChannel(
            code=school.code,
            business_number=school.number,
            gateway=self.gateway_for(school.code),
            directory=self.directory,
        )

    def school_for_delivery(self, phone_number_id: str) -> str | None:
        """Which school a delivery is for, from the number of ours it was addressed to.

        `None` in a single-school deployment, where there is nothing to choose between.

        An unrecognised `phone_number_id` returns a sentinel that no challenge can match,
        rather than `None`. Returning `None` would mean "the single school" to everything
        downstream, so an unmapped number — in practice a school onboarded at Meta and
        never added to `.env` — would have its parents resolved against whichever database
        happened to be the default. The sentinel makes the delivery fail as a mismatch
        instead, which is visible in the outcome log and harmless.
        """
        if not self.is_multi_school:
            return None
        try:
            return self.registry.by_phone_number_id(phone_number_id).code
        except UnknownSchool:
            logger.error(
                "A WhatsApp delivery arrived on phone_number_id %r, which no configured "
                "school owns. Parents messaging that number cannot sign in until it is "
                "added to %s and its per-school settings.",
                phone_number_id,
                SCHOOLS_VAR,
            )
            return UNATTRIBUTABLE_SCHOOL

    def close(self) -> None:
        """Release every pooled HTTP client. Called from the app's shutdown hook."""
        for gateway in (self.default_gateway, *self.by_school.values()):
            closer = getattr(gateway, "close", None)
            if callable(closer):
                closer()
        closer = getattr(self.directory, "close", None)
        if callable(closer):
            closer()


# ---------------------------------------------------------------------------
# Building it from the environment. The composition root's half of the work.
# ---------------------------------------------------------------------------


def build_registry(settings: Settings) -> SchoolRegistry:
    """Resolve `IDENTITY_SCHOOLS` and each school's number into a registry.

    Validation is deliberately asymmetric. A missing **number** is fatal: without it the
    click-to-chat link for that school opens WhatsApp's contact picker instead of a chat,
    and the parent is asked to choose who to send the school's verification code to — the
    failure `identity/env.py` exists to describe. Missing **credentials** are not fatal:
    that is the state every test and every laptop runs in, and the recording gateway keeps
    the flow working end to end there.
    """
    codes = settings.school_codes
    if not codes:
        return SchoolRegistry()

    schools: list[SchoolWhatsApp] = []
    suffixes: dict[str, str] = {}
    phone_ids: dict[str, str] = {}
    missing_numbers: list[str] = []

    for code in codes:
        if len(code) > MAX_CODE_LENGTH:
            raise SchoolsMisconfigured(
                f"{SCHOOLS_VAR} lists {code!r}, which is longer than the "
                f"{MAX_CODE_LENGTH} characters a school code may have."
            )

        suffix = env_suffix(code)
        clash = suffixes.get(suffix)
        if clash is not None:
            raise SchoolsMisconfigured(
                f"school codes {clash!r} and {code!r} both map to the environment suffix "
                f"{suffix!r}, so they would read the same {_NUMBER_PREFIX}_{suffix} and "
                "share one WhatsApp number. Rename one."
            )
        suffixes[suffix] = code

        number = school_env(_NUMBER_PREFIX, code)
        if not number:
            missing_numbers.append(f"{_NUMBER_PREFIX}_{suffix} (for school {code})")
            continue
        number = e164_or_raise(number, setting=f"{_NUMBER_PREFIX}_{suffix}")

        phone_number_id = school_env(_PHONE_ID_PREFIX, code)
        if phone_number_id:
            owner = phone_ids.get(phone_number_id)
            if owner is not None:
                # Two schools on one Meta number cannot be told apart on the way in, so
                # every parent of one of them would be resolved against the other's
                # database. Refuse rather than pick.
                raise SchoolsMisconfigured(
                    f"schools {owner!r} and {code!r} share the WhatsApp "
                    f"phone_number_id {phone_number_id!r}. Inbound messages cannot be "
                    "attributed to a school, so each school needs its own number."
                )
            phone_ids[phone_number_id] = code

        schools.append(
            SchoolWhatsApp(
                code=code,
                number=number,
                phone_number_id=phone_number_id,
                access_token=school_env(_TOKEN_PREFIX, code),
            )
        )

    if missing_numbers:
        raise SchoolsMisconfigured(
            f"{SCHOOLS_VAR} names schools with no WhatsApp number behind them: "
            + ", ".join(missing_numbers)
            + ". Every school needs its own number; there is no shared default, because "
            "a link without the right number opens WhatsApp's contact picker and the "
            "parent is asked to choose who to send the school's verification code to."
        )
    return SchoolRegistry(schools=tuple(schools))


def build_channels(settings: Settings, directory: GuardianDirectory) -> WhatsAppChannels:
    """Choose the gateways this process sends through, and install the webhook secrets.

    The environment is read here and in `api/deps.py`, and nowhere else. This is the
    composition root's work, so a misconfiguration is meant to stop the deploy rather than
    surface as a parent's login failing at eight in the morning.

    `e164_or_raise` is applied to every school's number at startup for exactly that reason:
    the national spelling `01288339613` produces `wa.me/01288339613`, which is a different
    number that does not exist, and the resulting failure is completely silent — the link
    opens, the chat is empty, no message ever arrives, and nothing logs anything.

    With no credentials the recording gateway stays in place and the flow still runs end to
    end, which is what makes this developable without a Meta account. It is a loud warning
    rather than a hard failure because that is also the state every test runs in.
    """
    registry = build_registry(settings)
    channels = WhatsAppChannels(
        registry=registry,
        directory=directory,
        # The secrets stay estate-wide: one Meta app delivers every school's messages to
        # one endpoint, and which school a delivery belongs to is read from its own
        # `phone_number_id` rather than from a separate endpoint per school.
        verify_token=settings.whatsapp_verify_token,
        app_secret=settings.whatsapp_app_secret,
    )

    if registry.is_multi_school:
        _configure_per_school(channels, settings)
    else:
        _configure_single_school(channels, settings)
    return channels


def _configure_single_school(channels: WhatsAppChannels, settings: Settings) -> None:
    number = settings.whatsapp_number
    if number:
        channels.business_number = e164_or_raise(
            number, setting="IDENTITY_WHATSAPP_NUMBER"
        )
    else:
        logger.warning(
            "IDENTITY_WHATSAPP_NUMBER is not set. Parent sign-in is DISABLED: without "
            "the school's number a click-to-chat link opens WhatsApp's contact picker "
            "instead of a chat, so /v1/auth/whatsapp/start will refuse rather than hand "
            "a parent a link that cannot work."
        )

    if settings.whatsapp_phone_number_id and settings.whatsapp_token:
        channels.default_gateway = CloudApiWhatsAppGateway(
            phone_number_id=settings.whatsapp_phone_number_id,
            access_token=settings.whatsapp_token,
            graph_version=settings.whatsapp_graph_version,
        )
        return

    # Without a Meta account the code is generated and then thrown away, which makes the
    # flow impossible to try end to end on a laptop. This turns it into a log line
    # instead. Off unless asked for, because the body IS the verification code: anywhere
    # a real parent can be verified, this writes their credential into a file that is
    # backed up, shipped to a log aggregator, and read by people who are not them.
    channels.default_gateway = RecordingWhatsAppGateway(
        log_bodies=settings.whatsapp_log_codes
    )
    logger.warning(
        "WhatsApp is not configured (IDENTITY_WHATSAPP_PHONE_NUMBER_ID and "
        "IDENTITY_WHATSAPP_TOKEN); verification codes are %s. Parent login by WhatsApp "
        "cannot reach a real phone in this state.",
        "WRITTEN TO THIS LOG (IDENTITY_WHATSAPP_LOG_CODES is on — never do this in "
        "production)"
        if settings.whatsapp_log_codes
        else "discarded. Set IDENTITY_WHATSAPP_LOG_CODES=true to read them here while "
        "developing",
    )


def _configure_per_school(channels: WhatsAppChannels, settings: Settings) -> None:
    """One number, one gateway, one webhook — several schools.

    What is per school is the pair that has to travel together: the number a parent
    messages, and the credentials a reply goes back out through. Split those and a code for
    one school's parent is sent from another school's number, arriving in a conversation
    the parent is not looking at.

    A school with no credentials gets no gateway of its own and falls back to the recording
    gateway, exactly as an unconfigured single-school deployment does. It is logged per
    school rather than once, because "WhatsApp is configured" stops being a single fact the
    moment there are several schools, and an estate where one branch silently cannot
    deliver codes is the failure worth naming.
    """
    # No single business number exists here. `channel_for` resolves each school's own from
    # the registry; this stays empty so anything still reading the process-wide value in a
    # multi-school deployment refuses rather than handing out one school's number to every
    # school's parents.
    channels.business_number = ""
    channels.default_gateway = RecordingWhatsAppGateway(
        log_bodies=settings.whatsapp_log_codes
    )

    live: list[str] = []
    recording: list[str] = []
    for school in channels.registry.schools:
        if school.can_send:
            channels.by_school[school.code] = CloudApiWhatsAppGateway(
                phone_number_id=school.phone_number_id,
                access_token=school.access_token,
                graph_version=settings.whatsapp_graph_version,
            )
            live.append(school.code)
        else:
            channels.by_school[school.code] = RecordingWhatsAppGateway(
                log_bodies=settings.whatsapp_log_codes
            )
            recording.append(school.code)

    logger.info(
        "WhatsApp is configured for %d school(s): %s deliver to real phones.",
        len(channels.registry.schools),
        ", ".join(live) or "none",
    )
    if recording:
        logger.warning(
            "These schools have no WhatsApp credentials and cannot deliver a verification "
            "code to a real phone: %s. Parent login is effectively DISABLED for them.",
            ", ".join(recording),
        )


__all__ = [
    "UNATTRIBUTABLE_SCHOOL",
    "WhatsAppChannels",
    "build_channels",
    "build_registry",
]
