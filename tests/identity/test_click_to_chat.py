"""The link a parent taps.

This exists because of a production failure that was invisible from inside the service:
every log line was fine, `/start` returned 201, and the link was a valid URL. Only the
phone number was missing from it — and `https://wa.me/?text=...` is a perfectly good
link that opens WhatsApp's CONTACT PICKER. The parent is asked to choose who to send the
school's verification code to, which in production they cannot possibly know.

So the assertions here are about the one thing that distinguishes a working sign-in from
a broken one: whether there is a number in the path.
"""
import pytest
from fastapi.testclient import TestClient

from identity.app import app
from identity.application.dto import SchoolChannel
from identity.application.services.whatsapp_login import WhatsAppLoginService
from identity.domain.errors import NotConfigured
from identity.domain.phone import click_to_chat_link
from identity.infrastructure.directory.fake import FakeGuardianDirectory
from identity.infrastructure.whatsapp.gateways import RecordingWhatsAppGateway
from tests.identity.conftest import install_channels


class TestTheLinkItself:
    def test_a_configured_number_opens_a_chat_with_that_number(self):
        link = click_to_chat_link("+201288339613", "SCHOOL VERIFY: ABC12345")

        assert link.startswith("https://wa.me/201288339613?")
        # The digits, with no '+' — wa.me rejects the plus.
        assert "wa.me/+" not in link

    @pytest.mark.parametrize("empty", ["", "   ", None])
    def test_no_number_is_refused_rather_than_rendered(self, empty):
        """`wa.me/?text=` is the bug. It is a valid URL, so nothing downstream can tell
        it apart from a working one — which is why this raises at the source."""
        with pytest.raises(ValueError, match="IDENTITY_WHATSAPP_NUMBER"):
            click_to_chat_link(empty, "SCHOOL VERIFY: ABC12345")

    def test_the_message_survives_its_reserved_characters(self):
        """An unescaped & or # truncates the message exactly where the nonce was, and
        what arrives looks almost right."""
        link = click_to_chat_link("+201288339613", "a&b#c d")

        assert "&" not in link.split("?text=", 1)[1]
        assert "#" not in link


class TestTheEndpoint:
    def test_it_refuses_to_start_a_sign_in_that_cannot_finish(self):
        """503, not 400: nothing the caller sent is wrong and nothing they can send will
        help. Refusing costs one message; not refusing costs every parent, silently."""
        service = WhatsAppLoginService(
            # `None` is safe precisely because of what this asserts: the refusal
            # happens before any repository method is called.
            challenges=None,
            channel_for=lambda code: SchoolChannel(
                code=None,
                business_number="",
                gateway=RecordingWhatsAppGateway(),
                directory=FakeGuardianDirectory(),
            ),
        )

        # Raised before the database is touched, so a misconfigured server does not
        # write a challenge nobody can ever answer.
        with pytest.raises(NotConfigured):
            service.start()

    def test_a_configured_server_hands_back_a_usable_link(self, db):
        with TestClient(app) as client:
            install_channels(client, business_number="+201288339613")
            response = client.post("/v1/auth/whatsapp/start")

        assert response.status_code == 201
        body = response.json()
        digits = body["link"].split("wa.me/", 1)[1].split("?", 1)[0]
        assert digits == "201288339613", "the link must name the school, not nobody"
        assert body["business_number"] == "+201288339613"

    def test_an_unconfigured_server_says_so_in_a_code_the_screen_can_render(self, db):
        with TestClient(app) as client:
            install_channels(client, business_number="")
            response = client.post("/v1/auth/whatsapp/start")

        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "not_configured"


class TestTheServiceReadsItsOwnEnvFile:
    def test_identity_loads_the_project_env(self):
        """The root cause. `identity/` is deployed as its own process, and nothing in it
        loaded the project's `.env` — so every os.getenv saw only the shell."""
        from identity.env import PROJECT_ROOT, load_env

        assert (PROJECT_ROOT / ".env.example").exists(), "PROJECT_ROOT points elsewhere"
        load_env()  # idempotent; safe to call again here
