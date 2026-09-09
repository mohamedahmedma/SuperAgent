"""Which variable names this service's SIS address and key.

There were two spellings for one service: `IDENTITY_SIS_BASE_URL` here and `SIS_BASE_URL`
in `records/`. Two names for one thing is not redundancy, it is a disagreement waiting to
happen — and a silent one. Fill in only the records spelling, which is the natural thing to
do when wiring the estate up from `records/`'s documentation, and identity keeps an empty
in-memory directory: the chat backend answers a parent's questions about marks perfectly,
and sign-in tells her that her number is not registered.

The specific name still wins, because a deployment may legitimately run identity against a
different SIS from the records facade. That was the only defensible reason for the second
name, and it is preserved rather than removed.
"""
import unittest

from identity.config import reset_settings, settings


class _SettingsCase(unittest.TestCase):
    """Restores the process's settings cache, whatever the test did to it."""

    def setUp(self):
        reset_settings()
        self.addCleanup(reset_settings)

    #: Every name that can answer "where is the SIS", so a case naming none of them really
    #: means none. Left implicit, this file passed alone and failed in the full run: a
    #: suite that had already exported SIS_BASE_URL made "neither set" untrue, and
    #: `patch.dict` adds without removing.
    SIS_NAMES = (
        "IDENTITY_SIS_BASE_URL",
        "IDENTITY_SIS_API_KEY",
        "SIS_BASE_URL",
        "SIS_API_KEY",
    )

    def _with(self, **env):
        """Resolve settings with exactly `env` set and every other SIS name unset.

        The second half matters as much as the first. These tests are about which name is
        read when another is absent, so a case must control ALL of them — otherwise it is
        asserting against whatever the previous suite happened to leave in the environment,
        and it passes or fails on collection order.
        """
        import os
        from unittest.mock import patch

        overrides = {name: "" for name in self.SIS_NAMES}
        overrides.update(env)

        with patch.dict(os.environ, overrides, clear=False):
            reset_settings()
            return settings()


class BaseUrlTests(_SettingsCase):
    def test_the_identity_specific_name_is_used_when_set(self):
        resolved = self._with(IDENTITY_SIS_BASE_URL="https://identity-sis.example")
        self.assertEqual("https://identity-sis.example", resolved.sis_base_url)

    def test_it_falls_back_to_the_shared_name(self):
        """The bug, stated directly: this used to resolve to "" and refuse every parent."""
        resolved = self._with(SIS_BASE_URL="https://sis.example")
        self.assertEqual("https://sis.example", resolved.sis_base_url)

    def test_the_specific_name_wins_over_the_shared_one(self):
        """Two SIS deployments is the one case that justified a second variable."""
        resolved = self._with(
            IDENTITY_SIS_BASE_URL="https://identity-sis.example",
            SIS_BASE_URL="https://records-sis.example",
        )
        self.assertEqual("https://identity-sis.example", resolved.sis_base_url)

    def test_a_blank_specific_name_falls_through_rather_than_winning(self):
        """`.env` files carry `FOO=` for something someone meant to disable.

        Treating that as "set to empty" would make the fallback unreachable for anyone who
        left the old line in place while filling in the new one — which is exactly what a
        careful operator does.
        """
        resolved = self._with(IDENTITY_SIS_BASE_URL="", SIS_BASE_URL="https://sis.example")
        self.assertEqual("https://sis.example", resolved.sis_base_url)

    def test_whitespace_only_is_also_unset(self):
        resolved = self._with(IDENTITY_SIS_BASE_URL="   ", SIS_BASE_URL="https://sis.example")
        self.assertEqual("https://sis.example", resolved.sis_base_url)

    def test_neither_set_stays_empty(self):
        """Empty is what selects the in-memory fake, which refuses everyone.

        That is the safe direction and must not change: a login that cannot succeed is a
        support call, whereas a login that succeeds against an empty directory is a
        stranger holding a token.
        """
        resolved = self._with()
        self.assertEqual("", resolved.sis_base_url)


class ApiKeyTests(_SettingsCase):
    """The key follows the address, so the pair cannot come from two different schools."""

    def test_the_identity_specific_key_is_used_when_set(self):
        resolved = self._with(IDENTITY_SIS_API_KEY="identity-key")
        self.assertEqual("identity-key", resolved.sis_api_key)

    def test_it_falls_back_to_the_shared_key(self):
        resolved = self._with(SIS_API_KEY="shared-key")
        self.assertEqual("shared-key", resolved.sis_api_key)

    def test_the_specific_key_wins(self):
        resolved = self._with(IDENTITY_SIS_API_KEY="identity-key", SIS_API_KEY="shared-key")
        self.assertEqual("identity-key", resolved.sis_api_key)

    def test_neither_set_stays_empty(self):
        self.assertEqual("", self._with().sis_api_key)


class DirectorySelectionTests(_SettingsCase):
    """What the resolved settings actually cause to be built.

    The setting is only interesting because of this: an empty base URL installs a fake that
    refuses every parent, and that is the failure the shared-name fallback exists to stop
    somebody walking into.
    """

    def test_the_shared_name_alone_is_enough_to_get_a_real_directory(self):
        from identity.app import _build_directory
        from identity.infrastructure.directory.fake import FakeGuardianDirectory

        directory = _build_directory(self._with(SIS_BASE_URL="https://sis.example"))
        self.assertNotIsInstance(directory, FakeGuardianDirectory)

    def test_nothing_configured_still_yields_the_refusing_fake(self):
        from identity.app import _build_directory
        from identity.infrastructure.directory.fake import FakeGuardianDirectory

        self.assertIsInstance(_build_directory(self._with()), FakeGuardianDirectory)


if __name__ == "__main__":
    unittest.main()
