"""The issuer and audience three services must agree on, and the minter that stamps them.

`identity/` mints tokens carrying `iss` and `aud`. `backend/`, `records/` and `identity/`
itself each check them — offline, against a public key, per request. Four points, one pair
of strings, and until this was consolidated each of the four carried its own copy of the
literal.

Four copies that agree look harmless, which is the whole problem. The failure arrives when
one deployment sets `IDENTITY_ISSUER` on identity and not on the backend: identity mints
`iss: acme-school`, the backend still expects `school-identity`, and every authenticated
request in the estate returns 401. Nothing in that 401 mentions an issuer. It reads exactly
like a broken signing key, and that is where the day goes.

Each service still reads its OWN environment — a deployment may legitimately run two
identity services, and this suite and the records suite set different values in one pytest
process. What is shared is the default, not the configuration.
"""
import os
import unittest
from unittest.mock import patch

import schoolauth


class DefaultAgreementTests(unittest.TestCase):
    """With nothing configured, all four must land on the same pair."""

    def _resolved(self):
        """The four values, each read the way its own service reads it."""
        import importlib

        import backend.infra.identity as backend_identity
        import identity.config as identity_config
        import records.config as records_config

        importlib.reload(backend_identity)
        records_config.reset_settings()
        identity_config.reset_settings()

        return {
            "backend": (backend_identity.ISSUER, backend_identity.AUDIENCE),
            "records": (
                records_config.settings().identity_issuer,
                records_config.settings().identity_audience,
            ),
            "identity": (
                identity_config.settings().issuer,
                identity_config.settings().audience,
            ),
        }

    def tearDown(self):
        """Put the process back the way it was found.

        `backend.infra.identity` captures ISSUER and AUDIENCE at import, so a reload
        performed inside a `patch.dict` keeps the patched values after the patch exits —
        and `tests/general/test_backend_auth.py` mints tokens against exactly those module
        constants. Reloading once more here, with the real environment restored, is what
        stops this file deciding whether a later one passes.
        """
        import importlib

        import backend.infra.identity as backend_identity
        import identity.config as identity_config
        import records.config as records_config

        importlib.reload(backend_identity)
        records_config.reset_settings()
        identity_config.reset_settings()

    def test_all_three_default_to_the_same_pair(self):
        cleared = {
            "IDENTITY_ISSUER": "",
            "IDENTITY_AUDIENCE": "",
        }
        with patch.dict(os.environ, cleared, clear=False):
            os.environ.pop("IDENTITY_ISSUER", None)
            os.environ.pop("IDENTITY_AUDIENCE", None)
            resolved = self._resolved()

        pairs = set(resolved.values())
        self.assertEqual(1, len(pairs), f"services disagree by default: {resolved}")

    def test_the_shared_default_is_what_they_land_on(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("IDENTITY_ISSUER", None)
            os.environ.pop("IDENTITY_AUDIENCE", None)
            resolved = self._resolved()

        expected = (schoolauth.DEFAULT_ISSUER, schoolauth.DEFAULT_AUDIENCE)
        for service, pair in resolved.items():
            self.assertEqual(expected, pair, f"{service} does not use the shared default")

    def test_one_setting_moves_all_three_together(self):
        """The case the four literals broke.

        Setting the variable once, as a deployment does, must move every service — not
        three of them.
        """
        with patch.dict(
            os.environ,
            {"IDENTITY_ISSUER": "acme-school", "IDENTITY_AUDIENCE": "acme-services"},
        ):
            resolved = self._resolved()

        for service, pair in resolved.items():
            self.assertEqual(("acme-school", "acme-services"), pair, f"{service} did not move")


class NoLiteralsRemainTests(unittest.TestCase):
    """Caught at the source, because a reintroduced literal passes every test above.

    It would agree with the others right up until somebody edited one of them, which is
    precisely the failure mode — so the guard has to be that no copy exists, not that the
    copies currently match.
    """

    def test_no_service_hardcodes_the_pair_any_more(self):
        from pathlib import Path

        for module_path in (
            "backend/infra/identity.py",
            "records/config.py",
            "identity/config.py",
        ):
            source = Path(module_path).read_text(encoding="utf-8")
            for literal in ('"school-identity"', '"school-services"'):
                self.assertNotIn(
                    literal,
                    source,
                    f"{module_path} carries its own copy of {literal}",
                )

    def test_the_defaults_are_named_in_the_shared_package(self):
        self.assertEqual("school-identity", schoolauth.DEFAULT_ISSUER)
        self.assertEqual("school-services", schoolauth.DEFAULT_AUDIENCE)


class ReplayTests(unittest.TestCase):
    """Why the audience is checked at all, kept beside the agreement it depends on."""

    def test_the_audience_is_part_of_the_config_every_verifier_is_given(self):
        """Without it, a token minted for one service is replayable against another.

        Same key, same issuer, different data — so this is not a formality, and a config
        object that made it optional would let a service quietly stop checking.
        """
        config = schoolauth.IdentityConfig(issuer="i", audience="a")
        self.assertEqual("a", config.audience)

        with self.assertRaises(TypeError):
            schoolauth.IdentityConfig(issuer="i")  # type: ignore[call-arg]


if __name__ == "__main__":
    unittest.main()
