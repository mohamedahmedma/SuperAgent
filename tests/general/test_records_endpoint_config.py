"""One answer to "where is the records facade", shared by the two callers that need it.

`backend/tools/records.py` fetches a child's marks; `backend/chat/child_roster.py` fetches
the list of children to ask about. Same service, same credentials — and until this was
consolidated, each called `os.getenv` with its own copy of the default, with a comment
explaining that importing one from the other would be a cycle. The explanation was right
and the conclusion was not: `backend.env` imports from neither.

Two copies of a default fails slowly and confusingly. Change one and a parent is offered a
list of children fetched from one facade and marks fetched from another, which does not
present as a configuration error — it presents as a child whose records have vanished.
"""
import importlib
import os
import unittest
from unittest.mock import patch

import backend.env as backend_env


class SharedReaderTests(unittest.TestCase):
    """`records_base_url()` itself."""

    def test_the_configured_value_is_used(self):
        with patch.dict(os.environ, {"RECORDS_BASE_URL": "https://records.aurexis.cc"}):
            self.assertEqual("https://records.aurexis.cc", backend_env.records_base_url())

    def test_a_trailing_slash_is_removed(self):
        """Paths are concatenated onto this, so a slash here produces `//v1/...`."""
        with patch.dict(os.environ, {"RECORDS_BASE_URL": "https://records.aurexis.cc/"}):
            self.assertEqual("https://records.aurexis.cc", backend_env.records_base_url())

    def test_several_trailing_slashes_are_removed(self):
        with patch.dict(os.environ, {"RECORDS_BASE_URL": "https://records.aurexis.cc///"}):
            self.assertEqual("https://records.aurexis.cc", backend_env.records_base_url())

    def test_unset_falls_back_to_the_documented_port(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RECORDS_BASE_URL", None)
            self.assertEqual("http://localhost:8100", backend_env.records_base_url())

    def test_blank_falls_back_rather_than_producing_a_bare_path(self):
        """The one behaviour that changed, stated so it is not a surprise.

        `os.getenv("RECORDS_BASE_URL", default)` returns `""` for `RECORDS_BASE_URL=`, and
        every request then went to a bare path and failed. Blank-is-unset is this module's
        documented rule; applying it here can only turn a guaranteed failure into a
        working default.
        """
        with patch.dict(os.environ, {"RECORDS_BASE_URL": ""}):
            self.assertEqual("http://localhost:8100", backend_env.records_base_url())

        with patch.dict(os.environ, {"RECORDS_BASE_URL": "   "}):
            self.assertEqual("http://localhost:8100", backend_env.records_base_url())

    def test_a_blank_api_key_stays_blank(self):
        """`records/` fails closed on an empty key — a 503, not an open door.

        Substituting anything here would turn a loud misconfiguration into a quiet one, so
        the fallback that applies to the URL deliberately does not apply to this.
        """
        with patch.dict(os.environ, {"RECORDS_API_KEY": ""}):
            self.assertEqual("", backend_env.records_api_key())

    def test_the_configured_api_key_is_used(self):
        with patch.dict(os.environ, {"RECORDS_API_KEY": "a-secret"}):
            self.assertEqual("a-secret", backend_env.records_api_key())


class BothCallersAgreeTests(unittest.TestCase):
    """The property that actually matters: the two modules cannot disagree.

    Asserted by reloading both against one environment, which is what a deployment does
    when it starts — these are module constants, read at import, exactly as before.
    """

    def _reload_both(self):
        import backend.chat.child_roster as roster
        import backend.tools.records as tool

        return importlib.reload(tool), importlib.reload(roster)

    def tearDown(self):
        # Leave the process holding modules built from the real environment again.
        self._reload_both()

    def test_they_resolve_to_the_same_facade(self):
        with patch.dict(os.environ, {"RECORDS_BASE_URL": "https://records.aurexis.cc"}):
            tool, roster = self._reload_both()
            self.assertEqual("https://records.aurexis.cc", tool.BASE_URL)
            self.assertEqual(tool.BASE_URL, roster.BASE_URL)

    def test_they_present_the_same_credential(self):
        with patch.dict(os.environ, {"RECORDS_API_KEY": "one-shared-secret"}):
            tool, roster = self._reload_both()
            self.assertEqual("one-shared-secret", tool.API_KEY)
            self.assertEqual(tool.API_KEY, roster.API_KEY)

    def test_neither_carries_its_own_copy_of_the_default_any_more(self):
        """The regression, caught at the source.

        A reintroduced literal would pass every test above — both modules would still
        agree, right up until somebody changed one of them.
        """
        from pathlib import Path

        for module_path in (
            "backend/tools/records.py",
            "backend/chat/child_roster.py",
        ):
            source = Path(module_path).read_text(encoding="utf-8")
            self.assertNotIn(
                'os.getenv("RECORDS_BASE_URL"',
                source,
                f"{module_path} reads RECORDS_BASE_URL itself again",
            )


if __name__ == "__main__":
    unittest.main()
