"""Whether runs reach LangSmith is an environment question, so boot has to answer it.

Nothing in this codebase turns tracing on; the SDK reads the environment. That makes a
deployment which believes it is tracing and is not completely silent — the runs never
arrive, and a switch that is off, a project nobody is watching and a key the endpoint
rejects all present identically.

It happened twice on this estate: tracing came back when the services were recreated
against a `.env` carrying the intended key, and went away on the next release, which
recreated them against a different one. The two files' `LANGSMITH_API_KEY` and
`LANGSMITH_PROJECT` differed, and nothing said so.

So: one line at boot, always, naming the project, the endpoint and the key by
fingerprint — and never the key itself, because the line has to be pasteable.
"""
import hashlib
import logging
import os
import unittest
from unittest.mock import patch

from backend.infra.tracing import describe_tracing, key_fingerprint, log_tracing_status

#: Deliberately not shaped like a real token. An `lsv2_pt_...` fixture, even an invented
#: one, is matched by GitHub's push protection and the push is rejected as a leaked
#: "LangSmith Personal Access Token". These tests only care that the string never reaches
#: the log line, so its shape is free — and a fixture that blocks a push is a bad fixture.
KEY = "langsmith-key-used-only-by-these-tests"


def env(**values):
    """Exactly these LangSmith variables and no others, whatever the host has set."""
    return patch.dict(os.environ, values, clear=True)


class TheSwitchDecidesWhatIsSaid(unittest.TestCase):
    def test_unset_reads_as_off_and_names_the_variable(self):
        with env():
            self.assertEqual("tracing off (LANGSMITH_TRACING=unset)", describe_tracing())

    def test_a_false_switch_reads_as_off_and_shows_the_value(self):
        """The value matters: `false` is someone's decision, `unset` is an omission."""
        with env(LANGSMITH_TRACING="false"):
            self.assertEqual("tracing off (LANGSMITH_TRACING=false)", describe_tracing())

    def test_the_spellings_the_sdk_accepts_all_turn_it_on(self):
        for value in ("true", "TRUE", "1", "yes", "on"):
            with self.subTest(value=value):
                with env(LANGSMITH_TRACING=value, LANGSMITH_API_KEY=KEY):
                    self.assertIn("tracing on", describe_tracing())

    def test_the_legacy_langchain_switch_is_honoured_and_named(self):
        """A deployment configured before the rename still traces, and the line has to
        say which variable did it — otherwise the operator edits the other one."""
        with env(LANGCHAIN_TRACING_V2="true", LANGCHAIN_API_KEY=KEY):
            line = describe_tracing()
        self.assertIn("tracing on", line)
        self.assertIn("LANGCHAIN_API_KEY fp", line)

    def test_an_off_line_names_the_legacy_variable_when_that_is_the_one_set(self):
        with env(LANGCHAIN_TRACING_V2="false"):
            self.assertEqual("tracing off (LANGCHAIN_TRACING_V2=false)", describe_tracing())


class TheLineNamesWhereRunsGo(unittest.TestCase):
    def test_it_names_the_project_and_the_endpoint_host(self):
        with env(
            LANGSMITH_TRACING="true",
            LANGSMITH_API_KEY=KEY,
            LANGSMITH_PROJECT="aurexis-school",
            LANGSMITH_ENDPOINT="https://eu.api.smith.langchain.com",
        ):
            line = describe_tracing()
        self.assertIn("'aurexis-school'", line)
        self.assertIn("eu.api.smith.langchain.com", line)

    def test_the_default_endpoint_is_stated_rather_than_left_blank(self):
        """Runs going to the US default while someone watches an EU workspace is one of
        the ways this disappears, so the line says where they went even when unset."""
        with env(LANGSMITH_TRACING="true", LANGSMITH_API_KEY=KEY):
            self.assertIn("api.smith.langchain.com", describe_tracing())

    def test_a_missing_key_is_called_out_rather_than_implied(self):
        with env(LANGSMITH_TRACING="true", LANGSMITH_PROJECT="aurexis-school"):
            line = describe_tracing()
        self.assertIn("LANGSMITH_API_KEY is unset", line)
        self.assertIn("no run will be ingested", line)

    def test_an_unset_project_says_default(self):
        with env(LANGSMITH_TRACING="true", LANGSMITH_API_KEY=KEY):
            self.assertIn("'default'", describe_tracing())


class TheKeyIsNeverPrinted(unittest.TestCase):
    def test_the_line_carries_a_fingerprint_and_no_part_of_the_key(self):
        with env(LANGSMITH_TRACING="true", LANGSMITH_API_KEY=KEY, LANGSMITH_PROJECT="p"):
            line = describe_tracing()
        self.assertNotIn(KEY, line)
        self.assertNotIn(KEY[-4:], line)
        self.assertIn(key_fingerprint(KEY), line)

    def test_the_fingerprint_is_what_the_documented_shell_command_produces(self):
        """DEVOPS.md compares .env files with `sha256sum | cut -c1-8`. A line that cannot
        be matched against that is a line nobody can act on."""
        expected = hashlib.sha256(KEY.encode("utf-8")).hexdigest()[:8]
        self.assertEqual(expected, key_fingerprint(KEY))
        self.assertEqual(8, len(key_fingerprint(KEY)))


class ItIsSaidAtBoot(unittest.TestCase):
    def test_it_logs_at_info_under_the_langsmith_name(self):
        with env(LANGSMITH_TRACING="true", LANGSMITH_API_KEY=KEY, LANGSMITH_PROJECT="p"):
            with self.assertLogs("backend.infra.tracing", level=logging.INFO) as caught:
                log_tracing_status()
        self.assertEqual(1, len(caught.records))
        self.assertIn("LangSmith:", caught.output[0])
        self.assertIn("tracing on", caught.output[0])

    def test_it_logs_when_tracing_is_off_too(self):
        """"Why are there no traces" is asked far more often than the reverse, and an
        absent line answers neither."""
        with env():
            with self.assertLogs("backend.infra.tracing", level=logging.INFO) as caught:
                log_tracing_status()
        self.assertIn("tracing off", caught.output[0])


if __name__ == "__main__":
    unittest.main()
