# -*- coding: utf-8 -*-
"""Searching in the corpus's language, and refusing to when the translation is not safe.

Hybrid retrieval fuses a dense ranking and a sparse one, and the sparse half is BM25,
which matches TERMS. An Arabic question against an English corpus shares almost none with
it, so RRF fuses one useful list with one close to noise. Measured on 349 school
questions, changing only the language the question is asked in: recalled 337 -> 343,
ranked@8 320 -> 338, ranked@4 273 -> 314.

The risk this buys is that the query is the only thing standing between the user and the
corpus. A translation that drops "105,000 EGP" does not fail loudly — it retrieves a
different year group's fee and the turn answers confidently from it, which is the same
class of failure as item 26. So the prompt asks, and the code VERIFIES, and anything it
cannot verify searches with the original.
"""
import unittest
from unittest.mock import patch

from backend.rag import query_translation as qt


class _Reply:
    def __init__(self, content):
        self.content = content


def _invoke(text):
    return lambda messages: _Reply(text)


class TranslationIsGatedTests(unittest.TestCase):
    def setUp(self):
        qt.reset_cache()
        self.addCleanup(qt.reset_cache)

    def test_disabled_returns_the_query_untouched(self):
        with patch.object(qt._RETRIEVAL, "query_translation_enabled", False):
            text, trace = qt.translate_for_search("مصاريف Year 3 كام؟")
        self.assertEqual("مصاريف Year 3 كام؟", text)
        self.assertFalse(trace["query_translated"])

    def test_a_question_already_in_the_corpus_language_costs_nothing(self):
        """The gate is a script count, not a model, so an English deployment never pays."""
        called = []

        def _spy(messages):
            called.append(messages)
            return _Reply("should not be reached")

        with patch.object(qt._RETRIEVAL, "query_translation_enabled", True):
            text, trace = qt.translate_for_search("What are the Year 3 fees?", invoke=_spy)
        self.assertEqual("What are the Year 3 fees?", text)
        self.assertEqual([], called)
        self.assertIn("already in en", trace["query_translation_reason"])

    def test_an_arabic_question_is_translated(self):
        with patch.object(qt._RETRIEVAL, "query_translation_enabled", True):
            text, trace = qt.translate_for_search(
                "مصاريف Year 3 كام؟", invoke=_invoke("What are the Year 3 fees?")
            )
        self.assertEqual("What are the Year 3 fees?", text)
        self.assertTrue(trace["query_translated"])

    def test_the_same_question_is_translated_once(self):
        """A turn that rewrites searches twice from the same base question."""
        calls = []

        def _count(messages):
            calls.append(1)
            return _Reply("What are the Year 3 fees?")

        with patch.object(qt._RETRIEVAL, "query_translation_enabled", True):
            qt.translate_for_search("مصاريف Year 3 كام؟", invoke=_count)
            _, trace = qt.translate_for_search("مصاريف Year 3 كام؟", invoke=_count)
        self.assertEqual(1, len(calls))
        self.assertEqual("memoized", trace["query_translation_reason"])


class WhatTheCodeVerifiesRatherThanAsksForTests(unittest.TestCase):
    """Each of these is a way a translation looks fine and destroys the retrieval."""

    def setUp(self):
        qt.reset_cache()
        self.addCleanup(qt.reset_cache)

    def _translate(self, question, reply):
        with patch.object(qt._RETRIEVAL, "query_translation_enabled", True):
            return qt.translate_for_search(question, invoke=_invoke(reply))

    def test_a_dropped_amount_is_refused(self):
        """A fee row is found by "105,000 EGP" and by nothing else."""
        text, trace = self._translate(
            "هل مصاريف 105,000 EGP دي للسنة كلها؟", "Are these fees for the whole year?"
        )
        self.assertEqual("هل مصاريف 105,000 EGP دي للسنة كلها؟", text)
        self.assertFalse(trace["query_translated"])

    def test_localised_digits_are_refused(self):
        text, trace = self._translate("مصاريف Year 3 كام؟", "What are the Year ٣ fees?")
        self.assertFalse(trace["query_translated"])

    def test_a_dropped_year_code_is_refused(self):
        text, trace = self._translate("مصاريف FS1 كام؟", "What are the nursery fees?")
        self.assertFalse(trace["query_translated"])

    def test_a_dropped_email_is_refused(self):
        text, trace = self._translate(
            "أبعت الإيصال على finance@aurexis.example؟", "Where do I send the receipt?"
        )
        self.assertFalse(trace["query_translated"])

    def test_an_answered_question_is_refused(self):
        """A translation is a rephrasing. Anything that answers or explains has invented
        terms the corpus never used, and every invented term is dilution."""
        text, trace = self._translate(
            "مصاريف Year 3 كام؟",
            "The Year 3 fees are 105,000 EGP for Egyptian students and 115,000 EGP for "
            "international students, payable in three instalments across the year, and "
            "the school also offers a sibling discount of 7 percent for second children.",
        )
        self.assertFalse(trace["query_translated"])

    def test_a_reply_still_in_arabic_is_refused(self):
        text, trace = self._translate("مصاريف Year 3 كام؟", "مصاريف السنة التالتة كام؟")
        self.assertFalse(trace["query_translated"])

    def test_a_translator_that_raises_costs_the_improvement_not_the_turn(self):
        def _boom(messages):
            raise RuntimeError("provider is down")

        with patch.object(qt._RETRIEVAL, "query_translation_enabled", True):
            text, trace = qt.translate_for_search("مصاريف Year 3 كام؟", invoke=_boom)
        self.assertEqual("مصاريف Year 3 كام؟", text)
        self.assertFalse(trace["query_translated"])

    def test_typography_is_normalised_rather_than_hoped_for(self):
        """An early run came back with "Pre-K" spelled using a non-breaking hyphen, which
        BM25 scores as a different token — the measurement would have been of the
        translator rather than of the retrieval."""
        text, trace = self._translate(
            "مصاريف Pre-K كام؟", "What are the Pre‑K fees?"
        )
        self.assertTrue(trace["query_translated"])
        self.assertIn("Pre-K", text)
        self.assertNotIn("‑", text)


class ProtectedTokensTests(unittest.TestCase):
    def test_amounts_codes_dates_and_times_are_protected(self):
        found = qt.protected_tokens("مصاريف Y03 بـ 105,000 EGP يوم 15/09 الساعة 7:45؟")
        self.assertIn("105,000", found)
        self.assertIn("Y03", found)
        self.assertIn("15/09", found)
        self.assertIn("7:45", found)

    def test_emails_and_urls_are_protected_whole(self):
        found = qt.protected_tokens(
            "ابعت على admissions@aurexis.example أو https://aurexis.example/admission"
        )
        self.assertIn("admissions@aurexis.example", found)
        self.assertTrue(any("aurexis.example/admission" in token for token in found))

    def test_ordinary_words_are_not_protected(self):
        """Proper nouns are a judgement, and a rule that cannot be checked mechanically
        does not belong in a verifier."""
        self.assertEqual(set(), qt.protected_tokens("ما هي مواعيد الدراسة؟"))


if __name__ == "__main__":
    unittest.main()


class OneCallDoesWhicheverJobsTheTurnNeedsTests(unittest.TestCase):
    """Resolution and translation are independent, so the gate is a union and the prompt
    is assembled from whichever half fired.

    Two small models reading the same message is one model call too many when both jobs
    are wanted, and gating either behind the other is wrong in both directions: an Arabic
    message that stands on its own needs translating and not resolving, and an English
    follow-up the reverse. Gating translation behind `needs_resolution` would have skipped
    most Arabic turns, which is most of the traffic."""

    def setUp(self):
        from backend.rag import query_translation

        query_translation.reset_cache()
        self.addCleanup(query_translation.reset_cache)

    @staticmethod
    def _config(**overrides):
        from backend.profiles.registry import load_profile

        settings = {"query_resolution_enabled": True, "query_resolution_max_chars": 24,
                    **overrides}
        return load_profile("school").agent.model_copy(update=settings)

    def _resolve(self, question, history, payload, **kw):
        from backend.chat.resolution import resolve_question

        seen = {}

        def _invoke(*args, **kwargs):
            seen.update(kwargs)
            return payload

        resolved = resolve_question(question, history, self._config(**kw), invoke=_invoke)
        return resolved, seen

    def test_an_arabic_standalone_question_is_translated_without_being_resolved(self):
        """The turn the union gate exists for. `needs_resolution` says no — the message
        carries its own subject — and translation still has to happen."""
        long_arabic = "ما هي مصاريف Year 3 للطالب المصري في العام الدراسي القادم؟"
        resolved, seen = self._resolve(
            long_arabic, [],
            {"question": long_arabic, "intent": "standalone",
             "search_text": "What are the Year 3 fees for an Egyptian student next year?"},
        )
        self.assertTrue(seen.get("translating"))
        self.assertFalse(seen.get("resolving"))
        self.assertEqual(
            "What are the Year 3 fees for an Egyptian student next year?", resolved.search_text
        )

    def test_the_translation_lands_in_the_memo_retrieval_reads(self):
        """What makes this ONE call rather than two: retrieval later asks for the same
        translation and finds it, without knowing the resolver exists."""
        from backend.rag import query_translation as qt

        long_arabic = "ما هي مصاريف Year 3 للطالب المصري في العام الدراسي القادم؟"
        self._resolve(
            long_arabic, [],
            {"question": long_arabic, "intent": "standalone",
             "search_text": "What are the Year 3 fees for an Egyptian student next year?"},
        )
        with patch.object(qt._RETRIEVAL, "query_translation_enabled", True):
            text, trace = qt.translate_for_search(long_arabic, invoke=_invoke("unused"))
        self.assertEqual("memoized", trace["query_translation_reason"])
        self.assertIn("Year 3 fees", text)

    def test_an_english_follow_up_is_resolved_without_being_translated(self):
        resolved, seen = self._resolve(
            "and for international?", [{"role": "user", "content": "what are the Year 3 fees?"}],
            {"question": "what are the Year 3 international fees?", "intent": "followup",
             "search_text": ""},
        )
        self.assertTrue(seen.get("resolving"))
        self.assertFalse(seen.get("translating"))
        self.assertEqual("", resolved.search_text)

    def test_an_arabic_follow_up_asks_for_both(self):
        resolved, seen = self._resolve(
            "وللدولي؟", [{"role": "user", "content": "مصاريف Year 3 كام؟"}],
            {"question": "مصاريف Year 3 للدولي كام؟", "intent": "followup",
             "search_text": "What are the Year 3 international fees?"},
        )
        self.assertTrue(seen.get("resolving"))
        self.assertTrue(seen.get("translating"))
        self.assertEqual("What are the Year 3 international fees?", resolved.search_text)

    def test_a_bad_translation_does_not_cost_the_resolution(self):
        """Each field is validated on its own. A translation that drops the one term that
        identifies the answer is discarded; the resolution it arrived with still stands."""
        resolved, _ = self._resolve(
            "وللدولي؟", [{"role": "user", "content": "مصاريف Year 3 كام؟"}],
            {"question": "مصاريف Year 3 للدولي كام؟", "intent": "followup",
             "search_text": "What are the international fees?"},
        )
        self.assertEqual("", resolved.search_text, "the translation dropped Year 3")
        self.assertTrue(resolved.resolved)
        self.assertEqual("مصاريف Year 3 للدولي كام؟", resolved.question)

    def test_a_bad_resolution_does_not_cost_the_translation(self):
        """And the other way round: an empty question is an abstention, but a verified
        translation is already in the memo and still useful to retrieval."""
        resolved, _ = self._resolve(
            "وللدولي؟", [{"role": "user", "content": "مصاريف Year 3 كام؟"}],
            {"question": "", "intent": "followup",
             "search_text": "What are the Year 3 international fees?"},
        )
        self.assertFalse(resolved.resolved)
        self.assertEqual("What are the Year 3 international fees?", resolved.search_text)

    def test_a_resolver_callable_that_predates_the_flags_still_resolves(self):
        """`invoke` is documented as replaceable per deployment, and it grew two keyword
        arguments. An implementation that does not accept them raises TypeError, which the
        caller's broad except would have swallowed as "resolver error" — resolution would
        have stopped happening silently and looked like a model gone quiet. It is asked
        what it accepts instead."""
        from backend.chat.resolution import resolve_question

        legacy = lambda *args: {  # noqa: E731 — the shape being tested is the signature
            "question": "مصاريف Year 3 للدولي كام؟", "intent": "followup", "constraints": [],
        }
        resolved = resolve_question(
            "وللدولي؟", [{"role": "user", "content": "مصاريف Year 3 كام؟"}],
            self._config(), invoke=legacy,
        )
        self.assertTrue(resolved.resolved, "an old callable must still resolve")
        self.assertEqual("مصاريف Year 3 للدولي كام؟", resolved.question)
        self.assertEqual("", resolved.search_text, "it was never asked to translate")

    def test_an_english_standalone_question_costs_no_call_at_all(self):
        from backend.chat.resolution import resolve_question

        calls = []

        def _spy(*args, **kwargs):
            calls.append(args)
            return {"question": "x", "intent": "standalone"}

        resolved = resolve_question(
            "What are the fees for Year 3 Egyptian students next year?", [],
            self._config(), invoke=_spy,
        )
        self.assertEqual([], calls)
        self.assertFalse(resolved.resolved)


class TheCorpusIsAskedBeforeAModelIsTests(unittest.TestCase):
    """Translation is the FALLBACK. A corpus published in the user's language makes the
    question moot — pairing already routes an Arabic question to the Arabic half, and
    translating it into English would send it to the half that was just superseded.

    Both gates are free: a script count and a memoized database read. A turn that needs
    no translation finds that out without spending a token."""

    def setUp(self):
        qt.reset_cache()
        qt.reset_coverage()
        self.addCleanup(qt.reset_cache)
        self.addCleanup(qt.reset_coverage)

    def _with_corpus(self, languages):
        from unittest.mock import MagicMock

        from backend.composition import Services, set_default_services

        pairs = MagicMock()
        pairs.has_language.side_effect = lambda code: code in languages
        set_default_services(Services(document_pairs=pairs))
        self.addCleanup(set_default_services, None)
        return pairs

    def test_an_arabic_corpus_means_no_translation_and_no_call(self):
        self._with_corpus({"ar"})
        called = []
        with patch.object(qt._RETRIEVAL, "query_translation_enabled", True):
            text, trace = qt.translate_for_search(
                "مصاريف Year 3 كام؟", invoke=lambda m: called.append(m) or _Reply("x")
            )
        self.assertEqual("مصاريف Year 3 كام؟", text)
        self.assertFalse(trace["query_translated"])
        self.assertIn("published in ar", trace["query_translation_reason"])
        self.assertEqual([], called, "the corpus answered it; no model was needed")

    def test_an_english_only_corpus_falls_back_to_translating(self):
        self._with_corpus({"en"})
        with patch.object(qt._RETRIEVAL, "query_translation_enabled", True):
            text, trace = qt.translate_for_search(
                "مصاريف Year 3 كام؟", invoke=_invoke("What are the Year 3 fees?")
            )
        self.assertTrue(trace["query_translated"])
        self.assertEqual("What are the Year 3 fees?", text)

    def test_the_corpus_is_not_asked_once_per_question(self):
        """A database read on the critical path of every Arabic question, answered once."""
        pairs = self._with_corpus({"ar"})
        with patch.object(qt._RETRIEVAL, "query_translation_enabled", True):
            for _ in range(5):
                qt.translate_for_search("مصاريف Year 3 كام؟")
        self.assertEqual(1, pairs.has_language.call_count)

    def test_an_uploaded_arabic_corpus_is_noticed_without_a_restart(self):
        """The reason this is a short-lived memo rather than a per-conversation flag: an
        admin who uploads the Arabic half should stop being charged for translations
        within minutes, not whenever the last long session happens to end."""
        pairs = self._with_corpus(set())
        self.assertFalse(qt.corpus_covers("ar", now=0.0))
        pairs.has_language.side_effect = lambda code: code == "ar"
        self.assertFalse(qt.corpus_covers("ar", now=10.0), "still inside the memo")
        self.assertTrue(qt.corpus_covers("ar", now=10_000.0), "the memo expired")

    def test_an_unreadable_pair_table_translates_rather_than_guessing(self):
        from unittest.mock import MagicMock

        from backend.composition import Services, set_default_services

        pairs = MagicMock()
        pairs.has_language.side_effect = RuntimeError("table is gone")
        set_default_services(Services(document_pairs=pairs))
        self.addCleanup(set_default_services, None)

        self.assertFalse(qt.corpus_covers("ar"))
