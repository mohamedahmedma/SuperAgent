"""Harmony transcript handling, against transcripts the provider actually produced.

Every LEAK_* constant below is a real `content` string captured from
`openai/gpt-oss-20b` on Together at the school profile's own `temperature=0.2`. They are
verbatim on purpose: the three leak shapes differ in ways that a hand-written sample
would smooth over, and the third one is the reason `Finalizer` exists at all.
"""
import unittest

from langchain_core.messages import AIMessageChunk

from backend.chat.finalize import Finalizer, finalize_text, message_text
from backend.chat.model_output import HarmonyFilter, has_harmony_markup, strip_harmony

# The whole envelope, literal tokens intact. Captured from the agent's ANSWERING call —
# the one whose content cannot be dropped, because the answer is inside it.
LEAK_FULL_ENVELOPE = (
    "<|channel|>analysis<|message|>We have two chunks: fees for first grade and fourth "
    'grade. The user asked "مصاريف ابني كام" meaning "how much are my son\'s expenses". '
    "We need to ask for grade? But we can provide general info. Provide both options."
    "<|end|><|start|>assistant<|channel|>final<|message|>"
    "رسوم الصف الأول الابتدائي 30,000 جنيه على ثلاث دفعات."
)

# Tokens eaten by the provider's parser, bare header left behind.
LEAK_BARE_HEADER = (
    "commentary to=functions.search_knowledge_base analysisNo data. We can respond: "
    '"I don\'t have that info."عذرًا، لا أملك معلومات حول مصاريف ابنك.'
)

# No marker at all. Unstrippable by any text rule — handled by WHERE it occurs.
LEAK_NAKED_PROSE = (
    "We need to see knowledge base. We need to see the result."
    "مصاريف ابنك تختلف حسب العمر، النشاطات التي يشارك فيها."
)

CLEAN_ANSWER = "رسوم الصف الأول الابتدائي 30,000 جنيه على ثلاث دفعات. [1]"


class HarmonyStripTests(unittest.TestCase):
    def test_plain_text_is_returned_byte_for_byte(self):
        """The common case, and the one that must never regress."""
        self.assertEqual(strip_harmony(CLEAN_ANSWER), CLEAN_ANSWER)
        self.assertFalse(has_harmony_markup(CLEAN_ANSWER))

    def test_empty_and_none_survive(self):
        self.assertEqual(strip_harmony(""), "")
        self.assertEqual(strip_harmony(None), None)

    def test_full_envelope_keeps_only_the_final_channel(self):
        out = strip_harmony(LEAK_FULL_ENVELOPE)
        self.assertEqual(out, "رسوم الصف الأول الابتدائي 30,000 جنيه على ثلاث دفعات.")
        self.assertNotIn("We have two chunks", out)
        self.assertNotIn("<|", out)

    def test_bare_header_is_stripped_from_the_front(self):
        out = strip_harmony(LEAK_BARE_HEADER)
        self.assertFalse(out.startswith("commentary"))
        self.assertNotIn("to=functions.", out)

    def test_an_answer_opening_with_the_word_analysis_is_not_eaten(self):
        """`to=functions.` is required precisely so prose cannot trip the header rule."""
        prose = "analysis of the fee schedule shows three instalments."
        self.assertEqual(strip_harmony(prose), prose)

    def test_commentary_channel_is_dropped_entirely(self):
        transcript = (
            "<|start|>assistant<|channel|>commentary to=functions.search_knowledge_base "
            '<|constrain|>json<|message|>{"query": "fees"}<|call|>'
        )
        self.assertEqual(strip_harmony(transcript), "")

    def test_a_body_with_no_channel_named_is_kept(self):
        """Permissive by design: suppressing a real answer is the worse failure."""
        self.assertEqual(
            strip_harmony("<|start|>assistant<|message|>مرحبا<|return|>"), "مرحبا"
        )


class HarmonyStreamingTests(unittest.TestCase):
    def _stream(self, text, size):
        harmony = HarmonyFilter()
        out = "".join(harmony.feed(text[i : i + size]) for i in range(0, len(text), size))
        return out + harmony.flush()

    def test_tokens_split_across_chunk_boundaries(self):
        """`<|chan` ending one delta and `nel|>` beginning the next must still parse."""
        expected = strip_harmony(LEAK_FULL_ENVELOPE)
        for size in (1, 2, 3, 5, 7, 13, 64):
            with self.subTest(chunk_size=size):
                self.assertEqual(self._stream(LEAK_FULL_ENVELOPE, size).strip(), expected)

    def test_plain_text_streams_through_unchanged_at_every_chunk_size(self):
        for size in (1, 3, 11):
            with self.subTest(chunk_size=size):
                self.assertEqual(self._stream(CLEAN_ANSWER, size), CLEAN_ANSWER)

    def test_a_lone_angle_bracket_is_not_held_forever(self):
        self.assertEqual(self._stream("5 < 6 and 7 > 6", 1), "5 < 6 and 7 > 6")

    def test_a_bare_header_is_stripped_at_every_chunk_size(self):
        """The gap that let a half-read header through.

        `strip_harmony` saw the whole string and stripped it, so the one-shot test
        passed while the STREAMING path — the only one production uses — was broken at
        every chunk size from 1 to 11. `_BARE_HEADER` matches an incomplete header
        (`to=functions.search_kn` satisfies `[\\w.-]+`), so an eager attempt stripped a
        prefix, declared itself settled, and published the remainder — `owledge_base` —
        as the first word of the answer.
        """
        expected = "عذرًا، لا أملك معلومات حول مصاريف ابنك."
        header = "commentary to=functions.search_knowledge_base " + expected
        for size in (1, 2, 3, 5, 7, 11, 23, 64, 97, 999):
            with self.subTest(chunk_size=size):
                out = self._stream(header, size).strip()
                self.assertEqual(out, expected)
                self.assertNotIn("to=functions", out)
                self.assertNotIn("owledge_base", out)

    def test_an_answer_shorter_than_a_header_still_arrives(self):
        """Nothing is emitted until the header question is settled, so a reply shorter
        than a header would be held for its whole life if `flush` did not settle it."""
        for size in (1, 3, 999):
            with self.subTest(chunk_size=size):
                self.assertEqual(self._stream("تمام.", size), "تمام.")


def _chunk(mid, text="", tool_calls=False):
    """One `AIMessageChunk` as the provider streams them."""
    kwargs = {"content": text, "id": mid}
    if tool_calls:
        kwargs["tool_call_chunks"] = [
            {"name": "search_knowledge_base", "args": '{"query":"x"}', "id": "c1", "index": 0}
        ]
    return AIMessageChunk(**kwargs)


class FinalizerTests(unittest.TestCase):
    def test_a_tool_calling_message_contributes_nothing(self):
        """The failure that showed a parent an invented fee before the tool had run."""
        final = Finalizer()
        seen = final.consider(_chunk("m1", tool_calls=True))
        seen += final.consider(_chunk("m1", "عذرًا، لا أستطيع العثور على معلومات."))
        seen += final.finish()
        self.assertEqual(seen, "")
        self.assertEqual(final.answer, "")
        self.assertEqual(final.as_trace()["finalize_dropped_tool_call_messages"], 1)

    def test_short_prose_before_the_tool_call_delta_never_escapes(self):
        """Content that arrives ahead of the delta is still caught, because the header
        hold in `HarmonyFilter` has not released it yet.

        Measured order puts the tool-call delta first, so this is belt-and-braces — but
        it is the strongest guarantee available and worth pinning: while the first
        `_HEADER_DECIDED_AFTER` characters are buffered, a message can still change its
        mind about being a tool call and take its prose back with it.
        """
        final = Finalizer()
        seen = final.consider(_chunk("m1", "We need to see the result."))
        seen += final.consider(_chunk("m1", tool_calls=True))
        seen += final.finish()
        self.assertEqual(seen, "")
        self.assertEqual(final.answer, "")
        self.assertEqual(final.as_trace()["finalize_dropped_tool_call_messages"], 1)

    def test_prose_long_enough_to_have_been_streamed_cannot_be_taken_back(self):
        """The honest limit. Past the buffer, text has reached the reader and no later
        chunk can unsay it — which is why the rule is enforced by message shape rather
        than by trying to retract."""
        long_prose = "We need to see the result. " * 8  # comfortably past the hold
        final = Finalizer()
        seen = final.consider(_chunk("m1", long_prose))
        self.assertNotEqual(seen, "")
        # From the moment the message identifies itself, nothing further escapes.
        seen_after = final.consider(_chunk("m1", tool_calls=True))
        seen_after += final.consider(_chunk("m1", "and more prose"))
        seen_after += final.finish()
        self.assertEqual(seen_after, "")

    def test_the_answering_message_survives_and_is_stripped(self):
        final = Finalizer()
        final.consider(_chunk("m1", tool_calls=True))
        final.consider(_chunk("m1", "invented answer before the tool ran"))
        final.note_tool_result()
        out = "".join(final.consider(_chunk("m2", part)) for part in _split(LEAK_FULL_ENVELOPE, 9))
        out += final.finish()
        self.assertEqual(out.strip(), "رسوم الصف الأول الابتدائي 30,000 جنيه على ثلاث دفعات.")
        self.assertNotIn("invented answer", out)
        self.assertNotIn("We have two chunks", out)
        trace = final.as_trace()
        self.assertEqual(trace["finalize_dropped_tool_call_messages"], 1)
        self.assertEqual(trace["finalize_harmony_messages"], 1)
        self.assertEqual(trace["finalize_tool_results"], 1)

    def test_naked_prose_is_caught_by_the_tool_call_rule(self):
        """The shape with no marker. No text rule finds it; the message shape does."""
        final = Finalizer()
        final.consider(_chunk("m1", tool_calls=True))
        for part in _split(LEAK_NAKED_PROSE, 8):
            self.assertEqual(final.consider(_chunk("m1", part)), "")
        self.assertEqual(final.finish(), "")

    def test_a_clean_turn_is_untouched(self):
        final = Finalizer()
        out = "".join(final.consider(_chunk("m1", part)) for part in _split(CLEAN_ANSWER, 4))
        out += final.finish()
        self.assertEqual(out, CLEAN_ANSWER)
        self.assertEqual(final.as_trace()["finalize_dropped_tool_call_messages"], 0)

    def test_chunks_with_no_id_still_stream(self):
        """A provider that omits ids collapses to one message — older behaviour, valid."""
        final = Finalizer()
        out = final.consider(AIMessageChunk(content="مرحبا"))
        self.assertEqual(out + final.finish(), "مرحبا")

    def test_message_text_reads_block_content(self):
        blocks = AIMessageChunk(content=[{"type": "text", "text": "أهلاً"}, {"type": "other"}])
        self.assertEqual(message_text(blocks), "أهلاً")


class SynchronousPathTests(unittest.TestCase):
    def test_same_two_rules_apply_off_the_stream(self):
        self.assertEqual(finalize_text(LEAK_FULL_ENVELOPE).strip(),
                         "رسوم الصف الأول الابتدائي 30,000 جنيه على ثلاث دفعات.")
        self.assertEqual(finalize_text("anything", has_tool_calls=True), "")
        self.assertEqual(finalize_text(CLEAN_ANSWER), CLEAN_ANSWER)


def _split(text, size):
    return [text[i : i + size] for i in range(0, len(text), size)]


# Captured from the live provider on the question that started all of this, with the
# `<|…|>` delimiters eaten and the channel name welded to the text that followed it.
# This is the shape the original bug report showed, and the one that still leaked after
# the token and bare-header rules were both in place.
GLUED_FAKE_TRANSCRIPT = (
    "analysisThe tool call failed due to missing query field. We need to call "
    "search_knowledge_base with query.assistantcommentary "
    'to=functions.search_knowledge_basejson{"query":"مصاريف ابني"}'
)
GLUED_FINAL_ANSWER = "finalرسوم الصف الأول الابتدائي للعام 2026 هي 30,000 جنيه على ثلاث دفعات. [1]"


class GluedTranscriptTests(unittest.TestCase):
    """Markers with their delimiters eaten. Found by the live eval, not by hand."""

    def _stream(self, text, size):
        harmony = HarmonyFilter()
        out = "".join(harmony.feed(text[i:i + size]) for i in range(0, len(text), size))
        return out + harmony.flush()

    def _at_every_size(self, text, expected):
        for size in (1, 2, 3, 5, 7, 11, 23, 64, 97, 999):
            with self.subTest(chunk_size=size):
                self.assertEqual(self._stream(text, size).strip(), expected.strip())

    def test_a_transcript_with_no_final_channel_yields_no_answer(self):
        """Reasoning plus a fabricated tool call and no answer channel. Returning the
        reasoning with its label removed would only make the leak harder to see."""
        self._at_every_size(GLUED_FAKE_TRANSCRIPT, "")

    def test_text_after_a_glued_final_marker_is_the_answer(self):
        self._at_every_size(
            GLUED_FINAL_ANSWER,
            "رسوم الصف الأول الابتدائي للعام 2026 هي 30,000 جنيه على ثلاث دفعات. [1]",
        )

    def test_reasoning_before_a_final_marker_is_dropped(self):
        self._at_every_size(
            "analysisI should answer.finalرسوم الصف الأول 30,000 جنيه. [1]",
            "رسوم الصف الأول 30,000 جنيه. [1]",
        )

    def test_a_final_marker_far_past_the_header_hold_is_still_found(self):
        """The hold releases at 96 characters; the answer channel can open later, so the
        message is kept whole until it does."""
        text = ("analysisI need to check the chunks. " + "More reasoning here. " * 8
                + "finalرسوم الصف الأول 30,000 جنيه. [1]")
        self._at_every_size(text, "رسوم الصف الأول 30,000 جنيه. [1]")

    def test_prose_that_merely_contains_a_channel_word_is_untouched(self):
        """The words are ordinary English. Only a marker WELDED to what follows it — no
        space, and not continuing the word in lowercase — is treated as markup."""
        for prose in (
            "analysis of the fee schedule shows three instalments.",
            "Finally, the fees are 30,000 EGP.",
            "The fees are 30,000 EGP for Year 1. [1]",
        ):
            with self.subTest(prose=prose):
                self._at_every_size(prose, prose)


if __name__ == "__main__":
    unittest.main()
