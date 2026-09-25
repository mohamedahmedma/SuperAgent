"""RAG_FIX_PLAN item 19: what a turn loads is exactly what its readers read.

A turn loads only the latest `history_window()` messages. That is safe only while every
reader of a turn's history reads a tail no longer than that window, so each reader here
is run over a long conversation and over just its window, and must say the same thing.
A reader added later that looks further back fails here instead of silently losing the
start of a parent's conversation.
"""
import unittest
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from backend.chat.context_messages import build_context_messages, history_window
from backend.chat.resolution import conversation_text
from backend.chat.signals import _last_user_text
from backend.profiles import get_profile


def _conversation(length: int) -> list:
    return [
        HumanMessage(content=f"question {n}") if n % 2 == 0 else AIMessage(content=f"answer {n}")
        for n in range(length)
    ]


class HistoryWindowTests(unittest.TestCase):
    def setUp(self):
        self.agent = get_profile().agent
        self.full = _conversation(40)
        self.window = self.full[-history_window(self.agent):]

    def test_the_agent_is_shown_the_same_history(self):
        def shown(history):
            return [(type(m).__name__, m.content) for m in build_context_messages(history, "now")]

        self.assertEqual(shown(self.full), shown(self.window))

    def test_the_resolver_and_the_classifier_read_the_same_dialogue(self):
        limit = self.agent.query_resolution_history_messages
        self.assertEqual(conversation_text(self.full, limit=limit), conversation_text(self.window, limit=limit))

    def test_the_scoring_fallback_finds_the_same_last_question(self):
        self.assertEqual(_last_user_text(self.full), _last_user_text(self.window))

    def test_the_window_is_the_longest_tail_any_reader_takes(self):
        wide_agent = SimpleNamespace(context_window_messages=14, query_resolution_history_messages=6)
        wide_resolver = SimpleNamespace(context_window_messages=6, query_resolution_history_messages=20)
        self.assertEqual(14, history_window(wide_agent))
        self.assertEqual(20, history_window(wide_resolver))
        self.assertEqual(2, history_window(SimpleNamespace(context_window_messages=0,
                                                           query_resolution_history_messages=0)))


if __name__ == "__main__":
    unittest.main()
