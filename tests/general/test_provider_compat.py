"""The provider's harmony parser, and the one message role that switches it off.

Every case here came from a live request. `openai/gpt-oss-*` on Together returns its
chain of thought in a dedicated `reasoning` field and a clean `content` — until the
conversation carries a `role: "tool"` message, after which `reasoning` disappears from
the response and the raw transcript arrives inside `content` instead. Measured on both
gpt-oss models, with and without tools bound, at two sampling configurations, with the
prompt as `system` and as `developer`, and with no prompt at all: only the message role
changes the outcome.

These tests assert on the outgoing PAYLOAD rather than on a response, for two reasons.
The bug is in what we send, so that is where the fix has to be checked; and the day the
endpoint parses this path correctly, `test_langchain_alone_sends_the_shape_that_breaks_it`
is what should start failing, which is the signal to delete the workaround rather than
carry it forever.
"""
import unittest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from backend.provider_compat import fold_messages, fold_tool_results_into_text

FEES = '{"chunks":[{"id":1,"text":"رسوم الصف الأول 30,000 جنيه."}]}'
GRADES = '{"student":"فاطمة","subjects":[{"name":"الفيزياء","pct":62.0}]}'


def _retrieved_once():
    """A turn that called one tool and is going back for the answer."""
    return [
        SystemMessage(content="You are the school's assistant."),
        HumanMessage(content="مصاريف ابني كام؟"),
        AIMessage(content="", tool_calls=[
            {"name": "search_knowledge_base", "args": {"query": "رسوم"}, "id": "c1"}]),
        ToolMessage(content=FEES, tool_call_id="c1"),
    ]


def _retrieved_twice():
    """Two rounds, which is what a question about fees AND grades produces."""
    return _retrieved_once() + [
        AIMessage(content="We need grades too.", tool_calls=[
            {"name": "get_student_records", "args": {"student_name": "فاطمة"}, "id": "c2"}]),
        ToolMessage(content=GRADES, tool_call_id="c2"),
    ]


def _payload(model, messages):
    return model._get_request_payload(messages, stop=None)["messages"]


def _model(folded):
    model = ChatOpenAI(model="openai/gpt-oss-20b", api_key="k", base_url="http://x")
    return fold_tool_results_into_text(model) if folded else model


class TheShapeThatBreaksTheParser(unittest.TestCase):
    def test_langchain_alone_sends_the_shape_that_breaks_it(self):
        """Pins the upstream behaviour being worked around. If this ever fails because
        the payload changed, re-measure before touching anything else."""
        sent = _payload(_model(False), _retrieved_once())
        self.assertEqual([m["role"] for m in sent],
                         ["system", "user", "assistant", "tool"])
        self.assertTrue(sent[2]["tool_calls"])


class TheFoldRemovesIt(unittest.TestCase):
    def test_no_tool_message_survives_the_fold(self):
        sent = _payload(_model(True), _retrieved_once())
        self.assertEqual([m["role"] for m in sent],
                         ["system", "user", "assistant", "user"])
        self.assertNotIn("tool", [m["role"] for m in sent])
        self.assertFalse(any(m.get("tool_calls") for m in sent))

    def test_the_retrieved_text_is_still_there_and_is_attributed(self):
        """The model has to be able to tell which result answers which part of the
        question, and the tool name is the only thing that says so — so it is carried
        over from the assistant turn before that turn is dropped."""
        sent = _payload(_model(True), _retrieved_once())
        self.assertIn(FEES, sent[-1]["content"])
        self.assertIn("search_knowledge_base", sent[-1]["content"])

    def test_the_pre_tool_narration_is_not_replayed(self):
        """Its content is what the model wrote BEFORE the tool ran — a guess at the
        answer. The gpt-oss model card says not to replay reasoning from earlier turns,
        and replaying it teaches the model that reasoning belongs in `content`."""
        sent = _payload(_model(True), _retrieved_twice())
        self.assertNotIn("We need grades too.", " ".join(m["content"] or "" for m in sent))
        self.assertFalse(any(m.get("tool_calls") for m in sent))

    def test_the_model_can_still_see_that_it_already_called_the_tool(self):
        """Dropping the turn outright loses the only record of the call, and a model that
        cannot see its own call makes it again — measured at 85 consecutive calls in one
        turn against an empty result. What it DID is restated; what it thought is not."""
        sent = _payload(_model(True), _retrieved_once())
        assistant = sent[2]
        self.assertEqual(assistant["role"], "assistant")
        self.assertIn("search_knowledge_base", assistant["content"])
        self.assertIn("رسوم", assistant["content"])

    def test_two_rounds_keep_both_results_in_order(self):
        sent = _payload(_model(True), _retrieved_twice())
        body = " ".join(m["content"] or "" for m in sent)
        self.assertIn(FEES, body)
        self.assertIn(GRADES, body)
        self.assertLess(body.index(FEES), body.index(GRADES))

    def test_results_that_arrived_together_stay_in_one_turn(self):
        """Otherwise a parallel tool call becomes a run of separate user messages, which
        reads as though the parent said all of them."""
        messages = [
            HumanMessage(content="مصاريف ودرجات؟"),
            AIMessage(content="", tool_calls=[
                {"name": "search_knowledge_base", "args": {}, "id": "c1"},
                {"name": "get_student_records", "args": {}, "id": "c2"}]),
            ToolMessage(content=FEES, tool_call_id="c1"),
            ToolMessage(content=GRADES, tool_call_id="c2"),
        ]
        sent = _payload(_model(True), messages)
        self.assertEqual([m["role"] for m in sent], ["user", "assistant", "user"])
        self.assertIn(FEES, sent[-1]["content"])
        self.assertIn(GRADES, sent[-1]["content"])

    def test_the_question_and_the_prompt_are_untouched(self):
        sent = _payload(_model(True), _retrieved_once())
        self.assertEqual(sent[0]["content"], "You are the school's assistant.")
        self.assertEqual(sent[1]["content"], "مصاريف ابني كام؟")


class ATurnWithNoToolPaysNothing(unittest.TestCase):
    def test_a_plain_conversation_is_returned_unchanged(self):
        plain = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
        self.assertIs(fold_messages(plain), plain)

    def test_the_wire_is_identical_with_and_without_the_fold(self):
        """Every turn before the first tool runs — the majority — must be byte-identical,
        so this cannot be the cause of a behaviour change nobody went looking for."""
        chat = [SystemMessage(content="You are the school's assistant."),
                HumanMessage(content="صباح الخير")]
        self.assertEqual(_payload(_model(False), chat), _payload(_model(True), chat))


class ItIsSafeToInstall(unittest.TestCase):
    def test_a_model_with_no_payload_hook_is_returned_untouched(self):
        """`init_chat_model` hands back a LAZY model when no model id is configured, and
        that object builds the real one on any unknown attribute — so the probe asks the
        class, and importing the agent without a model configured stays a no-op."""

        class Lazy:
            def __getattr__(self, name):
                raise TypeError("_init_chat_model_helper() missing 1 required argument")

        lazy = Lazy()
        self.assertIs(fold_tool_results_into_text(lazy), lazy)

    def test_installing_it_twice_folds_once(self):
        model = fold_tool_results_into_text(fold_tool_results_into_text(
            ChatOpenAI(model="openai/gpt-oss-20b", api_key="k", base_url="http://x")))
        sent = _payload(model, _retrieved_once())
        self.assertEqual([m["role"] for m in sent],
                         ["system", "user", "assistant", "user"])
        self.assertEqual(sent[-1]["content"].count("search_knowledge_base"), 1)


if __name__ == "__main__":
    unittest.main()
