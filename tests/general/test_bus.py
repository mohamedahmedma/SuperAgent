"""The command and query buses, and the behaviours that wrap a handler.

The registry half is nearly too small to test. What is worth pinning is the part a later
slice could break without noticing: that behaviours nest in the order they were given, that a
behaviour can refuse without the handler running, that a handler stays callable on its own,
and that an unregistered message says so usefully instead of raising KeyError.
"""

import logging
import unittest

from backend.application.behaviours import Logging, Timing
from backend.application.bus import CommandBus, QueryBus, UnregisteredMessage


class Ask:
    def __init__(self, text="hello"):
        self.text = text


class Count:
    pass


class _Handler:
    """A handler is an ordinary callable object; the bus never requires a base class."""

    def __init__(self, reply="answered"):
        self.reply = reply
        self.seen = []

    async def __call__(self, message):
        self.seen.append(message)
        return self.reply


class _Recording:
    """A behaviour that records when it entered and left, so nesting order is observable."""

    def __init__(self, name, log):
        self.name = name
        self.log = log

    async def __call__(self, message, nxt):
        self.log.append(f"enter {self.name}")
        result = await nxt(message)
        self.log.append(f"leave {self.name}")
        return result


class _Refusing:
    """A behaviour that short-circuits, the way admission control refuses at the door."""

    def __init__(self, reason="too busy"):
        self.reason = reason

    async def __call__(self, message, nxt):
        return self.reason


class CommandBusTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_command_reaches_its_handler_and_the_result_comes_back(self):
        handler = _Handler()
        bus = CommandBus()
        bus.register(Ask, handler)

        command = Ask()
        self.assertEqual("answered", await bus.execute(command))
        self.assertEqual([command], handler.seen)

    async def test_the_handler_stays_callable_on_its_own(self):
        """The whole reason the bus registers handlers instead of owning them.

        A test, or any caller that wants a stack trace naming the use case rather than the
        bus, holds the handler directly. If this ever stops working the bus has become the
        only route to the code, which is what registering was meant to avoid.
        """
        handler = _Handler()
        bus = CommandBus()
        bus.register(Ask, handler)

        command = Ask()
        self.assertEqual("answered", await handler(command))
        self.assertIs(handler, bus.handler_for(Ask))
        self.assertEqual([command], handler.seen)

    async def test_behaviours_nest_in_the_order_they_were_given(self):
        """First in the list is the outermost frame, which is what reading it suggests."""
        log = []
        bus = CommandBus(behaviours=[_Recording("outer", log), _Recording("inner", log)])
        bus.register(Ask, _Handler())

        await bus.execute(Ask())

        self.assertEqual(
            ["enter outer", "enter inner", "leave inner", "leave outer"],
            log,
        )

    async def test_a_behaviour_can_refuse_without_the_handler_running(self):
        handler = _Handler()
        bus = CommandBus(behaviours=[_Refusing("no leases left")])
        bus.register(Ask, handler)

        self.assertEqual("no leases left", await bus.execute(Ask()))
        self.assertEqual([], handler.seen)

    async def test_a_refusal_outside_still_skips_the_behaviours_inside_it(self):
        log = []
        bus = CommandBus(behaviours=[_Refusing(), _Recording("inner", log)])
        bus.register(Ask, _Handler())

        await bus.execute(Ask())

        self.assertEqual([], log)

    async def test_an_unregistered_message_names_itself_and_what_was_registered(self):
        bus = CommandBus()
        bus.register(Ask, _Handler())

        with self.assertRaises(UnregisteredMessage) as caught:
            await bus.execute(Count())

        self.assertIs(Count, caught.exception.message_type)
        self.assertIn("Count", str(caught.exception))
        self.assertIn("Ask", str(caught.exception))

    async def test_an_unregistered_message_is_a_lookup_error(self):
        """So a route that wants to map it to a 500 can catch the standard type."""
        self.assertTrue(issubclass(UnregisteredMessage, LookupError))

    def test_registering_twice_is_refused(self):
        """Two handlers for one command is a wiring mistake, and the symptom of letting the
        second win silently is that one of them simply never runs."""
        bus = CommandBus()
        bus.register(Ask, _Handler("first"))

        with self.assertRaises(ValueError) as caught:
            bus.register(Ask, _Handler("second"))

        self.assertIn("Ask", str(caught.exception))

    async def test_an_exception_from_the_handler_reaches_the_caller(self):
        """Behaviours observe failures; they do not swallow them."""

        class _Boom:
            async def __call__(self, message):
                raise RuntimeError("provider refused")

        log = []
        bus = CommandBus(behaviours=[_Recording("outer", log)])
        bus.register(Ask, _Boom())

        with self.assertRaises(RuntimeError):
            await bus.execute(Ask())
        # Entered, never left: the behaviour saw the call begin and the exception pass through.
        self.assertEqual(["enter outer"], log)


class QueryBusTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_query_returns_its_read_model(self):
        bus = QueryBus()
        bus.register(Count, _Handler(reply=7))
        self.assertEqual(7, await bus.ask(Count()))

    async def test_the_two_buses_keep_separate_registries(self):
        """A command bus and a query bus may both carry a type of the same name without one
        answering for the other."""
        commands, queries = CommandBus(), QueryBus()
        commands.register(Ask, _Handler("written"))
        queries.register(Ask, _Handler("read"))

        self.assertEqual("written", await commands.execute(Ask()))
        self.assertEqual("read", await queries.ask(Ask()))


class BehaviourTests(unittest.IsolatedAsyncioTestCase):
    async def test_timing_measures_a_handler_that_failed(self):
        """A use case that takes nine seconds and then raises is the one worth timing."""

        class _Boom:
            async def __call__(self, message):
                raise RuntimeError("nope")

        ticks = iter([1.0, 3.5])
        bus = CommandBus(behaviours=[Timing(clock=lambda: next(ticks))])
        bus.register(Ask, _Boom())

        with self.assertLogs("backend.application.behaviours.timing", level="INFO") as logs:
            with self.assertRaises(RuntimeError):
                await bus.execute(Ask())

        self.assertIn("2500.0 ms", logs.output[0])
        self.assertIn("Ask", logs.output[0])

    async def test_timing_says_nothing_below_its_threshold(self):
        ticks = iter([1.0, 1.01])
        bus = CommandBus(behaviours=[Timing(threshold_seconds=1.0, clock=lambda: next(ticks))])
        bus.register(Ask, _Handler())

        logger = logging.getLogger("backend.application.behaviours.timing")
        with self.assertNoLogs(logger, level="INFO"):
            await bus.execute(Ask())

    async def test_logging_never_writes_the_message_itself(self):
        """A command carries a parent's question, a child's name and an access token."""
        bus = CommandBus(behaviours=[Logging()])
        bus.register(Ask, _Handler())

        with self.assertLogs("backend.application.behaviours.logging_behaviour", "DEBUG") as logs:
            await bus.execute(Ask(text="what are Ahmed's marks"))

        joined = "\n".join(logs.output)
        self.assertNotIn("Ahmed", joined)
        self.assertIn("Ask", joined)

    async def test_logging_reports_a_failure_and_re_raises(self):
        class _Boom:
            async def __call__(self, message):
                raise ValueError("bad slot")

        bus = CommandBus(behaviours=[Logging()])
        bus.register(Ask, _Boom())

        with self.assertLogs("backend.application.behaviours.logging_behaviour", "DEBUG") as logs:
            with self.assertRaises(ValueError):
                await bus.execute(Ask())

        self.assertIn("ValueError", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
