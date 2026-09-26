"""A command bus and a query bus: a type-to-handler registry, and behaviours around it.

Sixty-odd lines of our own rather than a framework, because that is all a mediator is and a
dependency would not earn its place.

## A handler is an ordinary object and stays one

The bus does not own handlers, it only knows where they are. Every handler is constructed by
`bootstrap` with its collaborators, is callable on its own, and is registered here as well:

    handler = AskQuestion(pipeline=..., admission=...)
    bus.register(AskQuestionCommand, handler)
    ...
    await bus.execute(AskQuestionCommand(...))   # through the behaviours
    await handler(command)                        # or directly, in a test

That is deliberate. A mediator that is the ONLY route to a handler takes two things away:
go-to-definition stops reaching the code that runs, and a stack trace fills with bus frames
instead of naming the use case. Registering rather than owning keeps both. Tests call the
handler; routes call the bus, so they get the behaviours.

## Behaviours

Cross-cutting concerns compose around the handler, innermost last, so the first behaviour in
the list is the outermost frame:

    bus = CommandBus(behaviours=[Logging(), Timing()])

A behaviour receives the message, and `nxt` to continue the chain. It may short-circuit by not
calling `nxt` - which is how admission control refuses a turn at the door, and how a rate
limiter refuses without the handler ever seeing the message.

Later steps add the behaviours that need collaborators this layer does not have yet: the
transaction behaviour (6B, which needs the unit of work) and turn admission (6D).

## Why two buses and not one

A command changes something and a query does not, and the behaviours differ because of it: a
command wants a transaction, a query wants none. One bus would have to be told which kind it
was holding, and the registry is the natural place to record that instead. They share their
implementation and differ only in what they promise.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Protocol


class Handler(Protocol):
    """Anything callable with one message. Handlers are classes with `__call__`."""

    async def __call__(self, message: Any) -> Any: ...


class Behaviour(Protocol):
    """A cross-cutting concern wrapped around a handler.

    `nxt` continues the chain. Not calling it short-circuits, and whatever the behaviour
    returns becomes the result - which is how a refusal at the door is expressed.
    """

    async def __call__(self, message: Any, nxt: Callable[[Any], Awaitable[Any]]) -> Any: ...


class UnregisteredMessage(LookupError):
    """No handler for this message type.

    A programming error, not a runtime condition: it means `bootstrap` did not register
    something, so it says which type and what was registered instead of failing as a
    KeyError deep inside a route.
    """

    def __init__(self, message_type: type, known: Sequence[type]) -> None:
        names = ", ".join(sorted(t.__name__ for t in known)) or "nothing"
        super().__init__(f"No handler registered for {message_type.__name__}. Registered: {names}")
        self.message_type = message_type


class _Bus:
    """The shared half of both buses: a registry, and a behaviour pipeline around it."""

    #: What this bus carries, for error messages and for the two subclasses to differ on.
    kind = "message"

    def __init__(self, behaviours: Sequence[Behaviour] = ()) -> None:
        self._handlers: dict[type, Handler] = {}
        self._behaviours: tuple[Behaviour, ...] = tuple(behaviours)

    def register(self, message_type: type, handler: Handler) -> None:
        """Route `message_type` to `handler`.

        Registering twice is refused rather than silently winning, because two handlers for
        one command is a wiring mistake whose symptom would otherwise be that one of them
        never runs.
        """
        if message_type in self._handlers:
            raise ValueError(
                f"{message_type.__name__} already has a handler "
                f"({type(self._handlers[message_type]).__name__})"
            )
        self._handlers[message_type] = handler

    def handler_for(self, message_type: type) -> Handler:
        """The registered handler, so a caller may hold it directly."""
        try:
            return self._handlers[message_type]
        except KeyError:
            raise UnregisteredMessage(message_type, tuple(self._handlers)) from None

    async def _dispatch(self, message: Any) -> Any:
        handler = self.handler_for(type(message))

        async def call(m: Any) -> Any:
            return await handler(m)

        chain: Callable[[Any], Awaitable[Any]] = call
        # Reversed so that the first behaviour given is the outermost frame: the one a
        # reader of `CommandBus(behaviours=[Logging(), Timing()])` expects to see first.
        for behaviour in reversed(self._behaviours):
            chain = _wrap(behaviour, chain)
        return await chain(message)


def _wrap(
    behaviour: Behaviour, nxt: Callable[[Any], Awaitable[Any]]
) -> Callable[[Any], Awaitable[Any]]:
    """One link of the chain, as its own function so the closure captures this behaviour."""

    async def link(message: Any) -> Any:
        return await behaviour(message, nxt)

    return link


class CommandBus(_Bus):
    """Commands change something. One handler each; the result is whatever it returns.

    A handler may return a value - a new id, or an async iterator of turn events, which is
    how AskQuestion streams - so this is not fire-and-forget.
    """

    kind = "command"

    async def execute(self, command: Any) -> Any:
        return await self._dispatch(command)


class QueryBus(_Bus):
    """Queries change nothing and return a read model.

    Free to read with SQL shaped for the screen rather than through an aggregate, and
    returns a DTO, so a page needing six joins does not have to pretend to be a domain
    operation.
    """

    kind = "query"

    async def ask(self, query: Any) -> Any:
        return await self._dispatch(query)
