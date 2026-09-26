"""Identity service.

The one place in the system that decides *who someone is*. Every other service —
records, the chat backend — verifies a signed token and reads the answer. None of
them resolve identity themselves, and none of them can mint a token.

Runs and deploys independently. Nothing here imports `backend`, `records`, or
`frontend`, and nothing imports this.

## Layout

Clean/onion layering, the same shape `sis/` uses, so an engineer moving between the two
services meets one convention rather than two::

    domain/          the rules, with no I/O. No SQLAlchemy, no FastAPI, no environment.
    application/     the use cases, over ports they declare themselves.
      ports/           Protocols: repositories, the guardian directory, the gateway.
      services/        one module per feature — sessions, whatsapp_login, administration.
      dto.py           what a use case returns, as plain dataclasses.
    infrastructure/  everything that touches the outside world.
      db/              engine, tables, and the repositories that implement the ports.
      crypto/          password hashing, the signing key, JWT minting.
      whatsapp/        the Cloud API gateway, inbound parsing, per-school channels.
      directory/       the SIS client, and the in-memory fake it falls back to.
    api/             routers, wire schemas, the error mapping, and the wiring.
    config.py        every environment variable, read lazily in exactly one place.
    app.py           the composition root.

**Dependencies point inward.** `application/` imports `domain/`; `infrastructure/` and
`api/` import `application/`; nothing in `domain/` or `application/` imports either of
those, or `config.py`. That is what makes a use case testable by constructing it with
plain classes, and it is checked by `tests/identity/test_layering.py`.
"""

__version__ = "0.2.0"
