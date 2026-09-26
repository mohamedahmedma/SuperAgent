"""The HTTP edge. FastAPI and pydantic live here and nowhere below.

This is the only layer allowed to know that HTTP, an environment and a database all
exist at once. It reads `sis.config`, builds a `SqlAlchemyUnitOfWork`, hands the
resulting ports to a service, and translates whatever comes back — including the domain
errors that travel up unchanged — into a status code and a JSON envelope.

Nothing here is imported by `sis.domain` or `sis.application`. That direction is the
whole point: a use case is unit-testable with fake repositories precisely because it has
never heard of a `Request`, and the day this service grows a CLI importer or a queue
consumer, that entry point composes the same services without a web framework in reach.

The package deliberately exports nothing at import time. Routers pull in models,
parsers and the engine; re-exporting them here would drag the entire service into
`alembic`'s process the moment it imports `sis.api` for anything at all.
"""
