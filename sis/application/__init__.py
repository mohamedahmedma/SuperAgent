"""The use-case layer: what this service *does*, expressed without a framework.

Three sub-packages, and the direction of every dependency between them is the point:

* `dto/`      — plain dataclasses crossing the layer boundaries. Never pydantic; a DTO
                that is a pydantic model drags validation semantics and a JSON schema
                into code that is supposed to be callable from a unit test with no
                request in sight.
* `ports/`    — `Protocol` interfaces the services are written against. Structural, so
                a fake repository in a test satisfies them by having the right methods
                rather than by inheriting from anything.
* `services/` — the use cases. They depend on `domain/` and on `ports/`, and on nothing
                else. No sqlalchemy import, no `Session` parameter, no clock read that
                is not injected.

The rule that keeps this honest: **every service must be constructible with fake
repositories and exercised with no database**. If a test of a service needs Alembic to
have run, the dependency arrow has been drawn backwards somewhere in this package.
"""
