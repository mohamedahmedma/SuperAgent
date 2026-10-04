"""Use cases for the turn.

`commands/` is the write side: one Command object and one handler per use case, each
going through the domain and committing through a unit of work. `queries/` is the read
side: one Query object and one handler, free to read with SQL shaped for the screen and
returning a DTO rather than an aggregate. `services/` is what several handlers share.

Depends on `domain` and on `application.ports` only. It never imports an adapter.
"""
