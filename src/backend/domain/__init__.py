"""Pure rules and the errors they raise. Nothing here talks to anything.

Nothing here may import FastAPI, Starlette, SQLAlchemy, Alembic, LangChain, LangGraph,
Redis, pymilvus, httpx, requests, boto3 or openai. `.importlinter` enforces that rather
than trusting this docstring. Pydantic is allowed, because the sibling services' domains
already use it (`sis/domain/value_objects.py`, `sis/domain/grades.py`).

Flat, like the domains of `sis`, `identity` and `records`. A package appears here when there
is code to put in it, never ahead of it.
"""
