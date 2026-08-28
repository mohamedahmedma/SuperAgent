"""The driving side of the hexagon: HTTP in, contract out.

The only layer that knows FastAPI exists and — with `records/config.py` — the only one
that knows an environment does. `deps.py` composes a use case and checks the two
credentials; a router calls it and maps the result onto `schemas/`.
"""
