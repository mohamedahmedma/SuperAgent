"""The HTTP layer: routers, the wire schemas, the error mapping, and the wiring.

This is the only layer that knows FastAPI exists, and — with `identity/config.py` — the
only one that knows an environment does. `deps.py` composes a use case out of settings,
repositories and gateways; a router then calls that use case and maps the result.
"""
