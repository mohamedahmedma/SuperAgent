"""The use cases, over the ports they need.

Four reads and one link check. This layer knows about `domain/` and `ports/`, and nothing
about HTTP status codes, `httpx` or the environment — what it needs is handed to its
constructor by `api/deps.py`.

`assembly.py` is the one deliberate exception, and it is worth naming: it maps domain
values onto the **pydantic response models** in `api/schemas/`, so it imports upward. That
is where it belongs rather than in `domain/`, where it used to sit — a module that builds
wire objects is a mapper, not a rule, and pretending otherwise would have `domain/`
importing a serialisation library.
"""
