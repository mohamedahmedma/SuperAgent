"""The only package that is allowed to know every layer.

Today's `composition.py` moves here: the container that builds each collaborator once and
lazily, the app factory, and the typed settings. It wires the adapters in
`infrastructure/` to the ports in `application/ports/`, registers handlers on the bus, and
hands the result to `api/`.

Everything else in the backend depends inwards. This is where the arrows are allowed to
point anywhere, because something has to choose the concrete classes.
"""
