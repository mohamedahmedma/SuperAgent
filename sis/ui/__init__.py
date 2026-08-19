"""The registrar UI: server-rendered HTML, and a delivery mechanism for nothing else.

This package is the *second* way into the same use cases `sis/api/` already exposes. It
holds no rules. A page handler resolves a form into arguments, calls a service in
`sis/application/services/`, and renders the result; every decision about what is legal
was made underneath it and is shared, byte for byte, with the JSON API. A rule that
exists only here is a rule the API does not enforce, and the registrar would meet it as
"the website wouldn't let me, but the import did it anyway".

**The browser receives finished HTML.** There is no `fetch`, no JSON parsed in a page and
no client-side data layer. Forms POST to Python, which validates and redirects
(POST/Redirect/GET), so a refresh after an import re-renders a result instead of
re-committing it. JavaScript may only do things that are cosmetic — enabling a button,
confirming a destructive click — because a school with a locked-down browser or a
half-loaded page must still be able to enrol a child.

**A handler calls services, never repositories and never sqlalchemy.** That is the same
rule `sis/api/routers/` obeys, and it is what keeps `sis/ui/deps.py` and `sis/api/deps.py`
the only modules in either surface that know an environment and a database exist.

Nothing is imported here on purpose. `sis/app.py` mounts the routers by module path, and
an import in this file would drag Jinja2, the template directory and every router into
any process that so much as touched `sis.ui` — including the `export_openapi.py` run that
has no business loading a UI at all.
"""
from __future__ import annotations

__all__: list[str] = []
