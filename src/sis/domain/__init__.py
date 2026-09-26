"""The domain: every rule this service has, expressed without a framework.

Nothing under this package imports sqlalchemy, fastapi, pydantic or `sis.config`. That
is the constraint that makes `application/services/` unit-testable against fake
repositories with no database and no event loop — the moment one domain module reaches
for a session or a settings object, every test of every service needs one too.

This module re-exports the public surface of the seven domain modules so that callers
write `from sis.domain import Student, Percentage` rather than tracking which file each
type happens to live in. The submodules remain the source of truth and may always be
imported directly; this is a convenience, not a facade.

Why `import *` and not an explicit name list: the aggregate would then have to repeat
every class name in the domain, and a single spelling that drifts from its module makes
the *whole package* fail to import — taking the API, the repositories and the test suite
down with it, with a traceback that points here rather than at the module that changed.
Each module curates what it exposes; this file only unions them.

Import order below follows the dependency order — errors and values first, then the
aggregates built from them — so a reader can see the shape of the domain from this list
alone.

The union is then filtered down to names this package actually owns. A bare
`from x import *` re-exports whatever the submodule imported at *its* top — `math`, `re`,
`date`, `dataclass`, `field` — so without the filter `from sis.domain import *` quietly
rebinds `field` to `dataclasses.field` in the importing module, and the next reader of
that module cannot tell why a dataclass declaration two hundred lines down behaves
strangely. Filtering by provenance keeps the star import honest without maintaining a
name list that can drift.
"""
# ruff: noqa: F401, F403
import types as _types

from sis.domain.errors import *
from sis.domain.value_objects import *
from sis.domain.structure import *
from sis.domain.people import *
from sis.domain.grades import *
from sis.domain.imports import *
from sis.domain.auth import *

__all__ = sorted(
    name
    for name, value in list(globals().items())
    if not name.startswith("_")
    # Modules travel through a star import as ordinary names; drop them first, since a
    # module object's `__module__` is its *contents'* concern and answers nothing here.
    and not isinstance(value, _types.ModuleType)
    # Constants such as `PREFIX_LENGTH` are plain ints with no `__module__` at all; the
    # default keeps them, because a value defined in this package is what we are after
    # and only borrowed names (typing, datetime, dataclasses) carry a foreign one.
    and getattr(value, "__module__", "sis.domain").startswith("sis.domain")
)
