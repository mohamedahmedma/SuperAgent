"""The architecture, asserted rather than described.

A layering rule that lives only in a README is a rule that decays: the import that breaks
it is always locally reasonable — a service needs a timeout, `config` has one, and it is
one line. Six months later the use cases cannot be constructed without an environment and
nobody can point at the commit where that became true.

So the rules are checked by reading the imports out of the source with `ast`, which is
also why this file needs no fixtures and touches no database.

**What each rule actually protects**

`domain/` importing nothing outward is what makes "does eight bad passwords lock the
account" a function call rather than eight HTTP requests.

`application/` importing no framework is what lets `sis`-style fakes stand in for a
database, and what lets `import_legacy_accounts.py` reuse a use case without a request.

Neither importing `config` is the one that decays fastest, and the one with the sharpest
symptom: a use case that reads the environment behaves differently depending on which test
ran before it.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

IDENTITY = pathlib.Path(__file__).resolve().parents[2] / "identity"


def _modules(package: str) -> list[pathlib.Path]:
    return sorted((IDENTITY / package).rglob("*.py"))


def _imports(path: pathlib.Path) -> set[str]:
    """Every module this file imports, including inside functions.

    Function-level imports count. Deferring an import does not undo a dependency — it only
    hides it from the top of the file, which is precisely how a layering rule gets broken
    without anyone noticing.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


def _ids(paths: list[pathlib.Path]) -> list[str]:
    return [str(p.relative_to(IDENTITY)).replace("\\", "/") for p in paths]


#: Nothing in `domain/` may import any of these. `application/` may import none of them
#: either — it is the same list, because a use case that needs a database driver to be
#: constructed is a use case that cannot be unit-tested.
_OUTWARD = ("sqlalchemy", "fastapi", "httpx", "starlette", "pydantic", "jose")


@pytest.mark.parametrize("path", _modules("domain"), ids=_ids(_modules("domain")))
def test_the_domain_imports_nothing_outward(path: pathlib.Path) -> None:
    """The rules are pure, so they are testable by calling them."""
    for module in _imports(path):
        root = module.split(".")[0]
        assert root not in _OUTWARD, (
            f"{path.name} imports {module}. The domain layer holds rules, not I/O — "
            f"move whatever needs {root} into infrastructure/ and pass it in."
        )
        assert not module.startswith(("identity.application", "identity.infrastructure", "identity.api")), (
            f"{path.name} imports {module}, which points outward. Dependencies in this "
            f"service point inward: application -> domain, never the reverse."
        )


@pytest.mark.parametrize("path", _modules("application"), ids=_ids(_modules("application")))
def test_the_use_cases_import_no_framework(path: pathlib.Path) -> None:
    """A use case is constructed from ports, so a test builds one out of plain classes."""
    for module in _imports(path):
        root = module.split(".")[0]
        assert root not in _OUTWARD, (
            f"{path.name} imports {module}. Use cases talk to ports, not to {root} — "
            f"declare what you need in application/ports/ and implement it in "
            f"infrastructure/."
        )
        assert not module.startswith(("identity.infrastructure", "identity.api")), (
            f"{path.name} imports {module}. The concrete side implements the ports; the "
            f"use cases must not reach for it."
        )


@pytest.mark.parametrize(
    "path",
    _modules("domain") + _modules("application"),
    ids=_ids(_modules("domain") + _modules("application")),
)
def test_nothing_inward_reads_the_environment(path: pathlib.Path) -> None:
    """The rule with the sharpest symptom.

    A service that reads `os.getenv` cannot be tested without arranging the environment,
    and it then behaves differently depending on which test ran first. Everything a use
    case needs — a TTL, a round count, a lockout threshold — is a constructor argument,
    supplied by `api/deps.py`.
    """
    imports = _imports(path)
    assert "identity.config" not in imports, (
        f"{path.name} imports identity.config. Values a use case needs are passed into "
        f"its constructor by api/deps.py, which is the only layer that knows an "
        f"environment exists."
    )
    source = path.read_text(encoding="utf-8")
    assert "os.getenv" not in source and "os.environ" not in source, (
        f"{path.name} reads the environment directly. See identity/config.py."
    )


def test_the_composition_root_is_the_only_place_that_builds_the_world() -> None:
    """`app.py` and `api/deps.py` may read settings. Nothing else outside config may.

    Stated as a whitelist rather than a prohibition, because the failure mode is a *new*
    module quietly acquiring the habit — and a whitelist makes that a test failure with a
    name on it rather than a rule somebody has to remember at review time.
    """
    allowed = {"app.py", "deps.py", "session.py", "channels.py", "config.py"}
    offenders = []
    for path in sorted(IDENTITY.rglob("*.py")):
        if path.name in allowed or "tests" in path.parts:
            continue
        if "identity.config" in _imports(path):
            offenders.append(str(path.relative_to(IDENTITY)))
    assert not offenders, (
        f"These read configuration and should be handed it instead: {offenders}. "
        f"If one genuinely belongs in the composition root, add it to `allowed` above "
        f"with a reason."
    )
