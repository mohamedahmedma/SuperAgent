"""The hexagon, asserted rather than described.

A layering rule that lives only in a README decays: the import that breaks it is always
locally reasonable — an adapter needs a timeout, `config` has one, it is one line. Six
months later the use cases cannot be constructed without an environment and nobody can
point at the commit where that became true.

So the rules are read out of the source with `ast`, which is why this file needs no
fixtures and touches no network.

## Why these particular rules, for this particular service

Records is an **adapter between two services**, on the latency path of every parent
question. Two of the rules below are really about that:

`ports/` importing no adapter is what lets a use case be exercised against a dict instead
of a running SIS — the difference between a test suite that takes two seconds and one that
needs `docker compose up`.

`domain/` and `application/` importing no `httpx` is what keeps the count of HTTP clients
in this service at three, all of them built by the composition root with one pool size.
The bug this prevents is the one that was actually here: three clients, three different
answers to how wide the pool should be, and two of them wrong.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

RECORDS = pathlib.Path(__file__).resolve().parents[2] / "records"


def _modules(package: str) -> list[pathlib.Path]:
    return sorted((RECORDS / package).rglob("*.py"))


def _imports(path: pathlib.Path) -> set[str]:
    """Every module this file imports, including inside functions.

    Function-level imports count. Deferring an import does not undo a dependency — it only
    hides it from the top of the file, which is exactly how a layering rule gets broken
    without anyone noticing. Several adapters here legitimately defer `import httpx` to
    keep import time cheap, and this still sees it.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


def _ids(paths: list[pathlib.Path]) -> list[str]:
    return [str(p.relative_to(RECORDS)).replace("\\", "/") for p in paths]


_IO = ("httpx", "fastapi", "starlette", "urllib", "requests")

_DOMAIN = _modules("domain")
_PORTS = _modules("ports")
_APPLICATION = _modules("application")


@pytest.mark.parametrize("path", _DOMAIN, ids=_ids(_DOMAIN))
def test_the_domain_does_no_io(path: pathlib.Path) -> None:
    """The school's vocabulary is testable by calling it."""
    for module in _imports(path):
        root = module.split(".")[0]
        assert root not in _IO, (
            f"{path.name} imports {module}. The domain holds rules, not I/O — move "
            f"whatever needs {root} into adapters/ and pass it in."
        )
        assert not module.startswith(
            ("records.adapters", "records.api", "records.application", "records.ports")
        ), (
            f"{path.name} imports {module}, which points outward. The domain is the "
            f"centre of the hexagon and knows about none of the rings around it."
        )


@pytest.mark.parametrize("path", _PORTS, ids=_ids(_PORTS))
def test_a_port_names_no_implementation(path: pathlib.Path) -> None:
    """A port is what the use cases need, not how it is met.

    A port that imported its adapter would make the inversion decorative: `application/`
    would transitively pull in `httpx` and a SIS base URL just by declaring what it wants.
    """
    for module in _imports(path):
        assert module.split(".")[0] not in _IO, f"{path.name} imports {module}"
        assert not module.startswith(("records.adapters", "records.api")), (
            f"{path.name} imports {module}. Ports state the need; adapters/ meets it."
        )


@pytest.mark.parametrize("path", _APPLICATION, ids=_ids(_APPLICATION))
def test_the_use_cases_touch_no_transport(path: pathlib.Path) -> None:
    """A use case talks to ports, never to a socket.

    `api.schemas` is allowed, and only from `assembly.py`: it maps domain values onto the
    pydantic response models, which is a mapper's job. Every other module here would be
    reaching for the wire format without cause.
    """
    for module in _imports(path):
        assert module.split(".")[0] not in _IO, (
            f"{path.name} imports {module}. Use cases talk to ports — declare the need in "
            f"records/ports/ and implement it under records/adapters/."
        )
        assert not module.startswith("records.adapters"), (
            f"{path.name} imports {module}. The concrete side implements the ports; the "
            f"use cases must not reach for it."
        )
        if module.startswith("records.api"):
            assert path.name == "assembly.py", (
                f"{path.name} imports {module}. Only assembly.py may see the wire format, "
                f"because building response models is what it is for."
            )


@pytest.mark.parametrize(
    "path", _DOMAIN + _PORTS + _APPLICATION, ids=_ids(_DOMAIN + _PORTS + _APPLICATION)
)
def test_nothing_inward_reads_the_environment(path: pathlib.Path) -> None:
    """The rule with the sharpest symptom.

    A use case that reads `os.getenv` cannot be tested without arranging the environment,
    and it then behaves differently depending on which test ran first. Everything a use
    case needs — a policy, a timeout, a pool size — is a constructor argument supplied by
    `api/deps.py` or the composition root.
    """
    assert "records.config" not in _imports(path), (
        f"{path.name} imports records.config. Values a use case needs are handed to it, "
        f"not fetched — see records/api/deps.py."
    )
    source = path.read_text(encoding="utf-8")
    assert "os.getenv" not in source and "os.environ" not in source, (
        f"{path.name} reads the environment directly. See records/config.py."
    )


def test_only_the_composition_root_builds_the_world() -> None:
    """Which files may read configuration, as a whitelist.

    A whitelist rather than a prohibition, because the failure mode is a *new* module
    quietly acquiring the habit — and a whitelist makes that a test failure with a name on
    it rather than a rule somebody has to remember at review time.
    """
    allowed = {"app.py", "config.py", "deps.py", "identity.py", "http.py",
               "calendar.py", "directory.py", "grades.py"}
    offenders = [
        str(p.relative_to(RECORDS)).replace("\\", "/")
        for p in sorted(RECORDS.rglob("*.py"))
        if p.name not in allowed and "records.config" in _imports(p)
    ]
    assert not offenders, (
        f"These read configuration and should be handed it instead: {offenders}."
    )


def test_there_is_exactly_one_http_client_builder() -> None:
    """Three clients to one service, built one way.

    There used to be three `httpx.Client(...)` constructions with three different sets of
    limits, and two of them were wrong — see `records/config._DEFAULT_POOL_SIZE` for what
    that cost. Consolidating them is only durable if a fourth cannot appear quietly.
    """
    builders = [
        str(p.relative_to(RECORDS)).replace("\\", "/")
        for p in sorted(RECORDS.rglob("*.py"))
        if "httpx.Client(" in p.read_text(encoding="utf-8")
    ]
    assert builders == ["adapters/sis/http.py"], (
        f"httpx.Client is constructed in {builders}. One builder, in adapters/sis/http.py, "
        f"so the pool size and the no-retry/no-redirect rules cannot drift apart."
    )
