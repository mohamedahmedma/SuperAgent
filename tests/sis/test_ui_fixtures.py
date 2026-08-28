"""The console's test fixtures are the shapes the service really returns.

This file exists because of a bug that got all the way to a live walk against the real
database. `GET /students/{n}/placements` answers `{student_number, count, placements: [...]}`.
The record screen read it as a bare list, and the jsdom smoke test **agreed with the screen**:
its fixture was a bare list too. So the console was tested, green, and threw
`rows.filter is not a function` the first time it met the actual service.

A stub that is wrong in the same way as the code it tests is worse than no stub. It does not
merely fail to catch the bug — it certifies it.

So the fixtures live in `sis/frontend/tests/fixtures.json`, the smoke test reads them from
there, and this suite checks each one against the response model its route declares in
`app.openapi()`. The check is structural rather than a full JSON Schema validation: required
keys present, no keys the model does not declare, list where a list is declared, and the same
two rules applied recursively to nested objects and to the elements of every list. That is
enough to catch every shape mistake actually made so far, and it needs no new dependency.

Types are deliberately **not** checked. A fixture writing `"9"` where the model says integer is
a fixture that is slightly untidy; a fixture with the wrong keys is a test that lies. Only one
of those is worth failing a build over, and adding the other would mean modelling every
`date | None` in the schema.
"""
import json
from pathlib import Path

import pytest

from sis.app import app

FIXTURES = Path(__file__).resolve().parents[2] / "sis" / "frontend" / "tests" / "fixtures.json"

#: Answered by the app itself and outside the versioned contract, so there is no model to
#: compare it against.
UNVERSIONED = {"GET /health"}


def _spec() -> dict:
    return app.openapi()


def _load() -> dict:
    assert FIXTURES.is_file(), f"{FIXTURES} is missing — the smoke test reads its stubs from it"
    return json.loads(FIXTURES.read_text(encoding="utf-8"))


def _resolve(schema: dict, schemas: dict) -> dict:
    """Follow a `$ref`, and collapse the `anyOf` FastAPI emits for `X | None`."""
    seen = 0
    while "$ref" in schema and seen < 8:
        schema = schemas[schema["$ref"].rsplit("/", 1)[-1]]
        seen += 1
    if "anyOf" in schema:
        # `Model | None` — the null arm carries no properties, so the other one is the shape.
        for arm in schema["anyOf"]:
            candidate = _resolve(arm, schemas)
            if candidate.get("type") != "null" and (
                candidate.get("properties") or candidate.get("type") == "array"
            ):
                return candidate
    return schema


def _operation(route: str, spec: dict) -> dict | None:
    """The GET/POST operation a fixture key names, matching `{param}` segments."""
    method, path = route.split(" ", 1)
    actual = path.strip("/").split("/")
    for candidate, operations in spec["paths"].items():
        pattern = candidate.strip("/").split("/")
        if len(pattern) != len(actual):
            continue
        if all(a == b or b.startswith("{") for a, b in zip(actual, pattern)):
            return operations.get(method.lower())
    return None


def _check(value: object, schema: dict, schemas: dict, where: str) -> list[str]:
    """Structural comparison, recursive. Returns human-readable complaints."""
    schema = _resolve(schema, schemas)
    problems: list[str] = []

    if schema.get("type") == "array":
        if not isinstance(value, list):
            return [f"{where}: the route declares a list, the fixture has {type(value).__name__}"]
        items = schema.get("items")
        if items:
            for index, element in enumerate(value):
                problems += _check(element, items, schemas, f"{where}[{index}]")
        return problems

    properties = schema.get("properties")
    if not properties:
        # A free-form object (`dict[str, Any]`) or a bare scalar: nothing to compare.
        return problems

    if not isinstance(value, dict):
        return [f"{where}: the route declares an object, the fixture has {type(value).__name__}"]

    declared = set(properties)
    required = set(schema.get("required") or ())
    present = set(value)

    for name in sorted(required - present):
        problems.append(f"{where}: missing {name!r}, which the route always sends")
    for name in sorted(present - declared):
        near = sorted((d for d in declared if name in d or d in name), key=len)
        hint = f" (did you mean {near[0]!r}?)" if near else ""
        problems.append(f"{where}: has {name!r}, which the route never sends{hint}")

    for name in sorted(present & declared):
        problems += _check(value[name], properties[name], schemas, f"{where}.{name}")

    return problems


def _routes() -> list[str]:
    return [route for route in _load() if route not in UNVERSIONED]


@pytest.mark.parametrize("route", _routes())
def test_the_fixture_matches_what_the_route_declares(route: str) -> None:
    """One test per stubbed route, so a failure names the fixture rather than the file."""
    spec = _spec()
    schemas = spec["components"]["schemas"]
    fixtures = _load()

    operation = _operation(route, spec)
    assert operation is not None, (
        f"{route} matches no route in the service. Either the fixture is stale or the client "
        "changed its path — both make the smoke test a test of nothing."
    )

    response = operation["responses"]["200"]["content"]["application/json"]["schema"]
    problems = _check(fixtures[route], response, schemas, route)
    assert not problems, (
        "the smoke test's stub does not match the response model. A stub that is wrong the "
        "same way as the screen certifies the bug instead of catching it:\n  "
        + "\n  ".join(problems)
    )


def test_every_screen_the_smoke_test_walks_is_stubbed() -> None:
    """No fixture may be dead, and the walk may not rely on the empty default.

    Both halves are the same rule from opposite sides: a fixture nothing requests is a
    misleading record of what the console does, and a screen whose data comes from the
    catch-all is a screen tested against a shape the service never sends. The smoke test
    prints the second case; this makes the first one fail too.
    """
    smoke = (FIXTURES.parent / "smoke.mjs").read_text(encoding="utf-8")
    assert "fixtures.json" in smoke, (
        "smoke.mjs must read its stubs from fixtures.json — an inline copy is a second set of "
        "shapes to keep in step, which is exactly how the placements bug survived"
    )
    for route in _routes():
        path = route.split(" ", 1)[1]
        stem = path.rsplit("/", 1)[-1] or path
        assert stem, route
