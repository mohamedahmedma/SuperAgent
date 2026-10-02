"""Write the OpenAPI contract to `sis/openapi.json`.

    python -m sis.export_openapi

The generated file is committed on purpose. It is the artefact an integrator reads and
the thing client code is written against, so a change to it should show up as a
reviewable diff rather than as a surprise at runtime. Regenerate it whenever a route or
schema changes.

No database is needed to run this. The schema check lives in the app's lifespan, which
never starts here — rendering the contract must not require production credentials.
"""
import json
import pathlib

from sis.app import app

OUTPUT = pathlib.Path(__file__).parent / "openapi.json"


def main() -> None:
    OUTPUT.write_text(json.dumps(app.openapi(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
