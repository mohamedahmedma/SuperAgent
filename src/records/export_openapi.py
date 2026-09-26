"""Write the OpenAPI contract to `records/openapi.json`.

    python -m records.export_openapi

The generated file is committed on purpose. It is the artefact the agent's tool layer
is written against and the thing an integrator reads, so a change to it should show up
as a reviewable diff rather than as a surprise at runtime. Regenerate it whenever a
route or schema changes.
"""
import json
import pathlib

from records.app import app

OUTPUT = pathlib.Path(__file__).parent / "openapi.json"


def main() -> None:
    OUTPUT.write_text(json.dumps(app.openapi(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
