"""Write the OpenAPI contract to `identity/openapi.json`.

    python -m identity.export_openapi

Committed on purpose, for the same reason as the records contract: it is what an
integrator reads, so a change to it should be a reviewable diff rather than a
surprise at runtime.
"""
import json
import pathlib

from identity.app import app

OUTPUT = pathlib.Path(__file__).parent / "openapi.json"


def main() -> None:
    OUTPUT.write_text(json.dumps(app.openapi(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
