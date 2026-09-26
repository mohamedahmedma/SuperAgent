"""`python -m backend.graphs` — draw the turn flow, or only check the map.

Run on the way into the server the same way the schema upgrade is, from the image's CMD.
There it is off unless `GRAPHS_DRAW_ON_START` says otherwise, and it never fails the
chain: a diagram is not worth refusing to start for.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Draw the turn flow as a PNG.")
    parser.add_argument("--profile", help="profile to draw (default: ACTIVE_PROFILE from .env)")
    parser.add_argument("--out", type=Path, help="output directory (default: $GRAPHS_OUTPUT_DIR)")
    parser.add_argument("--parts", action="store_true", help="also draw each compiled graph on its own")
    parser.add_argument("--check", action="store_true", help="verify the map against the code, draw nothing")
    parser.add_argument(
        "--on-start",
        action="store_true",
        help="startup hook: do nothing unless GRAPHS_DRAW_ON_START is set, and never fail",
    )
    args = parser.parse_args(argv)

    # Both have to be set before `backend` is imported: `.env` is read at import and does
    # not override what is already set. Tracing off because building a graph is not a
    # conversation worth recording — the same reason conftest.py and reindex.py give.
    if args.profile:
        os.environ["ACTIVE_PROFILE"] = args.profile
    for tracing in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
        os.environ.setdefault(tracing, "false")
    from backend.env import env_bool, load_env

    load_env()

    if args.on_start and not env_bool("GRAPHS_DRAW_ON_START", False):
        return 0

    try:
        return _draw(args)
    except Exception as exc:
        if not args.on_start:
            raise
        print(f"turn-flow diagram skipped: {exc}")
        return 0


def _draw(args) -> int:
    from backend.graphs import flow

    stale = flow.stale_references()
    if stale:
        print("The flow map names code that no longer exists — update STEPS in backend/graphs/flow.py:")
        for ref in stale:
            print(f"  {ref}")
        return 0 if args.on_start else 2

    parts = flow.compile_parts()
    chart, _ = flow.build_flow(parts)
    if args.check:
        print(f"flow map OK for profile {parts.profile_name!r}: {len(chart.nodes)} steps, {len(chart.edges)} edges")
        return 0

    out = args.out or flow.output_dir()
    out.mkdir(parents=True, exist_ok=True)
    print(f"profile {parts.profile_name!r} -> {out}")
    flow._write("project_flow", chart.render(), out)
    # Always drawn beside the live one: a target nobody can put next to today's shape is
    # a target nobody checks their work against.
    flow._write("target_flow", flow.build_target().render(), out)
    if args.parts:
        for name, graph in ((f"agent_{parts.profile_name}", parts.agent), ("rag", parts.rag)):
            flow._write(name, graph.get_graph(xray=True).draw_mermaid(), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
