"""Dossier version backfill.

Run after bumping DOSSIER_VERSION:

    python -m backend.assets.backfill --dry-run     # report only, changes nothing
    python -m backend.assets.backfill               # apply

Two outcomes per stale row, and the distinction is the point of the whole mechanism:

* **migrated** — the upgrade only reshaped stored data, so it is applied in place.
  Free, and the whole corpus finishes in seconds.
* **marked_stale** — the upgrade needs fresh model output, which cannot be produced
  offline. The row is flagged for the extraction pipeline instead of being filled with
  invented data.

Never a full re-ingest: that was the failure mode this design exists to avoid.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from backend.assets.dossier import DOSSIER_VERSION
from backend.assets.store import AssetStore, BackfillReport

logger = logging.getLogger(__name__)


def run_backfill(
    store: AssetStore | None = None,
    target_version: int = DOSSIER_VERSION,
    batch_size: int = 200,
    dry_run: bool = False,
) -> BackfillReport:
    from backend.assets.store import get_asset_store

    active = store or get_asset_store()
    report = active.backfill(
        target_version=target_version,
        batch_size=batch_size,
        dry_run=dry_run,
    )
    logger.info(
        "Backfill to v%d %s: scanned=%d migrated=%d marked_stale=%d failed=%d",
        target_version,
        "(dry run)" if dry_run else "complete",
        report.scanned,
        report.migrated,
        report.marked_stale,
        report.failed,
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill asset dossiers to the current schema version.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would change, write nothing.")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument(
        "--target-version",
        type=int,
        default=DOSSIER_VERSION,
        help=f"Version to migrate up to (default: current DOSSIER_VERSION={DOSSIER_VERSION}).",
    )
    parser.add_argument("--stats", action="store_true", help="Print store statistics and exit.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from backend.env import load_env

    load_env()

    from backend.assets.store import get_asset_store

    store = get_asset_store()

    if args.stats:
        print(json.dumps(store.stats(), indent=2))
        return 0

    report = run_backfill(
        store=store,
        target_version=args.target_version,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )
    print(json.dumps(report.as_dict(), indent=2))
    # Non-zero on failures so a deploy pipeline running this as a migration step stops.
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
