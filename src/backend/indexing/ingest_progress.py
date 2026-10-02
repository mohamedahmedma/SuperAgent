"""Where a long ingest step reports how far it has got.

Ingestion runs on a background thread with no request waiting on it, so its progress
has nowhere to go unless the caller hands in somewhere to put it. Uploads pass a sink
that writes to the job tracker the admin UI polls; reindex scripts and tests pass
nothing and pay one attribute lookup per image.

Figure extraction is the only stage with anything to say so far, and it is the one that
needed saying: it is minutes of model calls inside a step that otherwise reports 5% once
and then nothing until the whole document is parsed.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover - import cycle at runtime, types only
    from backend.assets.pipeline import FigureReport

logger = logging.getLogger(__name__)


class IngestProgress:
    """No-op by default: a subclass overrides only the stages it cares about."""

    def figures_progress(self, done: int, total: int) -> None:
        """`done` of `total` images extracted. Called as each one lands."""

    def figures_finished(self, report: "FigureReport") -> None:
        """What extraction produced, once, when the stage is over."""


def report_progress(sink: Optional[IngestProgress], done: int, total: int) -> None:
    """Tell `sink` how far extraction has got, and never let it stop the ingest.

    A progress sink writes to a database the document being ingested does not depend
    on. Letting a failure there abort an upload would trade the whole document for the
    progress bar describing it.
    """
    if sink is None:
        return
    try:
        sink.figures_progress(done, total)
    except Exception:
        logger.exception("Ingest progress sink failed at %d/%d", done, total)


def report_finished(sink: Optional[IngestProgress], report: "FigureReport") -> None:
    """Hand `sink` the outcome, under the same rule as `report_progress`."""
    if sink is None:
        return
    try:
        sink.figures_finished(report)
    except Exception:
        logger.exception("Ingest progress sink failed on the final report")
