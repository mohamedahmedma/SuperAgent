"""Reads and writes `document_pairs`: which documents are the same thing in two languages.

The table's own docstring explains what a row IS. This module is the only thing that
writes one, so the invariants live here rather than as a comment on each caller:

  - a filename appears on AT MOST ONE row, on the side matching its language. Two rows
    claiming the same file would make "does this have a twin" answerable two ways.
  - a row with both sides empty is deleted, never left behind. An empty row is not a
    document with no files; it is a row nobody can see or fill, because the upload form
    creates a new row rather than offering the orphans.
  - `pair_id` is opaque and generated here. Nothing derives meaning from its text.

## Why nothing about language is written onto a chunk

An earlier shape of this put `doc_language` and a `pair_id` on every Milvus chunk and
filtered on those. It was dropped, and the reason is the reason this is a table at all:
a value written onto a chunk can only be set while the chunk is being written, so
pairing an English document with an Arabic one uploaded months later would mean
silently re-indexing a document nobody touched — and until that finished, the two
halves would disagree about whether they were a pair.

Routing therefore reads this table at query time and excludes filenames
(`superseded_filenames`). Pairing a row takes effect on the next question, no re-index,
and there is exactly one place that knows which documents are twins.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import List, Optional

from backend.chat.language import ARABIC, ENGLISH
from backend.db.models import DocumentPair
from backend.infra.database import SessionLocal

logger = logging.getLogger(__name__)

#: The column a language occupies. Used to turn a language code into a field name
#: without a branch at every call site.
_COLUMN = {ARABIC: "filename_ar", ENGLISH: "filename_en"}


def column_for(language: str) -> str:
    """The `document_pairs` column holding `language`'s file, or "" for anything else."""
    return _COLUMN.get(language or "", "")


def _as_dict(row: DocumentPair) -> dict:
    return {
        "pair_id": row.pair_id,
        "title": row.title,
        "filename_ar": row.filename_ar or "",
        "filename_en": row.filename_en or "",
        # Derived rather than stored: a stored flag is one more thing that can disagree
        # with the two columns beside it.
        "paired": bool(row.filename_ar and row.filename_en),
    }


def list_pairs() -> List[dict]:
    """Every row, newest first."""
    db = SessionLocal()
    try:
        rows = db.query(DocumentPair).order_by(DocumentPair.created_at.desc()).all()
        return [_as_dict(row) for row in rows]
    finally:
        db.close()


def get_pair(pair_id: str) -> Optional[dict]:
    db = SessionLocal()
    try:
        row = db.get(DocumentPair, (pair_id or "").strip())
        return _as_dict(row) if row else None
    finally:
        db.close()


def find_by_filename(filename: str) -> Optional[dict]:
    """The row holding `filename` on either side, or None.

    How a delete finds the row to clear, and how an upload notices it is replacing a
    file that already belongs to a pair.
    """
    name = (filename or "").strip()
    if not name:
        return None
    db = SessionLocal()
    try:
        row = (
            db.query(DocumentPair)
            .filter((DocumentPair.filename_ar == name) | (DocumentPair.filename_en == name))
            .first()
        )
        return _as_dict(row) if row else None
    finally:
        db.close()


def attach(pair_id: str, language: str, filename: str, title: str = "") -> dict:
    """Put `filename` on `language`'s side of `pair_id`, creating the row if needed.

    Detaches the file from any OTHER row first, so re-uploading a file under a new pair
    moves it rather than leaving it claimed twice — the at-most-one-row invariant. A
    file replacing itself on the same side is a no-op for that purpose.
    """
    column = column_for(language)
    if not column:
        raise ValueError(f"no document_pairs column for language {language!r}")
    name = (filename or "").strip()
    if not name:
        raise ValueError("filename is required")

    pair_id = (pair_id or "").strip() or f"p_{uuid.uuid4().hex[:16]}"
    db = SessionLocal()
    try:
        # Clear this filename off every other row, and off the other side of this one:
        # the same document cannot be both the Arabic and the English version.
        for row in db.query(DocumentPair).filter(
            (DocumentPair.filename_ar == name) | (DocumentPair.filename_en == name)
        ).all():
            if row.filename_ar == name and not (row.pair_id == pair_id and column == "filename_ar"):
                row.filename_ar = ""
            if row.filename_en == name and not (row.pair_id == pair_id and column == "filename_en"):
                row.filename_en = ""
            row.updated_at = datetime.utcnow()

        row = db.get(DocumentPair, pair_id)
        if row is None:
            row = DocumentPair(pair_id=pair_id, title=title or _default_title(name))
            db.add(row)
        elif title:
            row.title = title
        setattr(row, column, name)
        row.updated_at = datetime.utcnow()
        db.flush()
        result = _as_dict(row)

        _delete_empty_rows(db)
        db.commit()
        return result
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def detach(filename: str) -> Optional[dict]:
    """Clear `filename` off whichever row holds it. Returns the row as it now stands,
    or None when the row became empty and was removed."""
    name = (filename or "").strip()
    if not name:
        return None
    db = SessionLocal()
    try:
        row = (
            db.query(DocumentPair)
            .filter((DocumentPair.filename_ar == name) | (DocumentPair.filename_en == name))
            .first()
        )
        if row is None:
            return None
        if row.filename_ar == name:
            row.filename_ar = ""
        if row.filename_en == name:
            row.filename_en = ""
        row.updated_at = datetime.utcnow()
        db.flush()

        survived = bool(row.filename_ar or row.filename_en)
        result = _as_dict(row) if survived else None
        _delete_empty_rows(db)
        db.commit()
        return result
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _delete_empty_rows(db) -> None:
    db.query(DocumentPair).filter(
        (DocumentPair.filename_ar == "") & (DocumentPair.filename_en == "")
    ).delete(synchronize_session=False)


def _default_title(filename: str) -> str:
    stem = (filename or "").rsplit(".", 1)[0].strip()
    return stem[:255] or filename[:255]


def superseded_filenames(language: str) -> List[str]:
    """Files to EXCLUDE when answering in `language`: the other-language half of a pair
    whose `language` half also exists.

    This is the whole language-routing rule, and it is expressed as an exclusion rather
    than an inclusion on purpose. Filtering retrieval down to "documents in the asked
    language" would hide every document that exists in one language only — an
    English-only policy would become unanswerable in Arabic, which is the opposite of
    what routing is for. Excluding only the redundant twin leaves everything else
    eligible, so:

        pair with both sides,  asked in ar  -> the English half is excluded
        pair with both sides,  asked in en  -> the Arabic half is excluded
        Arabic side only,      asked in ar  -> nothing excluded
        English side only,     asked in ar  -> nothing excluded, and it answers

    An unknown or unsupported language excludes nothing, so a question the detector
    could not place still searches the whole corpus.

    ONE EXCEPTION, and it is the same principle applied to pictures. A translation is
    redundant only when it is redundant in full. A text-only Arabic rendering of an
    English handbook carries none of the handbook's figures, so excluding the English
    half on an Arabic question drops every image the corpus has — the parent gets the
    uniform described in words and no picture of it, and nothing in the answer says
    why. So a twin that carries a figure this side does NOT have is kept: redundant
    prose is a smaller cost than a missing picture, and this rule exists to prefer a
    language, never to restrict the corpus to one.
    """
    other = {ARABIC: ENGLISH, ENGLISH: ARABIC}.get(language or "")
    if not other:
        return []
    keep_column = getattr(DocumentPair, column_for(language))
    drop_column = getattr(DocumentPair, column_for(other))
    db = SessionLocal()
    try:
        rows = (
            db.query(keep_column, drop_column)
            .filter(keep_column != "", drop_column != "")
            .all()
        )
    finally:
        db.close()

    candidates = [(keep, drop) for keep, drop in rows if keep and drop]
    # Deployments that never pair anything return here, having touched one table and
    # asked nothing about assets — the feature still costs a single lookup until it
    # is actually used.
    if not candidates:
        return []
    return _without_the_sides_holding_unique_pictures(candidates)


def _without_the_sides_holding_unique_pictures(candidates: List[tuple]) -> List[str]:
    """`candidates` is [(kept filename, twin filename)]; returns the twins safe to drop.

    Safe means: every picture the twin can show, the kept side can show too. Compared
    by image content HASH, because the usual case is the same image embedded in both
    halves of a translated pair — those are genuinely redundant and must still be
    excluded, or pairing would stop narrowing anything the moment a document had a
    figure in it.
    """
    hashes = _displayable_hashes({name for pair in candidates for name in pair})
    if hashes is None:
        # Nothing can be shown, so nothing can be lost — the plain rule applies. See
        # _displayable_hashes for why this is the right direction to fail in.
        return [drop for _, drop in candidates]

    superseded = []
    for keep, drop in candidates:
        unique = hashes.get(drop, set()) - hashes.get(keep, set())
        if unique:
            logger.debug(
                "keeping %s alongside %s: it carries %d figure(s) %s cannot show",
                drop, keep, len(unique), keep,
            )
            continue
        superseded.append(drop)
    return superseded


def _displayable_hashes(filenames):
    """{filename: {sha256}} for the images each file can show, or None when unknowable.

    None is not an error path, it is "this deployment has no pictures at stake", and
    the caller then applies the plain exclusion. Both routes to it are that:

      - assets are off by profile, so no document has a figure to lose;
      - the asset table cannot be read, in which case `build_asset_references` cannot
        read it either and the turn would attach no image whichever side survived.

    So keeping a redundant translation eligible here would buy no picture and cost the
    narrowing that pairing exists for. That is the opposite of the trade this whole
    function makes when the pictures ARE real, and the difference is exactly whether a
    figure can actually reach the user.
    """
    try:
        from backend.profiles import get_profile

        if not get_profile().assets.enabled:
            return None
        from backend.assets.store import get_asset_store

        return get_asset_store().displayable_hashes_by_filename(filenames)
    except Exception:
        logger.exception(
            "could not read document assets; pairing without figure awareness"
        )
        return None
