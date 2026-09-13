"""Bilingual document pairs: which documents are the same thing in two languages.

The table's own docstring explains what a row IS. `DocumentPairService` is the only code
that writes one, so the invariants live here rather than as a comment at each caller:

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
from dataclasses import replace

from backend.application.ports.repositories import DocumentPairRecord
from backend.application.ports.unit_of_work import UnitOfWorkFactory
from backend.chat.language import ARABIC, ENGLISH
from backend.infra.unit_of_work import SqlAlchemyUnitOfWork

logger = logging.getLogger(__name__)

#: The record field holding each language's file.
_SIDES = {ARABIC: "filename_ar", ENGLISH: "filename_en"}
_TWIN = {ARABIC: ENGLISH, ENGLISH: ARABIC}


class DocumentPairService:
    """The bilingual entries of the knowledge base, and the routing rule built on them."""

    def __init__(self, unit_of_work: UnitOfWorkFactory = SqlAlchemyUnitOfWork) -> None:
        self._unit_of_work = unit_of_work

    def list_pairs(self) -> list[DocumentPairRecord]:
        """Every entry, newest first."""
        with self._unit_of_work() as uow:
            return list(uow.document_pairs.list_all())

    def get_pair(self, pair_id: str) -> DocumentPairRecord | None:
        with self._unit_of_work() as uow:
            return uow.document_pairs.get((pair_id or "").strip())

    def find_by_filename(self, filename: str) -> DocumentPairRecord | None:
        """The entry holding `filename` on either side, or None.

        How a delete finds the entry to clear, and how an upload notices it is replacing
        a file that already belongs to a pair.
        """
        name = (filename or "").strip()
        if not name:
            return None
        with self._unit_of_work() as uow:
            holders = uow.document_pairs.holding(name)
        return holders[0] if holders else None

    def attach(self, pair_id: str, language: str, filename: str, title: str = "") -> DocumentPairRecord:
        """Put `filename` on `language`'s side of `pair_id`, creating the entry if needed.

        Detaches the file from any OTHER entry first, so re-uploading a file under a new
        pair moves it rather than leaving it claimed twice — the at-most-one-row
        invariant. A file replacing itself on the same side is a no-op for that purpose.
        """
        side = self._side(language)
        name = (filename or "").strip()
        if not name:
            raise ValueError("filename is required")
        pair_id = (pair_id or "").strip() or f"p_{uuid.uuid4().hex[:16]}"

        with self._unit_of_work() as uow:
            pairs = uow.document_pairs
            # Clear this filename off every other entry, and off the other side of this
            # one: the same document cannot be both the Arabic and the English version.
            for holder in pairs.holding(name):
                cleared = replace(
                    holder,
                    **{
                        field: ""
                        for field in _SIDES.values()
                        if getattr(holder, field) == name
                        and not (holder.pair_id == pair_id and field == side)
                    },
                )
                if cleared != holder:
                    pairs.save(cleared)

            current = pairs.get(pair_id) or DocumentPairRecord(
                pair_id=pair_id, title=title or _default_title(name)
            )
            current = replace(current, **{side: name}, **({"title": title} if title else {}))
            pairs.save(current)
            pairs.delete_empty()
            uow.commit()
        return current

    def detach(self, filename: str) -> DocumentPairRecord | None:
        """Clear `filename` off whichever entry holds it. Returns the entry as it now
        stands, or None when it became empty and was removed."""
        name = (filename or "").strip()
        if not name:
            return None
        with self._unit_of_work() as uow:
            pairs = uow.document_pairs
            holders = pairs.holding(name)
            if not holders:
                return None
            holder = holders[0]
            cleared = replace(
                holder, **{field: "" for field in _SIDES.values() if getattr(holder, field) == name}
            )
            pairs.save(cleared)
            pairs.delete_empty()
            uow.commit()
        return None if cleared.empty else cleared

    def superseded_filenames(self, language: str) -> list[str]:
        """Files to EXCLUDE when answering in `language`: the other-language half of a
        pair whose `language` half also exists.

        This is the whole language-routing rule, and it is expressed as an exclusion
        rather than an inclusion on purpose. Filtering retrieval down to "documents in
        the asked language" would hide every document that exists in one language only —
        an English-only policy would become unanswerable in Arabic, which is the opposite
        of what routing is for. Excluding only the redundant twin leaves everything else
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
        other = _TWIN.get(language or "")
        if not other:
            return []
        with self._unit_of_work() as uow:
            rows = uow.document_pairs.paired_filenames()

        keeps_arabic = language == ARABIC
        candidates = [(ar, en) if keeps_arabic else (en, ar) for ar, en in rows]
        # Deployments that never pair anything return here, having touched one table and
        # asked nothing about assets — the feature still costs a single lookup until it
        # is actually used.
        if not candidates:
            return []
        return self._without_the_sides_holding_unique_pictures(candidates)

    @staticmethod
    def _side(language: str) -> str:
        side = _SIDES.get(language or "")
        if not side:
            raise ValueError(f"no document pair side for language {language!r}")
        return side

    def _without_the_sides_holding_unique_pictures(self, candidates: list[tuple]) -> list[str]:
        """`candidates` is [(kept filename, twin filename)]; returns the twins safe to drop.

        Safe means: every picture the twin can show, the kept side can show too. Compared
        by image content HASH, because the usual case is the same image embedded in both
        halves of a translated pair — those are genuinely redundant and must still be
        excluded, or pairing would stop narrowing anything the moment a document had a
        figure in it.
        """
        hashes = self._displayable_hashes({name for pair in candidates for name in pair})
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

    @staticmethod
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
            logger.exception("could not read document assets; pairing without figure awareness")
            return None


def _default_title(filename: str) -> str:
    stem = (filename or "").rsplit(".", 1)[0].strip()
    return stem[:255] or filename[:255]

