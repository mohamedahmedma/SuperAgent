"""Guardians, their numbers, and their links to children, over SQLAlchemy.

Implements `GuardianRepository` and `StudentGuardianRepository` from
`sis.application.ports.repositories`. Nothing here returns an ORM row: every method hands
back a `sis.domain` object, because the moment a service holds a `models.Guardian` it
holds a live session and the unit test meant to assert a rule needs a database again.

**The vocabulary gap this module closes.** The ports speak `Phone` and `StudentNumber`;
the tables join on surrogate integers, and a guardian's phone is not even on its own
table. Translating in both directions is this layer's entire job, and it is why every
bulk method starts by resolving numbers to ids in one query rather than per row.

**A guardian is found by *any* of her numbers.** `get`/`get_many` resolve through
`guardian_phones`, so a second upload quoting the WhatsApp line she gave last year finds
the same woman rather than creating a rival record holding half her children. Every
returned `Guardian` carries her full set of numbers, primary first.

**Statement counts are part of the contract.** A guardians upload carries hundreds of
rows, and the per-row shape turns one upload into a thousand round trips. Every bulk
method states its count above its body and none grows with the number of rows.

**Nothing here commits.** The transaction belongs to the unit of work.
"""
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import replace
from typing import Any
from uuid import uuid4

from sqlalchemy import delete, insert, select, update
from sqlalchemy.orm import Session

from sis.application.ports.repositories import StudentGuardianKey
from sis.domain.errors import UnknownReference
from sis.domain.guardians import Guardian, RelationshipType, StudentGuardian
from sis.domain.value_objects import Phone, StudentNumber
from sis.infrastructure.db import models

# See `people_repository._IN_CHUNK`: SQLite binds one parameter per `IN` element and
# refuses past SQLITE_MAX_VARIABLE_NUMBER, so "one query" here means one per chunk.
_IN_CHUNK = 400


def _chunked(values: Sequence[Any]) -> Iterator[Sequence[Any]]:
    """Split an `IN` list into batches SQLite will accept."""
    for start in range(0, len(values), _IN_CHUNK):
        yield values[start : start + _IN_CHUNK]


def _public_id() -> str:
    """An opaque handle for a guardian, stable for the life of the row.

    Random rather than sequential because it is what a parent-facing URL would carry one
    day, and a sequential id in a path invites walking it to enumerate every family in the
    school.
    """
    return uuid4().hex


def _to_guardian(row: models.Guardian, phones: Sequence[str]) -> Guardian:
    """One ORM row plus her numbers -> one domain `Guardian`."""
    return Guardian(
        phones=tuple(Phone(phone) for phone in phones),
        full_name_ar=row.full_name_ar,
        full_name_en=row.full_name_en,
        preferred_language=row.preferred_language,
        is_active=row.is_active,
    )


def _relationship(raw: str) -> RelationshipType:
    """A stored relationship, degraded to `OTHER` rather than raising.

    A value this build has no member for means the row was written by a later version. A
    contact list that refuses to render because one link says `step_parent` is worse than
    one that shows that link as `other` — the phone number, which is the part somebody
    needs at 8am, is right either way.
    """
    try:
        return RelationshipType(raw)
    except ValueError:
        return RelationshipType.OTHER


class SqlAlchemyGuardianRepository:
    """`GuardianRepository` over a session. People and their numbers; links are the sibling."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, phone: Phone) -> Guardian | None:
        found = self.get_many([phone])
        return found.get(str(phone))

    def get_many(self, phones: Collection[Phone]) -> Mapping[str, Guardian]:
        """Two statements: the queried numbers, then every number those guardians hold.

        Keyed by the number that was *asked about*, not by each guardian's primary one, so
        a caller holding the alternate line it searched on finds the answer it asked for.
        The same woman therefore appears under two keys when both her numbers were queried,
        which is correct: both questions have the same answer.
        """
        wanted = sorted({str(phone) for phone in phones})
        if not wanted:
            return {}

        # Which guardian each queried number reaches.
        owner_of: dict[str, int] = {}
        for chunk in _chunked(wanted):
            for row in self._session.execute(
                select(models.GuardianPhone.phone, models.GuardianPhone.guardian_id).where(
                    models.GuardianPhone.phone.in_(chunk)
                )
            ):
                owner_of[row.phone] = row.guardian_id
        if not owner_of:
            return {}

        guardian_ids = sorted(set(owner_of.values()))
        guardians = self._load(guardian_ids)
        return {
            phone: guardians[guardian_id]
            for phone, guardian_id in owner_of.items()
            if guardian_id in guardians
        }

    def _load(self, guardian_ids: Sequence[int]) -> dict[int, Guardian]:
        """Guardians by id, each carrying her full set of numbers, primary first."""
        rows: dict[int, models.Guardian] = {}
        for chunk in _chunked(guardian_ids):
            for row in self._session.scalars(
                select(models.Guardian).where(models.Guardian.id.in_(chunk))
            ):
                rows[row.id] = row

        numbers: dict[int, list[str]] = {}
        for chunk in _chunked(guardian_ids):
            for record in self._session.execute(
                select(
                    models.GuardianPhone.guardian_id,
                    models.GuardianPhone.phone,
                    models.GuardianPhone.is_primary,
                )
                .where(models.GuardianPhone.guardian_id.in_(chunk))
                # Primary first, then a stable order so two reads of one guardian never
                # disagree about which of her secondary numbers comes second.
                .order_by(
                    models.GuardianPhone.is_primary.desc(), models.GuardianPhone.phone
                )
            ):
                numbers.setdefault(record.guardian_id, []).append(record.phone)

        return {
            guardian_id: _to_guardian(row, numbers.get(guardian_id, ()))
            for guardian_id, row in rows.items()
            # A guardian with no numbers cannot be constructed and cannot be reached; she
            # is skipped rather than raising, so one orphaned row does not break a read of
            # four hundred families.
            if numbers.get(guardian_id)
        }

    def ids_for(self, phones: Collection[Phone]) -> Mapping[str, int]:
        """Surrogate ids by number — how the link repository resolves what the domain names."""
        wanted = sorted({str(phone) for phone in phones})
        found: dict[str, int] = {}
        for chunk in _chunked(wanted):
            for row in self._session.execute(
                select(models.GuardianPhone.phone, models.GuardianPhone.guardian_id).where(
                    models.GuardianPhone.phone.in_(chunk)
                )
            ):
                found[row.phone] = row.guardian_id
        return found

    def upsert_many(self, guardians: Sequence[Guardian]) -> Mapping[str, bool]:
        """Insert or update by primary phone; `True` marks the ones this call created.

        A fixed handful of statements for an upload of any size: one `SELECT` of the
        numbers already on file, one to load the matching rows, then at most one
        executemany `INSERT` and one `UPDATE` per table. A re-run that changes nothing
        issues only the reads.

        Rows whose names already match are left out of the `UPDATE` payload, so
        re-importing a guardians sheet does not stamp `updated_at` on nine hundred parents
        and destroy the only column that can answer "whose details changed in this import".

        Numbers **accumulate**. A sheet that mentions only her mobile adds nothing and
        removes nothing; it never drops the second line an earlier upload recorded, because
        an upload is a statement about what the school now knows, not a complete
        description of a person.
        """
        if not guardians:
            return {}

        # Last write wins within one file for the *names*, matching
        # `SqlAlchemyStudentRepository` -- but numbers are unioned rather than replaced.
        # A mother listed on two of her children's rows supplies her second phone on only
        # one of them, and a plain last-wins collapse would silently keep whichever row
        # the file happened to end with, discarding a number she gave the school.
        incoming: dict[str, Guardian] = {}
        for guardian in guardians:
            previous = incoming.get(guardian.identity)
            if previous is None:
                incoming[guardian.identity] = guardian
                continue
            phones = list(previous.phones)
            known = {str(phone) for phone in phones}
            for phone in guardian.phones:
                if str(phone) not in known:
                    phones.append(phone)
                    known.add(str(phone))
            incoming[guardian.identity] = replace(guardian, phones=tuple(phones))

        # Resolve every number the batch mentions -- not just the primaries -- because a
        # row whose primary is new may still name an alternate that already identifies an
        # existing guardian, and inserting a second row for her would trip the unique
        # index and reject the whole upload.
        mentioned = {
            str(phone) for guardian in incoming.values() for phone in guardian.phones
        }
        owner_of = self.ids_for([Phone(phone) for phone in sorted(mentioned)])
        existing = self._rows_by_id(sorted(set(owner_of.values())))

        to_insert: list[dict[str, Any]] = []
        to_update: list[dict[str, Any]] = []
        new_phones: list[dict[str, Any]] = []
        created: dict[str, bool] = {}

        for identity, guardian in incoming.items():
            guardian_id = next(
                (
                    owner_of[str(phone)]
                    for phone in guardian.phones
                    if str(phone) in owner_of
                ),
                None,
            )
            if guardian_id is None:
                created[identity] = True
                to_insert.append(
                    {
                        "public_id": _public_id(),
                        "full_name_ar": guardian.full_name_ar,
                        "full_name_en": guardian.full_name_en,
                        "preferred_language": guardian.preferred_language,
                        "is_active": guardian.is_active,
                    }
                )
                continue

            created[identity] = False
            row = existing.get(guardian_id)
            if row is not None and (
                row.full_name_ar != guardian.full_name_ar
                or row.full_name_en != guardian.full_name_en
                or row.preferred_language != guardian.preferred_language
                or row.is_active != guardian.is_active
            ):
                to_update.append(
                    {
                        "id": guardian_id,
                        "full_name_ar": guardian.full_name_ar,
                        "full_name_en": guardian.full_name_en,
                        "preferred_language": guardian.preferred_language,
                        "is_active": guardian.is_active,
                    }
                )
            for phone in guardian.phones:
                if str(phone) not in owner_of:
                    new_phones.append(
                        {
                            "guardian_id": guardian_id,
                            "phone": str(phone),
                            # Never promoted to primary: an existing guardian already has
                            # one, and moving it would change the identity every stored
                            # link names.
                            "is_primary": False,
                        }
                    )

        if to_insert:
            # `returning` rather than a second SELECT: the ids are needed immediately to
            # attach each new guardian's numbers, and re-querying by name would match the
            # wrong row whenever two new parents share one.
            inserted = self._session.scalars(
                insert(models.Guardian).returning(models.Guardian.id), to_insert
            ).all()
            for guardian_id, identity in zip(
                inserted, [k for k, v in created.items() if v], strict=True
            ):
                for position, phone in enumerate(incoming[identity].phones):
                    new_phones.append(
                        {
                            "guardian_id": guardian_id,
                            "phone": str(phone),
                            "is_primary": position == 0,
                        }
                    )
        if to_update:
            self._session.execute(update(models.Guardian), to_update)
        if new_phones:
            self._session.execute(insert(models.GuardianPhone), new_phones)
        return created

    def _rows_by_id(self, guardian_ids: Sequence[int]) -> dict[int, models.Guardian]:
        rows: dict[int, models.Guardian] = {}
        for chunk in _chunked(guardian_ids):
            for row in self._session.scalars(
                select(models.Guardian).where(models.Guardian.id.in_(chunk))
            ):
                rows[row.id] = row
        return rows


class SqlAlchemyStudentGuardianRepository:
    """`StudentGuardianRepository` over a session — the many-to-many and its permission."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def _select(self):
        """Links with the natural keys a domain `StudentGuardian` needs, in one statement.

        The join to `guardian_phones` is filtered to the primary number, because the
        domain names a guardian by her identity and an unfiltered join would return one
        row per number she holds — quietly duplicating a mother with two lines into two
        entries in her child's contact list.
        """
        return (
            select(
                models.Student.student_number,
                models.GuardianPhone.phone,
                models.StudentGuardian.relationship_type,
                models.StudentGuardian.relationship_label,
                models.StudentGuardian.is_primary_contact,
                models.StudentGuardian.can_view_records,
                models.StudentGuardian.restriction_note,
            )
            .select_from(models.StudentGuardian)
            .join(models.Student, models.Student.id == models.StudentGuardian.student_id)
            .join(
                models.GuardianPhone,
                (models.GuardianPhone.guardian_id == models.StudentGuardian.guardian_id)
                & models.GuardianPhone.is_primary.is_(True),
            )
        )

    @staticmethod
    def _to_link(row: Any) -> StudentGuardian:
        return StudentGuardian(
            student_number=row.student_number,
            guardian_phone=row.phone,
            relationship_type=_relationship(row.relationship_type),
            relationship_label=row.relationship_label,
            is_primary_contact=row.is_primary_contact,
            can_view_records=row.can_view_records,
            restriction_note=row.restriction_note,
        )

    def list_for_student(
        self, student_number: StudentNumber
    ) -> Sequence[StudentGuardian]:
        found = self.list_for_students([student_number])
        return found.get(str(student_number), ())

    def list_for_students(
        self, student_numbers: Collection[StudentNumber]
    ) -> Mapping[str, Sequence[StudentGuardian]]:
        """One query per chunk. Children with no guardians are absent, not mapped to `()`."""
        wanted = sorted({str(number) for number in student_numbers})
        if not wanted:
            return {}
        found: dict[str, list[StudentGuardian]] = {}
        for chunk in _chunked(wanted):
            for row in self._session.execute(
                self._select()
                .where(models.Student.student_number.in_(chunk))
                # Primary contact first — the number the office rings — then a stable
                # order so a contact list does not reshuffle between two reads.
                .order_by(
                    models.StudentGuardian.is_primary_contact.desc(),
                    models.GuardianPhone.phone,
                )
            ):
                found.setdefault(row.student_number, []).append(self._to_link(row))
        return {number: tuple(links) for number, links in found.items()}

    def list_students_for_guardian(
        self, phone: Phone, *, viewable_only: bool = True
    ) -> Sequence[StudentGuardian]:
        """Which children this number may ask about — the parent-facing query.

        Resolved through *any* of her numbers, so a parent who verifies the second line
        she gave the school sees the same children as one who verifies the first.
        """
        owner = self._session.scalars(
            select(models.GuardianPhone.guardian_id).where(
                models.GuardianPhone.phone == str(phone)
            )
        ).first()
        if owner is None:
            return ()

        stmt = self._select().where(models.StudentGuardian.guardian_id == owner)
        if viewable_only:
            # Applied in SQL rather than filtered afterwards: a restricted child must
            # never be loaded into a process that is answering a parent, so there is no
            # object for a later bug to leak.
            stmt = stmt.where(models.StudentGuardian.can_view_records.is_(True))
        return tuple(
            self._to_link(row)
            for row in self._session.execute(
                stmt.order_by(models.Student.student_number)
            )
        )

    def upsert_many(
        self, links: Sequence[StudentGuardian]
    ) -> Mapping[StudentGuardianKey, bool]:
        """Insert or update links in bulk; `True` marks the ones this call created.

        Resolves student numbers and guardian phones to ids in two queries, then at most
        one executemany `INSERT` and one `UPDATE`. Unchanged links are left out of the
        payload for the reason `upsert_many` on students leaves them out: `updated_at` is
        the only column that can answer "what did this import actually change".

        Raises `UnknownReference` when a number resolves to nothing. Defensive rather than
        expected — the import service has already rejected unknown students and writes
        guardians before links — but a link written against a guessed id would attach a
        real child to the wrong adult, so it fails loudly instead.
        """
        if not links:
            return {}

        incoming: dict[StudentGuardianKey, StudentGuardian] = {
            link.identity: link for link in links
        }

        student_ids = self._student_ids({key[0] for key in incoming})
        guardian_ids = SqlAlchemyGuardianRepository(self._session).ids_for(
            [Phone(key[1]) for key in incoming]
        )

        resolved: dict[StudentGuardianKey, tuple[int, int]] = {}
        for key in incoming:
            number, phone = key
            if number not in student_ids:
                raise UnknownReference(f"no student {number}", field="student_number")
            if phone not in guardian_ids:
                raise UnknownReference(f"no guardian on {phone}", field="phone")
            resolved[key] = (student_ids[number], guardian_ids[phone])

        existing = self._existing_rows(set(resolved.values()))

        to_insert: list[dict[str, Any]] = []
        to_update: list[dict[str, Any]] = []
        created: dict[StudentGuardianKey, bool] = {}
        for key, link in incoming.items():
            pair = resolved[key]
            fields = {
                "relationship_type": link.relationship_type.value,
                "relationship_label": link.relationship_label,
                "is_primary_contact": link.is_primary_contact,
                "can_view_records": link.can_view_records,
                "restriction_note": link.restriction_note,
            }
            row = existing.get(pair)
            if row is None:
                created[key] = True
                to_insert.append(
                    {"student_id": pair[0], "guardian_id": pair[1], **fields}
                )
                continue
            created[key] = False
            if any(getattr(row, name) != value for name, value in fields.items()):
                to_update.append({"id": row.id, **fields})

        if to_insert:
            self._session.execute(insert(models.StudentGuardian), to_insert)
        if to_update:
            self._session.execute(update(models.StudentGuardian), to_update)
        return created

    def unlink(self, student_number: StudentNumber, phone: Phone) -> bool:
        """Remove one link. `False` when there was nothing to remove."""
        student_ids = self._student_ids({str(student_number)})
        guardian_ids = SqlAlchemyGuardianRepository(self._session).ids_for([phone])
        if str(student_number) not in student_ids or str(phone) not in guardian_ids:
            return False
        result = self._session.execute(
            delete(models.StudentGuardian).where(
                models.StudentGuardian.student_id == student_ids[str(student_number)],
                models.StudentGuardian.guardian_id == guardian_ids[str(phone)],
            )
        )
        return bool(result.rowcount)

    def _student_ids(self, numbers: Collection[str]) -> dict[str, int]:
        wanted = sorted(set(numbers))
        found: dict[str, int] = {}
        for chunk in _chunked(wanted):
            for row in self._session.execute(
                select(models.Student.student_number, models.Student.id).where(
                    models.Student.student_number.in_(chunk)
                )
            ):
                found[row.student_number] = row.id
        return found

    def _existing_rows(
        self, pairs: Collection[tuple[int, int]]
    ) -> dict[tuple[int, int], models.StudentGuardian]:
        """Existing links for the pairs this call touches.

        Filtered by student id alone rather than by the composite pair: a tuple `IN` is
        not portable across SQLite and PostgreSQL, and the extra rows are one family's
        worth of guardians, discarded in memory.
        """
        student_ids = sorted({student_id for student_id, _ in pairs})
        found: dict[tuple[int, int], models.StudentGuardian] = {}
        for chunk in _chunked(student_ids):
            for row in self._session.scalars(
                select(models.StudentGuardian).where(
                    models.StudentGuardian.student_id.in_(chunk)
                )
            ):
                found[(row.student_id, row.guardian_id)] = row
        return found
