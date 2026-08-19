"""Guardian import: a child, an adult responsible for her, and how to reach him.

One row is one *pairing*, never one person. A mother, a father and a big brother for one
child are three rows sharing a student number, which is what lets a child have any number
of guardians without the file growing a column per parent — and a sheet with fixed
`mother_phone`/`father_phone` columns is exactly the shape that has nowhere to put the
grandmother who actually collects her.

The one thing that does *not* work that way is a second number for the same adult, which
is `alt_phone` on her own row. The primary phone is her identity (see
`sis.domain.guardians`), so two rows carrying two unfamiliar numbers are two different
people by definition; no name-matching can safely decide otherwise, and a service that
tried would merge two families on a coincidence of spelling.
"""
from dataclasses import dataclass

from sis.domain.guardians import RelationshipType
from sis.domain.value_objects import Phone, StudentNumber


@dataclass(frozen=True, slots=True)
class ParsedGuardianRow:
    """One guardian line the parser understood.

    Values are already value objects, as in `ParsedRosterRow`: a `Phone` here has been
    normalised to E.164 at the boundary, so the identity two rows are matched on is
    settled before any service compares them. Anything needing stored data — does this
    child exist, is this number already someone else's — is deliberately absent, because
    a parser that consulted the database would make one file parse differently on
    different days and break the preview's promise.

    `relationship_type` cannot fail a row. Unrecognised text buckets to
    `RelationshipType.OTHER` and survives verbatim in `relationship_label`, so closing
    the vocabulary costs nothing a human typed.
    """

    line_number: int
    student_number: StudentNumber
    phone: Phone
    full_name_ar: str
    full_name_en: str
    relationship_type: RelationshipType = RelationshipType.OTHER
    relationship_label: str = ""
    #: A second number for the *same* adult — a mobile plus a WhatsApp-only line. `None`
    #: when the column was blank or absent, which is the ordinary case.
    alt_phone: Phone | None = None
    is_primary_contact: bool = False
    #: Whether this guardian may read the child's academic records. Defaults to `True`
    #: here and to `False` in the database, and the asymmetry is deliberate: a registrar
    #: who uploaded a family sheet and reviewed it in preview has stated who these people
    #: are, while a link created by code that never asked must grant nothing. A sheet
    #: naming a court-restricted guardian sets the column to false explicitly.
    can_view_records: bool = True
    restriction_note: str = ""

    @property
    def has_name(self) -> bool:
        """A guardian must be nameable in at least one script to be worth a row."""
        return bool(self.full_name_ar.strip() or self.full_name_en.strip())

    @property
    def phones(self) -> tuple[Phone, ...]:
        """Her numbers, primary first — the shape `Guardian` is built from."""
        if self.alt_phone is None or str(self.alt_phone) == str(self.phone):
            return (self.phone,)
        return (self.phone, self.alt_phone)


@dataclass(frozen=True, slots=True)
class GuardianPreviewCommand:
    """Parse and validate a guardians upload, writing an ImportBatch and nothing else.

    Carries no target of any kind — no academic year, no class, no term — unlike its
    roster and grades counterparts. Every row already names its own child and its own
    guardian, so there is nothing for an upload-wide parameter to narrow, and inventing
    one would only create a way for a file to be committed against the wrong scope.

    `content` is bytes already in memory. No path, because the API layer must never write
    a file of children's and parents' names to disk for a parser to read back.
    """

    filename: str
    content: bytes
    actor: str


@dataclass(frozen=True, slots=True)
class GuardianCommitCommand:
    """Apply a previously previewed batch.

    Carries the batch id rather than the file, for the reason `RosterCommitCommand` does:
    commit replays what the registrar actually saw.
    """

    batch_id: str
    actor: str
