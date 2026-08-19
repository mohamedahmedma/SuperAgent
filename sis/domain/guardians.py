"""The adults responsible for a child, and which of them may read her records.

Two entities on purpose, for the same reason `Student` and `ClassEnrolment` are two
(see `sis.domain.people`). `Guardian` is a person — a name and the numbers she can be
reached on. `StudentGuardian` is a *relationship*, and every fact that only makes sense
about a pairing lives there: a man is `FATHER` to one child on this roll and `GUARDIAN`
to another he has custody of but is not related to, so `relationship_type` cannot be a
column on the person. Neither can permission: the same grandmother may read one
grandchild's report and be barred by a court order from another's.

**The phone is the identity.** A school runs no parent-numbering system for an importer
to key on, so the number is what recognises the same mother across three of her
children's rows and across next term's re-upload. That is why `Phone` normalises as
aggressively as it does — if two spellings of one number fail to collide, she becomes
two people, each holding half her children.

`can_view_records` is the field to be careful with. It is separate from the link
existing because a guardian who is legitimately on file — an emergency contact, a parent
who attends events — can be legally barred from seeing academic records. Modelling that
as "delete the row" loses the contact; modelling it as a flag keeps both facts and gives
`restriction_note` somewhere to name the court order. It defaults to `False` here, so a
link built by code that never considered the question grants nothing; the importer sets
it deliberately, because a registrar reviewing a preview has considered the question.

Framework-free like the rest of `sis/domain/`: stdlib and sibling value objects only,
no clock, no database.
"""
from dataclasses import dataclass
from enum import StrEnum

from sis.domain.errors import ValidationError
from sis.domain.value_objects import Phone, StudentNumber


class RelationshipType(StrEnum):
    """Who a guardian is to a child. A closed vocabulary, with the words kept beside it.

    Closed because this is the half that gets *counted*: "how many mothers are on file",
    "show me every child whose only contact is a sibling". Free text cannot be counted —
    "mother", "Mother" and "الأم" tally as three — and it cannot be translated, since a
    string typed into a spreadsheet appears in no translation table the UI owns.

    The cost of a closed list is that it cannot describe every family, which is what
    `StudentGuardian.relationship_label` is for: "big brother" buckets to `SIBLING` here
    and survives verbatim there. Degrading an unrecognised word to `OTHER` is therefore
    lossless, which is why it is safe — unlike guessing at an unfamiliar *grade*, where
    the guess would replace a fact nobody could recover.
    """

    MOTHER = "mother"
    FATHER = "father"
    GUARDIAN = "guardian"
    SIBLING = "sibling"
    GRANDPARENT = "grandparent"
    OTHER = "other"


def _coerce(entity: object, field: str, kind: type) -> None:
    """Replace a raw value with its value object, in place, during `__post_init__`.

    The same contract `sis.domain.people` gives its entities, and it exists for the same
    reason: annotating `phone: Phone` documents an intention that nothing enforces, so a
    caller passing the string `"+201001234567"` builds a `Guardian` that compares unequal
    to every `Phone("+201001234567")` in the service.
    """
    raw = getattr(entity, field)
    if not isinstance(raw, kind):
        object.__setattr__(entity, field, kind(raw))


def _clean_name(raw: object, *, field: str) -> str:
    """Names are labels, so anything is allowed in them except a lie about emptiness."""
    if raw is None:
        return ""
    if not isinstance(raw, str):
        raise ValidationError(f"{field} must be text", field=field)
    return raw.strip()


@dataclass(frozen=True, slots=True)
class Guardian:
    """One adult: the numbers that reach her, and her names.

    `phones` is a tuple rather than a single value because one parent legitimately has
    two — a mobile and a number that only takes WhatsApp is the ordinary case, and a
    school that can store one of them loses the other or duplicates the person. The
    **first is primary**: it is the number the rest of the system treats as her handle,
    and the one a `StudentGuardian` names.

    Deliberately bare of anything about a child. Which children she is responsible for,
    what she is to each of them, and which of their records she may read are all
    `StudentGuardian` — see this module's docstring for why none of that can be a column
    here.
    """

    phones: tuple[Phone, ...]
    full_name_ar: str = ""
    full_name_en: str = ""
    preferred_language: str = "ar"
    is_active: bool = True

    def __post_init__(self) -> None:
        phones = tuple(
            phone if isinstance(phone, Phone) else Phone(phone)
            for phone in self.phones
        )
        if not phones:
            raise ValidationError(
                "a guardian needs at least one phone number", field="phone"
            )
        # Checked rather than silently de-duplicated: the same number twice on one
        # guardian is a registrar filling both columns with one value, and quietly
        # collapsing it would leave her looking as though she had supplied a second way
        # to be reached when she had not.
        seen = {str(phone) for phone in phones}
        if len(seen) != len(phones):
            raise ValidationError(
                "the same phone number is listed twice for one guardian", field="phone"
            )
        object.__setattr__(self, "phones", phones)

        name_ar = _clean_name(self.full_name_ar, field="full_name_ar")
        name_en = _clean_name(self.full_name_en, field="full_name_en")
        # Both names are wanted but only one is required, for the same reason `Student`
        # accepts one: a sheet exported from an Arabic-first system has no English
        # column, and refusing it makes the registrar type transliterations by hand.
        if not name_ar and not name_en:
            raise ValidationError(
                "a guardian needs a name in Arabic or English", field="full_name_ar"
            )
        object.__setattr__(self, "full_name_ar", name_ar)
        object.__setattr__(self, "full_name_en", name_en)

    @property
    def primary_phone(self) -> Phone:
        """The number that identifies her. Always present; `__post_init__` guarantees it."""
        return self.phones[0]

    @property
    def identity(self) -> str:
        """The natural key a repository upserts on."""
        return str(self.primary_phone)

    def reachable_on(self, phone: Phone | str) -> bool:
        """Is this one of her numbers? Any of them, not only the primary one."""
        return str(phone) in {str(known) for known in self.phones}


@dataclass(frozen=True, slots=True)
class StudentGuardian:
    """One child's link to one guardian: what they are to each other, and what she may see.

    The many-to-many that the whole feature exists for. A child ordinarily has more than
    one guardian and a guardian often has more than one child, so this cannot be a column
    on either side — a `students.guardian_phone` column holds the father and loses the
    mother, and answers "who do I call" with whichever of them was written last.

    Identity is `(student_number, guardian_phone)`: one real person has exactly one
    relationship to one child, so a second row for the same pair updates this one rather
    than creating a rival.

    The guardian is named by phone rather than by a database id for the same reason
    `ClassEnrolment` names a class by `class_code`: the domain speaks in the natural keys
    a spreadsheet actually contains, and resolving those to foreign keys is the
    repository's job, on write.
    """

    student_number: StudentNumber | str
    guardian_phone: Phone | str
    relationship_type: RelationshipType = RelationshipType.OTHER
    # The registrar's own words — "big brother", "الجدة لأم". Kept verbatim beside the
    # bucketed type so closing the vocabulary costs no information.
    relationship_label: str = ""
    is_primary_contact: bool = False
    # Default deny. See the module docstring: a link created by code that never asked the
    # question must not grant a reading no human authorised.
    can_view_records: bool = False
    # Why access was restricted — a court order reference, a school decision, a date.
    # Never rendered to a parent; it exists for the registrar and for an audit.
    restriction_note: str = ""

    def __post_init__(self) -> None:
        _coerce(self, "student_number", StudentNumber)
        _coerce(self, "guardian_phone", Phone)
        if not isinstance(self.relationship_type, RelationshipType):
            try:
                object.__setattr__(
                    self, "relationship_type", RelationshipType(self.relationship_type)
                )
            except ValueError as error:
                # Raised, not degraded. Free text from a *spreadsheet* degrades to OTHER
                # in the parser, where the original survives in the label; by the time a
                # value reaches this constructor it came from code or from a stored row,
                # and an unknown member there is a bug rather than a typing registrar.
                raise ValidationError(
                    f"unknown relationship type {self.relationship_type!r}",
                    field="relationship_type",
                ) from error
        object.__setattr__(
            self, "relationship_label", _clean_name(self.relationship_label, field="relationship_label")
        )
        object.__setattr__(
            self, "restriction_note", _clean_name(self.restriction_note, field="restriction_note")
        )

    @property
    def identity(self) -> tuple[str, str]:
        """`(student_number, guardian_phone)` — what a repository upserts on."""
        return (str(self.student_number), str(self.guardian_phone))

    def differs_from(self, other: "StudentGuardian") -> bool:
        """Would storing `self` over `other` change anything a human stated?

        Used to tell `UPDATED` from `UNCHANGED` in an import preview. Identity fields are
        excluded because two links that differ there are not versions of one fact; they
        are two facts.
        """
        return (
            self.relationship_type is not other.relationship_type
            or self.relationship_label != other.relationship_label
            or self.is_primary_contact != other.is_primary_contact
            or self.can_view_records != other.can_view_records
            or self.restriction_note != other.restriction_note
        )
