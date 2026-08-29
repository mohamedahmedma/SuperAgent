"""What a class is called, derived from what a class *is*.

A school writes `1/2 ب` on a door and everyone in the building knows it means the second
classroom of the first primary grade. A database that stores only that string knows
nothing: it cannot list the primary rungs, cannot answer "every first-grade class", and
sorts `10/1 ب` before `2/1 ب`. So the structured facts are what is stored —

    educational system   arabic | language      (which section of the school)
    stage                garden | primary | preparatory | secondary
    grade number         1, 2, 3 …              (the rung within that stage)
    section number       1, 2, 3 …              (which room on that rung)
    label                free text              ("A", "Simba Class")

— and the string on the door is *generated* from them, in whichever language is being
read. That is the rule this module implements, in both directions: `render_*` builds the
name, `parse_arabic_class_code` reads an existing one back into the facts, so a school
already holding a list of codes can be imported without retyping.

**The order of the two numbers in an Arabic code is grade first, then classroom.**
`1/2 ب` is grade one, room two. This is stated because it is the one thing about the
format that cannot be inferred from an example where both numbers are `1`, and reading it
the other way round silently swaps a school's entire class list.

**Language sections are not a translation of Arabic sections.** They run a different
ladder — a continuous Grade 1..14 count rather than a per-stage one — and a KG room is
named "Simba Class" rather than numbered at all. So the label a school types always wins
over anything generated: `render_class_label` returns the stored label when there is one.
Generation exists to fill the blank, never to overrule.

Nothing here reads a database, a clock or the environment.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from sis.domain.errors import ValidationError
from sis.domain.structure import Stage


class EducationalSystemKind(StrEnum):
    """Which naming ladder a section of the school runs on.

    Two kinds, and a school may run both at once — that is the case this whole dimension
    exists for. The *code* and *names* of a section are a school's own (`AR`, `LANG`,
    "National Section", "قسم اللغات"); the kind is what tells this module which rules
    apply, so a school can call its language section anything and still get `Grade 3 A`.

    `UNSPECIFIED` is what every rung was before this dimension existed, and what a rung
    belongs to in a school that does not divide itself. It renders from the stored label
    and generates nothing, which is exactly the previous behaviour.
    """

    UNSPECIFIED = "unspecified"
    ARABIC = "arabic"
    LANGUAGE = "language"


# The letter an Egyptian school appends to a class code to name the stage.
#
# KG             foundation stage in both UI languages
# ب  ابتدائي     primary
# ع  إعدادي      preparatory
# ث  ثانوي       secondary
#
# Stored as the source of both directions so the parser and the renderer cannot drift —
# a mapping written twice is a mapping that disagrees with itself the first time a stage
# is added.
STAGE_LETTER: Final[dict[Stage, str]] = {
    Stage.GARDEN: "ر",
    Stage.PRIMARY: "ب",
    Stage.PREPARATORY: "ع",
    Stage.SECONDARY: "ث",
}

STAGE_BY_LETTER: Final[dict[str, Stage]] = {
    letter: stage for stage, letter in STAGE_LETTER.items()
}

# What a stage is called on screen. The adjective form ("الابتدائي") attaches to a grade —
# "الصف الأول الابتدائي" — while the noun form ("المرحلة الابتدائية") names the division
# itself in a heading. Arabic needs both and they are not interchangeable; English gets by
# with one word, which is why a naive one-word-per-stage table reads wrong in Arabic.
STAGE_NAMES: Final[dict[Stage, dict[str, str]]] = {
    Stage.GARDEN: {
        "en": "KG",
        "en_long": "KG",
        "ar": "KG",
        "ar_long": "KG",
    },
    Stage.PRIMARY: {
        "en": "Primary",
        "en_long": "Primary School",
        "ar": "الابتدائي",
        "ar_long": "المرحلة الابتدائية",
    },
    Stage.PREPARATORY: {
        "en": "Preparatory",
        "en_long": "Preparatory School",
        "ar": "الإعدادي",
        "ar_long": "المرحلة الإعدادية",
    },
    Stage.SECONDARY: {
        "en": "Secondary",
        "en_long": "Secondary School",
        "ar": "الثانوي",
        "ar_long": "المرحلة الثانوية",
    },
    Stage.UNSPECIFIED: {
        "en": "",
        "en_long": "",
        "ar": "",
        "ar_long": "",
    },
}

# Masculine ordinals, which is the form that agrees with الصف and الفصل. Up to fourteen
# because that is the longest ladder in scope (a language section counting Grade 1..14);
# past the end the numeral is used, which is ugly but true rather than wrong.
_ARABIC_ORDINALS: Final[tuple[str, ...]] = (
    "الأول",
    "الثاني",
    "الثالث",
    "الرابع",
    "الخامس",
    "السادس",
    "السابع",
    "الثامن",
    "التاسع",
    "العاشر",
    "الحادي عشر",
    "الثاني عشر",
    "الثالث عشر",
    "الرابع عشر",
)

_ENGLISH_ORDINALS: Final[tuple[str, ...]] = (
    "First",
    "Second",
    "Third",
    "Fourth",
    "Fifth",
    "Sixth",
    "Seventh",
    "Eighth",
    "Ninth",
    "Tenth",
    "Eleventh",
    "Twelfth",
    "Thirteenth",
    "Fourteenth",
)

# `1/2 ب`, and the shapes a person actually types: any spacing, an ASCII or Arabic slash,
# and the letter optionally absent when the stage is already known from context.
_ARABIC_CODE = re.compile(
    r"^\s*(?P<grade>\d{1,2})\s*[//؟\\-]\s*(?P<section>\d{1,2})\s*(?P<letter>[رﺭبﺏعﻉثﺙ])?\s*$"
)

# `Grade 3 A`, `G3A`, `3A` — the shapes a language section writes. The trailing letter is
# the room, not the stage, which is why this is a separate pattern rather than a flag.
_LATIN_CODE = re.compile(
    r"^\s*(?:grade\s*|g)?(?P<grade>\d{1,2})\s*[-/ ]?\s*(?P<section>[A-Za-z]{1,2})\s*$",
    re.IGNORECASE,
)


def arabic_ordinal(number: int) -> str:
    """`1` -> `الأول`. Past the table, the numeral — true beats prettily wrong."""
    if 1 <= number <= len(_ARABIC_ORDINALS):
        return _ARABIC_ORDINALS[number - 1]
    return str(number)


def english_ordinal(number: int) -> str:
    """`1` -> `First`. Same fallback rule as the Arabic."""
    if 1 <= number <= len(_ENGLISH_ORDINALS):
        return _ENGLISH_ORDINALS[number - 1]
    return str(number)


def stage_name(stage: Stage, language: str = "en", *, long: bool = False) -> str:
    """The stage on screen. `long` gives the noun form that names the division itself."""
    names = STAGE_NAMES.get(stage, STAGE_NAMES[Stage.UNSPECIFIED])
    key = "ar" if str(language).lower().startswith("ar") else "en"
    return names[f"{key}_long"] if long else names[key]


@dataclass(frozen=True, slots=True)
class ClassCoordinates:
    """The structured facts behind a class name. What `parse_*` returns and `render_*` takes.

    `grade_number` and `section_number` are the two numbers in `1/2 ب`. Both are optional
    because a school may have rungs it has not numbered yet, and a half-known class must
    still be representable — the renderer degrades to whatever it was given rather than
    inventing the missing half.
    """

    stage: Stage = Stage.UNSPECIFIED
    grade_number: int | None = None
    section_number: int | None = None
    kind: EducationalSystemKind = EducationalSystemKind.UNSPECIFIED
    # The room label a school typed: "A", "Simba Class". Wins over generation, always.
    label_en: str = ""
    label_ar: str = ""

    def __post_init__(self) -> None:
        for name in ("grade_number", "section_number"):
            value = getattr(self, name)
            if value is None:
                continue
            # A bool is an int, and `grade_number=True` would silently mean grade one.
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValidationError(f"{name} must be a positive whole number", field=name)


def parse_arabic_class_code(code: str) -> ClassCoordinates:
    """Read `1/2 ب` back into its facts. Grade first, room second — see the module docstring.

    Raises rather than guessing on anything that is not that shape. A code this cannot
    read is a code the school writes differently, and silently returning "grade 1, room 1"
    would file every one of those classes under the same rung.
    """
    match = _ARABIC_CODE.match(str(code or ""))
    if match is None:
        raise ValidationError(
            f"{code!r} is not an Arabic class code; expected the form '1/2 ب' "
            "(grade, then classroom, then the stage letter)",
            field="code",
        )
    letter = match.group("letter") or ""
    return ClassCoordinates(
        stage=STAGE_BY_LETTER.get(letter, Stage.UNSPECIFIED),
        grade_number=int(match.group("grade")),
        section_number=int(match.group("section")),
        kind=EducationalSystemKind.ARABIC,
    )


def parse_language_class_code(code: str) -> ClassCoordinates:
    """Read `Grade 3 A` / `G3A` / `3A` into its facts.

    The room comes back as a label rather than a number: a language section names rooms
    with letters, and turning `A` into `1` would make the two sections' `section_number`
    columns mean different things.
    """
    match = _LATIN_CODE.match(str(code or ""))
    if match is None:
        raise ValidationError(
            f"{code!r} is not a language-section class code; expected the form 'Grade 3 A'",
            field="code",
        )
    return ClassCoordinates(
        grade_number=int(match.group("grade")),
        label_en=match.group("section").upper(),
        kind=EducationalSystemKind.LANGUAGE,
    )


def render_class_code(
    *,
    kind: EducationalSystemKind,
    stage: Stage,
    grade_number: int | None,
    section_number: int | None = None,
    label_en: str = "",
) -> str:
    """The identifier a school writes on the door. Immutable once stored (decision 7).

    Generated here so a console that offers "stage, grade, room" produces the same string
    a registrar would have typed, rather than a second convention nobody recognises.
    """
    if kind is EducationalSystemKind.ARABIC:
        if grade_number is None or section_number is None:
            raise ValidationError(
                "an Arabic class code needs both a grade and a classroom number",
                field="code",
            )
        letter = STAGE_LETTER.get(stage, "")
        return f"{grade_number}/{section_number} {letter}".strip()

    if kind is EducationalSystemKind.LANGUAGE:
        if grade_number is None:
            raise ValidationError("a language class code needs a grade", field="code")
        room = (label_en or "").strip().replace(" ", "")
        return f"G{grade_number}{room}" if room else f"G{grade_number}"

    # Unspecified: there is no convention to follow, so the label is the code.
    return (label_en or "").strip()


def render_grade_name(
    *,
    kind: EducationalSystemKind,
    stage: Stage,
    grade_number: int | None,
    language: str = "en",
) -> str:
    """The rung: `First Grade Primary` / `الصف الأول الابتدائي` / `Grade 9`.

    The two ladders genuinely name rungs differently and that is not a translation choice.
    An Arabic section counts within a stage, so the stage is part of the rung's name and
    "الصف الأول" alone is ambiguous across four of them. A language section counts
    straight through, so `Grade 9` already identifies one rung and appending the stage
    would be noise.
    """
    arabic_reading = str(language).lower().startswith("ar")
    if grade_number is None:
        return stage_name(stage, language, long=True)

    # KG counts on its own in both ladders and deliberately keeps the same short label
    # in Arabic and English. This is checked before the per-kind branches because it is
    # the one rule the two sections genuinely share.
    if stage is Stage.GARDEN:
        return f"KG {grade_number}"

    if kind is EducationalSystemKind.LANGUAGE:
        if arabic_reading:
            return f"الصف {arabic_ordinal(grade_number)}"
        return f"Grade {grade_number}"

    if arabic_reading:
        return f"الصف {arabic_ordinal(grade_number)} {stage_name(stage, 'ar')}".strip()
    return f"{english_ordinal(grade_number)} Grade {stage_name(stage, 'en')}".strip()


def render_class_label(coordinates: ClassCoordinates, language: str = "en") -> str:
    """The short name of the room, on its own: `A`, `Simba Class`, `الفصل الثاني`.

    The stored label wins whenever there is one — that is the configurable-KG-name
    requirement, and it is also just the rule that a school's own words beat generated
    ones. Generation fills the blank.
    """
    arabic_reading = str(language).lower().startswith("ar")
    stored = coordinates.label_ar if arabic_reading else coordinates.label_en
    # Fall back to the other language's label before generating: a room called
    # "Simba Class" with no Arabic label is still called Simba Class in Arabic.
    stored = (stored or coordinates.label_en or coordinates.label_ar or "").strip()
    if stored:
        return stored
    if coordinates.section_number is None:
        return ""
    if arabic_reading:
        return f"الفصل {arabic_ordinal(coordinates.section_number)}"
    return f"Class {coordinates.section_number}"


def _is_room_code(room: str) -> bool:
    """`A`, `B2` — a room identified by a letter, rather than named after something."""
    return len(room) <= 2 and room.isalnum()


def render_class_title(coordinates: ClassCoordinates, language: str = "en") -> str:
    """The full, unambiguous title of one class, in the language being read.

        1/2 ب   ->  Class 2 — Second Grade Primary
                    الفصل الثاني — الصف الأول الابتدائي
        G3A     ->  Grade 3 A
        KG      ->  Simba Class — KG

    An em dash between the room and the rung rather than a comma or a slash: it survives
    being read right-to-left without the punctuation appearing to belong to the wrong
    half, which a comma does.

    Both halves are optional, and either alone is a valid title. A class with a label and
    no rung is "Simba Class"; a rung with no room is "First Grade Primary". Joining is
    the last step so a missing half never leaves a dangling separator.
    """
    room = render_class_label(coordinates, language)
    rung = render_grade_name(
        kind=coordinates.kind,
        stage=coordinates.stage,
        grade_number=coordinates.grade_number,
        language=language,
    )
    if coordinates.kind is EducationalSystemKind.LANGUAGE and room and rung and _is_room_code(room):
        # `Grade 3 A`, not `A — Grade 3`. A language section writes the rung and the room
        # as one phrase, and splitting them reads as two separate facts about a class.
        #
        # Only when the room is a *code*. A named room — "Simba Class", which is what a
        # kindergarten uses — is a name in its own right, and "KG 1 Simba Class" runs two
        # nouns together where "Simba Class — KG 1" reads as a room and the rung it is on.
        return f"{rung} {room}"
    parts = [part for part in (room, rung) if part]
    return " — ".join(parts)


__all__ = [
    "ClassCoordinates",
    "EducationalSystemKind",
    "STAGE_BY_LETTER",
    "STAGE_LETTER",
    "STAGE_NAMES",
    "arabic_ordinal",
    "english_ordinal",
    "parse_arabic_class_code",
    "parse_language_class_code",
    "render_class_code",
    "render_class_label",
    "render_class_title",
    "render_grade_name",
    "stage_name",
]
