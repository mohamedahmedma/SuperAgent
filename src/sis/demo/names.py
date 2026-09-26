"""Fictional people, in Arabic and in English. Nothing here belongs to a real person.

Every name is drawn from these pools and combined deterministically, so the same seed run
produces the same roster twice — a demo whose student numbers move between runs is a demo
nobody can write a test or a bug report against.

Two lists, not one transliterated into the other: a demo school that shows the same string
twice under an Arabic and an English heading proves nothing about the bilingual handling,
which is one of the things this data exists to exercise. So each child gets a genuine
Arabic name *and* its Latin-script rendering, paired here rather than generated.

The pool is deliberately small. Four hundred children from ninety name parts means
repeated full names, which is realistic — schools have three Ahmed Mohameds — and it is
the case a roster screen has to handle without ambiguity, because the student number is
what actually identifies a child.
"""
from __future__ import annotations

from typing import Final

# (Arabic, Latin) pairs. Given names a school in Egypt would actually see, invented as
# combinations rather than taken from anybody.
BOY_NAMES: Final[tuple[tuple[str, str], ...]] = (
    ("أحمد", "Ahmed"),
    ("محمد", "Mohamed"),
    ("يوسف", "Youssef"),
    ("عمر", "Omar"),
    ("خالد", "Khaled"),
    ("مازن", "Mazen"),
    ("كريم", "Karim"),
    ("زياد", "Ziad"),
    ("طارق", "Tarek"),
    ("سيف", "Seif"),
    ("آدم", "Adam"),
    ("حسن", "Hassan"),
    ("مصطفى", "Mostafa"),
    ("عبد الرحمن", "Abdelrahman"),
    ("مروان", "Marwan"),
    ("بلال", "Belal"),
    ("رامي", "Ramy"),
    ("نادر", "Nader"),
)

GIRL_NAMES: Final[tuple[tuple[str, str], ...]] = (
    ("فاطمة", "Fatma"),
    ("مريم", "Mariam"),
    ("سارة", "Sara"),
    ("نور", "Nour"),
    ("هنا", "Hana"),
    ("ليلى", "Laila"),
    ("جنى", "Jana"),
    ("ملك", "Malak"),
    ("سلمى", "Salma"),
    ("رنا", "Rana"),
    ("دينا", "Dina"),
    ("ياسمين", "Yasmin"),
    ("أمينة", "Amina"),
    ("حبيبة", "Habiba"),
    ("رقية", "Roqaya"),
    ("تاليا", "Talia"),
    ("لينا", "Lina"),
    ("زينة", "Zeina"),
)

# Father's name, then family name. Egyptian full names are given + father + family, which
# is why two parts are combined rather than one surname being appended.
MIDDLE_NAMES: Final[tuple[tuple[str, str], ...]] = (
    ("أحمد", "Ahmed"),
    ("محمود", "Mahmoud"),
    ("إبراهيم", "Ibrahim"),
    ("سامي", "Samy"),
    ("فؤاد", "Fouad"),
    ("عادل", "Adel"),
    ("رفعت", "Refaat"),
    ("مجدي", "Magdy"),
    ("شريف", "Sherif"),
    ("هشام", "Hisham"),
    ("وليد", "Walid"),
    ("عصام", "Essam"),
)

FAMILY_NAMES: Final[tuple[tuple[str, str], ...]] = (
    ("الشناوي", "El-Shennawy"),
    ("عبد الله", "Abdallah"),
    ("منصور", "Mansour"),
    ("الفقي", "El-Feqi"),
    ("سليمان", "Soliman"),
    ("زكي", "Zaki"),
    ("الجندي", "El-Gindy"),
    ("رشدي", "Roshdy"),
    ("النجار", "El-Naggar"),
    ("قنديل", "Kandil"),
    ("شاهين", "Shahin"),
    ("الديب", "El-Deeb"),
    ("بدوي", "Badawy"),
    ("حجازي", "Hegazy"),
    ("العيسوي", "El-Eisawy"),
)


def student_name(index: int, *, female: bool) -> tuple[str, str]:
    """Build one child's `(Arabic, English)` full name from a stable index.

    Prime-ish strides across the four pools so consecutive children in a class do not all
    share a family name — the readable failure of `index % len(pool)` on every part at
    once is a register that looks like one extended family.
    """
    given = GIRL_NAMES if female else BOY_NAMES
    first_ar, first_en = given[index % len(given)]
    middle_ar, middle_en = MIDDLE_NAMES[(index * 5) % len(MIDDLE_NAMES)]
    family_ar, family_en = FAMILY_NAMES[(index * 7) % len(FAMILY_NAMES)]
    return (
        f"{first_ar} {middle_ar} {family_ar}",
        f"{first_en} {middle_en} {family_en}",
    )


def guardian_name(index: int, *, female: bool, family: tuple[str, str]) -> tuple[str, str]:
    """A parent's name, sharing the child's family name — which is how a roster reads."""
    given = GIRL_NAMES if female else BOY_NAMES
    first_ar, first_en = given[(index * 3) % len(given)]
    return f"{first_ar} {family[0]}", f"{first_en} {family[1]}"


def family_of(index: int) -> tuple[str, str]:
    """The family name `student_name` gave this index, so a guardian can share it."""
    return FAMILY_NAMES[(index * 7) % len(FAMILY_NAMES)]


__all__ = [
    "BOY_NAMES",
    "FAMILY_NAMES",
    "GIRL_NAMES",
    "MIDDLE_NAMES",
    "family_of",
    "guardian_name",
    "student_name",
]
