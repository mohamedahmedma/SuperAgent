"""A parent and a child, as much of each as this service is entitled to hold.

Both are values, not rows. This service stores neither: it asks the school's system of
record, uses the answer to mint one token, and forgets it. What is kept on an account is
the handle — `guardian_external_id` — and nothing else about the family.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GuardianRef:
    """A parent, named by a handle rather than by the number that found her.

    Carries no phone number on purpose. Once this is in hand the number has done its job,
    and everything downstream — the token, the account row, the audit line — refers to her
    by `public_id`. A number is PII that changes; a handle is neither.
    """

    public_id: str
    full_name_ar: str = ""
    full_name_en: str = ""
    preferred_language: str = "ar"

    @property
    def display_name(self) -> str:
        """Her name in whichever script the school recorded, Arabic first."""
        return self.full_name_ar or self.full_name_en


@dataclass(frozen=True, slots=True)
class ChildRef:
    """One child, as much of her as belongs in a token and no more.

    A name to greet her by, the year she is in, and whether she is a son or a daughter —
    exactly what is needed to understand a parent who writes "my son" rather than a name.

    Nothing about her RECORD is here: no marks, no attendance, no birth date, no contact
    details. This travels in a bearer token that lives in a browser and rides every
    request into every access log, so what it carries is the minimum that makes the
    feature work, and the reader is expected to fetch anything else it needs.
    """

    student_id: str
    full_name_ar: str = ""
    full_name_en: str = ""
    year_level: str = ""
    gender: str = "unspecified"

    @property
    def display_name(self) -> str:
        return self.full_name_ar or self.full_name_en or self.student_id

    def as_claim(self) -> dict:
        """The compact form that goes into the token.

        Short keys, because this is paid on every request a parent's browser makes.
        """
        return {
            "id": self.student_id,
            "ar": self.full_name_ar,
            "en": self.full_name_en,
            "yr": self.year_level,
            "g": self.gender,
        }


__all__ = ["ChildRef", "GuardianRef"]
