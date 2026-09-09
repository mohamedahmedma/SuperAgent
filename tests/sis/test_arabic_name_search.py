"""Finding a child whose name is spelled the other way.

A registrar types `فاطمه` and the roster says `فاطمة`; types `احمد` and the roster says
`أحمد`. Those are one name each. Which spelling reached this service was decided by
whoever typed the spreadsheet, and the child has no say in it — so a search that compares
the raw strings answers "no such child" about a child who is plainly on the register, and
the registrar's next move is to give up and edit the name by hand until the box finds it.

Two properties are pinned here, and the second is the one that breaks quietly:

**Both directions.** Every case below is asserted from both spellings, because a fold
applied to the typed term alone matches nothing at all — the query becomes a string the
column does not contain. Symmetric folding is the whole mechanism.

**Nothing else widens.** `%` is still a literal, an empty box still answers with silence,
and an English name is still matched as it always was. A search that quietly became a
prefix-of-anything match would pass a test that only checked the Arabic.
"""
import pytest
from fastapi.testclient import TestClient

from tests.sis.conftest import registrar_headers
from sis.domain.arabic import fold_for_search


@pytest.fixture()
def registrar() -> dict[str, str]:
    return registrar_headers()


def _add(client: TestClient, headers: dict[str, str], number: str, arabic: str, english: str = "") -> None:
    created = client.post(
        "/v1/students",
        json={"student_number": number, "full_name_ar": arabic, "full_name_en": english},
        headers=headers,
    )
    assert created.status_code in (200, 201), created.text


# ---------------------------------------------------------------------------
# The fold itself, with no database in the way
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("one", "other"),
    [
        ("فاطمة", "فاطمه"),  # teh marbuta / heh
        ("أحمد", "احمد"),  # alef with hamza / bare alef
        ("إيمان", "ايمان"),  # alef with hamza below
        ("آية", "ايه"),  # alef with madda, and the teh marbuta with it
        ("ليلى", "ليلي"),  # alef maksura / yeh
        ("رؤوف", "رووف"),  # waw with hamza
        ("فائز", "فايز"),  # yeh with hamza
        ("مُحَمَّد", "محمد"),  # fully vowelled, as a document that was typeset carries it
        ("عــلي", "علي"),  # tatweel, as justified Arabic carries it
    ],
)
def test_two_spellings_of_one_name_fold_onto_one_key(one: str, other: str) -> None:
    assert fold_for_search(one) == fold_for_search(other)


def test_two_different_names_do_not_fold_together() -> None:
    """The fold is lossy on orthography and must stay lossless on identity.

    `حسن` and `حسين` differ by a letter that carries the name; `سارة` and `سارا` are two
    spellings of one name and are allowed to meet. A fold that merged the first pair would
    put the wrong child on screen, which is worse than not finding her at all.
    """
    assert fold_for_search("حسن") != fold_for_search("حسين")
    assert fold_for_search("عمر") != fold_for_search("عمرو")
    assert fold_for_search("سارة") == fold_for_search("ساره")


def test_a_latin_name_passes_through_untouched() -> None:
    """None of these characters occurs in it, so the English half is folded by doing nothing."""
    assert fold_for_search("Sara Mohamed Ali") == "Sara Mohamed Ali"
    assert fold_for_search("") == ""


# ---------------------------------------------------------------------------
# And over HTTP, which is where the registrar meets it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("on_file", "typed"),
    [
        ("فاطمة رفعت الجندي", "فاطمه"),
        ("فاطمه رفعت الجندي", "فاطمة"),
        ("أحمد سيد قنديل", "احمد"),
        ("احمد سيد قنديل", "أحمد"),
        ("ليلى حسن", "ليلي"),
        ("مُحَمَّد عادل", "محمد"),
    ],
)
def test_a_name_is_found_however_either_side_spells_it(
    client: TestClient, registrar: dict[str, str], on_file: str, typed: str
) -> None:
    _add(client, registrar, "10432", on_file)
    found = client.get("/v1/students", params={"q": typed}, headers=registrar)
    assert found.status_code == 200, found.text
    assert [row["student_number"] for row in found.json()["students"]] == ["10432"], (
        f"{typed!r} did not find {on_file!r}"
    )


def test_the_english_name_and_the_number_still_match_as_they_did(
    client: TestClient, registrar: dict[str, str]
) -> None:
    """Folding is additive. Everything that matched before this existed still matches."""
    _add(client, registrar, "10432", "أحمد سيد", "Ahmed Sayed")

    assert client.get("/v1/students", params={"q": "ahmed"}, headers=registrar).json()["count"] == 1
    assert client.get("/v1/students", params={"q": "Ahmed"}, headers=registrar).json()["count"] == 1
    assert client.get("/v1/students", params={"q": "10432"}, headers=registrar).json()["count"] == 1


def test_a_wildcard_is_still_a_character_and_not_a_wildcard(
    client: TestClient, registrar: dict[str, str]
) -> None:
    """The escaping survives the fold, which runs before it.

    `%` in the box used to mean "every child in the school", which reads as a broken filter.
    Folding happens first and introduces no wildcard of its own, so the escape still lands.
    """
    _add(client, registrar, "10432", "فاطمة رفعت", "Fatma Refaat")
    assert client.get("/v1/students", params={"q": "%"}, headers=registrar).json()["count"] == 0
    assert client.get("/v1/students", params={"q": "_"}, headers=registrar).json()["count"] == 0
    assert client.get("/v1/students", params={"q": ""}, headers=registrar).json()["count"] == 0
