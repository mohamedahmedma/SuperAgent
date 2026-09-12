"""The child's name, as something to take OUT of a retrieval query.

The knowledge base is the school's own published material: it is written once, for every
family, and it does not contain a single pupil's name. A search for «مصاريف علي حسن»
therefore spends two of its four terms on strings the corpus cannot hold, and the place
that hurts most is the sparse half — a name is a RARE term, so its IDF is high, and BM25
will happily rank a permission slip that merely mentions another Ali above the fee
schedule that answers the question.

The name still has to reach everything else. Which child this turn is about decides
which record is read (`planned_child_id`), how the answer is worded, and what the parent
is asked when two children match. So this is not redaction for privacy and it is not a
rewrite of the message: it is one derived view of the question, for retrieval only, and
`backend/rag/pipeline.py:_search_query` is the only caller.

That is the same shape the child's YEAR already has, for the same measured reason — see
`_search_query`'s docstring. Conditions and identity travel BESIDE the question; only
words the corpus might actually contain travel inside it.

## Why this is not a name detector

There is no Arabic NER here, and no list of given names. Both would be the wrong tool:
they answer "is this a name" when the question is "is this MY CHILD'S name", and the
second has an authoritative answer already — the roster the records facade returned for
this guardian, which `backend/chat/child_resolution.py` has just matched the message
against. A closed set of at most a handful of strings, established before the agent ran.

So nothing here guesses. It removes strings the turn has already PROVEN name this
parent's child, and abstains everywhere else.

## The folding problem, which is the whole difficulty

`backend/text_matching.name_key` folds ى onto ي, because a school's SIS spells Arabic
however the registrar typed it and a parent has no way to know which way. That folding
is what makes «ليلي» find «ليلى». It also makes the preposition على and the given name
علي the same string — and this deployment's roster carries a child called علي. The same
collision reaches آية, which folds onto the Egyptian «إيه» (what), and عمر, which is also
the word for an age.

`text_matching` already states the rule from the other side: its BM25 stop list leaves
على out on purpose, "never add a token that is also a plausible Egyptian given name".
`_ALSO_ORDINARY_WORDS_NATURAL` below is that same list read the other way round — names
this corpus's own vocabulary can collide with — and it does not decide what a token
MEANS. It decides how much evidence is needed before a token is cut out of somebody's
question:

  * an unambiguous name (أحمد, ليلى, فاطمة) is cut wherever it appears, folded match
    included, because no Arabic sentence contains it by accident;
  * an ambiguous one is cut only where something OTHER than the folded key says it is a
    name — the parent spelled it the way the roster spells it, or a relationship word
    introduces it («ابني على») — and then only once, because the second occurrence of a
    homograph in one sentence is far more likely to be the ordinary word.

Every abstention leaves the query exactly as it was, which is what this system did
before this module existed. That asymmetry is deliberate: leaving a name in a query
costs some precision, while cutting a preposition — or the word «عمر» out of a question
about a child's age — changes what was asked, and nothing downstream can tell that it
happened.

## What it deliberately does not reach

A name carrying a clitic — «لعلي», «وليلى» — is one token here and is left alone. Undoing
those prefixes is not free: stripping ف off «فعلي» (actual) produces the name, which is
the same false positive this module spends its length avoiding, and the sparse half
already converges the two through `search_key`'s stemmer. The case that matters in
practice is the bare name, because both ways one arrives are bare: a parent typing
«مصاريف علي», and `chat/resolve_question.j2` writing the name in for a pronoun.

Nothing here reads a clock, a database, a model or the environment.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import List, Sequence, Set, Tuple

from backend.text_matching import name_key
from backend.text_normalization import sanitize_text

#: Word tokens, spelled exactly as `text_matching._TOKEN` spells them so that what counts
#: as one word is decided the same way for matching and for cutting.
_TOKEN = re.compile(r"\w+", re.UNICODE)

#: Sentence punctuation that a removed word can leave stranded behind a space. Both the
#: Arabic marks and their Latin counterparts, because a parent's keyboard supplies
#: whichever it is set to and one message often carries both.
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([؟،؛?!,.:;])")

#: Given names in this deployment's population that are ALSO ordinary Arabic words, or
#: that fold onto one. Written in natural spelling and folded at import, exactly as
#: `text_matching.arabic_stop_words_for_analyzer` is, because the comparison happens on
#: folded keys and a list written in surface form would silently never match.
#:
#: The rule for adding one: it earns its place by being a word this corpus, or a parent
#: writing to this school, would use with no child in mind. Being a common name is not
#: enough — أحمد and محمد are everywhere and mean nothing else, so they belong in the
#: fast path that cuts every occurrence.
#:
#: A missing entry costs a cut that should not have happened; a spurious entry costs a
#: name left in a query. The first is worse, so the list errs long.
_ALSO_ORDINARY_WORDS_NATURAL = (
    "على", "علي",      # the preposition, and the name it folds onto
    "عمر",             # an age — «عمر الطالب» is a question this corpus answers
    "نجاح",            # passing, a word a school corpus is built out of
    "حسن",             # good, fine
    "ملك",             # a king, and property — «ملك المدرسة»
    "أمل",             # hope
    "نور",             # light
    "هنا",             # here
    "آية",             # a verse; folds onto «إيه», the Egyptian "what"
    "إيمان",           # faith
    "فرح",             # joy
    "سعيد",            # happy
    "كريم",            # generous
    "جميل",            # beautiful
    "عادل",            # fair, just
    "شريف",            # honourable
    "هاجر",            # she emigrated
    "ندى",             # dew
    "سما",             # sky
    "هدى",             # guidance
    "رحمة",            # mercy
    "بسمة",            # a smile
    "أمير",            # a prince
    "رضا",             # contentment
    "وفاء",            # loyalty
    "صفاء",            # clarity
    "حياة",            # life
    "سلام",            # peace
    "عبير",            # fragrance
    "منى",             # wishes
    "جنى",             # a harvest
    "زين",             # fine, adorned
)

#: Words that introduce a child by name. One of these immediately before a token is the
#: evidence that lets an ambiguous name be cut: «ابني على» is a boy, «خصم على» is not.
#:
#: Deliberately only the token IMMEDIATELY before. Widening the window to two buys
#: «ابني الكبير على» and pays for it with «ابني في على», which is not a sentence but is
#: the kind of thing messages reach this function as.
_RELATIONSHIP_WORDS_NATURAL = (
    "ابني", "إبني", "ابنتي", "بنتي", "ولدي", "ابن", "بنت",
    "الطالب", "الطالبة", "التلميذ", "التلميذة",
    "son", "daughter", "child", "student", "pupil",
)


@lru_cache(maxsize=1)
def _also_ordinary_words() -> frozenset:
    return frozenset(
        key for key in (name_key(word) for word in _ALSO_ORDINARY_WORDS_NATURAL) if key
    )


@lru_cache(maxsize=1)
def _relationship_words() -> frozenset:
    return frozenset(
        key for key in (name_key(word) for word in _RELATIONSHIP_WORDS_NATURAL) if key
    )


def _exact_key(text: str) -> str:
    """The spelling-PRESERVING key: what the writer actually typed, compared fairly.

    `sanitize_text` and nothing more — so PDF damage, tatweel and invisible marks cannot
    make one spelling look like two, while the hamza, the teh marbuta and the alef
    maksura that tell على from علي all survive. Casefolded, so an English name compares
    the way `name_key` would compare it.

    This is precisely the distinction `name_key` exists to destroy, which is why it is
    built here rather than imported: retrieval needs both readings of a token — the
    forgiving one to find a child, the strict one to be sure about a homograph.
    """
    return " ".join(sanitize_text(text or "").casefold().split())


def name_surfaces(*, reference: str, child_name: str = "", label: str = "") -> Tuple[str, ...]:
    """The strings this turn has proven are the child's name, longest first.

    `reference` is `RequestSignals.child_reference`, and the gate is that it must be
    `named`. Every other value — a pin, an only child, "my son", the conversation
    carrying the subject — means the message contains no name to remove, so there is
    nothing to look for, and looking anyway is how a question about «الخصم على الصف
    الثالث» would lose its preposition to a child nobody mentioned.

    `named` is not taken on the classifier's word either: `signals._read_child` demotes
    it to `context` unless the name it reported actually occurs in the text it read, so
    by the time this runs the occurrence is established.

    Both the name AS WRITTEN and the roster's own spelling of it are returned. They are
    usually one string and cost nothing when they are; when they differ it is because a
    parent and a registrar spelled one child two ways, which is the case the whole of
    `child_resolution` exists to survive.

    The label's FIRST token is offered on its own — that is the given name, the part a
    parent actually types. The rest are patronymics: «حسن» in «علي حسن» is the father, it
    is shared by every sibling, and on its own it is an ordinary adjective. It goes as
    part of the full name and never by itself.
    """
    if reference != "named":
        return ()
    surfaces: List[str] = []

    def add(candidate: str) -> None:
        text = " ".join(str(candidate or "").split())
        if text and not any(_exact_key(text) == _exact_key(seen) for seen in surfaces):
            surfaces.append(text)

    add(child_name)
    add(label)
    label_tokens = _TOKEN.findall(label or "")
    if len(label_tokens) > 1:
        add(label_tokens[0])
    # Longest first, so «علي حسن» is taken as one name before «علي» is weighed as a
    # homograph on its own — a two-token match needs no ambiguity argument at all.
    surfaces.sort(key=lambda text: len(_TOKEN.findall(text)), reverse=True)
    return tuple(surfaces)


def _spans(text: str) -> List[Tuple[int, int, str]]:
    return [(match.start(), match.end(), match.group(0)) for match in _TOKEN.finditer(text)]


def _introduced(spans: Sequence[Tuple[int, int, str]], index: int) -> bool:
    """Whether the token at `index` is introduced by a relationship word."""
    return index > 0 and name_key(spans[index - 1][2]) in _relationship_words()


def _windows_to_cut(
    spans: Sequence[Tuple[int, int, str]], text: str, surface: str, taken: Set[int]
) -> List[int]:
    """Which token positions this surface claims. The whole decision lives here.

    Returns the starting index of each window to remove. `taken` holds positions a longer
    surface has already claimed, so «علي» does not re-examine the «علي» inside an «علي
    حسن» that has already gone.
    """
    width = len(_TOKEN.findall(surface))
    if width < 1:
        return []
    exact: List[int] = []
    folded: List[int] = []
    for start in range(0, len(spans) - width + 1):
        if any(start + offset in taken for offset in range(width)):
            continue
        window = text[spans[start][0]:spans[start + width - 1][1]]
        if _exact_key(window) == _exact_key(surface):
            exact.append(start)
        elif name_key(window) == name_key(surface):
            folded.append(start)

    # A name of two or more words is not a phrase a sentence produces by accident, so it
    # needs no argument beyond having matched.
    if width > 1 or name_key(surface) not in _also_ordinary_words():
        return exact + folded

    # Ambiguous, and therefore evidence-led. A relationship word is the strongest thing
    # available; the parent having spelled it the roster's way is the next. A folded-only
    # match with neither is exactly the «خصم على الأخ» case, and is left alone.
    introduced = [start for start in exact + folded if _introduced(spans, start)]
    candidates = introduced or exact
    # One occurrence at most: a homograph written twice in one question is far more
    # likely to be the name once and the word once than the name twice.
    return candidates[:1]


def strip_child_names(text: str, surfaces: Sequence[str]) -> Tuple[str, int]:
    """`text` with the child's name cut out of it, and how many cuts were made.

    Cut from the ORIGINAL string, never from a folded copy of it. The result is read by
    the embedder and by the rewrite model, and `backend/text_matching.py` states the rule
    they both depend on: folded Arabic has lost the orthography those models were trained
    on. Matching may fold; what comes back out of here is the parent's own spelling, minus
    some words.

    A cut that would empty the question is refused. «علي؟» — a parent naming a child and
    nothing else — is not a search for the empty string; it is a question this function
    has nothing to contribute to.
    """
    if not text or not surfaces:
        return text, 0
    spans = _spans(text)
    if not spans:
        return text, 0

    taken: Set[int] = set()
    cuts = 0
    for surface in surfaces:
        width = len(_TOKEN.findall(surface))
        for start in _windows_to_cut(spans, text, surface, taken):
            taken.update(range(start, start + width))
            cuts += 1
    if not cuts:
        return text, 0

    # Rebuilt from the gaps AROUND the tokens that go, so the punctuation and spacing the
    # parent wrote survives — the question keeps reading like a question, which is what
    # the dense half is scoring.
    pieces: List[str] = []
    cursor = 0
    for index, (start, end, _token) in enumerate(spans):
        if index not in taken:
            continue
        pieces.append(text[cursor:start])
        cursor = end
    pieces.append(text[cursor:])
    stripped = " ".join("".join(pieces).split())
    # The gap a cut token leaves in front of its own punctuation. Closed because the
    # dense half embeds this string and «مصاريف ؟» is not a sentence anybody wrote.
    stripped = _SPACE_BEFORE_PUNCTUATION.sub(r"\1", stripped).strip()
    if not _TOKEN.search(stripped):
        return text, 0
    return stripped, cuts


__all__ = ["name_surfaces", "strip_child_names"]
