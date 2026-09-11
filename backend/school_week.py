"""Which day a parent asked about — decided here, never by the model.

«إيه حصص بكره؟» is the commonest timetable question this deployment sees, and it is one
a language model cannot answer: it has no clock. Asked to work out which weekday
"tomorrow" is, it either invents one or reads the whole week out and leaves the parent
to find the row — which is what it did before this module existed.

So the resolution is arithmetic and lives here. The model's only job is to repeat the
words the parent used; turning «بكره» into `sunday` is a date and a lookup table, and
both are things code is better at than a model and cheaper at than a round trip.

Two halves, and they are deliberately separate:

`day_phrase` reads a phrase out of free text. Lexical, folded through the same
`name_key` the roster and subject matchers use, and with no clock in it at all — its
answer for a given message is the same answer forever, which is what makes it safe to
run in the planner and carry alongside a turn.

`resolve_day` turns that phrase into a weekday, and is the only half that needs to know
what day it is. A weekday the parent NAMED needs no clock — a timetable is a recurring
weekly plan, so every Sunday carries the same lessons — and only the relative words
(today, tomorrow, yesterday) spend the date at all.

## What this deliberately does not know

**Holidays.** A timetable is the class's standing plan for a weekday, not a claim that
the school opens on a particular date. «بكره» resolving to Sunday says what Sunday
normally holds; whether that Sunday is a holiday is a calendar fact, and the calendar
is in the knowledge base. `records_result.j2` says so to the model on every narrowed
answer, because the plausible thing to say — "he has Arabic first thing tomorrow" — is
false on the morning of a public holiday, and a parent would act on it.

**Which days the school opens.** That arrives with the timetable payload, per school,
and the caller checks the resolved day against it. A Friday resolved here is a real
Friday; whether it is a school day is not this module's fact to hold.

## The ambiguous words

Arabic day names collide with number words: «التلات» is Tuesday and also "the three",
«الأربع» is Wednesday and also "the four". The dialect spellings are kept where they are
overwhelmingly the day in a question about a schedule («التلات»), and dropped where they
are more number than day (bare «الثلاث», «الأربع»). A wrong narrowing here is recoverable
and visible — the parent is shown which day the answer is for and can say otherwise —
whereas failing to recognise the word a parent actually types is silent.

«اليوم» is the same problem in the other direction: it is "today" and also just "the
day", as in «اليوم الدراسي بيخلص امتى» — the school day. Phrases are matched
longest-first, so listing that one as recognised-but-not-a-day is all it takes to keep
it out.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime

from backend.env import school_timezone
from backend.text_matching import name_key

logger = logging.getLogger(__name__)

#: The week's keys, exactly as SIS spells them (`sis.domain.structure.WorkingDay`) and as
#: the facade relays them. Matched against a payload's own day strings, so they cannot be
#: prettied up here.
WEEKDAYS = (
    "saturday", "sunday", "monday", "tuesday", "wednesday", "thursday", "friday",
)

#: `date.weekday()` order — Monday is 0. Written out rather than derived from `WEEKDAYS`
#: above, which is in a school's reading order and not Python's.
_BY_INDEX = (
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
)

#: Phrases that mean a day RELATIVE to now, and how many days from today they are.
_RELATIVE = {
    "today": 0,
    "tomorrow": 1,
    "day after tomorrow": 2,
    "yesterday": -1,
}

#: Every phrase this recognises, folded, mapped to what it means. A phrase mapping to
#: `""` is recognised as NOT a day — see «اليوم الدراسي» in the module docstring.
#:
#: Keys are already folded (`name_key`): أ/إ/آ have become ا, ة has become ه, and the
#: diacritics are gone. Writing them folded means the table is what the matcher actually
#: compares against, rather than something that has to be transformed on the way in and
#: could disagree with itself.
_PHRASES: dict[str, str] = {
    # --- relative, English ---
    "today": "today",
    "tomorrow": "tomorrow",
    "day after tomorrow": "day after tomorrow",
    "yesterday": "yesterday",
    # --- relative, Arabic, Egyptian dialect included ---
    "اليوم": "today",
    "النهارده": "today",
    "انهارده": "today",
    "بكره": "tomorrow",
    "بكرا": "tomorrow",
    "غدا": "tomorrow",
    "بعد بكره": "day after tomorrow",
    "بعد بكرا": "day after tomorrow",
    "بعد غد": "day after tomorrow",
    "بعد الغد": "day after tomorrow",
    "امبارح": "yesterday",
    "امس": "yesterday",
    "البارحه": "yesterday",
    # --- named weekdays, English ---
    **{day: day for day in WEEKDAYS},
    # --- named weekdays, Arabic ---
    "السبت": "saturday",
    "الاحد": "sunday",
    "الحد": "sunday",
    "الاثنين": "monday",
    "الاتنين": "monday",
    "الثلاثاء": "tuesday",
    "الثلاثا": "tuesday",
    "التلاتاء": "tuesday",
    "التلات": "tuesday",
    "الاربعاء": "wednesday",
    "الاربعا": "wednesday",
    "الخميس": "thursday",
    "الجمعه": "friday",
    # --- recognised, and not a day ---
    # "the school day", which every one of these questions is otherwise about.
    "اليوم الدراسي": "",
    "the school day": "",
}

#: The longest phrase above, in tokens. Scanning tries this many first, so «بعد بكره»
#: beats «بكره» and «اليوم الدراسي» beats «اليوم».
_LONGEST = max(len(phrase.split()) for phrase in _PHRASES)

#: A word, in any script. Everything else is a separator — «بكرة؟» is two tokens' worth of
#: characters and one word, and the question mark is the parent's, not part of the day.
_TOKENS = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class AskedDay:
    """One weekday, and how the parent got to it."""

    #: The school's own key — "sunday". What a payload's rows are matched on.
    key: str
    #: The phrase as this module canonicalised it: "tomorrow", or "sunday" when named.
    phrase: str
    #: "today" | "tomorrow" | "yesterday" | "day after tomorrow", or "" when the parent
    #: named the day outright. Wording depends on it: "tomorrow (Sunday)" answers what
    #: was asked, while "Sunday" alone leaves the parent checking a calendar.
    relative: str = ""


def day_phrase(text: str) -> str:
    """The day phrase in `text`, canonicalised — or `""` when it names no day.

    Leftmost-longest over the folded tokens. Whole tokens rather than substrings, which
    is what keeps «الأحد» out of «الواحدة» and costs nothing, since Arabic attaches the
    article to the front of the word and leaves the token intact.

    Split on punctuation and not merely on spaces, because «بكرة؟» is how the question
    actually arrives — a parent ends it with the Arabic question mark and no space, and
    matching whole space-delimited tokens missed every one of them.

    Pure and literal — no model, no clock, no I/O. The same message gives the same
    answer on every run, which is what lets the planner call it and hand the result to a
    tool as a settled argument.
    """
    tokens = _TOKENS.findall(name_key(text or ""))
    if not tokens:
        return ""
    for start in range(len(tokens)):
        for size in range(min(_LONGEST, len(tokens) - start), 0, -1):
            meaning = _PHRASES.get(" ".join(tokens[start : start + size]))
            if meaning is None:
                continue
            # Recognised. A phrase that means no day ends the scan rather than letting a
            # shorter reading of its own words match — «اليوم الدراسي» must not fall
            # through to «اليوم».
            return meaning
    return ""


def today_at_school() -> date:
    """Today where the SCHOOL is, which is the only place a school day happens.

    Not the server's clock and not the parent's: a guardian reading this from another
    timezone still asks about tomorrow at their child's school, and around midnight the
    three answers differ.
    """
    return datetime.now(school_timezone()).date()


def resolve_day(phrase: str, *, today: date | None = None) -> AskedDay | None:
    """The weekday `phrase` names, or None when it names none.

    `phrase` may be a canonical token from `day_phrase` or the parent's own words — both
    go through the same lookup, so a day the planner extracted and a day the model
    repeated out of the message resolve identically. That is the point of routing both
    through one function: two matchers would eventually disagree about «بكره», and the
    disagreement would surface as an answer about the wrong day.

    `today` is injectable so the behaviour is testable without waiting for Tuesday.
    """
    meaning = day_phrase(phrase)
    if not meaning:
        return None
    if meaning in WEEKDAYS:
        # A named day needs no date at all — see the module docstring.
        return AskedDay(key=meaning, phrase=meaning)
    offset = _RELATIVE.get(meaning)
    if offset is None:  # pragma: no cover - every phrase that is not a weekday is relative
        return None
    stamp = date.fromordinal((today or today_at_school()).toordinal() + offset)
    return AskedDay(key=_BY_INDEX[stamp.weekday()], phrase=meaning, relative=meaning)
