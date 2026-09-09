"""The Harmony transcript format, and how to get an answer back out of it.

Format knowledge only. Nothing here knows what a turn is, which messages matter, or
what a user is allowed to see — that is policy, and it lives in
`backend/chat/finalize.py`. This module is pure and takes no context, for the same
reason `backend/chat/child_resolution.py` is: every rule below is then a unit test
rather than a live conversation somebody has to reproduce.

`MODEL=openai/gpt-oss-*` does not emit a plain string. It emits the OpenAI **Harmony**
transcript, which carries three channels in one response:

    analysis    the model's private reasoning
    commentary  tool calls and their preambles
    final       the answer, and the only channel a user may ever see

A provider is supposed to parse that and hand back `content` holding the final channel
alone. Together's endpoint does not do it reliably: measured against
`openai/gpt-oss-20b` at the school profile's own `temperature=0.2`, the analysis channel
reached `content` in **4 of 5** identical calls, in three different shapes —

  * the whole envelope, literal tokens included:
        `<|channel|>analysis<|message|>We have two chunks...<|end|>`
        `<|start|>assistant<|channel|>final<|message|>عذرًا...`
  * the tokens eaten, a bare channel header left behind:
        `commentary to=functions.search_knowledge_base analysisNo data.عذرًا...`
  * nothing but the reasoning prose, run straight into the answer with no separator:
        `We need to see the result.مصاريف ابنك تختلف...`

The first two are marked and this module strips them. The third has no marker, so no
text rule can find the seam without risking a cut through a real answer — it is handled
in `finalize.py` instead, by where it occurs rather than by what it says.

Text with no Harmony token in it is returned unchanged, byte for byte. A provider that
parses its own format correctly, or a model that never spoke Harmony, must not pay for
this and must not be at risk from it.
"""
from __future__ import annotations

import re

#: The Harmony control tokens. Every one begins `<|`, which is what lets the streaming
#: hold-back below look only at the tail after the last `<`.
TOKENS = (
    "<|start|>",
    "<|end|>",
    "<|message|>",
    "<|channel|>",
    "<|constrain|>",
    "<|call|>",
    "<|return|>",
)

_MAX_TOKEN = max(len(token) for token in TOKENS)

#: Channels the format defines. Anything else in a header is a role or a recipient.
_CHANNELS = ("analysis", "commentary", "final")

#: The only channel whose body is the answer.
_FINAL = "final"

#: A header that survived with its tokens stripped. `to=functions.` is the anchor and it
#: is required: it cannot occur in Arabic or English prose, so requiring it is what keeps
#: this from eating an answer that merely opens with the word "analysis". Anchored at the
#: start, because a header only ever appears where a message begins.
_BARE_HEADER = re.compile(
    r"^\s*(?:analysis|commentary|final)?\s*to=functions\.[\w.-]+\s*"
    r"(?:<\|constrain\|>\w*)?\s*(?:analysis|commentary|final)?\s*"
)


#: A channel or role word with its `<|…|>` delimiters eaten, glued straight onto what
#: follows it — `analysisThe tool call failed`, `finalرسوم الصف`, `assistantcommentary`.
#: The join is the signal: prose puts a space or a stop after "analysis", and a model
#: writing "Finally" continues the word in lowercase Latin, which the lookahead excludes.
_GLUED = r"(?:analysis|commentary|final|assistant)"
_GLUED_NEXT = r"(?=[A-Z؀-ۿ\d{\[\"]|" + _GLUED + r")"
_GLUED_MARKER = re.compile(r"(?:^|(?<=[\s.}\"']))" + _GLUED + _GLUED_NEXT)
_GLUED_FINAL = re.compile(r"(?:^|(?<=[\s.}\"']))final" + _GLUED_NEXT)


def has_harmony_markup(text: str) -> bool:
    """Whether `text` carries anything this module would act on."""
    if not text:
        return False
    return (
        any(token in text for token in TOKENS)
        or bool(_BARE_HEADER.match(text))
        or bool(_GLUED_MARKER.search(text))
    )


def split_glued_transcript(text: str) -> str:
    """Recover the answer from a transcript whose `<|…|>` delimiters were eaten.

    The measured shape the other rules miss. Two real captures, both from the live
    provider on the question that started all of this:

        analysisThe tool call failed due to missing query field. We need to call
        search_knowledge_base with query.assistantcommentary
        to=functions.search_knowledge_basejson{"query":"مصاريف ابني"}

        finalرسوم الصف الأول الابتدائي للعام 2026 هي 30,000 جنيه على ثلاث دفعات. [1]

    Neither carries a token, so the state machine never engages, and neither opens with
    `to=functions.`, so `_BARE_HEADER` does not match. What they do carry is the channel
    NAME, welded to the text that followed it — and Harmony says what that means:

      * text after the last `final` marker is the answer, and everything before it is
        the transcript that led there;
      * markers with NO `final` among them mean the model never opened an answer
        channel. The content is analysis and a fabricated tool call. There is no answer
        in it to recover, so returning none is the honest result rather than handing
        back reasoning with its label removed — which would only make the leak harder
        to see.

    Text with no glued marker is returned unchanged.
    """
    if not text:
        return text
    finals = list(_GLUED_FINAL.finditer(text))
    if finals:
        return text[finals[-1].end():]
    if _GLUED_MARKER.search(text):
        return ""
    return text


class HarmonyFilter:
    """Keeps the final channel out of a Harmony transcript, one stream chunk at a time.

    Stateful because the tokens split across chunks: `<|chan` can end one delta and
    `nel|>` begin the next, and a filter that judged each chunk alone would emit the
    halves. `feed` therefore holds back any tail that could still become a token and
    releases it on the following chunk, or at `flush`.

    Starts in the emitting state, so a response containing no Harmony markup at all
    passes through untouched — the common case, and the one that must not regress.
    """

    __slots__ = (
        "_buf", "_emitting", "_in_header", "_header",
        "_saw_markup", "_settled", "_transcript",
    )

    def __init__(self) -> None:
        self._buf = ""
        # Emit until told otherwise: plain content is a body, not a header.
        self._emitting = True
        self._in_header = False
        self._header = ""
        self._saw_markup = False
        self._settled = False
        # Holding a whole message that looks like a transcript, waiting to learn whether
        # it ever opens a final channel. See `_settle_header`.
        self._transcript = False

    def feed(self, text: str) -> str:
        """Consume one delta; return the part of it a user may see (often empty)."""
        if not text:
            return ""
        self._buf += text
        if self._transcript:
            # This message opened with channel markers and no answer channel yet. Hold
            # everything — and keep the buffer INTACT while holding, because the decision
            # is about the message as a whole. An earlier revision emptied it on entering
            # this state, so each later delta arrived looking like fresh prose, exited the
            # hold, and published the remainder of a fabricated transcript.
            finals = list(_GLUED_FINAL.finditer(self._buf))
            if not finals:
                return ""
            self._buf = self._buf[finals[-1].end():]
            self._transcript = False
            return self._drain(hold_partial=True)
        # The bare-header form carries no token to trigger the state machine, so it is
        # stripped off the front of the message before anything else looks at the buffer.
        #
        # Nothing is emitted until that question is settled, and the hold is the whole
        # point: a header is only recognisable once all ~45 characters of it have
        # arrived, and this stream delivers one or two characters at a time. Draining
        # eagerly published "commentary to=functions.search_knowledge_base" a character
        # at a time and then had nothing left to strip — the regex matched a buffer the
        # reader had already seen. Measured: correct at 64-character chunks, broken at
        # every size from 1 to 11.
        #
        # It costs the first `_HEADER_DECIDED_AFTER` characters of a message, once, and
        # only until the answer is longer than a header could be.
        if not self._settled:
            if "<|" in self._buf or len(self._buf) >= _HEADER_DECIDED_AFTER:
                self._settle_header()
                if self._transcript:
                    # Settling put the message into the hold. Draining here would
                    # publish the very transcript the hold exists to withhold.
                    return ""
            else:
                return ""
        return self._drain(hold_partial=True)

    def _settle_header(self) -> None:
        """Decide once whether the buffer opens with a bare header, and strip it if so.

        Deferred until the buffer is longer than any header can be, because the pattern
        will happily match an INCOMPLETE one: at 33 characters
        `to=functions.search_kn` satisfies `[\\w.-]+`, so an eager attempt stripped a
        half-read header, declared the question settled, and then published the rest of
        it — `owledge_base` — as the opening of the answer. Matching a prefix of a
        pattern is not matching the pattern; waiting is what makes the difference
        observable.
        """
        self._settled = True
        stripped = _BARE_HEADER.sub("", self._buf, count=1)
        if stripped != self._buf:
            self._saw_markup = True
            self._buf = stripped
        # And the delimiter-eaten form, which carries no token and no `to=functions.`
        # anchor for the rule above to find. Same hold applies: the decision needs the
        # whole opening of the message, which is why it lives here.
        recovered = split_glued_transcript(self._buf)
        if recovered != self._buf:
            self._saw_markup = True
            if recovered:
                # A final channel was found; what follows it is the answer.
                self._buf = recovered
            else:
                # Markers but no final channel YET. The message may still open one, so
                # the buffer is KEPT and re-examined on every delta rather than dropped.
                self._transcript = True

    def flush(self) -> str:
        """Release whatever is still held. Call once, when the message ends."""
        if not self._settled:
            # A message shorter than a header is still a message. Settle it now, on
            # everything that arrived, rather than emitting a buffer nobody inspected.
            self._settle_header()
        if self._transcript:
            # The message ended without ever opening a final channel. It carried
            # reasoning and a fabricated tool call and no answer, so it contributes none.
            self._buf = ""
            return ""
        out = self._drain(hold_partial=False)
        self._buf = ""
        return out

    @property
    def saw_markup(self) -> bool:
        """Whether this message carried Harmony markup. For the trace, not for policy."""
        return self._saw_markup

    def _drain(self, *, hold_partial: bool) -> str:
        out: list = []
        while self._buf:
            index, token = self._next_token(self._buf)
            if token is None:
                break
            # Everything before the token belongs to whatever region we are in now.
            self._consume(self._buf[:index], out)
            self._buf = self._buf[index + len(token) :]
            self._saw_markup = True
            self._settled = True
            self._apply(token)

        if hold_partial:
            keep = self._partial_token_len(self._buf)
            if keep:
                self._consume(self._buf[: len(self._buf) - keep], out)
                self._buf = self._buf[len(self._buf) - keep :]
                return "".join(out)
        self._consume(self._buf, out)
        self._buf = ""
        return "".join(out)

    @staticmethod
    def _next_token(buf: str):
        best_index, best_token = -1, None
        for token in TOKENS:
            found = buf.find(token)
            if found != -1 and (best_index == -1 or found < best_index):
                best_index, best_token = found, token
        return best_index, best_token

    def _consume(self, text: str, out: list) -> None:
        if not text:
            return
        if self._in_header:
            self._header += text
        elif self._emitting:
            out.append(text)

    def _apply(self, token: str) -> None:
        if token in ("<|channel|>", "<|start|>"):
            # A new header begins. `<|start|>` carries the role and `<|channel|>` the
            # channel, and a message may carry both — so the buffer is reset by whichever
            # comes second, leaving the channel name as the last thing written.
            self._in_header = True
            self._header = ""
            self._emitting = False
        elif token == "<|constrain|>":
            self._in_header = True
        elif token == "<|message|>":
            # The header is complete; it decides whether this body is the answer.
            self._emitting = self._header_is_final(self._header)
            self._in_header = False
            self._header = ""
        elif token in ("<|end|>", "<|call|>", "<|return|>"):
            # Between messages. Nothing here is a body, so nothing here is emitted.
            self._in_header = False
            self._header = ""
            self._emitting = False

    @staticmethod
    def _header_is_final(header: str) -> bool:
        """Whether the body after this header is the answer.

        Permissive on purpose. A header naming a channel is believed; a header naming
        none — a role on its own, which some parsers leave behind — is read as the final
        channel rather than dropped. Suppressing a real answer is a worse failure than
        letting an unlabelled line through, and every leak this module was built from
        labelled its channel.
        """
        cleaned = re.sub(r"to=\S+", " ", header)
        named = [word for word in re.findall(r"[a-z]+", cleaned.lower()) if word in _CHANNELS]
        if not named:
            return True
        return named[-1] == _FINAL

    @staticmethod
    def _partial_token_len(buf: str) -> int:
        """How many trailing characters could still grow into a token."""
        tail = buf[-(_MAX_TOKEN - 1) :] if len(buf) >= _MAX_TOKEN else buf
        start = tail.rfind("<")
        if start == -1:
            return 0
        candidate = tail[start:]
        return len(candidate) if any(t.startswith(candidate) for t in TOKENS) else 0


#: How much text has to arrive before the leading-header question is decided. It must
#: exceed the LONGEST header the format can produce — a channel word, a `to=functions.`
#: recipient, an optional `<|constrain|>json`, and another channel word run to roughly
#: seventy characters — because deciding early means deciding against a partial match.
#: Only the first chunk of a message waits, and only until the answer is longer than a
#: header could have been.
_HEADER_DECIDED_AFTER = 96


def strip_harmony(text: str) -> str:
    """One-shot form of `HarmonyFilter`, for a response that is already complete."""
    if not text or not has_harmony_markup(text):
        return text
    harmony = HarmonyFilter()
    return (harmony.feed(text) + harmony.flush()).strip()


__all__ = [
    "HarmonyFilter",
    "TOKENS",
    "has_harmony_markup",
    "split_glued_transcript",
    "strip_harmony",
]
