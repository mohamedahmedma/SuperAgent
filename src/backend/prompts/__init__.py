"""Prompt templates: the shipped prompt text, composed per turn.

Prompts moved out of the profile YAML because a profile answers "what is this
deployment" and a prompt answers "how does the system talk" — and only the first is
something a domain owner should have to write. A profile now selects and parameterises
prompts; it no longer has to carry them.

## Composition, and why it saves anything

The agent's system prompt is paid on every turn, so it must carry only what is true on
every turn. Everything else is conditional, and there are three places to put a
conditional instruction, in increasing order of how cheap it is:

  1. **The system prompt.** Paid on every turn regardless. Only invariants go here.
  2. **A capability fragment** (this module). Rendered only when the capability is
     bound for THIS turn, so a turn narrowed to search-only never pays for the rest.
  3. **A tool description or a tool result** (backend/tools/). The framework sends a
     description only when its tool is bound, and a result only when the tool ran —
     so anything about one tool, or about what retrieval actually returned, is free
     on every turn it does not apply to.

Rung 3 is why this package is small. Most per-tool instruction text already lives with
its tool; what remains here is the cross-cutting text that no single tool owns.

## Rules for templates

Templates receive **booleans and strings, never objects to interrogate**. A template
that computes `{% if scope == 'out_of_domain' and certainty >= 2 %}` has become policy
that no test covers and no type checker sees. Decisions belong in
backend/chat/turn_policy.py, which is pure and tested; templates only lay out the text
those decisions selected.

Values are passed as render arguments, never rendered as templates themselves — Jinja
does not recursively expand a variable's contents, so a persona containing `{{ ... }}`
renders literally rather than executing. That is what keeps profile-supplied text data
rather than code. If templates ever become user-supplied (as opposed to profile
*selection* being user-driven), this must move to `SandboxedEnvironment`.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

TEMPLATE_ROOT = Path(__file__).parent / "templates"


@lru_cache(maxsize=1)
def _environment() -> Environment:
    """The process-wide template environment. Templates compile once and are cached by
    Jinja itself, so rendering per turn costs a dictionary lookup and a string join."""
    return Environment(
        loader=FileSystemLoader(str(TEMPLATE_ROOT)),
        # The setting that matters most here. Jinja's default renders an unknown
        # variable as an empty string, silently — so a typo in `{{ citation_rule }}`
        # would ship a prompt with the citation contract missing, no error, and asset
        # attribution quietly degraded downstream (it parses [n] out of the answer).
        # StrictUndefined turns that into an exception the tests catch instead.
        undefined=StrictUndefined,
        # These are prompts, not markup. Autoescaping would turn `&` into `&amp;` and
        # quotes into entities inside text a model reads literally.
        autoescape=False,
        # Whitespace control: without these, every `{% if %}` line leaves a blank line
        # behind, and blank lines are tokens.
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render(template_name: str, **context: Any) -> str:
    """Render a template by path relative to `templates/`, e.g. "agent/system.j2"."""
    return _environment().get_template(template_name).render(**context).strip()


def resolve(override: str, template_name: str, **context: Any) -> str:
    """Profile-supplied prompt text when set, otherwise the shipped template.

    This is what makes the migration out of profile YAML incremental and reversible: a
    deployment that already tuned a prompt in its profile keeps that text and never
    notices the template exists, while everything else picks up the shipped one.

    Overrides interpolate `{name}` with str.replace rather than str.format. Three call
    sites used format() before, and format() raises on any literal brace in the prompt
    — a JSON example, a `{}` in a code snippet — which made a perfectly reasonable
    override fail at runtime instead of at load. replace() cannot raise, handles the
    same `{name}` placeholders these prompts actually use, and matches what the other
    sites (persona, extraction prompts) were already doing.
    """
    if override:
        text = override
        for key, value in context.items():
            text = text.replace("{" + key + "}", str(value))
        return text.strip()
    return render(template_name, **context)


def template_names() -> list[str]:
    """Every shipped template. Used by the tests that render all of them."""
    return sorted(_environment().list_templates(extensions=["j2"]))


__all__ = ["TEMPLATE_ROOT", "render", "resolve", "template_names"]
