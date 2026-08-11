"""Attribute schema: one declaration, four derived artifacts.

A domain's attribute vocabulary is declared once in its profile:

    attributes:
      - {name: color,    type: string,  multi: true, values: [red, blue, ...]}
      - {name: price,    type: number,  unit: EGP}
      - {name: in_stock, type: boolean}

and everything downstream is generated from it:

    build_extraction_model()  -> the structured-output schema the extractor asks for
    build_filter_model()      -> the tool argument schema the agent fills in
    normalize()               -> coercion and closed-vocabulary validation at ingest
    matches() / SQL           -> filter evaluation at query time

That is what makes the system domain-agnostic. Selling shoes and cataloguing lab
equipment differ by a YAML block, not by a code path — and because the filter model is
generated from the same declaration, the agent literally cannot invent a filter key
that the store does not index.
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

from pydantic import BaseModel, ConfigDict, Field, create_model

logger = logging.getLogger(__name__)


class AttributeType(str, Enum):
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"


class AttributeSpec(BaseModel):
    """One attribute in a domain's vocabulary."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    type: AttributeType = AttributeType.STRING
    description: str = ""
    # Closed vocabulary. When set, extraction is constrained to these values and a
    # filter naming anything else is rejected — which is what stops "crimson" and
    # "red" splitting one facet in two.
    values: List[str] = Field(default_factory=list)
    # True when one entity can legitimately carry several values (colours, materials).
    multi: bool = False
    filterable: bool = True
    unit: str = ""

    def python_type(self) -> type:
        return {
            AttributeType.STRING: str,
            AttributeType.NUMBER: float,
            AttributeType.BOOLEAN: bool,
        }[self.type]

    def prompt_line(self) -> str:
        parts = [f"- {self.name} ({self.type.value}"]
        if self.multi:
            parts.append(", one or more")
        parts.append(")")
        line = "".join(parts)
        if self.description:
            line += f": {self.description}"
        if self.values:
            line += f" — one of: {', '.join(self.values)}"
        if self.unit:
            line += f" (unit: {self.unit})"
        return line


class NumberRange(BaseModel):
    """Inclusive bounds. Either end may be omitted for an open-ended filter."""

    model_config = ConfigDict(extra="forbid")

    min: Optional[float] = None
    max: Optional[float] = None

    def contains(self, value: float) -> bool:
        if self.min is not None and value < self.min:
            return False
        if self.max is not None and value > self.max:
            return False
        return True

    def is_empty(self) -> bool:
        return self.min is None and self.max is None


class AttributeSchema:
    """The compiled vocabulary. Built once per profile and reused."""

    def __init__(self, specs: Iterable[AttributeSpec]):
        self._specs: Dict[str, AttributeSpec] = {}
        for spec in specs:
            if spec.name in self._specs:
                raise ValueError(f"Duplicate attribute {spec.name!r} in the schema")
            self._specs[spec.name] = spec

    @classmethod
    def from_config(cls, entities_config) -> "AttributeSchema":
        return cls(entities_config.attributes)

    def __len__(self) -> int:
        return len(self._specs)

    def __bool__(self) -> bool:
        return bool(self._specs)

    @property
    def specs(self) -> List[AttributeSpec]:
        return list(self._specs.values())

    def get(self, name: str) -> Optional[AttributeSpec]:
        return self._specs.get(name)

    def names(self) -> List[str]:
        return list(self._specs)

    def filterable(self) -> List[AttributeSpec]:
        return [spec for spec in self._specs.values() if spec.filterable]

    # -- prompt -----------------------------------------------------------------

    def describe(self) -> str:
        """The vocabulary, rendered for an extraction prompt."""
        return "\n".join(spec.prompt_line() for spec in self._specs.values())

    # -- generated models -------------------------------------------------------

    def build_extraction_model(self, name: str = "EntityAttributes") -> Type[BaseModel]:
        """Structured-output schema for the extractor.

        Every field is optional: an extractor that cannot see a value must be able to
        omit it rather than being forced to invent one.
        """
        fields: Dict[str, tuple] = {}
        for spec in self._specs.values():
            annotation = List[spec.python_type()] if spec.multi else Optional[spec.python_type()]
            default = Field(
                default_factory=list if spec.multi else None,
                description=spec.prompt_line().lstrip("- "),
            ) if spec.multi else Field(default=None, description=spec.prompt_line().lstrip("- "))
            fields[spec.name] = (annotation, default)
        return create_model(name, __config__=ConfigDict(extra="ignore"), **fields)

    def build_filter_model(self, name: str = "EntityFilters") -> Type[BaseModel]:
        """Tool-argument schema the agent fills in.

        Generated from the same declaration as extraction, so the agent sees the real
        vocabulary — including closed value lists — in the tool signature it was going
        to call anyway. That is what makes attribute filtering cost zero extra LLM
        calls: the slot filling happens inside a call already being made.
        """
        fields: Dict[str, tuple] = {}
        for spec in self.filterable():
            if spec.type is AttributeType.NUMBER:
                annotation, description = Optional[NumberRange], f"Range filter on {spec.name}"
                if spec.unit:
                    description += f" in {spec.unit}"
            elif spec.type is AttributeType.BOOLEAN:
                annotation, description = Optional[bool], f"Exact match on {spec.name}"
            else:
                # Always a list: "any of these" is the natural shape for a catalogue
                # filter and collapses the one-vs-many case for the model.
                annotation = Optional[List[str]]
                description = f"Match any of these {spec.name} values"
                if spec.values:
                    description += f" (allowed: {', '.join(spec.values)})"
            fields[spec.name] = (annotation, Field(default=None, description=description))
        return create_model(name, __config__=ConfigDict(extra="forbid"), **fields)

    # -- normalisation ----------------------------------------------------------

    def normalize(self, raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Coerce extracted values onto the schema, dropping anything unusable.

        Silently dropping is deliberate: a model that returns "burgundy" for a closed
        colour vocabulary has produced a value nothing can filter on, and storing it
        would create a facet that looks real but matches no query.
        """
        if not isinstance(raw, dict):
            return {}
        normalized: Dict[str, Any] = {}
        for key, value in raw.items():
            spec = self._specs.get(key)
            if spec is None or value is None:
                continue
            coerced = self._coerce(spec, value)
            if coerced is not None and coerced != []:
                normalized[key] = coerced
        return normalized

    def _coerce(self, spec: AttributeSpec, value: Any) -> Any:
        if spec.multi:
            values = value if isinstance(value, (list, tuple, set)) else [value]
            coerced = [self._coerce_single(spec, item) for item in values]
            return [item for item in coerced if item is not None]
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        return self._coerce_single(spec, value)

    def _coerce_single(self, spec: AttributeSpec, value: Any) -> Any:
        if value is None:
            return None
        try:
            if spec.type is AttributeType.NUMBER:
                return float(value)
            if spec.type is AttributeType.BOOLEAN:
                if isinstance(value, bool):
                    return value
                text = str(value).strip().lower()
                if text in ("true", "yes", "1", "in stock"):
                    return True
                if text in ("false", "no", "0", "out of stock"):
                    return False
                return None
            text = str(value).strip()
            if not text:
                return None
            if spec.values:
                lowered = text.lower()
                for allowed in spec.values:
                    if allowed.lower() == lowered:
                        return allowed  # canonical casing from the vocabulary
                return None
            return text
        except (TypeError, ValueError):
            return None

    # -- evaluation -------------------------------------------------------------

    def matches(self, attributes: Optional[Dict[str, Any]], filters: Optional[Dict[str, Any]]) -> bool:
        """Whether an entity's attributes satisfy a filter set (AND across keys)."""
        if not filters:
            return True
        attributes = attributes or {}
        for key, condition in filters.items():
            if condition is None:
                continue
            spec = self._specs.get(key)
            if spec is None:
                continue
            if not self._matches_one(spec, attributes.get(key), condition):
                return False
        return True

    def _matches_one(self, spec: AttributeSpec, value: Any, condition: Any) -> bool:
        # A filtered attribute the entity simply does not have is a non-match: a
        # shopper asking for red shoes should not be shown items of unknown colour.
        if value is None or value == []:
            return False

        if spec.type is AttributeType.NUMBER:
            bounds = condition if isinstance(condition, NumberRange) else NumberRange.model_validate(condition)
            if bounds.is_empty():
                return True
            values = value if isinstance(value, list) else [value]
            return any(bounds.contains(float(item)) for item in values)

        if spec.type is AttributeType.BOOLEAN:
            return bool(value) == bool(condition)

        wanted = condition if isinstance(condition, (list, tuple, set)) else [condition]
        wanted_lower = {str(item).strip().lower() for item in wanted if item is not None}
        if not wanted_lower:
            return True
        held = value if isinstance(value, list) else [value]
        return any(str(item).strip().lower() in wanted_lower for item in held)

    def validate_filters(self, filters: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[str]]:
        """Split a filter payload into (usable, rejected-reasons).

        Rejections are returned rather than raised so the tool can tell the agent what
        it got wrong and let it retry, instead of failing the user's turn.
        """
        usable: Dict[str, Any] = {}
        problems: List[str] = []
        for key, condition in (filters or {}).items():
            if condition is None:
                continue
            spec = self._specs.get(key)
            if spec is None:
                problems.append(f"unknown attribute {key!r} (known: {', '.join(self.names()) or 'none'})")
                continue
            if not spec.filterable:
                problems.append(f"{key!r} is not filterable")
                continue
            if spec.type is AttributeType.NUMBER:
                try:
                    bounds = condition if isinstance(condition, NumberRange) else NumberRange.model_validate(condition)
                except Exception:
                    problems.append(f"{key!r} expects a range like {{min, max}}")
                    continue
                if not bounds.is_empty():
                    usable[key] = bounds
                continue
            if spec.type is AttributeType.BOOLEAN:
                usable[key] = bool(condition)
                continue

            wanted = condition if isinstance(condition, (list, tuple, set)) else [condition]
            cleaned = [str(item).strip() for item in wanted if item is not None and str(item).strip()]
            if spec.values:
                allowed = {item.lower(): item for item in spec.values}
                canonical = [allowed[item.lower()] for item in cleaned if item.lower() in allowed]
                rejected = [item for item in cleaned if item.lower() not in allowed]
                if rejected:
                    problems.append(
                        f"{key!r} does not accept {', '.join(repr(item) for item in rejected)} "
                        f"(allowed: {', '.join(spec.values)})"
                    )
                cleaned = canonical
            if cleaned:
                usable[key] = cleaned
        return usable, problems


def build_attribute_schema(entities_config) -> AttributeSchema:
    return AttributeSchema.from_config(entities_config)
