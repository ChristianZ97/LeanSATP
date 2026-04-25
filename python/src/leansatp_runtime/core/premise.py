"""Premise (lemma/theorem) handling — LeanDojo serialized format."""

import re
from dataclasses import dataclass


@dataclass
class Premise:
    """A premise from LeanDojo serialized format: "<a>full_name</a> code"."""

    full_name: str
    code: str
    raw: str

    @classmethod
    def from_leandojo_format(cls, serialized: str) -> "Premise":
        match = re.match(r"<a>(.+?)</a>\s*(.*)", serialized, re.DOTALL)
        if match:
            return cls(
                full_name=match.group(1).strip(),
                code=match.group(2).strip(),
                raw=serialized,
            )
        return cls(full_name=serialized.strip(), code="", raw=serialized)

    def __str__(self) -> str:
        return self.full_name

    def __repr__(self) -> str:
        return f"Premise({self.full_name})"


def premise_name(p) -> str:
    """Extract `full_name` from a Premise-like object, with empty-string fallback."""
    if p is None:
        return ""
    return getattr(p, "full_name", str(p))
