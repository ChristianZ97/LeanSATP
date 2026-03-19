# core/premise.py
"""
Premise (Lemma) Handling Module v1.1.0 (Simplified Single GPU)

Handles LeanDojo format premises.
Removed all distributed training code for simplicity.
"""

import re
from typing import List
from dataclasses import dataclass


@dataclass
class Premise:
    """
    Represents a premise (lemma/theorem) from LeanDojo format.

    Attributes:
        full_name: Fully qualified name (e.g., "Nat.add_comm")
        code: The code/signature of the premise
        raw: Original serialized string
    """

    full_name: str
    code: str
    raw: str

    @classmethod
    def from_leandojo_format(cls, serialized: str) -> "Premise":
        """
        Parse a premise from LeanDojo serialized format.

        Format: "<a>full_name</a> code"
        """
        match = re.match(r"<a>(.+?)</a>\s*(.*)", serialized, re.DOTALL)
        if match:
            full_name = match.group(1).strip()
            code = match.group(2).strip()
            return cls(full_name=full_name, code=code, raw=serialized)
        else:
            return cls(full_name=serialized.strip(), code="", raw=serialized)

    def __str__(self) -> str:
        return self.full_name

    def __repr__(self) -> str:
        return f"Premise({self.full_name})"


def load_premises(filepath: str) -> List[Premise]:
    """
    Load premises from a text file (one per line in LeanDojo format).
    """
    premises = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                premises.append(Premise.from_leandojo_format(line))
    print(f"Loaded {len(premises)} premises from {filepath}")
    return premises
