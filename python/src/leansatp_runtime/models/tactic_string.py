"""Render policy action vectors into a Lean 4 aesop tactic string."""

from typing import List, Optional

from ..core import Premise
from ..core.premise import premise_name
from .components.heads import (
    CONFIG_BINARY_KEYS,
    CONFIG_LEVEL_KEYS,
    DEFAULT_AESOP_CONFIG,
    LEVEL_VALUES,
    SAFE_TACTICS,
    UNSAFE_TACTICS,
)


def _to_list(x):
    if x is None:
        return None
    return x.tolist() if hasattr(x, "tolist") else x


def to_lean4_string(
    safe_actions,
    unsafe_actions,
    lemma_actions=None,
    lemma_premises: Optional[List[Premise]] = None,
    config_level_actions=None,
    config_binary_actions=None,
    tactic_name: str = "aesop",
) -> str:
    """Render action vectors to a complete `<tactic_name> (config := {...})` block.

    `tactic_name` lets callers emit `satp (config := …)` for the SATP tactic
    cascade while keeping the same configuration grammar as plain `aesop`.

    Config-emission follows the trainer's skip-when-default canonical form:
    only fields whose value differs from ``DEFAULT_AESOP_CONFIG[key]`` are
    written into the config block, so the tactic string round-trips losslessly
    through the buffer's ``from_lean4_string``.
    """
    if not tactic_name or not tactic_name.strip():
        raise ValueError("tactic_name must be a non-empty string")
    tactic_name = tactic_name.strip()

    safe_actions = _to_list(safe_actions)
    unsafe_actions = _to_list(unsafe_actions)
    lemma_actions = _to_list(lemma_actions)
    config_level_actions = _to_list(config_level_actions)
    config_binary_actions = _to_list(config_binary_actions)

    level_values: dict = {}
    for i, key in enumerate(CONFIG_LEVEL_KEYS):
        if config_level_actions is not None and i < len(config_level_actions):
            level_values[key] = LEVEL_VALUES[key][int(config_level_actions[i])]
        else:
            level_values[key] = DEFAULT_AESOP_CONFIG[key]

    binary_values: dict = {}
    for i, key in enumerate(CONFIG_BINARY_KEYS):
        if config_binary_actions is not None and i < len(config_binary_actions):
            binary_values[key] = bool(config_binary_actions[i])
        else:
            binary_values[key] = bool(DEFAULT_AESOP_CONFIG[key])

    config_lines: List[str] = []
    for key in CONFIG_LEVEL_KEYS:
        v = level_values[key]
        default = DEFAULT_AESOP_CONFIG[key]
        if v is None and default is None:
            continue
        if v == default:
            continue
        config_lines.append(f"    {key:<23} := {v}")
    for key in CONFIG_BINARY_KEYS:
        v = binary_values[key]
        if v == DEFAULT_AESOP_CONFIG[key]:
            continue
        config_lines.append(f"    {key:<23} := {'true' if v else 'false'}")

    if not config_lines:
        aesop_config = f"  {tactic_name}"
    else:
        body = "\n".join(config_lines)
        aesop_config = f"  {tactic_name} (config := {{\n{body}\n  }})"

    rule_entries: List[tuple] = []

    # Safe rules: priority 0 = disabled, 1-4 → Lean priority 4,3,2,1.
    for idx, priority in enumerate(safe_actions or []):
        if priority > 0:
            name = SAFE_TACTICS[idx]
            rule = f"    (add safe {5 - priority} (by {name}))"
            rule_entries.append((0, -priority, name, rule))

    # Unsafe rules: priority 0 disabled, 1-4 → 70/80/90/100%.
    UNSAFE_PROB = {1: 70, 2: 80, 3: 90, 4: 100}
    for idx, priority in enumerate(unsafe_actions or []):
        if priority > 0:
            name = UNSAFE_TACTICS[idx]
            rule = f"    (add unsafe {UNSAFE_PROB[priority]}% (by {name}))"
            rule_entries.append((1, -priority, name, rule))

    # Lemma rules: priority 0 disabled, 1-4 → 10/20/30/40%.
    if lemma_actions and lemma_premises:
        for priority, premise in zip(lemma_actions, lemma_premises):
            if priority <= 0 or premise is None:
                continue
            name = premise_name(premise)
            if not name.strip():
                continue
            pct = priority * 10
            rule = (
                f"    (add unsafe {pct}% (by first | apply {name} | "
                f"rw [{name}] | simp only [{name}]))"
            )
            rule_entries.append((2, -priority, name, rule))

    if not rule_entries:
        return aesop_config

    rule_entries.sort(key=lambda x: (x[0], x[1], x[2]))
    return aesop_config + "\n" + "\n".join(r[3] for r in rule_entries)
