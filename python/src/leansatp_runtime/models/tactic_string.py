"""Render policy action vectors into a Lean 4 aesop tactic string."""

from typing import List, Optional

from ..core import Premise
from ..core.premise import premise_name
from .components.heads import SAFE_TACTICS, UNSAFE_TACTICS

# Default Aesop config (referenced when level/binary actions are absent).
_DEFAULT_AESOP_CONFIG = {
    "maxRuleApplicationDepth": 30,
    "maxRuleApplications": 200,
    "maxNormIterations": 100,
    "enableSimp": True,
    "useSimpAll": True,
}

# Config levels are offset in steps of LEVEL_STEP from the default.
_LEVEL_STEP = 20


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
    """
    if not tactic_name or not tactic_name.strip():
        raise ValueError("tactic_name must be a non-empty string")
    tactic_name = tactic_name.strip()

    safe_actions = _to_list(safe_actions)
    unsafe_actions = _to_list(unsafe_actions)
    lemma_actions = _to_list(lemma_actions)
    config_level_actions = _to_list(config_level_actions)
    config_binary_actions = _to_list(config_binary_actions)

    if config_level_actions is not None and len(config_level_actions) >= 3:
        max_depth = (
            _DEFAULT_AESOP_CONFIG["maxRuleApplicationDepth"]
            + _LEVEL_STEP * config_level_actions[0]
        )
        max_apps = (
            _DEFAULT_AESOP_CONFIG["maxRuleApplications"]
            + _LEVEL_STEP * config_level_actions[1]
        )
        max_norm = (
            _DEFAULT_AESOP_CONFIG["maxNormIterations"]
            + _LEVEL_STEP * config_level_actions[2]
        )
    else:
        max_depth = _DEFAULT_AESOP_CONFIG["maxRuleApplicationDepth"]
        max_apps = _DEFAULT_AESOP_CONFIG["maxRuleApplications"]
        max_norm = _DEFAULT_AESOP_CONFIG["maxNormIterations"]

    if config_binary_actions is not None and len(config_binary_actions) >= 2:
        enable_simp = bool(config_binary_actions[0])
        use_simp_all = bool(config_binary_actions[1])
    else:
        enable_simp = _DEFAULT_AESOP_CONFIG["enableSimp"]
        use_simp_all = _DEFAULT_AESOP_CONFIG["useSimpAll"]

    aesop_config = f"""  {tactic_name} (config := {{
    maxRuleApplicationDepth := {max_depth}
    maxRuleApplications     := {max_apps}
    maxNormIterations       := {max_norm}
    enableSimp              := {"true" if enable_simp else "false"}
    useSimpAll              := {"true" if use_simp_all else "false"}
  }})"""

    rule_entries: List[tuple] = []

    # Safe rules: priority 0 = disabled, 1-4 → Lean priority 4,3,2,1.
    for idx, priority in enumerate(safe_actions):
        if priority > 0:
            name = SAFE_TACTICS[idx]
            rule = f"    (add safe {5 - priority} (by {name}))"
            rule_entries.append((0, -priority, name, rule))

    # Unsafe rules: priority 0 disabled, 1-4 → 70/80/90/100%.
    UNSAFE_PROB = {1: 70, 2: 80, 3: 90, 4: 100}
    for idx, priority in enumerate(unsafe_actions):
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
