"""Decode/render checks for the vendored v2 policy — no checkpoint, no GPU."""

import numpy as np
import pytest
import torch

from leansatp_runtime.core import Premise
from leansatp_runtime.models.policy_v2 import (
    LEMMA_HOST_POOL,
    N_LEMMA_DECISION,
    N_TACTICS,
    TACTIC_POOL,
    PremiseRetriever,
    build_policy_v2,
    decode_decision,
    decode_lemma_decision,
    to_lean4_string,
)


def test_decode_decision_off_and_bounds():
    assert decode_decision(0) is None
    assert decode_decision(-1) is None
    assert decode_decision(1) == ("norm", 0)
    assert decode_decision(3) == ("norm", 2)
    assert decode_decision(4) == ("safe", 0)
    assert decode_decision(7) == ("unsafe", 0)
    assert decode_decision(9) == ("unsafe", 2)
    assert decode_decision(10) is None  # beyond N_TYPE×N_PRIORITY


def test_decode_lemma_decision_off_and_bounds():
    assert decode_lemma_decision(0) is None
    assert decode_lemma_decision(1) == (0, "norm", 0)
    assert decode_lemma_decision(180) == (19, "unsafe", 2)
    assert decode_lemma_decision(N_LEMMA_DECISION) is None  # host 20 out of range


def test_render_all_off_is_bare_head():
    tactic = to_lean4_string([0] * N_TACTICS, None, None, None, None)
    assert tactic == "  aesop"


def test_render_tactic_name_override():
    tactic = to_lean4_string([0] * N_TACTICS, tactic_name="satp")
    assert tactic == "  satp"


def test_render_tactic_rules_and_priorities():
    decisions = [0] * N_TACTICS
    decisions[0] = 1  # ring → norm 100
    decisions[4] = 9  # linarith → unsafe 90%
    tactic = to_lean4_string(decisions)
    assert TACTIC_POOL[0] == "ring" and TACTIC_POOL[4] == "linarith"
    assert tactic == (
        "  aesop\n    (add norm 100 (by ring))\n    (add unsafe 90% (by linarith))"
    )


def test_render_config_skip_default_and_builtin_clause():
    # levels: maxRuleApplications=2400, depth=30 (default → skipped),
    # normIters=520, maxGoals=1024; binary: enableBuiltin=False → rule_sets.
    tactic = to_lean4_string(
        [0] * N_TACTICS,
        config_level=[7, 2, 5, 1],
        config_binary=[1, 1, 1, 1, 0],
    )
    assert tactic == (
        "  aesop (config := {\n"
        "    maxRuleApplications := 2400\n"
        "    maxNormIterations := 520\n"
        "    maxGoals := 1024\n"
        "  }) (rule_sets := [-builtin])"
    )


def test_render_lemma_host_and_ordering():
    premise = Premise.from_leandojo_format(
        "<a>Nat.add_comm</a> theorem Nat.add_comm ..."
    )
    decisions = [0] * N_TACTICS
    decisions[0] = 1  # ring → norm 100 (tactic tier sorts before lemma tier)
    lemma = [0] * 32
    lemma[0] = 1  # host 0 = "apply {L}", norm 100
    tactic = to_lean4_string(decisions, lemma, [premise] + [None] * 31)
    assert LEMMA_HOST_POOL[0] == "apply {L}"
    assert tactic == (
        "  aesop\n"
        "    (add norm 100 (by ring))\n"
        "    (add norm 100 (by apply Nat.add_comm))"
    )


def test_build_rejects_drifted_decode_metadata():
    ckpt = {
        "model_state_dict": {},
        "tactic_pool": {"pool": TACTIC_POOL, "lemma_host_pool": ["apply {L}"]},
    }
    with pytest.raises(RuntimeError, match="lemma_host_pool"):
        build_policy_v2(ckpt, cache_dir=".", device="cpu")


def test_render_lemma_none_premise_skipped():
    lemma = [0] * 32
    lemma[3] = 1  # premise slot is None (padded retrieval) → no rule
    tactic = to_lean4_string([0] * N_TACTICS, lemma, [None] * 32)
    assert tactic == "  aesop"


def test_retriever_rejects_wrong_width_cache(tmp_path):
    np.save(tmp_path / "premise_embeddings.npy", np.zeros((3, 7), dtype=np.float32))
    (tmp_path / "mathlib4_premises.txt").write_text(
        "<a>a</a> x\n<a>b</a> y\n<a>c</a> z\n"
    )
    r = PremiseRetriever(
        str(tmp_path),
        encode_fn=None,
        hidden_size=1472,
        device=torch.device("cpu"),
        embeddings_on_device=False,
    )
    with pytest.raises(ValueError, match="wrong-encoder"):
        r.load()
