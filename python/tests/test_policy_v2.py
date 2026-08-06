"""Decode/render checks for the vendored v2 policy — no checkpoint, no GPU."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from leansatp_runtime.core import Premise
from leansatp_runtime.hf_pin import HF_REPO, REVISION
from leansatp_runtime.models.policy_v2 import (
    LEMMA_HOST_POOL,
    LEMMA_K,
    MAX_SEQUENCE_LENGTH,
    N_LEMMA_DECISION,
    N_TACTICS,
    TACTIC_POOL,
    TYPE_PRIORITY_VALUES,
    PremiseRetriever,
    _INFER,
    build_policy_v2,
    decode_decision,
    decode_lemma_decision,
    to_lean4_string,
)
from leansatp_runtime.service import DEFAULT_CHECKPOINT

# ── Era anchors ────────────────────────────────────────────────────────────
# Everything else in this file derives its expectations from the constants the
# pin happens to supply, so the suite is internally consistent under *any*
# era — including one where the pin points somewhere nobody intended. These
# two tests are the only place an era is asserted, and they are meant to fail
# on a bump so the bumper is forced to update them deliberately.


def test_pin_is_the_v427_era():
    """The pin names one artifact set, and the loaded decode surface is its own.

    Needs no checkpoint, so unlike the test below it also runs in CI.

    Deliberately not skipped under SATP_INFER_SOURCE: pointing the decode
    source at another era while the pin still says v4.27 is precisely the
    mismatch this test exists to catch, and it is invisible everywhere else
    (verified 2026-08-06 — the rest of the suite passes unchanged against
    v2's infer.py). A local copy of the *pinned* file is byte-identical and
    still passes, so offline work is not penalised.

    REVISION is asserted, not just HF_REPO: it is env-overridable
    (SATP_HF_REVISION), and two revisions of the same repo are two different
    training runs. The digest covers pool *contents*, which lengths alone
    miss — a 19-host pool with one host renamed is still 19 hosts.

    What this cannot do: satp-policy-v2-alphaproof ships a byte-identical
    decode surface, so no assertion over constants can separate it. Only the
    REVISION literal and the checkpoint's own run id can — hence both.
    """
    assert (HF_REPO, REVISION) == (
        "ChristianZ97/satp-policy-v4.27",
        "5a4d2f1bd6731dd5b68ad95264094284206e609f",
    ), "hf_pin.py no longer names the v4.27 artifact set this suite was written for"
    assert (
        len(LEMMA_HOST_POOL),
        N_LEMMA_DECISION,
        len(TACTIC_POOL),
        LEMMA_K,
        MAX_SEQUENCE_LENGTH,
    ) == (19, 172, 28, 32, 1024)
    # v2's 20th host; its absence is what makes this era's pool a strict prefix
    assert "positivity [{L}]" not in LEMMA_HOST_POOL

    surface = json.dumps(
        [list(LEMMA_HOST_POOL), list(TACTIC_POOL), TYPE_PRIORITY_VALUES],
        sort_keys=True,
    )
    digest = hashlib.sha256(surface.encode()).hexdigest()[:16]
    assert digest == "c862d653945539f7", (
        f"decode surface loaded from {_INFER.__file__} is not v4.27's "
        f"(digest {digest}), but the pin says {HF_REPO}@{REVISION[:8]}"
    )


@pytest.mark.skipif(
    not Path(DEFAULT_CHECKPOINT).exists(), reason="no checkpoint in the cache dir"
)
def test_local_checkpoint_is_the_pinned_file():
    """The weights on disk are byte-for-byte the file HF serves at the pin.

    Cardinality is not identity: every v4.27 training run, and
    satp-policy-v2-alphaproof, share one decode surface (19 hosts / 172
    decisions). Checking constants alone accepts all of them, which is how a
    half-finished bump — pin moved, cache dir still holding the previous
    run's weights — stays green.

    The anchor is HF's own record, not anything derived from training: the
    digest below is the `lfs.sha256` that
    `HfApi().repo_info(HF_REPO, revision=REVISION, files_metadata=True)`
    reports for best_checkpoint.pt, confirmed equal to this file on
    2026-08-06. Deliberately not the checkpoint's `wandb_run_id`: W&B runs
    get deleted, so that field can stop meaning anything while the weights
    stay valid.

    Cost: ~10 s to hash 1.07 GB. It only runs where the checkpoint already
    exists, and it is the one check standing between a pin bump and a
    multi-day run on the wrong weights.
    """
    h = hashlib.sha256()
    with open(DEFAULT_CHECKPOINT, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    assert h.hexdigest() == (
        "867372b6ef698da376973b5de933e962c1582924a0b89592f7ada827063fa85a"
    ), (
        f"{DEFAULT_CHECKPOINT} is not the best_checkpoint.pt that "
        f"{HF_REPO}@{REVISION[:8]} serves (got {h.hexdigest()[:16]}…)"
    )

    # Cheap cross-check that the pin's decode constants and the checkpoint's
    # own metadata agree — the same comparison build_policy_v2 makes, run
    # here so a mismatch surfaces before anyone loads a model.
    # weights_only=False is required, not sloppy: the checkpoint carries
    # numpy_random_state, which the safe loader refuses. map_location="meta"
    # keeps this to a metadata read (~0.02 s) and never initialises CUDA.
    ck = torch.load(
        DEFAULT_CHECKPOINT, map_location="meta", mmap=True, weights_only=False
    )
    meta = ck.get("tactic_pool") or {}
    for key, want in (
        ("lemma_host_pool", list(LEMMA_HOST_POOL)),
        ("pool", list(TACTIC_POOL)),
        ("type_priority_values", TYPE_PRIORITY_VALUES),
        ("lemma_k", LEMMA_K),
    ):
        assert key in meta, (
            f"{DEFAULT_CHECKPOINT}: tactic_pool has no {key!r} — "
            "build_policy_v2 would skip this cross-check silently"
        )
        got = meta[key]
        if isinstance(want, list):
            got = list(got)
        elif isinstance(want, dict):
            got = {k: list(v) for k, v in got.items()}
        assert got == want, (
            f"{DEFAULT_CHECKPOINT}: tactic_pool[{key!r}] disagrees with the "
            f"constants from {_INFER.__file__} — checkpoint and pin are "
            "different eras"
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
    # Derived, not hardcoded — but note what that does and does not buy. The
    # HF infer.py writes N_LEMMA_HOST and N_LEMMA_DECISION as their own
    # literals, independent of len(LEMMA_HOST_POOL), so this stays a real
    # three-way consistency check (head width ↔ decode bounds ↔ render pool)
    # rather than a tautology. What it gives up is the era anchor: a literal
    # boundary fails loudly on a bump, which is useful — that job now belongs
    # to test_pin_is_the_v427_era above, where it is explicit.
    last_host = len(LEMMA_HOST_POOL) - 1
    assert decode_lemma_decision(0) is None
    assert decode_lemma_decision(1) == (0, "norm", 0)
    assert decode_lemma_decision(N_LEMMA_DECISION - 1) == (last_host, "unsafe", 2)
    assert decode_lemma_decision(N_LEMMA_DECISION) is None  # one past the last host


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
