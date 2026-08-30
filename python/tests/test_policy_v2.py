"""Decode/render checks for the vendored v2 policy — no checkpoint, no GPU."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from leansatp_runtime.core import Premise
from leansatp_runtime.hf_pin import CHECKPOINT_SHA256, HF_REPO, REVISION
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
        "8ed997e1526e58106a960966b06721dbf313c6a4",
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

    The anchor is HF's own record, not anything derived from training:
    `CHECKPOINT_SHA256` is the `lfs.sha256` that
    `HfApi().repo_info(HF_REPO, revision=REVISION, files_metadata=True)`
    reports for best_checkpoint.pt. Deliberately not the checkpoint's
    `wandb_run_id`: W&B runs get deleted, so that field can stop meaning
    anything while the weights stay valid.

    Imported, never copied here. A local literal is how this guard failed on
    2026-08-12: the revision was bumped one line away and this copy of the
    digest stayed on the superseded file, so the test kept approving exactly
    the skew it exists to catch. One definition, in hf_pin.py, next to the
    revision it describes.

    Cost: ~10 s to hash 1.07 GB. It only runs where the checkpoint already
    exists, and it is the one check standing between a pin bump and a
    multi-day run on the wrong weights.
    """
    h = hashlib.sha256()
    with open(DEFAULT_CHECKPOINT, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    assert h.hexdigest() == CHECKPOINT_SHA256, (
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


def test_startup_rejects_a_checkpoint_that_is_not_the_pinned_one(tmp_path, monkeypatch):
    """``ensure_local_checkpoint`` refuses wrong weights, wherever they sit.

    The digest test above only fires when someone runs the suite. The
    2026-08-12 incident did not involve running the suite: a Bridge auto-start
    picked up whatever sat at the default path and served it for five minutes.
    So the enforcement lives in the startup path, and this covers it —
    including the branch-switch case that motivates it, since ``cache/`` is
    shared with LeanSATP's ``main`` branch and is gitignored, so its contents
    outlive a checkout.

    The relocated case is the one worth spelling out. ``SATP_CACHE_DIR`` is a
    supported knob (both setup.sh scripts), Bridge derives
    ``<dir>/best_checkpoint.pt`` from it, and an earlier version of this guard
    verified only the hardcoded default path — so a relocated cache degraded to
    a warning that auto-start throws away with the child's stderr. Identity is
    a property of the bytes, not of where they live.

    Hermetic on purpose: a fake file, not the real superseded checkpoint, which
    exists on one machine and would make this pass for the wrong reason
    everywhere else.
    """
    from leansatp_runtime import service

    impostor = tmp_path / "best_checkpoint.pt"
    impostor.write_bytes(b"not the pinned weights")

    # (a) at the default path
    monkeypatch.setattr(service, "DEFAULT_CHECKPOINT", str(impostor))
    with pytest.raises(RuntimeError, match="is not the best_checkpoint.pt"):
        service.ensure_local_checkpoint(str(impostor))

    # (b) relocated — a different path, same wrong bytes, still rejected
    relocated = tmp_path / "elsewhere"
    relocated.mkdir()
    moved = relocated / "best_checkpoint.pt"
    moved.write_bytes(b"not the pinned weights")
    monkeypatch.setattr(service, "DEFAULT_CHECKPOINT", str(tmp_path / "unrelated.pt"))
    with pytest.raises(RuntimeError, match="is not the best_checkpoint.pt"):
        service.ensure_local_checkpoint(str(moved))

    # (c) waived explicitly: allowed through, and /health still tells the truth
    assert service.ensure_local_checkpoint(str(moved), allow_unverified=True) == str(
        moved
    )
    assert service._LOADED_CHECKPOINT["matches_pin"] is False
    assert service._LOADED_CHECKPOINT["checkpoint"] == str(moved)
    # Reported separately from matches_pin: Bridge refuses an unexplained
    # mismatch but accepts a waived one when the caller also opts in, so
    # "wrong" and "wrong on purpose" have to be distinguishable over HTTP.
    assert service._LOADED_CHECKPOINT["unverified_waived"] is True


def test_revision_override_does_not_waive_verification(tmp_path, monkeypatch):
    """An inherited env var must not be able to switch the guard off.

    ``SATP_HF_REVISION`` used to waive the digest comparison on its own, on the
    reasoning that no digest is known for an arbitrary revision. That turns "I
    cannot check" into "no need to check", and it is inherited: a Bridge-spawned
    child picks the variable up from whatever shell started its parent, so a
    leftover or mistyped value silently disabled the guard on the auto-start
    path — the one path this whole guard exists for. Not knowing is a stop.
    """
    from leansatp_runtime import hf_pin, service

    impostor = tmp_path / "best_checkpoint.pt"
    impostor.write_bytes(b"not the pinned weights")
    monkeypatch.setattr(service, "DEFAULT_CHECKPOINT", str(impostor))
    monkeypatch.setattr(hf_pin, "is_default_revision", lambda: False)

    with pytest.raises(RuntimeError, match="no checkpoint digest is known"):
        service.ensure_local_checkpoint(str(impostor))

    # Only the explicit argument gets through.
    assert service.ensure_local_checkpoint(str(impostor), allow_unverified=True) == str(
        impostor
    )


@pytest.mark.parametrize(
    "on_pin_rev,right_bytes,flag,starts,matches,waived",
    [
        (True, True, False, True, True, False),
        (True, True, True, True, True, False),  # nothing to waive
        (True, False, False, False, None, None),
        (True, False, True, True, False, True),
        (False, True, False, False, None, None),
        (False, True, True, True, False, True),  # right bytes, wrong revision: waiver, not a match
        (False, False, False, False, None, None),
        (False, False, True, True, False, True),
    ],
)
def test_identity_truth_table(
    tmp_path, monkeypatch, on_pin_rev, right_bytes, flag, starts, matches, waived
):
    """All eight combinations of (revision, bytes, flag), not just the common one.

    Each earlier test pinned down one branch, and the branch none of them
    covered was the one that mattered: a non-default revision whose checkpoint
    bytes happen to equal the pinned digest reported matches_pin=true, so Bridge
    took the "verified" path and never asked the caller to consent — while that
    revision selects a different infer.py, i.e. a different decode surface.

    matches_pin is now revision AND digest, and unverified_waived is derived
    from that rather than from whether the flag was passed, so a flag on an
    already-verified service does not force callers to opt in for nothing.

    Hermetic: two small files stand in for the checkpoints and the expected
    digest is monkeypatched to whichever one plays "the pinned bytes". The
    first version required the real 1.07 GB artifact, which is gitignored and
    never fetched by CI — so every row silently skipped there, and the cell
    this table exists to protect had no protection at all.
    """
    from leansatp_runtime import hf_pin, service

    pinned = tmp_path / "pinned.pt"
    pinned.write_bytes(b"the pinned weights")
    other = tmp_path / "other.pt"
    other.write_bytes(b"some other weights")
    pinned_digest = hashlib.sha256(pinned.read_bytes()).hexdigest()

    monkeypatch.setattr(hf_pin, "CHECKPOINT_SHA256", pinned_digest)
    monkeypatch.setattr(hf_pin, "is_default_revision", lambda: on_pin_rev)
    path = str(pinned if right_bytes else other)

    if not starts:
        with pytest.raises(RuntimeError):
            service.ensure_local_checkpoint(path, allow_unverified=flag)
        return

    assert service.ensure_local_checkpoint(path, allow_unverified=flag) == path
    assert service._LOADED_CHECKPOINT["matches_pin"] is matches
    assert service._LOADED_CHECKPOINT["unverified_waived"] is waived


@pytest.mark.skipif(
    not Path(DEFAULT_CHECKPOINT).exists(), reason="no checkpoint in the cache dir"
)
def test_health_fingerprint_reports_the_pinned_identity():
    """What /health publishes after a good load.

    A bare ``{"ok": true}`` cannot separate the intended fleet from a daemon
    that outlived a pin bump, and Bridge reuses any healthy listener without
    asking what it holds. These are the fields that make the difference
    checkable by whoever probes the port.
    """
    from leansatp_runtime import service

    service.ensure_local_checkpoint(DEFAULT_CHECKPOINT)
    got = service._LOADED_CHECKPOINT
    assert got["checkpoint_sha256"] == CHECKPOINT_SHA256
    assert got["matches_pin"] is True
    assert (got["repo"], got["revision"]) == (HF_REPO, REVISION)


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
