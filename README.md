# LeanSATP

LeanSATP provides the `satp` and `satp?` Lean tactics for running a learned
SATP policy that emits an `aesop` configuration for the current proof goal.
Lean still checks the final proof; the Python service only proposes the tactic.

This branch is a standalone Lean `v4.26.0` environment. Run Lean from the
LeanSATP repository root so Lake uses this repo's toolchain and bundled deps.

## Branches & Pinned Environment

| branch | policy checkpoint (HF, pinned) | eval dataset (HF, pinned) | Lean environment |
|---|---|---|---|
| `main` | SATP v1 (paper) | miniF2F (paper) | Lean `v4.17.0-rc1` + DSP-Plus Mathlib fork (see `main` README) |
| `v2` (this branch) | [`ChristianZ97/satp-policy-v2`](https://huggingface.co/ChristianZ97/satp-policy-v2) `449f192d` | [`ChristianZ97/minif2f-satp`](https://huggingface.co/datasets/ChristianZ97/minif2f-satp) `32ed7f63` | standalone `v4.26.0` (below) |
| `alphaproof` | [`ChristianZ97/satp-policy-v2-alphaproof`](https://huggingface.co/ChristianZ97/satp-policy-v2-alphaproof) `a654c92d` | [`ChristianZ97/minif2f-satp-alphaproof`](https://huggingface.co/datasets/ChristianZ97/minif2f-satp-alphaproof) `81a8abfa` | standalone `v4.26.0` (below) |

Environment shared by the `v2` and `alphaproof` branches:

- Lean toolchain `leanprover/lean4:v4.26.0` (`lean-toolchain`); Lake uses the
  bundled deps under `deps/` (no remote Mathlib cache).
- Pinned deps: mathlib4 `2df2f0150c` (stock, zero Mathlib changes), aesop
  `2f6d238` (+ local `bfsScore` rule set), LeanCopilot `4cdef7b` (`v4.26.0`,
  no fork).
- Checkpoint and inference code are downloaded at the pinned HF revision by
  `./setup.sh`; `python/src/leansatp_runtime/hf_pin.py` is the single pin.
  The model-card numbers on this branch were produced at dataset snapshot
  `32ed7f63`.
- Verification: `lake` builds in this environment are the ground truth; all
  reported numbers are cross-checked set-identical against an independent
  kimina-lean-server running stock Mathlib on the same `v4.26.0` toolchain.
- A proof counts only with zero `sorry` and a clean `#print axioms` audit
  (no `sorryAx`).

## Results (v2)

Tactic-level, as defined by `reproduce.py` in the pinned HF repo:
miniF2F-test **92/244**, reproduced twice bit-identically, all 92 rebuilt by
`lake`; miniF2F-valid 91/244 (kimina reports 94; 3 close only via `sorryAx`
and are rejected by the axiom audit). The Lean-side `satp?` tactic reproduces
the same 92-problem test set end to end.

Full DSP pipeline (draft → sketch → prove cascade) on miniF2F-test (244):

| mode | proved / 244 |
|---|---|
| `satp` cascade, deterministic ×1 | 132 |
| `satpbfsaesop` ×1 (n=3) | 185 / 185 / 187 |
| `satpbfsaesop4x` (n=3) | 189 / 189 / 192 |

A fourth ×1 run (179) is excluded under a pre-registered rule: the disk
filled (ENOSPC) mid-run. Pipeline numbers are produced by the private
companion repo `SATP-DSP-Eval`, pinned at commit `b84f4bde165e`; this branch
supplies the SATP policy service those runs call.

Draft/sketch provenance: all drafts and sketches were generated once under
the v1 paper environment (Lean `v4.17.0-rc1` + DSP-Plus Mathlib fork,
Qwen3.5-27B-FP8 drafter/sketcher) and are reused unchanged in every v2 run —
the sketch axis is held fixed, so v1 and v2 numbers are directly comparable.
Only proving and verification run in this branch's `v4.26.0` environment.
31/244 test statements use pre-v4.26 big-operator syntax (`∑ x in s, …`); a
purely syntactic compatibility macro (`_BIGOP_COMPAT` in `SATP-DSP-Eval`'s
`scripts/build_stages.py`, injected at the prove/e2e stage only) lets those
statements and the reused sketches elaborate under `v4.26` without editing
either.

## Setup

```bash
git clone --branch v2 https://github.com/ChristianZ97/LeanSATP.git
cd LeanSATP
./setup.sh
```

`./setup.sh` builds the bundled Lean environment under `deps/`, wires the local
LeanCopilot/CTranslate2 paths into Lake, and downloads SATP v2 assets into
`cache_v2/`. No shell export is needed for normal Lean use.

## Start The Service

Manual startup is recommended before the first `satp` call:

```bash
uv run -m leansatp_runtime.service --serve \
  --checkpoint cache_v2/best_checkpoint.pt \
  --cache-dir cache_v2/ \
  --host 127.0.0.1 \
  --port 5177
```

`satp` can auto-start the service, but manual startup avoids first-use
cold-start timeouts.

`satp` sends the current Lean goal state to the policy. The policy does not
consume theorem-form input; Lean elaborates theorem declarations before the
tactic runs.

Single source of truth: the v2 inference implementation (model, greedy
decode, tactic-string rendering) is `infer.py` in the
[`ChristianZ97/satp-policy-v2`](https://huggingface.co/ChristianZ97/satp-policy-v2)
HF repo — the same file that reproduces the model card numbers. The service
downloads it at a pinned revision (see
`python/src/leansatp_runtime/hf_pin.py`) together with the checkpoint, so
this repo hosts no second copy of the inference code.
Evaluation numbers are defined by the HF repo's `reproduce.py`; this repo is
the way you *use* the policy from Lean.

## SATP Example

Create `Demo.lean`:

```lean
import Mathlib
import LeanSATP

example (n : Nat) : n = n := by
  satp
```

Run it from the LeanSATP root:

```bash
lake env lean Demo.lean
```

Use `satp?` instead of `satp` when evaluating: it prints the exact generated
tactic as a "Try this" suggestion and fails instead of falling back to plain
`aesop`.
