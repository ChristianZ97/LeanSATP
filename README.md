# LeanSATP

LeanSATP provides the `satp` and `satp?` Lean tactics for running a learned
SATP policy that emits an `aesop` configuration for the current proof goal.
Lean still checks the final proof; the Python service only proposes the tactic.

This branch is a standalone Lean `v4.27.0` environment. Run Lean from the
LeanSATP repository root so Lake uses this repo's toolchain and bundled deps.

## Branches & Pinned Environment

| branch | policy checkpoint (HF, pinned) | eval dataset (HF, pinned) | Lean environment |
|---|---|---|---|
| `main` | SATP v1 (paper) | miniF2F (paper) | Lean `v4.17.0-rc1` + DSP-Plus Mathlib fork (see `main` README) |
| `v2` | [`ChristianZ97/satp-policy-v2`](https://huggingface.co/ChristianZ97/satp-policy-v2) `449f192d` | [`ChristianZ97/minif2f-satp`](https://huggingface.co/datasets/ChristianZ97/minif2f-satp) `32ed7f63` | standalone `v4.26.0` (below) |
| `alphaproof` | [`ChristianZ97/satp-policy-v2-alphaproof`](https://huggingface.co/ChristianZ97/satp-policy-v2-alphaproof) `a654c92d` | [`ChristianZ97/minif2f-satp-alphaproof`](https://huggingface.co/datasets/ChristianZ97/minif2f-satp-alphaproof) `81a8abfa` | standalone `v4.26.0` (below) |
| `v4.27` (this branch) | [`ChristianZ97/satp-policy-v4.27`](https://huggingface.co/ChristianZ97/satp-policy-v4.27) `5a4d2f1b` | [`ChristianZ97/minif2f-satp-v4.27`](https://huggingface.co/datasets/ChristianZ97/minif2f-satp-v4.27) `94424f13` | standalone `v4.27.0` |

This branch's environment (`v4.27`):

- Lean toolchain `leanprover/lean4:v4.27.0` (`lean-toolchain`); Lake uses the
  bundled deps under `deps/` (no remote Mathlib cache).
- Pinned deps: mathlib4 `a3a10db0e9` (stock, zero Mathlib changes), aesop
  `cb837cc` (+ local `bfsScore` rule set), LeanCopilot `v4.27.0` (no fork).
  All four are stamped in `deps/.pins`; `./setup.sh` wipes and rebuilds when
  the stamp does not match.
- Checkpoint and inference code are downloaded at the pinned HF revision by
  `./setup.sh`; `python/src/leansatp_runtime/hf_pin.py` is the single pin.
  The model-card numbers for this policy were produced at dataset snapshot
  `94424f13` of `ChristianZ97/minif2f-satp-v4.27`.
- Verification: `lake` builds in this environment are the ground truth. The
  v4.27 model card's 13 numbers were reproduced against `lake` on 2026-08-06
  (both seed checkpoints, five ablation rows each, plus three policy-free
  baselines), and every counted proof passed a `#print axioms` audit.
- A proof counts only with zero `sorry` and a clean `#print axioms` audit
  (no `sorryAx`). The audit is not optional: an `aesop` rule naming an
  unknown lemma can close any goal with a synthetic `sorry` that compiles
  cleanly and leaves no "sorry" in the compiler output.

The `v2` and `alphaproof` branches share a different environment — Lean
`v4.26.0`, mathlib4 `2df2f0150c`, aesop `2f6d238`, LeanCopilot `4cdef7b` —
and a 20-entry lemma-host pool. See those branches' own READMEs.

## Results

### v4.27 (this branch's pin)

Tactic-level, as defined by `reproduce.py` in the pinned HF repo, verified by
`lake` with a `#print axioms` audit on every counted proof:

| checkpoint | miniF2F-test |
|---|---|
| `best_checkpoint.pt` (= `ckpt_8964.pt`, the default) | **99/244** |
| `ckpt_1827.pt` | 97/244 |

Full-pipeline (draft → sketch → prove) numbers for this era are not in yet;
the `bfsaesop` baseline on the same 244 is 178 / 181 / 178 (n=3, mean 179.0).
Do not read the v2 table below as a v4.27 result.

### v2 (the `v2` branch — different era, kept for reference)

Tactic-level, as defined by `reproduce.py` in that branch's pinned HF repo:
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
companion repo `SATP-DSP-Eval`, pinned at commit `d10ef888f5f2` — the HEAD
under which every run in the table executed, whose LeanSATP submodule
gitlink resolves to this branch; this branch supplies the SATP policy
service those runs call.

Draft/sketch provenance for that table: all drafts and sketches were
generated once under the v1 paper environment (Lean `v4.17.0-rc1` +
DSP-Plus Mathlib fork, Qwen3.5-27B-FP8 drafter/sketcher) and reused
unchanged in every v2 run — the sketch axis is held fixed, so v1 and v2
numbers are directly comparable. Only proving and verification ran in the
`v2` branch's `v4.26.0` environment. 31/244 test statements use pre-v4.26
big-operator syntax (`∑ x in s, …`); a purely syntactic compatibility macro
(`_BIGOP_COMPAT` in `SATP-DSP-Eval`'s `scripts/build_stages.py`, injected at
the prove/e2e stage only) lets those statements and the reused sketches
elaborate under `v4.26` without editing either. The v4.27 era regenerates
its own drafts/sketches against the `minif2f-satp-v4.27` dataset, so that
compatibility macro is not part of this branch's chain.

## Setup

```bash
git clone --branch v4.27 https://github.com/ChristianZ97/LeanSATP.git
cd LeanSATP
./setup.sh
```

`./setup.sh` builds the bundled Lean environment under `deps/`, wires the local
LeanCopilot/CTranslate2 paths into Lake, and downloads the pinned v4.27 SATP
assets into `cache_v427/`. No shell export is needed for normal Lean use.
The era lives in the directory name on purpose — `cache_v2/` holds the v2
policy and its 20-host pool, and pointing the service at the wrong one is
refused rather than silently served.

## Start The Service

Manual startup is recommended before the first `satp` call:

```bash
uv run -m leansatp_runtime.service --serve \
  --checkpoint cache_v427/best_checkpoint.pt \
  --cache-dir cache_v427/ \
  --host 127.0.0.1 \
  --port 5177
```

`satp` can auto-start the service, but manual startup avoids first-use
cold-start timeouts.

`satp` sends the current Lean goal state to the policy. The policy does not
consume theorem-form input; Lean elaborates theorem declarations before the
tactic runs.

Single source of truth: the inference implementation (model, greedy
decode, tactic-string rendering) is `infer.py` in the pinned
[`ChristianZ97/satp-policy-v4.27`](https://huggingface.co/ChristianZ97/satp-policy-v4.27)
HF repo — the same file that reproduces the model card numbers, and the
source of the decode constants (`LEMMA_HOST_POOL`, `N_LEMMA_DECISION`, …)
that make a checkpoint from another era refuse to load. The service
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
