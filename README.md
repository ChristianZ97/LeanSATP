# LeanSATP

LeanSATP provides the `satp` and `satp?` Lean tactics for running a learned
SATP policy that emits an `aesop` configuration for the current proof goal.
Lean still checks the final proof; the Python service only proposes the tactic.

This branch is a standalone Lean `v4.26.0` environment. Run Lean from the
LeanSATP repository root so Lake uses this repo's toolchain and bundled deps.

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
downloads it at a pinned revision (see `models/policy_v2.py`) together with
the checkpoint, so this repo hosts no second copy of the inference code.
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
