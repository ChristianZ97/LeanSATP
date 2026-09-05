# LeanSATP

LeanSATP is the Lean 4 package implementing **SATP** (*Steering Aesop for Theorem Proving*) — a framework for learning `aesop` configuration, formulated as a contextual multi-armed bandit and supervised directly by `aesop`'s deterministic execution and Lean 4's verification. The `satp` tactic queries a local inference service for a goal-tailored `aesop` configuration and runs it, falling back to plain `aesop` if the service is unavailable; `satp?` additionally prints the executed tactic as a "Try this" suggestion and throws on failure instead of falling back. The service only proposes the configuration — Lean executes and checks everything as usual.

**Paper:** [OpenReview](https://openreview.net/forum?id=VncRNbkX2q)

**Poster:** [NeSy 2026](kyjeng_conf_NeSy_2026_poster.pdf)

This branch is a standalone Lean `v4.27.0` environment. Run Lean from the repository root so Lake uses this repo's toolchain and bundled deps.

## Environment

- Lean toolchain `leanprover/lean4:v4.27.0` (`lean-toolchain`); Lake uses the bundled deps under `deps/` (no remote Mathlib cache).
- Pinned deps: mathlib4 `a3a10db0e9` (stock, zero Mathlib changes), aesop `cb837cc` (+ local `bfsScore` rule set via `patches/aesop-bfsscore.patch`), LeanCopilot `v4.27.0` (no fork). All four are stamped in `deps/.pins`; `./setup.sh` wipes and rebuilds when the stamp does not match.
- The SATP policy checkpoint and its inference code (`infer.py` — model, greedy decode, tactic-string rendering) are downloaded at the pinned revision `8ed997e1` of [`ChristianZ97/satp-policy-v4.27`](https://huggingface.co/ChristianZ97/satp-policy-v4.27); `python/src/leansatp_runtime/hf_pin.py` is the single pin, and this repo hosts no second copy of the inference code.
- Evaluation is defined by `reproduce.py` in the pinned policy repo against [`ChristianZ97/minif2f-satp-v4.27`](https://huggingface.co/datasets/ChristianZ97/minif2f-satp-v4.27) `94424f13`; detailed numbers and their caveats live on the model card. A proof counts only with zero `sorry` and a clean `#print axioms` audit (no `sorryAx`).

Machine-checked proofs from the full draft → sketch → prove pipeline (MiniF2F, ProofNet#, PutnamBench) are published in [`ChristianZ97/LeanSATP-Eval`](https://github.com/ChristianZ97/LeanSATP-Eval); every proof there compiles standalone on stock Mathlib.

## Setup

```bash
git clone https://github.com/ChristianZ97/LeanSATP.git
cd LeanSATP
./setup.sh
```

`./setup.sh` builds the bundled Lean environment under `deps/`, wires the local LeanCopilot/CTranslate2 paths into Lake, and downloads the pinned SATP assets into `cache/`. No shell export is needed for normal Lean use.

## Start the Service

Manual startup is recommended before the first `satp` call:

```bash
uv run -m leansatp_runtime.service --serve \
  --checkpoint cache/best_checkpoint.pt \
  --cache-dir cache/ \
  --host 127.0.0.1 \
  --port 5177
```

`satp` can auto-start the service, but manual startup avoids first-use cold-start timeouts.

`satp` sends the current Lean goal state to the policy; Lean elaborates theorem declarations before the tactic runs, so the policy always sees a tactic state.

## Example

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

Use `satp?` instead of `satp` when evaluating: it prints the exact generated tactic as a "Try this" suggestion and fails instead of falling back to plain `aesop`.
