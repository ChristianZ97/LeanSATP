# LeanSATP

> This branch preserves LeanSATP as used in the paper (v1). Current development and the latest release live on [`main`](https://github.com/ChristianZ97/LeanSATP/tree/main).

LeanSATP is the Lean 4 package implementing **SATP** (*Steering Aesop for Theorem Proving*) — a framework for learning `aesop` configuration, formulated as a contextual multi-armed bandit and supervised directly by `aesop`'s deterministic execution and Lean 4's verification. The `satp` tactic queries a local inference service for a goal-tailored `aesop` configuration and runs it, falling back to plain `aesop` if the service is unavailable; `satp?` additionally prints the executed tactic as a "Try this" suggestion and throws on failure instead of falling back. The service only generates the configuration — Lean executes and checks it as usual.

On MiniF2F-Test, `satp` achieves a solve rate of 32.6% ± 0.3%, against 10.7% for plain `aesop`; see the paper for details.

LeanSATP builds with Lean `v4.17.0-rc1`. Mathlib (together with Aesop and LeanCopilot) is pulled transitively from the commit-pinned [DSP-Plus Mathlib fork](https://github.com/caochenrui/mathlib4/tree/dsp+), so the whole dependency tree is reproducible.

## Adding LeanSATP to Your Project

1. Use Lean `v4.17.0-rc1` (`lean-toolchain` must contain `leanprover/lean4:v4.17.0-rc1`) and add LeanSATP to your `lakefile.toml`:

   ```toml
   [[require]]
   name = "LeanSATP"
   git = "https://github.com/ChristianZ97/LeanSATP.git"
   rev = "legacy"
   ```

2. Install [`uv`](https://docs.astral.sh/uv/):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. From the LeanSATP package root (`.lake/packages/LeanSATP` when used as a dependency), run:

   ```bash
   ./setup.sh
   ```

   This installs the Python dependencies, fetches the Lean dependencies, and downloads the model checkpoint from [`ChristianZ97/satp-policy-goal`](https://huggingface.co/ChristianZ97/satp-policy-goal) into `cache/`.

## Usage

Start the inference service (recommended before first use — the checkpoint takes a few seconds to load):

```bash
uv run -m leansatp_runtime.service --serve \
  --checkpoint cache/best_checkpoint.pt \
  --cache-dir cache/ \
  --host 127.0.0.1 \
  --port 5177
```

Then:

```lean
import Mathlib
import LeanSATP

example : True := by
  satp
```

If no server is running, `satp` launches one in the background automatically; on any failure it logs a warning and falls back to plain `aesop`.
