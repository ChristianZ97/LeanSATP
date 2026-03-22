# LeanSATP

**LeanSATP** is the Lean 4 package implementing the **SATP** (*Steering Aesop for Theorem Proving*) policy — a learned agent that dynamically configures `aesop` on a per-theorem basis via a local PyTorch inference service.

The `satp` tactic queries a local SATP inference service for an `aesop` configuration tailored to the current proof goal, then runs the resulting tactic inside Lean. The Python service itself does not perform formal verification: it only generates the tactic. Lean then executes that tactic and checks the proof as usual. If the Python side is unavailable, `satp` logs one warning and falls back to plain `aesop`.

## Highlights

- 📄 Paper: *SATP: Steering Aesop for Theorem Proving* (under review)
- **28.7%** pass rate on MiniF2F-Test — surpasses `hammer` (26.6%) at **3.3× faster**
- **0.2B parameters**, runs fully locally with a single forward pass
- Drop-in fallback inside DSP pipelines; raises end-to-end coverage to **40.2%** (+44% relative)

LeanSATP is in an early stage of development and is therefore subject to breaking changes. There is currently a version of LeanSATP compatible with the stable version of Lean `v4.26.0` (and the corresponding version of Mathlib).

## Adding LeanSATP to Your Project

To add LeanSATP from its standalone repository to an existing project with a `lakefile.toml` file, add the following:

```toml
[[require]]
name = "LeanSATP"
git = "https://github.com/ChristianZ97/LeanSATP.git"
rev = "main"

[[require]]
name = "mathlib"
scope = "leanprover-community"
rev = "v4.26.0"
```

The file `lean-toolchain` should contain the following:

```text
leanprover/lean4:v4.26.0
```

If you have a project with a `lakefile.lean` instead of `lakefile.toml`, you can use this instead:

```lean
require LeanSATP from git "https://github.com/ChristianZ97/LeanSATP.git" @ "main"

require mathlib from git "https://github.com/leanprover-community/mathlib4.git" @ "v4.26.0"
```

Then use `lake update` to fetch LeanSATP and the corresponding versions of Lean and Mathlib. The following example should then compile without any warnings or errors:

```lean
import Mathlib
import LeanSATP

example : True := by
  satp
```

On first import, LeanSATP automatically downloads the checkpoint `ChristianZ97/SATP-aesop-policy` from Hugging Face. Internet access is required for this step.

If CUDA is visible but unusable on the current machine, the inference service automatically downgrades itself to CPU and continues serving requests. This fallback happens both during startup and during inference, and once the service has downgraded it stays on CPU for the rest of the process lifetime. This keeps `satp` on the SATP path instead of failing over to plain `aesop` for GPU compatibility issues.

## Requirements

LeanSATP has two runtime layers:

- **Lean**: Lean `v4.26.0`, Mathlib `v4.26.0`, and `import Mathlib` in any file that uses `satp` (the tactic pool requires Mathlib tactics).
- **Python**: either [`uv`](https://docs.astral.sh/uv/) on your `PATH`, or a Python `>=3.10` environment with the dependencies from this repository's [pyproject.toml](pyproject.toml).

The recommended path is to install `uv`. LeanSATP will then provision its Python runtime on first use via `uv run`.

If you do not want to use `uv`, install the runtime manually from the copied dependency folder inside your project:

```bash
cd .lake/packages/LeanSATP
python3 -m pip install -e .
```


## Starting SATP

There are two ways to start SATP:

- **Recommended: start the SATP server manually first.** This avoids the first-use cold-start timeout and lets you see the generated proof sketch in the service terminal.
- **On-demand: let `satp` start it automatically.** If no SATP server is running, the Lean wrapper will try to launch one in the background. If that launch fails, or if the service later fails to answer, `satp` logs a warning and falls back to plain `aesop`.

The SATP server always listens on `127.0.0.1:5177` by default, and `satp` talks to `http://127.0.0.1:5177/infer`.

## Manual Startup

Because the model checkpoint takes several seconds to load, **start the inference service manually before running any `satp` tactic** the first time. This avoids a timeout on the first invocation.

**Terminal 1** — start the inference service and wait until it is ready:

```bash
cd /path/to/LeanSATP   # or .lake/packages/LeanSATP if used as a dependency
uv run -m leansatp_runtime.service --serve \
  --checkpoint hf://ChristianZ97/SATP-aesop-policy/best_checkpoint.pt \
  --cache-dir cache/ \
  --host 127.0.0.1 \
  --port 5177
```

When the service is ready, it prints a Kimina-style startup block including:

- `LeanSATP service running on http://127.0.0.1:5177`
- `Try me with:`
- a copy-pasteable `curl` example for `/infer`

If you want to force CPU explicitly, prefix the command with `CUDA_VISIBLE_DEVICES=`:

```bash
CUDA_VISIBLE_DEVICES= uv run -m leansatp_runtime.service --serve \
  --checkpoint hf://ChristianZ97/SATP-aesop-policy/best_checkpoint.pt \
  --cache-dir cache/ \
  --host 127.0.0.1 \
  --port 5177
```

**Terminal 2** — once the service is ready, run Lean as usual:

```bash
lake env lean YourFile.lean
# or
lake build
```

If you are using LeanSATP through another Lean project, run the same command from the copied package directory:

```bash
cd .lake/packages/LeanSATP
uv run -m leansatp_runtime.service --serve \
  --checkpoint hf://ChristianZ97/SATP-aesop-policy/best_checkpoint.pt \
  --cache-dir cache/ \
  --host 127.0.0.1 \
  --port 5177
```

## Automatic Startup via `satp`

If no SATP server is already running, invoking `satp` makes LeanSATP try to start one automatically in the background using the package's own Python runtime.

The automatic path is convenient, but it has two limitations:

- the first cold start can take long enough to hit the Lean-side timeout
- you do not see the SATP service logs unless you start it manually yourself

So the exact behavior is:

- server already running: `satp` reuses it
- server not running: `satp` tries to launch it automatically
- launch or inference failure: `satp` logs a warning and falls back to plain `aesop`

**Optional sanity check** — verify the Python service directly before involving Lean:

```bash
curl --request POST \
  --url http://localhost:5177/infer \
  --header 'Content-Type: application/json' \
  --data '{"goal":"True"}' | jq
```

The service logs each request as `→ request ...` and `← response ...`, and on successful inference it prints the full generated Lean proof sketch, including the `theorem ... := by` header and the generated `aesop` configuration.

## Components

LeanSATP consists of the following components:

- **Lean wrapper** — a standalone Lean 4 package exposing the `satp` tactic
- **PyTorch inference service** — a bundled runtime in `python/src/leansatp_runtime`
- **Model checkpoint** — [`ChristianZ97/SATP-aesop-policy`](https://huggingface.co/ChristianZ97/SATP-aesop-policy) on Hugging Face
- **Lean proof search** — [Aesop](https://github.com/leanprover-community/aesop)


## Usage

The syntax for invoking the `satp` tactic is `by satp [lemmas]`. The `lemmas` argument is optional and can be used to pass an explicit list of lemmas whose names are appended to the generated `aesop` script as extra unsafe rules.

### Examples

- `satp` — run the model-backed SATP pipeline
- `satp [Nat.add_comm, Nat.add_assoc]` — run SATP and append explicit lemmas


### Retrieval

LeanSATP supports an optional runtime retrieval mode.

- **Checkpoint-only mode** (default): LeanSATP downloads the SATP checkpoint from Hugging Face and runs even if no retrieval assets are present.
- **Runtime retrieval mode**: enabled automatically when the local cache directory contains retrieval assets. LeanSATP does **not** currently auto-download or build retrieval assets. If you want runtime retrieval, you must provide the cache files yourself. The runtime looks for:
    - `premise_embeddings.npy`
    - `premises_raw.npy`
    - `bm25_index.pkl` (optional; used only for hybrid retrieval)

If `premise_embeddings.npy` and `premises_raw.npy` are absent, LeanSATP falls back to no-retrieval mode. If `bm25_index.pkl` is absent, LeanSATP still uses dense retrieval when the dense assets are present.


## Debugging

If `satp` logs a fallback warning, the message describes the specific failure:

- **No Python runtime found**: install `uv` with `curl -LsSf https://astral.sh/uv/install.sh | sh`, then run `uv sync`.
- **Service failed to spawn**: run `uv sync` inside the LeanSATP package directory to reinstall dependencies.
- **Service did not respond (timeout)**: the model is still loading. Start the service manually in a separate terminal (see [Manual Startup](#manual-startup)) and wait until it is ready before invoking `satp`. Also check for a port conflict with `lsof -i :5177`.
- **CUDA/device compatibility errors**: LeanSATP now auto-downgrades to CPU when CUDA is visible but unusable. To force CPU from the start, run the service with `CUDA_VISIBLE_DEVICES=`.
- **Want to inspect what SATP actually generated**: start the service manually and watch its terminal. Successful requests print the full proof sketch that LeanSATP is about to execute.

You can disable the import-time checkpoint download by setting the environment variable:

```text
SATP_SKIP_IMPORT_DOWNLOAD=1
```
