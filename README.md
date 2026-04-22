# LeanSATP

**LeanSATP** is the Lean 4 package implementing the **SATP** (*Steering Aesop for Theorem Proving*) policy — a learned agent that dynamically configures `aesop` on a per-theorem basis via a local PyTorch inference service.

The `satp` tactic queries a local SATP inference service for an `aesop` configuration tailored to the current proof goal, then runs the resulting tactic inside Lean. The Python service itself does not perform formal verification: it only generates the tactic. Lean then executes that tactic and checks the proof as usual. If the Python side is unavailable, `satp` logs one warning and falls back to plain `aesop`.

Two tactic variants are exposed:

- `satp` — user-facing. Falls back to plain `aesop` on transient failures (cold server start, policy miss).
- `satp?` — evaluation variant. Prints the exact tactic the policy ran as a "Try this" suggestion, and throws on any failure instead of falling back. Useful when measuring SATP's isolated contribution (`first | ((satp?); done) | sorry` leaves unclosed gaps as `sorry` rather than mixing in `aesop`'s coverage).

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

From the standalone repository root, run `./setup.sh` once to install Python dependencies, fetch Mathlib, and download the SATP checkpoint into the explicit `cache/` directory. The inference service expects a local checkpoint file and does not auto-download it at startup.

If CUDA is visible but unusable on the current machine, the inference service automatically downgrades itself to CPU and continues serving requests. This fallback happens both during startup and during inference, and once the service has downgraded it stays on CPU for the rest of the process lifetime. This keeps `satp` on the SATP path instead of failing over to plain `aesop` for GPU compatibility issues.

## Requirements

LeanSATP has two runtime layers:

- **Lean**: Lean `v4.26.0`, Mathlib `v4.26.0`, and `import Mathlib` in any file that uses `satp` (the tactic pool requires Mathlib tactics).
- **Python**: either [`uv`](https://docs.astral.sh/uv/) on your `PATH`, or a Python `>=3.10` environment with the dependencies from this repository's [pyproject.toml](pyproject.toml).

The recommended path is to install `uv`, then run `./setup.sh` from the repository root.

If you do not want to use `uv`, install the runtime manually from the copied dependency folder inside your project:

```bash
cd .lake/packages/LeanSATP
python3 -m pip install -e .
```


## Starting SATP

There are two ways to start SATP after `./setup.sh` has completed:

- **Recommended: start the SATP server manually first.** This avoids the first-use cold-start timeout and lets you see the generated proof sketch in the service terminal.
- **On-demand: let `satp` start it automatically.** If no SATP server is running, the Lean wrapper will try to launch one in the background. If that launch fails, or if the service later fails to answer, `satp` logs a warning and falls back to plain `aesop`.

The SATP server always listens on `127.0.0.1:5177` by default, and `satp` talks to `http://127.0.0.1:5177/infer`.

## Manual Startup

Because the model checkpoint takes several seconds to load, **start the inference service manually before running any `satp` tactic** the first time. This avoids a timeout on the first invocation.

**Terminal 1** — start the inference service and wait until it is ready:

```bash
cd /path/to/LeanSATP   # or .lake/packages/LeanSATP if used as a dependency
./setup.sh
uv run -m leansatp_runtime.service --serve \
  --checkpoint cache/best_checkpoint.pt \
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
  --checkpoint cache/best_checkpoint.pt \
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
./setup.sh
uv run -m leansatp_runtime.service --serve \
  --checkpoint cache/best_checkpoint.pt \
  --cache-dir cache/ \
  --host 127.0.0.1 \
  --port 5177
```

## Automatic Startup via `satp`

If no SATP server is already running, invoking `satp` makes LeanSATP try to start one automatically in the background using the package's own Python runtime.

The automatic path is convenient, but it has two limitations:

- the first cold start can take long enough to hit the Lean-side timeout
- you do not see the SATP service logs unless you start it manually yourself
- it assumes `./setup.sh` has already populated the local checkpoint and Lean dependencies

So the exact behavior is:

- server already running: `satp` reuses it
- server not running: `satp` tries to launch it automatically
- launch or inference failure: `satp` logs a warning and falls back to plain `aesop`

**Optional sanity check** — verify the Python service directly before involving Lean:

```bash
curl --request POST \
  --url http://localhost:5177/infer \
  --header 'Content-Type: application/json' \
  --data '{"formal_statement":"theorem t : True := by"}' | jq
```

The service logs each request as `→ request ...` and `← response ...`, and on successful inference it prints the full generated Lean proof sketch, including the `theorem ... := by` header and the generated `aesop` configuration.

### Environment variables

The Lean wrapper picks these up at `satp`/`satp?` call time; unset values fall back to the defaults shown.

| Variable | Default | Purpose |
|----------|---------|---------|
| `SATP_CHECKPOINT` | `<pkg>/cache/best_checkpoint.pt` | Override the local checkpoint file path |
| `SATP_CACHE_DIR` | `<pkg>/cache` | Override the cache directory (checkpoint, retrieval assets) |
| `SATP_SERVER_HOST` | `127.0.0.1` | SATP inference server host the Lean wrapper dials |
| `SATP_SERVER_PORT` | `5177` | SATP inference server port |
| `SATP_REQUEST_TIMEOUT` | `120` | Per-request timeout (seconds) for the curl call to `/infer` |

## Components

LeanSATP consists of the following components:

- **Lean wrapper** — a standalone Lean 4 package exposing the `satp` tactic
- **PyTorch inference service** — a bundled runtime in `python/src/leansatp_runtime`
- **Model checkpoint** — [`ChristianZ97/SATP-aesop-policy`](https://huggingface.co/ChristianZ97/SATP-aesop-policy) on Hugging Face
- **Lean proof search** — [Aesop](https://github.com/leanprover-community/aesop)


## Optional: `bfsaesop` (BFS tree search, opt-in)

`LeanSATP.BFS` exports a `bfsaesop` macro that drives `aesop`'s BFS search with LeanCopilot's `tacGen` as the only rule. This is kept off the top-level `import LeanSATP` so callers who only need `satp`/`satp?` do NOT transitively require LeanCopilot.

To use it:

1. Add LeanCopilot to your project's lakefile (LeanSATP does not declare it as a dependency).
2. Start a BFS-Prover aggregator on `localhost:23338` (or point `LeanCopilot.suggest_tactics.model` at your own generator).
3. Import the module and call the macro:

```lean
import Mathlib
import LeanSATP.BFS

example (a b : Nat) : a + b = b + a := by
  bfsaesop
```

The macro internally binds `LeanCopilot.suggest_tactics.model := "BFS-Prover"` via `set_option ... in`, so the generator is pinned at the tactic's use site regardless of the caller's current option state. `bfsaesop` uses `rule_sets := [bfs, -builtin, -default]` to exclude Aesop's built-ins and Mathlib's `@[aesop]` attributes, leaving only the LeanCopilot tacGen path — every closure is attributable to the external model.


## Usage

The syntax for invoking the `satp` tactic is `by satp [lemmas]`. The bracketed `lemmas` list is accepted for API back-compatibility but is currently **ignored** — the policy model does not ingest user hints and no post-inference Aesop rule injection is performed. `satp` and `satp [h1, h2]` produce the same behavior. To re-enable hint injection, reconnect `lemmaNames` inside `runSatpCascade` in `LeanSATP/Bridge.lean` (and expect a corresponding server-side change if hints are meant to reach the model).

### Examples

- `satp` — run the model-backed SATP pipeline
- `satp [Nat.add_comm, Nat.add_assoc]` — same as above; the bracket list is parsed then discarded


### Retrieval

LeanSATP supports an optional runtime retrieval mode.

- **Checkpoint-only mode** (default): LeanSATP runs from the local checkpoint installed by `./setup.sh`, even if no retrieval assets are absent upstream.
- **Runtime retrieval mode**: enabled automatically when the local cache directory contains retrieval assets. `./setup.sh` fetches any retrieval assets it finds on the same HF repo as the checkpoint (`ChristianZ97/SATP-aesop-policy`). Missing files are skipped silently. The runtime looks for:
    - `premise_embeddings.npy`
    - `premises_raw.npy`
    - `bm25_index.pkl` (optional; used only for hybrid retrieval)

Pass `--skip-retrieval` to `leansatp_runtime.service --download-only` if you want checkpoint-only setup.

If `premise_embeddings.npy` and `premises_raw.npy` are absent, LeanSATP falls back to no-retrieval mode. If `bm25_index.pkl` is absent, LeanSATP still uses dense retrieval when the dense assets are present.


## Debugging

If `satp` logs a fallback warning, the message describes the specific failure:

- **No Python runtime found**: install `uv` with `curl -LsSf https://astral.sh/uv/install.sh | sh`, then run `./setup.sh`.
- **Service failed to spawn**: run `./setup.sh` inside the LeanSATP package directory to reinstall dependencies and fetch Mathlib.
- **Missing checkpoint**: run `./setup.sh` to download `cache/best_checkpoint.pt`.
- **Service did not respond (timeout)**: the model is still loading. Start the service manually in a separate terminal (see [Manual Startup](#manual-startup)) and wait until it is ready before invoking `satp`. Also check for a port conflict with `lsof -i :5177`.
- **CUDA/device compatibility errors**: LeanSATP now auto-downgrades to CPU when CUDA is visible but unusable. To force CPU from the start, run the service with `CUDA_VISIBLE_DEVICES=`.
- **Want to inspect what SATP actually generated**: start the service manually and watch its terminal. Successful requests print the full proof sketch that LeanSATP is about to execute.
