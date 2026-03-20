# LeanSATP

LeanSATP is an automated reasoning tool for Lean that calls a local PyTorch SATP policy, asks it for an `aesop` configuration, and runs the resulting tactic inside Lean. The `satp` tactic provided by LeanSATP tries the model-backed configuration first and falls back to plain `aesop` if the Python side is unavailable.

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
import LeanSATP

example : True := by
  satp
```

The first time you `import LeanSATP`, LeanSATP best-effort prefetches the fixed Hugging Face checkpoint `ChristianZ97/SATP-aesop-policy`. The first time you run `satp`, LeanSATP starts its bundled local PyTorch inference service from this package's own Python runtime.

No external kimina server is required for this import path.

If the Python environment, checkpoint, or inference service is unavailable, LeanSATP logs one warning and falls back to plain `aesop`.

## Requirements

LeanSATP has two runtime layers:

- **Lean**: Lean `v4.26.0` and the matching Mathlib version.
- **Python**: either [`uv`](https://docs.astral.sh/uv/) on your `PATH`, or a Python `>=3.10` environment with the dependencies from this repository's [pyproject.toml](pyproject.toml).

The recommended path is to install `uv`. Then LeanSATP can provision its Python runtime on first use via `uv run`.

If you do not want to use `uv`, install the runtime manually from the copied dependency folder inside your project:

```bash
cd .lake/packages/LeanSATP
python3 -m pip install -e .
```

On first use, LeanSATP downloads the fixed Hugging Face checkpoint `ChristianZ97/SATP-aesop-policy`. Internet access is therefore required the first time the checkpoint is fetched.

## First-Time Setup

Because the model checkpoint takes several seconds to load, **start the inference service manually before running any `satp` tactic** the first time. This avoids a timeout on the first invocation.

**Terminal 1** — start the inference service and wait until it is ready:

```bash
cd /path/to/LeanSATP   # or .lake/packages/LeanSATP if used as a dependency
uv run python -m leansatp_runtime.service --serve \
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

On subsequent uses, `satp` will attempt to start the service automatically in the background. The service only needs to be started manually the first time, or after the process has been killed.

## Components

Currently, LeanSATP consists of/depends on the following components:

- **Lean wrapper**
  - A standalone Lean package
- **PyTorch inference service**
  - A bundled runtime in `python/src/leansatp_runtime`
- **Model checkpoint**
  - The fixed Hugging Face checkpoint [`ChristianZ97/SATP-aesop-policy`](https://huggingface.co/ChristianZ97/SATP-aesop-policy)
- **Lean proof search**
  - [Aesop](https://github.com/leanprover-community/aesop)

## Usage

The syntax for invoking the `satp` tactic is `by satp [lemmas]`. The `lemmas` argument is optional and can be used to pass an explicit list of lemmas whose names are appended to the generated `aesop` script as extra unsafe rules.

### Examples

You can use:

- `satp` to run the model-backed SATP pipeline
- `satp [Nat.add_comm, Nat.add_assoc]` to run SATP and append explicit lemmas

### Retrieval

LeanSATP still supports runtime retrieval, but retrieval is optional rather than required.

- **Checkpoint-only mode**: the default path. LeanSATP downloads the SATP checkpoint from Hugging Face and runs even if no retrieval assets are present.
- **Runtime retrieval mode**: enabled automatically when the local cache directory contains retrieval assets.

LeanSATP does **not** currently auto-download or build retrieval assets for the user. If you want runtime retrieval, you must provide the cache files yourself. The runtime looks for:

- `premise_embeddings.npy`
- `premises_raw.npy`
- `bm25_index.pkl` (optional; used only for hybrid retrieval)

If `premise_embeddings.npy` and `premises_raw.npy` are absent, LeanSATP falls back to no-retrieval mode. If `bm25_index.pkl` is absent, LeanSATP still uses dense retrieval when the dense assets are present.

### Debugging

If `satp` logs a fallback warning, the message describes the specific failure:

- **No Python runtime found**: install `uv` with `curl -LsSf https://astral.sh/uv/install.sh | sh`, then run `uv sync`.
- **Service failed to spawn**: run `uv sync` inside the LeanSATP package directory to reinstall dependencies.
- **Service did not respond (timeout)**: the model is still loading. Start the service manually in a separate terminal (see [First-Time Setup](#first-time-setup)) and wait until it is ready before invoking `satp`. Also check for a port conflict with `lsof -i :5177`.

You can disable the import-time checkpoint prefetch by setting:

```text
SATP_SKIP_IMPORT_DOWNLOAD=1
```
