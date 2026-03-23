#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CACHE_DIR="$ROOT_DIR/cache"
CHECKPOINT_PATH="${SATP_CHECKPOINT:-${SATP_CACHE_DIR:-$DEFAULT_CACHE_DIR}/best_checkpoint.pt}"
CACHE_DIR="$(dirname "$CHECKPOINT_PATH")"

cd "$ROOT_DIR"
mkdir -p "$CACHE_DIR"

echo "[LeanSATP] Syncing Python environment with uv..."
uv sync

echo "[LeanSATP] Fetching Lean dependencies with lake update..."
# Pre-create ProofWidgets build dir so mathlib's post-update hook can prune
# without error (needed when mathlib is resolved from a local path).
mkdir -p .lake/packages/proofwidgets/.lake/build/lib
lake update

echo "[LeanSATP] Downloading checkpoint into $CHECKPOINT_PATH ..."
uv run python -m leansatp_runtime.service --download-only \
  --checkpoint "$CHECKPOINT_PATH" \
  --cache-dir "$CACHE_DIR"

echo "[LeanSATP] Setup complete."
echo "[LeanSATP] Checkpoint: $CHECKPOINT_PATH"
echo "[LeanSATP] Cache dir: $CACHE_DIR"
