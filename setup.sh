#!/usr/bin/env bash
# ============================================================================
# LeanSATP/setup.sh  —  STANDALONE Lean v4.27 environment for satp + bfsaesop
# ----------------------------------------------------------------------------
# `git clone <this repo> && ./setup.sh` gives a self-contained v4.27 toolchain:
# no DSP-Plus, no kimina-lean-server, no parent SATP-DSP-Eval needed.
#
# caochenrui's dsp+ fork (the v4.17 line) only exists at v4.17, so we replicate
# its lakefile structure on v4.27 OFFICIAL upstream sources, built into deps/:
#
#   deps/aesop        = leanprover-community/aesop @ cb837cc + bfsScore patch
#                       (mathlib v4.27's own pinned aesop rev)
#                       (patches/aesop-bfsscore.patch in this repo)
#   deps/LeanCopilot  = lean-dojo/LeanCopilot     @ v4.27.0  (+ CTranslate2)
#   deps/mathlib4     = leanprover-community/mathlib4 @ a3a10db0e9 (v4.27.0,
#                       = ChristianZ97/*-satp-v4.27 dataset pin),
#                       lakefile rewired to require the two above + link CT2
#   (this package)    = LeanSATP; its lakefile requires mathlib from deps/mathlib4
#
# The dsp+ mathlib CANNOT use the official olean cache (modified lakefile shifts
# Cache/Hashing.getRootHash -> 0/N hits), so Mathlib is built from source ONCE
# (~1-2 h). Everything else is minutes.
#
# Usage:
#   ./setup.sh                   # build the whole env + download the checkpoint
#   FORCE=1 ./setup.sh           # wipe deps/ and rebuild every stage
#   SKIP_SATP_CKPT=1 ./setup.sh  # skip the 1.1 GB SATP v4.27 checkpoint download
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPS="$ROOT/deps"
CT2LIB="$DEPS/LeanCopilot/.lake/build/lib"
PATCH="$ROOT/patches/aesop-bfsscore.patch"
FORCE="${FORCE:-0}"

TOOLCHAIN="leanprover/lean4:v4.27.0"
AESOP_URL="https://github.com/leanprover-community/aesop.git"
AESOP_COMMIT="cb837cc26236ada03c81837bebe0acd9c70ced7d"
LEANCOPILOT_URL="https://github.com/lean-dojo/LeanCopilot.git"
LEANCOPILOT_TAG="v4.27.0"
MATHLIB_URL="https://github.com/leanprover-community/mathlib4.git"
MATHLIB_COMMIT="a3a10db0e9d66acbebf76c5e6a135066525ac900"

# Default checkpoint = the published v4.27 policy. Keep this in step with
# hf_pin.py / service.py's DEFAULT_CHECKPOINT_SOURCE — the three must name the
# same era or the service loads weights its decode constants can't read
# (build_policy_v2 refuses rather than mis-rendering). The revision inside
# hf_pin.py is what selects the file; this line only names the repo, so the two
# cannot drift apart the way a tagged cache directory did (see service.py).
SATP_CKPT_SOURCE="${SATP_CKPT_SOURCE:-hf://ChristianZ97/satp-policy-v4.27/best_checkpoint.pt}"
SATP_CACHE_DIR="${SATP_CACHE_DIR:-$ROOT/cache}"

log() { printf '\n\033[1m[LeanSATP setup] %s\033[0m\n' "$*"; }
die() { printf '\033[31m[LeanSATP setup] ERROR: %s\033[0m\n' "$*" >&2; exit 1; }

log "0. prereqs"
command -v lake  >/dev/null || die "lake not found (install elan)"
command -v uv    >/dev/null || die "uv not found"
command -v git   >/dev/null || die "git not found"
command -v cmake >/dev/null || die "cmake not found (needed for CTranslate2)"
command -v cc    >/dev/null || die "c compiler not found"
[ -f "$PATCH" ] || die "missing $PATCH"
elan toolchain install "$TOOLCHAIN" >/dev/null 2>&1 || true
mkdir -p "$DEPS"
# Cross-era safety: deps/ and .lake/ are gitignored, so they survive git branch
# switches. If the recorded pin set differs — or is absent while build artifacts
# exist (an in-place v4.26 -> v4.27 upgrade) — stale oleans would poison the new
# toolchain. Wipe and rebuild everything in that case.
# One stamp for all stages — coarse wipe over per-stage invalidation; the
# mathlib stage dominates rebuild cost anyway.
PINS="$TOOLCHAIN|$AESOP_COMMIT|$LEANCOPILOT_TAG|$MATHLIB_COMMIT"
if [ "$(cat "$DEPS/.pins" 2>/dev/null || true)" != "$PINS" ]; then
  if [ -n "$(ls -A "$DEPS" 2>/dev/null)" ] || [ -d "$ROOT/.lake" ]; then
    # An unstamped tree whose checkouts already match every pin (a build from a
    # pre-stamp v4.27 commit) is same-era: seed the stamp, keep the build.
    LC_WANT="$(git -C "$DEPS/LeanCopilot" rev-parse "$LEANCOPILOT_TAG^{commit}" 2>/dev/null || true)"
    if [ "$(git -C "$DEPS/aesop" rev-parse HEAD 2>/dev/null)" = "$AESOP_COMMIT" ] \
       && [ "$(git -C "$DEPS/mathlib4" rev-parse HEAD 2>/dev/null)" = "$MATHLIB_COMMIT" ] \
       && [ -n "$LC_WANT" ] && [ "$(git -C "$DEPS/LeanCopilot" rev-parse HEAD 2>/dev/null)" = "$LC_WANT" ]; then
      log "unstamped tree matches the pin set — seeding stamp, keeping existing build"
    else
      log "pin set changed/unknown — wiping deps/ and .lake/ for a clean rebuild"
      rm -rf "$DEPS" "$ROOT/.lake"
      mkdir -p "$DEPS"
    fi
  fi
  printf '%s' "$PINS" > "$DEPS/.pins"
fi
echo "  LeanSATP: $ROOT"
echo "  deps:     $DEPS"

# --- 1. aesop-dsp+ ----------------------------------------------------------
log "1. deps/aesop  (aesop @ ${AESOP_COMMIT:0:7} + bfsScore patch)"
if [ "$FORCE" = 1 ] || [ ! -f "$DEPS/aesop/.lake/build/lib/lean/Aesop.olean" ]; then
  rm -rf "$DEPS/aesop"
  git clone --quiet "$AESOP_URL" "$DEPS/aesop"
  git -C "$DEPS/aesop" checkout --quiet "$AESOP_COMMIT"
  git -C "$DEPS/aesop" apply "$PATCH"
  echo "  patch: $(git -C "$DEPS/aesop" diff --stat | tail -1)"
  ( cd "$DEPS/aesop" && lake build )
else echo "  already built — skip"; fi

# --- 2. LeanCopilot + CTranslate2 ------------------------------------------
log "2. deps/LeanCopilot @ ${LEANCOPILOT_TAG}  (+ CTranslate2 native)"
patch_openblas() {
  local ch; ch=$(find "$DEPS/LeanCopilot/.lake/build/OpenBLAS" -maxdepth 2 -name common.h 2>/dev/null | head -1)
  [ -z "$ch" ] && return 1
  grep -qE '^#ifdef[[:space:]]+HAVE_C11' "$ch" && \
    sed -i 's/^#ifdef[[:space:]]\+HAVE_C11/#if defined(HAVE_C11) \&\& !defined(__cplusplus)/' "$ch" && \
    echo "  patched OpenBLAS common.h"
}
if [ "$FORCE" = 1 ] || [ ! -f "$CT2LIB/libleanffi.so" ]; then
  rm -rf "$DEPS/LeanCopilot"
  git clone --quiet "$LEANCOPILOT_URL" "$DEPS/LeanCopilot"
  git -C "$DEPS/LeanCopilot" checkout --quiet "$LEANCOPILOT_TAG"
  ( cd "$DEPS/LeanCopilot"
    if ! lake build 2>&1 | tee "$DEPS/_lc_build.log"; then
      if grep -qiE 'stdatomic|HAVE_C11|common\.h|_Atomic' "$DEPS/_lc_build.log"; then
        echo "  OpenBLAS/gcc wall — patch + resume"; patch_openblas
        rm -rf .lake/build/CTranslate2; lake build
      else die "LeanCopilot build failed (see deps/_lc_build.log)"; fi
    fi )
  [ -f "$CT2LIB/libleanffi.so" ] || die "libleanffi.so not produced"
else echo "  already built — skip"; fi

# --- 3. mathlib-dsp+ --------------------------------------------------------
log "3. deps/mathlib4 @ ${MATHLIB_COMMIT:0:10}  (rewired lakefile, source build ~1-2h)"
if [ "$FORCE" = 1 ] || [ ! -f "$DEPS/mathlib4/.lake/build/lib/lean/Mathlib.olean" ]; then
  rm -rf "$DEPS/mathlib4"
  git clone --filter=blob:none --quiet "$MATHLIB_URL" "$DEPS/mathlib4"
  git -C "$DEPS/mathlib4" checkout --quiet "$MATHLIB_COMMIT"
  python3 - "$DEPS/mathlib4/lakefile.lean" "$DEPS" <<'PY'
import sys
lf, deps = sys.argv[1], sys.argv[2]
s = open(lf).read()
assert 'require "leanprover-community" / "aesop" @ git "master"' in s
assert 'require "leanprover-community" / "plausible" @ git "main"' in s
assert 'package mathlib where\n' in s
s = s.replace('require "leanprover-community" / "aesop" @ git "master"',
              'require aesop from "../aesop"')
s = s.replace('require "leanprover-community" / "plausible" @ git "main"',
              'require "leanprover-community" / "plausible" @ git "main"\n'
              'require LeanCopilot from "../LeanCopilot"')
s = s.replace('package mathlib where\n',
              'package mathlib where\n'
              f'  moreLinkArgs := #["-L{deps}/LeanCopilot/.lake/build/lib", "-lctranslate2"]\n')
open(lf,'w').write(s); print("  lakefile rewired (../aesop, ../LeanCopilot, -lctranslate2)")
PY
  ( cd "$DEPS/mathlib4"
    lake update aesop LeanCopilot
    echo "  building full Mathlib from source (~1-2h) ..."
    lake build )
  [ -f "$DEPS/mathlib4/.lake/build/lib/lean/Mathlib.olean" ] || die "Mathlib.olean not produced"
else echo "  already built — skip"; fi

# --- 4. LeanSATP (this package: satp + bfsaesop) ---------------------------
log "4. LeanSATP  (satp Bridge + bfsaesop BFS)"
grep -q 'deps/mathlib4' "$ROOT/lakefile.lean" || die "lakefile.lean must require mathlib from deps/mathlib4"
cd "$ROOT"
uv sync
mkdir -p .lake/packages/proofwidgets/.lake/build/lib   # mathlib post-update hook prune-safety
lake update mathlib
# Build the FULL Mathlib closure too — NOT just the subset `import LeanSATP`
# pulls. Standalone `import Mathlib` must resolve so the satp policy's emitted
# Mathlib tactics (ring / nlinarith / omega / field_simp / …) elaborate; without
# this the transitive deps (Qq / ProofWidgets / plausible / …) stay unbuilt under
# .lake/packages and `import Mathlib` dies with "unknown module prefix 'Qq'".
# Mathlib is a path-dep to the already-built deps/mathlib4, so this only fills in
# those small missing deps — it reuses deps/mathlib4's oleans, no Mathlib rebuild.
LD_LIBRARY_PATH="$CT2LIB:${LD_LIBRARY_PATH:-}" lake build Mathlib LeanSATP LeanSATP.Bridge LeanSATP.BFS LeanSATP.RulesetInit
[ -f "$ROOT/.lake/build/lib/lean/LeanSATP/BFS.olean" ] || die "LeanSATP.BFS did not build"
[ -n "$(find .lake/packages/Qq/.lake/build/lib -name 'Qq.olean' 2>/dev/null)" ] \
  || die "Mathlib closure incomplete (Qq unbuilt) — standalone 'import Mathlib' would fail"

# --- 5. SATP v4.27 policy checkpoint ------------------------------------------
log "5. SATP v4.27 checkpoint  ($SATP_CACHE_DIR/best_checkpoint.pt)"
if [ "${SKIP_SATP_CKPT:-0}" = 1 ]; then
  echo "  SKIP_SATP_CKPT=1 — skipping"
else
  # Only catches the plain v2 repo by name. It cannot catch a same-cardinality
  # sibling (satp-policy-v2-alphaproof is also 19/172) or a local path — those
  # pass every automatic check, so the pin is the only thing that identifies
  # which training run gets served.
  case "$SATP_CKPT_SOURCE" in *satp-policy-v2/*)
    echo "  WARNING: v4.26-era policy (satp-policy-v2) on a v4.27 env. Its decode"
    echo "           constants differ (LEMMA_HOST_POOL 20 vs 19), so the service"
    echo "           will refuse to load it against this branch's pin."
    echo "           There is no env override for the repo — switching eras means"
    echo "           editing HF_REPO in python/src/leansatp_runtime/hf_pin.py, or"
    echo "           using the v2 branch. Unset SATP_CKPT_SOURCE for v4.27." ;;
  esac
  # Downloading is not the same as having a usable install: only the checkpoint
  # is asserted below. Retrieval assets fetch best-effort and skip silently, and
  # a policy served without them raises on every /infer.
  # always run: hf_hub_download is cache-backed (no-op at an unchanged pin),
  # and a bumped SATP_HF_REVISION must refresh ckpt + assets + infer.py together
  uv run -m leansatp_runtime.service --download-only \
    --checkpoint "$SATP_CACHE_DIR/best_checkpoint.pt" \
    --checkpoint-source "$SATP_CKPT_SOURCE" --cache-dir "$SATP_CACHE_DIR"
  [ -f "$SATP_CACHE_DIR/best_checkpoint.pt" ] || die "checkpoint download failed"
fi

log "STANDALONE v4.27 ENV READY — satp + bfsaesop build on v4.27"
