"""PyTorch-backed SATP inference server."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import errno
import hashlib
import io
import json
import os
import re
import signal
import shutil
import sys
import textwrap
import warnings
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
from typing import Any, Optional

_DEFAULT_MAX_INFLIGHT = 8
_DEFAULT_INFER_CACHE_SIZE = 10000
_SEMAPHORE_ACQUIRE_TIMEOUT = 45.0  # must stay below Bridge.lean's curl timeout
DEFAULT_CHECKPOINT_SOURCE = "hf://ChristianZ97/satp-policy-v4.27/best_checkpoint.pt"
_PACKAGE_ROOT = Path(__file__).resolve().parents[3]
# One directory per checkout, named for nothing: this branch serves exactly the
# checkpoint ``hf_pin.py`` pins, so an era or run tag in the path buys nothing
# and actively misleads. It used to read ``cache_v427``, and 2026-08-12 showed
# why that is worse than a plain name — the pin moved *within* the era
# (best_checkpoint.pt: 867372b6… -> 1c03fdd5…) while the directory kept its
# v4.27 tag, so a stale file sat at the default path looking correct and
# ``ensure_local_checkpoint`` (existence-only) started it without complaint.
# The guard that actually works is ``test_policy_v2.py``'s SHA-256 assertion
# against HF's own ``lfs.sha256`` for the pinned revision, not the folder name.
DEFAULT_CACHE_DIR = str(_PACKAGE_ROOT / "cache")
DEFAULT_CHECKPOINT = str(Path(DEFAULT_CACHE_DIR) / "best_checkpoint.pt")

# Set once from argv; never varies per call, so it is process policy rather
# than a parameter to thread through serve -> engine -> load_policy.
_ALLOW_UNVERIFIED_CHECKPOINT = False

# What this process actually loaded, filled in by ensure_local_checkpoint and
# reported by /health. A liveness probe that only says {"ok": true} cannot tell
# a correct daemon from one that outlived a pin bump — Bridge reuses any
# healthy listener (Bridge.lean's ensureServerRunning), so on 2026-08-12 a
# leftover daemon was indistinguishable from the intended fleet. The digest is
# already computed at load, so publishing it costs nothing.
_LOADED_CHECKPOINT: dict[str, object] = {}

# Retrieval assets live on the same HF repo as the checkpoint, under a
# ``premises/`` subfolder. The dense pair (embeddings + raw) is required
# for retrieval; the BM25 index is optional. On disk we flatten to
# ``cache_dir/<basename>`` so ``policy.py``'s ``load_premise_embeddings``
# (which reads ``cache_dir/premise_embeddings.npy`` etc.) finds them.
DEFAULT_RETRIEVAL_FILES = (
    "premises/premise_embeddings.npy",  # v1 (satp-policy-goal) layout
    "premises/premises_raw.npy",
    "premises/bm25_index.pkl",
    "cache/premise_embeddings.npy",  # v2 / v4.27 layout; absent names skip silently
    "cache/mathlib4_premises.txt",
    # names flatten to basenames — don't point two ckpt generations at one cache dir
)

_torch = None
_AesopPolicy = None
_to_lean4_string = None
_MAX_SEQUENCE_LENGTH = None
_LoRAConfig = None
_RichConsole = None
_RichSyntax = None
_LOG_LEVEL_COLORS = {
    "INFO": "blue",
    "WARNING": "yellow",
    "ERROR": "red",
}
_CUDA_ERROR_MARKERS = (
    "cuda error",
    "cudnn",
    "cublas",
    "cuda-capable",
    "cuda capability",
    "cuda driver",
    "cuda runtime",
    "no kernel image is available",
    "not compatible with the current pytorch installation",
    "torch not compiled with cuda enabled",
)


def _ensure_imports() -> None:
    """Import heavyweight LeanSATP runtime modules lazily."""
    global _torch, _AesopPolicy, _to_lean4_string, _MAX_SEQUENCE_LENGTH, _LoRAConfig
    if _torch is not None:
        return

    import torch as _t

    _torch = _t

    from leansatp_runtime.models import AesopPolicy as _AP, to_lean4_string as _tl
    from leansatp_runtime.models.components.premise_encoder import MAX_SEQUENCE_LENGTH
    from leansatp_runtime.models.components.lora_adapter import LoRAConfig as _LC

    _AesopPolicy, _to_lean4_string, _MAX_SEQUENCE_LENGTH, _LoRAConfig = (
        _AP,
        _tl,
        MAX_SEQUENCE_LENGTH,
        _LC,
    )


def _parse_hf_checkpoint_source(checkpoint_source: str) -> tuple[str, str]:
    """Split an hf:// checkpoint source into repo id and filename."""
    parts = checkpoint_source[len("hf://") :].split("/", 2)
    if len(parts) != 3:
        raise ValueError(
            "hf:// checkpoint paths must look like 'hf://<org>/<repo>/<filename>'"
        )
    return f"{parts[0]}/{parts[1]}", parts[2]


def _normalize_local_path(path: str) -> Path:
    """Expand a user-provided local path without requiring it to exist yet."""
    return Path(path).expanduser().resolve(strict=False)


def ensure_local_checkpoint(
    checkpoint_path: str = DEFAULT_CHECKPOINT, *, allow_unverified: bool = False
) -> str:
    """Require the checkpoint to exist, and to be the pinned one, before
    starting inference.

    Existence alone was the whole check until 2026-08-12, when it let a stale
    file start a service that then served the *previous* policy for five
    minutes without a single warning — a Bridge auto-start filled the gap
    between killing one fleet and launching the next. Cardinality guards do not
    catch that (every v4.27 run, and satp-policy-v2-alphaproof, share one
    decode surface), and neither does the directory name: ``cache/`` is what
    LeanSATP's ``main`` branch uses too, it is gitignored, and its contents
    therefore survive a branch switch in the same checkout.

    Identity is checked against the *revision*, never against the path. The
    first version of this guard only verified ``DEFAULT_CHECKPOINT`` on the
    theory that any other path had been typed by someone who meant it — which
    is wrong for the supported knob: ``SATP_CACHE_DIR`` relocates the cache,
    Bridge derives ``<dir>/best_checkpoint.pt`` from it (runtimeConfigFromEnv),
    and the intent there is still "serve the pin", just from elsewhere. Under
    the path rule a relocated cache silently degraded to a warning that Bridge
    auto-start discards with the child's stderr. So: hash every local
    checkpoint, and let only an explicit ``--allow-unverified-checkpoint``
    waive the comparison. An *inherited* setting must never be able to waive
    it, which is why an overridden ``SATP_HF_REVISION`` now fails closed rather
    than excusing itself — env vars reach a Bridge-spawned child from whatever
    shell started its parent, so "auto-start never passes the flag" only means
    something once the flag is the sole waiver.

    ~10 s to hash 1.07 GB, once per service start (single call site, the eager
    model-load path) — noise next to loading the weights, and it is what lets
    /health publish a real fingerprint instead of a bare liveness bit.
    """
    if checkpoint_path.startswith("hf://"):
        raise FileNotFoundError(
            "checkpoint path must be a local file, not an hf:// URI; "
            "run ./setup.sh to download the checkpoint first"
        )

    resolved = _normalize_local_path(checkpoint_path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {resolved}; run ./setup.sh to download it"
        )
    if not resolved.is_file():
        raise FileNotFoundError(f"checkpoint path is not a file: {resolved}")

    from leansatp_runtime.hf_pin import (
        CHECKPOINT_SHA256,
        HF_REPO,
        REVISION,
        is_default_revision,
    )

    h = hashlib.sha256()
    with open(resolved, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    got = h.hexdigest()
    matches_pin = got == CHECKPOINT_SHA256

    # Exactly one thing waives verification, and it is an argument someone
    # typed. An overridden SATP_HF_REVISION used to waive it too, on the
    # reasoning that we hold no digest for an arbitrary revision — but that
    # turns "I do not know" into "go ahead", and env vars are inherited: a
    # Bridge-spawned child picks up a leftover or mistyped SATP_HF_REVISION
    # from the shell that launched the parent, which made the claim "auto-start
    # is always verified" false. Not knowing is now a hard stop; say so and
    # make the operator opt in.
    waived = (
        "--allow-unverified-checkpoint was passed"
        if (allow_unverified or _ALLOW_UNVERIFIED_CHECKPOINT)
        else None
    )

    if waived is None and not is_default_revision():
        raise RuntimeError(
            f"SATP_HF_REVISION pins {REVISION[:8]}, and no checkpoint digest is "
            f"known for that revision, so {resolved} (sha256 {got[:16]}…) cannot "
            "be verified. Unset SATP_HF_REVISION, or pass "
            "--allow-unverified-checkpoint to serve it unverified."
        )
    if waived is None and not matches_pin:
        raise RuntimeError(
            f"{resolved} is not the best_checkpoint.pt that "
            f"{HF_REPO}@{REVISION[:8]} serves: sha256 {got[:16]}… but the pin "
            f"expects {CHECKPOINT_SHA256[:16]}…. Run ./setup.sh to refresh it, "
            "or pass --allow-unverified-checkpoint if you really mean to serve "
            "other weights."
        )
    if waived is not None and not matches_pin:
        log_server(
            "WARNING",
            f"{resolved} is not the pinned checkpoint (sha256 {got[:16]}… vs "
            f"{CHECKPOINT_SHA256[:16]}…); serving it anyway because {waived}",
        )

    _LOADED_CHECKPOINT.clear()
    _LOADED_CHECKPOINT.update(
        {
            "checkpoint": str(resolved),
            "checkpoint_sha256": got,
            "matches_pin": matches_pin,
            "repo": HF_REPO,
            "revision": REVISION,
        }
    )
    return str(resolved)


def ensure_checkpoint_download(
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    checkpoint_source: str = DEFAULT_CHECKPOINT_SOURCE,
) -> str:
    """Download the SATP checkpoint into an explicit local cache path."""
    destination = _normalize_local_path(checkpoint_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if checkpoint_source.startswith("hf://"):
        from huggingface_hub import hf_hub_download

        from leansatp_runtime.hf_pin import revision_for

        repo_id, filename = _parse_hf_checkpoint_source(checkpoint_source)
        # Resolved *outside* the suppression block on purpose: it warns when
        # repo_id is unpinned, and _suppress_startup_noise redirects both
        # stdout and stderr — inside, the one call site that fetches the
        # actual weights would be the only silent one.
        revision = revision_for(repo_id)
        with _suppress_startup_noise():
            downloaded = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    # pin the whole bundle (source + weights + assets) to one
                    # immutable revision — see hf_pin.py
                    revision=revision,
                )
            )
    else:
        downloaded = _normalize_local_path(checkpoint_source)
        if not downloaded.exists() or not downloaded.is_file():
            raise FileNotFoundError(f"checkpoint source not found: {downloaded}")

    if downloaded.resolve() != destination.resolve():
        shutil.copy2(downloaded, destination)

    return str(destination)


def ensure_retrieval_download(
    cache_dir: str = DEFAULT_CACHE_DIR,
    checkpoint_source: str = DEFAULT_CHECKPOINT_SOURCE,
    filenames: tuple[str, ...] = DEFAULT_RETRIEVAL_FILES,
) -> list[str]:
    """Best-effort fetch of retrieval assets from the checkpoint's HF repo.

    Files that are not present upstream are skipped silently so retrieval
    remains optional. Returns the paths of the assets that ended up in
    `cache_dir`.
    """
    if not checkpoint_source.startswith("hf://"):
        return []

    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError

    from leansatp_runtime.hf_pin import revision_for

    repo_id, _ = _parse_hf_checkpoint_source(checkpoint_source)
    revision = revision_for(repo_id)
    destination_dir = Path(cache_dir).expanduser().resolve(strict=False)
    destination_dir.mkdir(parents=True, exist_ok=True)

    downloaded: list[str] = []
    for filename in filenames:
        # Flatten any HF subdirectory so policy.py's
        # ``cache_dir/<basename>`` convention keeps working no matter
        # how the HF repo is organised.
        target = destination_dir / Path(filename).name
        try:
            with _suppress_startup_noise():
                source_path = Path(
                    hf_hub_download(
                        repo_id=repo_id, filename=filename, revision=revision
                    )
                )
        except (EntryNotFoundError, HfHubHTTPError):
            continue
        if source_path.resolve() != target.resolve():
            shutil.copy2(source_path, target)
        downloaded.append(str(target))
    return downloaded


def preferred_device() -> str:
    """Choose the default runtime device."""
    _ensure_imports()
    return "cuda" if _torch.cuda.is_available() else "cpu"


def _error_messages(exc: BaseException) -> list[str]:
    messages: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = str(current).strip()
        if text:
            messages.append(text)
        current = current.__cause__ or current.__context__
    return messages


def is_cuda_failure(exc: BaseException) -> bool:
    """Return true when an exception looks like a CUDA/device runtime failure."""
    combined = "\n".join(_error_messages(exc)).lower()
    return any(marker in combined for marker in _CUDA_ERROR_MARKERS)


def short_error_summary(exc: BaseException, limit: int = 240) -> str:
    """Compact an exception chain into a short log-friendly summary."""
    summary = " | ".join(_error_messages(exc)) or exc.__class__.__name__
    summary = " ".join(summary.split())
    if len(summary) <= limit:
        return summary
    return summary[: limit - 3] + "..."


def _ensure_rich() -> None:
    """Import rich lazily for Kimina-style terminal rendering."""
    global _RichConsole, _RichSyntax
    if _RichConsole is not None:
        return
    try:
        from rich.console import Console as _Console
        from rich.syntax import Syntax as _Syntax
    except ImportError:
        _RichConsole = False
        _RichSyntax = False
        return
    _RichConsole = _Console
    _RichSyntax = _Syntax


@contextmanager
def _suppress_startup_noise():
    """Hide third-party model-loading chatter during server startup."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    with (
        redirect_stdout(io.StringIO()),
        redirect_stderr(io.StringIO()),
        warnings.catch_warnings(),
    ):
        warnings.filterwarnings(
            "ignore",
            message=r".*cuda capability.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*Minimum and Maximum cuda capability supported.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*Please install PyTorch with a following CUDA.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*is not compatible with the current PyTorch installation.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*You are sending unauthenticated requests to the HF Hub.*",
            category=UserWarning,
        )
        yield


def _supports_color(stream) -> bool:
    """Enable colored terminal output only for interactive terminals unless forced."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    isatty = getattr(stream, "isatty", None)
    return bool(isatty and isatty())


def _make_console(stream):
    """Create a rich console configured like Kimina's terminal output."""
    _ensure_rich()
    if not _RichConsole:
        return None
    return _RichConsole(
        file=stream,
        force_terminal=True,
        color_system="truecolor",
        highlight=False,
        soft_wrap=True,
    )


def log_server(
    level: str,
    message: str,
    *,
    stream=None,
    enable_color: bool | None = None,
) -> None:
    """Emit Kimina-style prefixed logs from the SATP server."""
    stream = stream or sys.stderr
    if enable_color is None:
        enable_color = _supports_color(stream)

    prefix = f"{level:<8}"
    if enable_color:
        console = _make_console(stream)
        if console is not None:
            color = _LOG_LEVEL_COLORS.get(level, "white")
            console.print(f"[{color}]{prefix}[/{color}] {message}")
            return

    print(f"{prefix} {message}", file=stream)


_TRAINED_CHECKPOINT_REPO = "ChristianZ97/satp-policy-goal"


def _expected_lemma_k() -> int:
    from leansatp_runtime.models.components.heads import DEFAULT_LEMMA_K

    return DEFAULT_LEMMA_K


def _validate_checkpoint_shape(state_dict) -> None:
    """Fail fast when the checkpoint's hardcoded architecture parameters disagree."""
    calib_key = "lemma_heads.calibration.0.weight"
    expected_k = _expected_lemma_k()
    if calib_key not in state_dict:
        raise RuntimeError(
            f"checkpoint is missing {calib_key!r}; runtime targets {_TRAINED_CHECKPOINT_REPO}"
        )
    ckpt_lemma_k = state_dict[calib_key].shape[0]
    if ckpt_lemma_k != expected_k:
        raise RuntimeError(
            f"checkpoint LEMMA_K={ckpt_lemma_k} disagrees with runtime LEMMA_K={expected_k}; "
            f"runtime targets {_TRAINED_CHECKPOINT_REPO}"
        )


# T5 feed-forward sub-layer linears live at `…DenseReluDense.{wi_0,wi_1,wo}`.
# Training regimes through 2026-05 LoRA-wrapped them alongside attention; the
# `only_DPO_RL` regime folded the wrapped FF weights back into plain Linears
# before upload. The runtime auto-detects which format is on disk so both
# layouts load without an explicit config flag.
_FF_LORA_NAMES = ("wi_0", "wi_1", "wo")


def _detect_lora_targets(state_dict) -> tuple[str, ...]:
    """Return LoRA target_modules that reproduce the checkpoint's wrapper layout.

    Default (attention-only) matches current training. When FF sub-layers ship
    LoRA-decomposed keys (older checkpoints), extend targets to include
    ``wi_0/wi_1/wo`` so module construction recreates the same wrapper
    hierarchy and ``load_state_dict`` finds every key.
    """
    targets = list(_LoRAConfig().target_modules)
    has_ff_lora = any(
        f".DenseReluDense.{name}.lora_A" in key
        for key in state_dict
        for name in _FF_LORA_NAMES
    )
    if has_ff_lora:
        targets.extend(_FF_LORA_NAMES)
    return tuple(targets)


def _validate_state_dict_load(incompatible_keys) -> None:
    """Reject a state_dict load that is not a perfect parameter-name match."""
    missing = list(incompatible_keys.missing_keys)
    unexpected = list(incompatible_keys.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(
            "checkpoint architecture mismatch: "
            f"missing_keys={missing[:3]!r} ({len(missing)} total), "
            f"unexpected_keys={unexpected[:3]!r} ({len(unexpected)} total); "
            f"runtime targets {_TRAINED_CHECKPOINT_REPO}"
        )


def load_policy(checkpoint_path: str, cache_dir: str, device: str | None = None):
    """Load the LeanSATP policy model once and keep it resident."""
    _ensure_imports()

    checkpoint_path = ensure_local_checkpoint(checkpoint_path)
    device = device or preferred_device()

    # weights_only=False: the ckpt embeds numpy objects; acceptable because
    # every hf:// artifact is pinned to one immutable revision (hf_pin.py)
    ckpt = _torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    # v2 ckpts carry tactic_heads.tactic_emb, v1 tactic_heads.safe_group.*;
    # route on the fingerprint (same spirit as the FF-LoRA auto-detect below).
    if "tactic_heads.tactic_emb" in state_dict:
        from leansatp_runtime.models import policy_v2 as _pv2

        log_server(
            "INFO",
            "[LeanSATP] v2 checkpoint detected (factored joint-action heads); "
            "using policy_v2 runtime",
        )
        with _suppress_startup_noise():
            model = _pv2.build_policy_v2(ckpt, cache_dir, device=device)
        try:
            model.load_premise_embeddings()
            retrieval_enabled = True
        except FileNotFoundError as exc:
            retrieval_enabled = False
            log_server(
                "INFO",
                f"[LeanSATP] Retrieval disabled: {exc}",
            )
        model.to(device)
        model.eval()
        return model, device, retrieval_enabled

    _validate_checkpoint_shape(state_dict)

    lora_targets = _detect_lora_targets(state_dict)
    if set(lora_targets) != set(_LoRAConfig().target_modules):
        log_server(
            "INFO",
            f"[LeanSATP] checkpoint LoRA targets: {lora_targets} (legacy FF-LoRA detected)",
        )

    with _suppress_startup_noise():
        model = _AesopPolicy(
            use_lora=True,
            lora_config=_LoRAConfig(target_modules=lora_targets),
            device=device,
            cache_dir=cache_dir,
        )
    incompatible = model.load_state_dict(state_dict, strict=False)
    _validate_state_dict_load(incompatible)
    try:
        model.load_premise_embeddings()
        retrieval_enabled = True
    except FileNotFoundError as exc:
        retrieval_enabled = False
        log_server(
            "INFO",
            f"[LeanSATP] Retrieval disabled: {exc}",
        )
    model.to(device)
    model.eval()
    return model, device, retrieval_enabled


def render_full_proof(formal_statement: str, tactic: str) -> str:
    """Combine the theorem prompt and generated tactic into a proof sketch."""
    return formal_statement.rstrip() + "\n" + textwrap.indent(tactic.rstrip(), "  ")


def _trace_label(policy_input: str) -> str:
    return "Policy input" if "⊢" in policy_input else "Full proof"


def _indent_block(block: str, prefix: str = "    ") -> str:
    """Indent a multi-line block for human-readable stderr traces."""
    cleaned = block.rstrip() or "<empty>"
    return textwrap.indent(cleaned, prefix)


def render_full_proof_trace(
    formal_statement: str,
    tactic: str,
    *,
    device: str,
) -> str:
    """Render the generated proof as a plain-text stderr block."""
    label = _trace_label(formal_statement)
    return "\n".join(
        [
            f"[LeanSATP] {label} ({device}):",
            _indent_block(render_full_proof(formal_statement, tactic)),
        ]
    )


def print_full_proof_trace(
    formal_statement: str,
    tactic: str,
    *,
    device: str,
    stream=None,
    enable_color: bool | None = None,
) -> None:
    """Print the generated proof using Kimina-style rich formatting when possible."""
    stream = stream or sys.stderr
    if enable_color is None:
        enable_color = _supports_color(stream)

    _ensure_rich()
    if enable_color and _RichConsole and _RichSyntax:
        console = _make_console(stream)
        label = _trace_label(formal_statement)
        log_server(
            "INFO",
            f"[bold magenta][LeanSATP] {label}[/bold magenta] "
            f"([bold yellow]{device}[/bold yellow]):",
            stream=stream,
            enable_color=True,
        )
        console.print(
            _RichSyntax(
                render_full_proof(formal_statement, tactic),
                "lean",
                theme="monokai",
                line_numbers=False,
                word_wrap=True,
            )
        )
        return

    print(
        f"INFO     {render_full_proof_trace(formal_statement, tactic, device=device)}",
        file=stream,
    )


def _curl_example_host(host: str) -> str:
    """Choose a copy-paste-friendly host for local curl examples."""
    if host in {"0.0.0.0", "::", ""}:
        return "localhost"
    if host == "127.0.0.1":
        return "localhost"
    return host


def render_try_me_message(host: str, port: int) -> str:
    """Render a Kimina-style curl hint for manual server checks."""
    curl_host = _curl_example_host(host)
    return "Try me with:\n" + textwrap.indent(
        "curl --request POST \\\n"
        f"  --url http://{curl_host}:{port}/infer \\\n"
        "  --header 'Content-Type: application/json' \\\n"
        '  --data \'{"formal_statement":"⊢ True"}\' | jq\n',
        "  ",
    )


def _interrupt_server(_signum, _frame) -> None:
    """Convert termination signals into the same clean shutdown path as Ctrl+C."""
    raise KeyboardInterrupt


def append_user_lemmas(
    tactic: str,
    user_lemmas: Optional[list[str]],
    priority_pct: int = 40,
) -> str:
    """Append explicit user lemmas as extra unsafe Aesop rules."""
    if not user_lemmas:
        return tactic

    seen: set[str] = set()
    extra_rules: list[str] = []
    for lemma_name in user_lemmas:
        lemma_name = (lemma_name or "").strip()
        if not lemma_name or lemma_name in seen:
            continue
        seen.add(lemma_name)
        extra_rules.append(
            "    "
            f"(add unsafe {priority_pct}% "
            f"(by first | apply {lemma_name} | rw [{lemma_name}] | simp only [{lemma_name}]))"
        )

    if not extra_rules:
        return tactic
    return tactic.rstrip() + "\n" + "\n".join(extra_rules)


# Pattern for retrieval-generated lemma rules in an aesop config.  These are
# emitted by `_to_lean4_string` with the same shape as `append_user_lemmas`
# (apply / rw / simp only) and always land below 50% priority (lemma tier).
# Tactic-tier rules (safe 1, unsafe 55-100%, or multi-line wrappers for
# things like `by field_simp`) never match this pattern.
_RETRIEVAL_LEMMA_RE = re.compile(
    r"^\s*\(add\s+unsafe\s+(\d+)%\s+"
    r"\(by\s+first\s+\|\s+apply\s+\S+\s+\|\s+rw\s+\[\S+\]\s+\|\s+simp\s+only\s+\[\S+\]\)\)\s*$"
)


def strip_retrieval_rules(tactic: str) -> str:
    """Drop retrieval-sourced lemma rules (<50%) from an aesop config.

    Used by the satp? cascade's Layer 1: when we want the sketch-provided
    hypotheses to be the sole lemma source, we first remove the
    policy-attention retrieved premises from the config so they don't
    compete with or outrank the hints.  Tactic-tier entries
    (safe / unsafe 55-100%, multi-line wrappers) are preserved.
    """
    kept: list[str] = []
    for line in tactic.split("\n"):
        match = _RETRIEVAL_LEMMA_RE.match(line)
        if match and int(match.group(1)) < 50:
            continue
        kept.append(line)
    return "\n".join(kept)


def policy_tactic(
    model_and_device,
    formal_statement: str,
    *,
    tactic_name: str = "aesop",
    user_lemmas: Optional[list[str]] = None,
    user_lemma_priority: Optional[int] = None,
    strip_retrieval: bool = False,
) -> str:
    """Greedy policy inference. ``strip_retrieval`` only acts here for v2
    (decode-time); the v1 path ignores it and keeps the caller-side text strip."""
    _ensure_imports()

    model, device, retrieval_enabled = model_and_device

    if getattr(model, "arch", "v1") == "v2":
        from leansatp_runtime.models import policy_v2 as _pv2

        tactic = _pv2.policy_tactic_v2(
            model,
            device,
            formal_statement,
            retrieval_enabled=retrieval_enabled,
            tactic_name=tactic_name,
            strip_retrieval=strip_retrieval,
        )
        kwargs: dict = {}
        if user_lemma_priority is not None:
            kwargs["priority_pct"] = user_lemma_priority
        return append_user_lemmas(tactic, user_lemmas, **kwargs)

    top_k_premises = None
    lemma_scores = None
    lemma_embs = None
    if retrieval_enabled and model.has_premise_cache():
        top_k_premises, lemma_scores, lemma_embs = model.retrieve(formal_statement)

    inputs = model.tokenizer(
        [formal_statement],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=_MAX_SEQUENCE_LENGTH,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with _torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            lemma_scores=lemma_scores.unsqueeze(0).to(device)
            if lemma_scores is not None
            else None,
            lemma_embeddings=lemma_embs.unsqueeze(0).to(device)
            if lemma_embs is not None
            else None,
        )

    tactic_logits = outputs["tactic_logits"]
    safe_actions = tactic_logits["safe"][0].argmax(dim=-1).tolist()
    unsafe_actions = tactic_logits["unsafe"][0].argmax(dim=-1).tolist()

    lemma_logits = outputs["lemma_logits"][0]
    residual_logits = outputs.get("residual_logits")
    if residual_logits is not None:
        lemma_logits = lemma_logits + residual_logits[0]
    lemma_actions = lemma_logits.argmax(dim=-1).tolist()

    config_logits = outputs["config_logits"]
    config_level = config_logits["level"][0].argmax(dim=-1).tolist()
    config_binary = (config_logits["binary"][0].squeeze(-1) > 0).int().tolist()

    # --- SATP component ablation (default OFF; leave-one-out study) ---
    # Each flag nulls exactly one emitted component so the rest of the config
    # is identical to the full model. Read per-call from env; with no flag set
    # this path is byte-identical to the unablated emit (live services unaffected).
    if os.environ.get("SATP_ABLATE_RETRIEVAL") == "1":
        lemma_actions = [0] * len(lemma_actions)  # drop lemma/premise rules
    if os.environ.get("SATP_ABLATE_TACTIC_PRIO") == "1":
        safe_actions = [0] * len(safe_actions)  # drop learned safe rules
        unsafe_actions = [0] * len(unsafe_actions)  # drop learned unsafe rules
    if os.environ.get("SATP_ABLATE_BUDGET") == "1":
        config_level = None  # -> DEFAULT_AESOP_CONFIG budget

    tactic = _to_lean4_string(
        safe_actions=safe_actions,
        unsafe_actions=unsafe_actions,
        lemma_actions=lemma_actions,
        lemma_premises=top_k_premises,
        config_level_actions=config_level,
        config_binary_actions=config_binary,
        tactic_name=tactic_name,
    )
    kwargs: dict = {}
    if user_lemma_priority is not None:
        kwargs["priority_pct"] = user_lemma_priority
    return append_user_lemmas(tactic, user_lemmas, **kwargs)


class SATPInferenceEngine:
    """Resident LeanSATP inference engine."""

    def __init__(
        self,
        checkpoint_path: str = DEFAULT_CHECKPOINT,
        cache_dir: str = DEFAULT_CACHE_DIR,
    ):
        self.checkpoint_path = checkpoint_path
        self.cache_dir = cache_dir
        self._lock = threading.Lock()
        self._downgrade_logged = False
        self.preferred_device = preferred_device()
        self.active_device = self.preferred_device
        self.retrieval_enabled = False
        self.model_and_device = self._load_with_fallback(self.preferred_device)

    def _log_cpu_downgrade(self, exc: BaseException, *, stage: str) -> None:
        if self._downgrade_logged:
            return
        summary = short_error_summary(exc)
        log_server(
            "WARNING",
            f"[LeanSATP] Warning: falling back from cuda to cpu during {stage}: {summary}",
        )
        self._downgrade_logged = True

    def _set_model_state(self, model_and_device) -> None:
        self.model_and_device = model_and_device
        self.active_device = model_and_device[1]
        self.retrieval_enabled = model_and_device[2]

    def _log_inference_trace(self, formal_statement: str, tactic: str) -> None:
        print_full_proof_trace(
            formal_statement,
            tactic,
            device=self.active_device,
            stream=sys.stderr,
        )

    def _load_engine(self, device: str):
        model_and_device = load_policy(
            self.checkpoint_path, self.cache_dir, device=device
        )
        self._set_model_state(model_and_device)
        return model_and_device

    def _load_with_fallback(self, device: str):
        try:
            return self._load_engine(device)
        except Exception as exc:
            if device != "cuda" or not is_cuda_failure(exc):
                raise
            self._log_cpu_downgrade(exc, stage="startup")
            return self._load_engine("cpu")

    def _downgrade_to_cpu_locked(self, exc: BaseException, *, stage: str) -> None:
        if self.active_device != "cuda":
            return
        self._log_cpu_downgrade(exc, stage=stage)
        self._load_engine("cpu")

    def infer(
        self,
        *,
        formal_statement: str,
        user_lemmas: Optional[list[str]] = None,
        tactic_name: str = "aesop",
        user_lemma_priority: Optional[int] = None,
        strip_retrieval: bool = False,
        hint_priority: Optional[int] = None,
    ) -> dict[str, Any]:
        formal_statement = (formal_statement or "").strip()
        if not formal_statement:
            raise ValueError("infer requires a non-empty formal_statement")

        # Run the policy WITHOUT appending user lemmas; user_lemma append
        # and retrieval strip are handled below so the order is explicit:
        #   policy output  →  (optional) strip retrieval rules  →  append hints
        policy_kwargs = dict(tactic_name=tactic_name, strip_retrieval=strip_retrieval)
        with self._lock:
            try:
                tactic = policy_tactic(
                    self.model_and_device,
                    formal_statement,
                    **policy_kwargs,
                )
            except Exception as exc:
                if self.active_device != "cuda" or not is_cuda_failure(exc):
                    raise
                self._downgrade_to_cpu_locked(exc, stage="inference")
                tactic = policy_tactic(
                    self.model_and_device,
                    formal_statement,
                    **policy_kwargs,
                )

        # Layer 1 of the satp? cascade: we want the sketch-provided hint
        # lemmas to be the only lemma source, so drop the retrieval
        # suggestions the policy just emitted.  Tactic-tier rules
        # (safe / unsafe ≥55%) survive.
        if strip_retrieval:
            tactic = strip_retrieval_rules(tactic)

        # Append hint lemmas at the caller's requested priority.  If the
        # caller (old API) passed user_lemma_priority but not hint_priority,
        # honour the former for backward compatibility.
        effective_priority = (
            hint_priority
            if hint_priority is not None
            else (user_lemma_priority if user_lemma_priority is not None else 40)
        )
        tactic = append_user_lemmas(
            tactic, user_lemmas, priority_pct=effective_priority
        )

        with self._lock:
            self._log_inference_trace(formal_statement, tactic)
        return {
            "tactic": tactic,
            "formal_statement": formal_statement,
        }

    def summary(self) -> str:
        """Describe the currently loaded inference engine state."""
        retrieval = "enabled" if self.retrieval_enabled else "disabled"
        return (
            "SATP inference engine initialized with: "
            f"PREFERRED_DEVICE=[bold]{self.preferred_device}[/bold], "
            f"ACTIVE_DEVICE=[bold]{self.active_device}[/bold], "
            f"RETRIEVAL=[bold]{retrieval}[/bold]"
        )


def _optional_str_field(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if value and not isinstance(value, str):
        raise ValueError(f"request field {key!r} must be a string when provided")
    return value or ""


class _SATPRequestHandler(BaseHTTPRequestHandler):
    server: "_SATPHTTPServer"

    def log_message(self, format: str, *args) -> None:
        return

    def _write_json(self, status: int, payload: dict[str, Any]) -> bool:
        """Write a JSON response. Returns False if the client already hung up."""
        body = json.dumps(payload).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            log_server(
                "WARNING",
                f"client disconnected before response {status} {self.command} {self.path}",
            )
            return False
        log_server("INFO", f"← response {status} {self.command} {self.path}")
        return True

    def _log_request_start(self, detail: str = "") -> None:
        suffix = f" [bold magenta]{detail}[/bold magenta]" if detail else ""
        log_server("INFO", f"→ request {self.command} {self.path}{suffix}")

    def do_GET(self) -> None:
        self._log_request_start()
        if self.path != "/health":
            self._write_json(404, {"ok": False, "error": "not found"})
            return
        # Liveness plus identity. `{"ok": true}` alone cannot distinguish the
        # intended fleet from a daemon that outlived a pin bump, and Bridge
        # reuses any healthy listener without asking what it loaded. The model
        # is loaded eagerly before the socket accepts, so these fields are
        # always populated by the time this can answer.
        self._write_json(200, {"ok": True, **_LOADED_CHECKPOINT})

    def do_POST(self) -> None:
        if self.path != "/infer":
            self._log_request_start()
            self._write_json(404, {"ok": False, "error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
            formal_statement = _optional_str_field(payload, "formal_statement")
            if not formal_statement:
                raise ValueError(
                    "request body must contain string field 'formal_statement'"
                )

            name = payload.get("name") or None
            detail = name or formal_statement.split("\n", 1)[0]
            self._log_request_start(detail)

            user_lemmas = payload.get("user_lemmas") or []
            tactic_name = payload.get("tactic_name", "aesop")
            user_lemma_priority = payload.get("user_lemma_priority")
            # Two new per-request knobs used by the satp? 3-layer cascade in
            # Bridge.lean.  strip_retrieval removes the retrieval-sourced
            # (<50%) lemma rules from the policy's aesop config before
            # returning; hint_priority controls the priority_pct used when
            # the hint lemmas are appended on top.  Defaults preserve the
            # pre-cascade single-call behaviour.
            strip_retrieval = bool(payload.get("strip_retrieval") or False)
            hint_priority = payload.get("hint_priority")
            if hint_priority is None:
                hint_priority = 40  # match legacy append_user_lemmas default
            else:
                hint_priority = int(hint_priority)

            # LRU cache lookup before acquiring the semaphore — cache hits
            # dodge GPU work entirely.  Key shape matches engine.infer's
            # kwargs so different (tactic_name, hints, priority, strip,
            # hint_priority) still miss cleanly.
            cache_key = (
                formal_statement,
                tactic_name,
                tuple(user_lemmas),
                user_lemma_priority,
                strip_retrieval,
                hint_priority,
            )
            cached = None
            if self.server.infer_cache_max > 0:
                with self.server.infer_cache_lock:
                    if cache_key in self.server.infer_cache:
                        cached = self.server.infer_cache[cache_key]
                        self.server.infer_cache.move_to_end(cache_key)
                        self.server.infer_cache_hits += 1
                    else:
                        self.server.infer_cache_misses += 1
            if cached is not None:
                log_server(
                    "INFO",
                    f"cache HIT ({self.server.infer_cache_hits} hits / "
                    f"{self.server.infer_cache_misses} misses)",
                )
                self._write_json(200, {"ok": True, **cached})
                return

            # Gate concurrent inference on a bounded semaphore.  Without
            # this, ThreadingHTTPServer spins up one thread per request;
            # the GIL + single-GPU forward pass then turn dozens of
            # in-flight requests into a stampede that pushes per-call
            # latency past Bridge.lean's curl timeout.  Acquire with a
            # bound below the client's timeout so a starving request gets
            # a clean 503 rather than hanging up the thread.
            acquired = self.server.inference_sem.acquire(
                timeout=_SEMAPHORE_ACQUIRE_TIMEOUT
            )
            if not acquired:
                log_server(
                    "WARNING",
                    f"inference queue full ({self.server.max_inflight} in-flight); "
                    f"returning 503 for {detail}",
                )
                self._write_json(
                    503,
                    {"ok": False, "error": "SATP server busy (max in-flight reached)"},
                )
                return
            try:
                result = self.server.engine.infer(
                    formal_statement=formal_statement,
                    user_lemmas=user_lemmas,
                    tactic_name=tactic_name,
                    user_lemma_priority=user_lemma_priority,
                    strip_retrieval=strip_retrieval,
                    hint_priority=hint_priority,
                )
                if self.server.infer_cache_max > 0:
                    with self.server.infer_cache_lock:
                        self.server.infer_cache[cache_key] = result
                        self.server.infer_cache.move_to_end(cache_key)
                        while (
                            len(self.server.infer_cache) > self.server.infer_cache_max
                        ):
                            self.server.infer_cache.popitem(last=False)
                self._write_json(200, {"ok": True, **result})
            finally:
                self.server.inference_sem.release()
        except (BrokenPipeError, ConnectionResetError):
            log_server(
                "WARNING",
                f"client disconnected mid-request {self.command} {self.path}",
            )
        except Exception as exc:
            self._write_json(500, {"ok": False, "error": str(exc)})


class _SATPHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address,
        engine: SATPInferenceEngine | None = None,
        *,
        bind_and_activate: bool = True,
    ):
        super().__init__(
            server_address,
            _SATPRequestHandler,
            bind_and_activate=bind_and_activate,
        )
        self.engine = engine
        # Bounded semaphore caps concurrent inference.  Overridable via
        # SATP_MAX_INFLIGHT; default 8 keeps GIL + single-GPU forward
        # within throughput before latency blows past client timeouts.
        try:
            max_inflight = int(
                os.environ.get("SATP_MAX_INFLIGHT", _DEFAULT_MAX_INFLIGHT)
            )
        except ValueError:
            max_inflight = _DEFAULT_MAX_INFLIGHT
        if max_inflight < 1:
            max_inflight = _DEFAULT_MAX_INFLIGHT
        self.max_inflight = max_inflight
        self.inference_sem = threading.BoundedSemaphore(max_inflight)

        # LRU cache of /infer results keyed by (formal_statement, tactic_name,
        # user_lemmas tuple, user_lemma_priority).  A single DSP eval typically
        # re-requests the same formal statement many times (different sketch
        # attempts rewrite the same theorem header, retries re-send the same
        # goal).  Policy forward is deterministic given inputs, so a cache hit
        # short-circuits both the semaphore and the GPU work.
        try:
            infer_cache_size = int(
                os.environ.get("SATP_INFER_CACHE_SIZE", _DEFAULT_INFER_CACHE_SIZE)
            )
        except ValueError:
            infer_cache_size = _DEFAULT_INFER_CACHE_SIZE
        if infer_cache_size < 0:
            infer_cache_size = 0
        self.infer_cache_max = infer_cache_size
        self.infer_cache: "OrderedDict[tuple, dict]" = OrderedDict()
        self.infer_cache_lock = threading.Lock()
        self.infer_cache_hits = 0
        self.infer_cache_misses = 0


def _is_address_in_use(exc: BaseException) -> bool:
    """Return true when a socket bind failed because the port is occupied."""
    return isinstance(exc, OSError) and exc.errno == errno.EADDRINUSE


def _log_port_in_use_help(host: str, port: int) -> None:
    """Explain how to recover when the SATP server port is already occupied."""
    log_server(
        "WARNING",
        f"[LeanSATP] Port {port} is already in use on {host}.",
    )
    log_server(
        "WARNING",
        "[LeanSATP] Another process is already listening on the SATP server port.",
    )
    log_server(
        "WARNING",
        "[LeanSATP] If that is an existing SATP server, reuse it instead of starting a second copy.",
    )
    log_server(
        "INFO",
        f"[LeanSATP] To inspect the current listener: lsof -i :{port}",
    )
    log_server(
        "INFO",
        f"[LeanSATP] To stop it and restart SATP: fuser -k {port}/tcp",
    )


def _log_missing_checkpoint_help(exc: FileNotFoundError) -> None:
    """Explain how to recover when the local checkpoint is missing."""
    log_server("ERROR", f"[LeanSATP] {exc}")
    log_server(
        "INFO",
        "[LeanSATP] Run ./setup.sh from the repository root to install dependencies, fetch mathlib, and download the checkpoint.",
    )


def serve(
    *,
    host: str,
    port: int,
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> int:
    """Start the resident LeanSATP HTTP server."""
    pid = os.getpid()
    server: _SATPHTTPServer | None = None
    previous_sigint = None
    previous_sigterm = None
    exit_code = 0

    log_server("INFO", f"Started server process [{pid}]")
    log_server("INFO", "Waiting for application startup.")
    try:
        server = _SATPHTTPServer((host, port), bind_and_activate=False)
        server.server_bind()

        engine = SATPInferenceEngine(
            checkpoint_path=checkpoint_path,
            cache_dir=cache_dir,
        )
        server.engine = engine
        server.server_activate()

        previous_sigint = signal.getsignal(signal.SIGINT)
        previous_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, _interrupt_server)
        signal.signal(signal.SIGTERM, _interrupt_server)
        log_server("INFO", engine.summary())
        log_server(
            "INFO",
            f"Inference concurrency cap: "
            f"MAX_INFLIGHT=[bold]{server.max_inflight}[/bold] "
            f"(override with SATP_MAX_INFLIGHT env var)",
        )
        log_server(
            "INFO",
            f"Inference LRU cache: "
            f"SIZE=[bold]{server.infer_cache_max}[/bold] entries "
            f"(override with SATP_INFER_CACHE_SIZE env var; 0 disables)",
        )
        log_server("INFO", "Application startup complete.")
        log_server(
            "INFO",
            f"Running [bold]LeanSATP Server[/bold] on [bold]http://{host}:{port}[/bold] (Press CTRL+C to quit)",
        )
        log_server("INFO", render_try_me_message(host, port))
        server.serve_forever()
    except KeyboardInterrupt:
        log_server("INFO", "Shutting down")
    except FileNotFoundError as exc:
        _log_missing_checkpoint_help(exc)
        exit_code = 1
    except OSError as exc:
        if _is_address_in_use(exc):
            _log_port_in_use_help(host, port)
            exit_code = 1
        else:
            raise
    finally:
        log_server("INFO", "Waiting for application shutdown.")
        if server is not None:
            server.server_close()
        if previous_sigint is not None:
            signal.signal(signal.SIGINT, previous_sigint)
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)
        log_server("INFO", "Application shutdown complete.")
        log_server("INFO", f"Finished server process [{pid}]")
    return exit_code


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LeanSATP inference server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5177)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--checkpoint-source", default=DEFAULT_CHECKPOINT_SOURCE)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--allow-unverified-checkpoint",
        action="store_true",
        help=(
            "serve a checkpoint whose SHA-256 does not match the pinned "
            "revision (era comparisons, self-trained weights). Bridge "
            "auto-start never passes this, so an auto-started service is "
            "always verified."
        ),
    )
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument(
        "--skip-retrieval",
        action="store_true",
        help="Skip downloading retrieval assets (premise embeddings, raw premises, BM25 index) during --download-only",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    global _ALLOW_UNVERIFIED_CHECKPOINT
    _ALLOW_UNVERIFIED_CHECKPOINT = args.allow_unverified_checkpoint

    if args.download_only:
        resolved = ensure_checkpoint_download(args.checkpoint, args.checkpoint_source)
        retrieval_paths: list[str] = []
        if not args.skip_retrieval:
            retrieval_paths = ensure_retrieval_download(
                args.cache_dir, args.checkpoint_source
            )
        # prefetch + validate the pinned inference source so a later offline
        # service start never has to contact HF (importing runs the download);
        # only for the v2 repo — v1/custom sources never touch the v2 pin
        from leansatp_runtime.hf_pin import HF_REPO as _pin_repo

        if args.checkpoint_source.startswith(f"hf://{_pin_repo}/"):
            from leansatp_runtime.models import policy_v2 as _pv2  # noqa: F401

        print(
            json.dumps(
                {
                    "ok": True,
                    "checkpoint_path": resolved,
                    "retrieval_paths": retrieval_paths,
                }
            )
        )
        return 0

    if args.serve:
        return serve(
            host=args.host,
            port=args.port,
            checkpoint_path=args.checkpoint,
            cache_dir=args.cache_dir,
        )

    log_server("ERROR", "Use --serve or --download-only")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
