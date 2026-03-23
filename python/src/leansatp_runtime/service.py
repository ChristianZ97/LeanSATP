"""PyTorch-backed SATP inference service."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import errno
import io
import json
import os
import re
import signal
import shutil
import sys
import textwrap
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
from typing import Any, Optional

DEFAULT_CHECKPOINT_SOURCE = "hf://ChristianZ97/SATP-aesop-policy/best_checkpoint.pt"
_PACKAGE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR = str(_PACKAGE_ROOT / "cache")
DEFAULT_CHECKPOINT = str(Path(DEFAULT_CACHE_DIR) / "best_checkpoint.pt")

_torch = None
_config = None
_AesopPolicy = None
_to_lean4_string = None
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
    global _torch, _config, _AesopPolicy, _to_lean4_string, _LoRAConfig
    if _torch is not None:
        return

    import torch as _t

    _torch = _t

    from leansatp_runtime.config import config as _c
    from leansatp_runtime.models.components import LoRAConfig as _LC
    from leansatp_runtime.models.policy import (
        AesopPolicy as _AP,
        to_lean4_string as _tl,
    )

    _config, _AesopPolicy, _to_lean4_string, _LoRAConfig = _c, _AP, _tl, _LC


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


def ensure_local_checkpoint(checkpoint_path: str = DEFAULT_CHECKPOINT) -> str:
    """Require a local checkpoint file to exist before starting inference."""
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

        repo_id, filename = _parse_hf_checkpoint_source(checkpoint_source)
        with _suppress_startup_noise():
            downloaded = Path(hf_hub_download(repo_id=repo_id, filename=filename))
    else:
        downloaded = _normalize_local_path(checkpoint_source)
        if not downloaded.exists() or not downloaded.is_file():
            raise FileNotFoundError(f"checkpoint source not found: {downloaded}")

    if downloaded.resolve() != destination.resolve():
        shutil.copy2(downloaded, destination)

    return str(destination)


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
    """Hide third-party model-loading chatter during service startup."""
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


def log_service(
    level: str,
    message: str,
    *,
    stream=None,
    enable_color: bool | None = None,
) -> None:
    """Emit Kimina-style prefixed logs from the SATP service."""
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


def load_policy(checkpoint_path: str, cache_dir: str, device: str | None = None):
    """Load the LeanSATP policy model once and keep it resident."""
    _ensure_imports()

    checkpoint_path = ensure_local_checkpoint(checkpoint_path)
    device = device or preferred_device()

    ckpt = _torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    calib_key = "lemma_heads.calibration.0.weight"
    if calib_key in state_dict:
        ckpt_lemma_k = state_dict[calib_key].shape[0]
        if ckpt_lemma_k != _config.LEMMA_K:
            log_service(
                "INFO",
                f"[LeanSATP] Overriding LEMMA_K: {_config.LEMMA_K} -> {ckpt_lemma_k}",
            )
            _config.LEMMA_K = ckpt_lemma_k

    use_lora = getattr(_config, "USE_LORA", False)
    lora_cfg = None
    if use_lora:
        lora_cfg = _LoRAConfig(
            r=getattr(_config, "LORA_R", 16),
            lora_alpha=getattr(_config, "LORA_ALPHA", 32),
            lora_dropout=getattr(_config, "LORA_DROPOUT", 0.1),
            target_modules=getattr(
                _config, "LORA_TARGET_MODULES", ("q", "k", "v", "o")
            ),
        )

    with _suppress_startup_noise():
        model = _AesopPolicy(
            freeze_base=True,
            use_lora=use_lora,
            lora_config=lora_cfg,
            device=device,
            cache_dir=cache_dir,
        )
    model.load_state_dict(state_dict, strict=False)
    try:
        model.load_premise_embeddings()
        retrieval_enabled = True
    except FileNotFoundError as exc:
        retrieval_enabled = False
        log_service(
            "INFO",
            f"[LeanSATP] Retrieval disabled: {exc}",
        )
    model.to(device)
    model.eval()
    return model, device, retrieval_enabled


def _sanitize_name(name: str, index: int) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_']", "_", (name or "").strip())
    cleaned = cleaned or f"h{index}"
    if cleaned[0].isdigit():
        cleaned = f"h_{cleaned}"
    return cleaned


def build_formal_statement(goal: str, hypotheses: list[dict[str, str]]) -> str:
    """Convert the current Lean goal state into a theorem-like prompt."""
    lines = ["theorem satp_goal"]
    for i, hyp in enumerate(hypotheses):
        name = _sanitize_name(hyp.get("name", ""), i)
        hyp_type = (hyp.get("type", "") or "").strip()
        if hyp_type:
            lines.append(f"  ({name} : {hyp_type})")
    lines.append(f"  : {(goal or '').strip()} := by")
    return "\n".join(lines)


def render_full_proof(formal_statement: str, tactic: str) -> str:
    """Combine the theorem prompt and generated tactic into a proof sketch."""
    return formal_statement.rstrip() + "\n" + textwrap.indent(tactic.rstrip(), "  ")


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
    return "\n".join(
        [
            f"[LeanSATP] Full proof ({device}):",
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
        log_service(
            "INFO",
            f"[bold magenta][LeanSATP] Full proof[/bold magenta] "
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
    """Render a Kimina-style curl hint for manual service checks."""
    curl_host = _curl_example_host(host)
    return "Try me with:\n" + textwrap.indent(
        "curl --request POST \\\n"
        f"  --url http://{curl_host}:{port}/infer \\\n"
        "  --header 'Content-Type: application/json' \\\n"
        '  --data \'{"goal":"True"}\' | jq\n',
        "  ",
    )


def _interrupt_service(_signum, _frame) -> None:
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


def policy_tactic(
    model_and_device,
    formal_statement: str,
    *,
    tactic_name: str = "aesop",
    user_lemmas: Optional[list[str]] = None,
) -> str:
    """Generate a LeanSATP tactic by greedy policy inference."""
    _ensure_imports()

    model, device, retrieval_enabled = model_and_device
    top_k_premises = None
    lemma_scores = None
    lemma_embs = None
    if retrieval_enabled and model.has_premise_cache():
        top_k_premises, lemma_scores, lemma_embs = model.retrieve(
            formal_statement, k=_config.LEMMA_K
        )

    inputs = model.tokenizer(
        [formal_statement],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=_config.MAX_SEQUENCE_LENGTH,
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

    tactic = _to_lean4_string(
        safe_actions=safe_actions,
        unsafe_actions=unsafe_actions,
        lemma_actions=lemma_actions,
        lemma_premises=top_k_premises,
        config_level_actions=config_level,
        config_binary_actions=config_binary,
        tactic_name=tactic_name,
    )
    return append_user_lemmas(tactic, user_lemmas)


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
        log_service(
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
        goal: str,
        hypotheses: Optional[list[dict[str, str]]] = None,
        user_lemmas: Optional[list[str]] = None,
        tactic_name: str = "aesop",
    ) -> dict[str, Any]:
        hypotheses = hypotheses or []
        formal_statement = build_formal_statement(goal, hypotheses)
        with self._lock:
            try:
                tactic = policy_tactic(
                    self.model_and_device,
                    formal_statement,
                    tactic_name=tactic_name,
                    user_lemmas=user_lemmas,
                )
            except Exception as exc:
                if self.active_device != "cuda" or not is_cuda_failure(exc):
                    raise
                self._downgrade_to_cpu_locked(exc, stage="inference")
                tactic = policy_tactic(
                    self.model_and_device,
                    formal_statement,
                    tactic_name=tactic_name,
                    user_lemmas=user_lemmas,
                )
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
            f"PREFERRED_DEVICE={self.preferred_device}, "
            f"ACTIVE_DEVICE={self.active_device}, "
            f"RETRIEVAL={retrieval}"
        )


class _SATPRequestHandler(BaseHTTPRequestHandler):
    server: "_SATPHTTPServer"

    def log_message(self, format: str, *args) -> None:
        return

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        log_service("INFO", f"← response {status} {self.command} {self.path}")

    def _log_request_start(self) -> None:
        log_service("INFO", f"→ request {self.command} {self.path}")

    def do_GET(self) -> None:
        self._log_request_start()
        if self.path != "/health":
            self._write_json(404, {"ok": False, "error": "not found"})
            return
        self._write_json(200, {"ok": True})

    def do_POST(self) -> None:
        self._log_request_start()
        if self.path != "/infer":
            self._write_json(404, {"ok": False, "error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
            goal = payload.get("goal", "")
            if not goal or not isinstance(goal, str):
                raise ValueError("request body must contain string field 'goal'")

            result = self.server.engine.infer(
                goal=goal,
                hypotheses=payload.get("hypotheses") or [],
                user_lemmas=payload.get("user_lemmas") or [],
                tactic_name=payload.get("tactic_name", "aesop"),
            )
            self._write_json(200, {"ok": True, **result})
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


def _is_address_in_use(exc: BaseException) -> bool:
    """Return true when a socket bind failed because the port is occupied."""
    return isinstance(exc, OSError) and exc.errno == errno.EADDRINUSE


def _log_port_in_use_help(host: str, port: int) -> None:
    """Explain how to recover when the SATP service port is already occupied."""
    log_service(
        "WARNING",
        f"[LeanSATP] Port {port} is already in use on {host}.",
    )
    log_service(
        "WARNING",
        "[LeanSATP] Another process is already listening on the SATP service port.",
    )
    log_service(
        "WARNING",
        "[LeanSATP] If that is an existing SATP server, reuse it instead of starting a second copy.",
    )
    log_service(
        "INFO",
        f"[LeanSATP] To inspect the current listener: lsof -i :{port}",
    )
    log_service(
        "INFO",
        f"[LeanSATP] To stop it and restart SATP: fuser -k {port}/tcp",
    )


def _log_missing_checkpoint_help(exc: FileNotFoundError) -> None:
    """Explain how to recover when the local checkpoint is missing."""
    log_service("ERROR", f"[LeanSATP] {exc}")
    log_service(
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
    """Start the resident LeanSATP HTTP service."""
    pid = os.getpid()
    server: _SATPHTTPServer | None = None
    previous_sigint = None
    previous_sigterm = None
    exit_code = 0

    log_service("INFO", f"Started server process [{pid}]")
    log_service("INFO", "Waiting for application startup.")
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
        signal.signal(signal.SIGINT, _interrupt_service)
        signal.signal(signal.SIGTERM, _interrupt_service)
        log_service("INFO", engine.summary())
        log_service("INFO", "Application startup complete.")
        log_service(
            "INFO",
            f"LeanSATP service running on http://{host}:{port} (Press CTRL+C to quit)",
        )
        log_service("INFO", render_try_me_message(host, port))
        server.serve_forever()
    except KeyboardInterrupt:
        log_service("INFO", "Shutting down")
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
        log_service("INFO", "Waiting for application shutdown.")
        if server is not None:
            server.server_close()
        if previous_sigint is not None:
            signal.signal(signal.SIGINT, previous_sigint)
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)
        log_service("INFO", "Application shutdown complete.")
        log_service("INFO", f"Finished server process [{pid}]")
    return exit_code


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LeanSATP inference service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5177)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--checkpoint-source", default=DEFAULT_CHECKPOINT_SOURCE)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if args.download_only:
        resolved = ensure_checkpoint_download(args.checkpoint, args.checkpoint_source)
        print(json.dumps({"ok": True, "checkpoint_path": resolved}))
        return 0

    if args.serve:
        return serve(
            host=args.host,
            port=args.port,
            checkpoint_path=args.checkpoint,
            cache_dir=args.cache_dir,
        )

    log_service("ERROR", "Use --serve or --download-only")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
