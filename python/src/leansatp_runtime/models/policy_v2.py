"""v2-architecture policy adapter — the implementation lives on Hugging Face.

"v2" here names the **architecture** (factored joint-action heads, detected
via the ``tactic_heads.tactic_emb`` fingerprint), not the checkpoint era: the
v4.27 policy shares this architecture and loads through this same module.
Which era is served is decided by ``hf_pin.py`` alone.

The single source of truth for inference (model classes, greedy decode,
tactic-string rendering) is ``infer.py`` in the pinned HF model repo — the
same file that reproduces that repo's model card numbers. Note the decode
constants travel with it: ``LEMMA_HOST_POOL`` is 20 entries on v2 and 19 on
v4.27, so pin and checkpoint must always move together.
This module downloads that file at a **pinned revision**,
imports it, and adapts it to the two entry points the service consumes
(``build_policy_v2`` + ``policy_tactic_v2``), so the tactic path and the
card's eval harness cannot drift apart: same inference, one code.

Bumping the pin: update ``hf_pin.py`` (or set ``SATP_HF_REVISION``), rerun
``./setup.sh`` so ckpt + assets + source refresh together, restart the
service, and re-run the Step-A parity check (dataset goal_states →
``/infer`` must byte-match the HF harness decodes, 244/244).
``SATP_INFER_SOURCE`` may also point at a local file for offline work.
Known residual (HF-repo side, owner's call): infer.py itself pulls the
third-party ByT5 base architecture/tokenizer without a revision; the trunk
weights come from the pinned checkpoint, so only tokenizer/config drift of
``kaiyuy/leandojo-lean4-retriever-byt5-small`` is exposed.

LeanSATP-only deltas kept here (not in the HF file): the ``tactic_name``
head override, the SATP_ABLATE_* leave-one-out gates, decode-time
``strip_retrieval`` (v2 lemma lines never match the v1 text-strip regex),
and the retriever's startup wrong-encoder width check.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from ..hf_pin import HF_REPO, REVISION, revision_for

_INFER_SOURCE = os.environ.get("SATP_INFER_SOURCE", f"hf://{HF_REPO}/infer.py")


def _load_infer_module():
    source = _INFER_SOURCE
    if source.startswith("hf://"):
        from huggingface_hub import hf_hub_download

        parts = source[len("hf://") :].split("/", 2)
        if len(parts) != 3:
            raise ValueError(
                "SATP_INFER_SOURCE must look like 'hf://<org>/<repo>/<filename>'"
            )
        repo_id = f"{parts[0]}/{parts[1]}"
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=parts[2],
                revision=revision_for(repo_id),
            )
        except Exception as exc:  # offline / bad pin — do NOT masquerade as a
            # missing checkpoint (service maps FileNotFoundError to that help)
            raise RuntimeError(
                f"pinned inference source {source}@{REVISION} unavailable: {exc}; "
                "run ./setup.sh (its download step prefetches it) or point "
                "SATP_INFER_SOURCE at a local copy"
            ) from exc
    else:
        path = Path(source).expanduser()
        if not path.is_file():
            raise RuntimeError(f"SATP_INFER_SOURCE not found: {path}")
    spec = importlib.util.spec_from_file_location("leansatp_hf_infer", path)
    module = importlib.util.module_from_spec(spec)
    # register before exec so the module's classes are picklable
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


_INFER = _load_infer_module()

# Re-exported decode/render surface (tests + service) — all owned by the HF
# file; nothing below reimplements decode semantics.
LEMMA_HOST_POOL = _INFER.LEMMA_HOST_POOL
LEMMA_K = _INFER.LEMMA_K
MAX_SEQUENCE_LENGTH = _INFER.MAX_SEQUENCE_LENGTH
N_LEMMA_DECISION = _INFER.N_LEMMA_DECISION
N_TACTICS = _INFER.N_TACTICS
TACTIC_POOL = _INFER.TACTIC_POOL
TYPE_PRIORITY_VALUES = _INFER.TYPE_PRIORITY_VALUES
AesopPolicyV2 = _INFER.AesopPolicy
AesopPolicyV2.arch = "v2"  # class-level, matching the old vendored class
decode_decision = _INFER.decode_decision
decode_lemma_decision = _INFER.decode_lemma_decision
premise_name = _INFER.premise_name


class PremiseRetriever(_INFER.PremiseRetriever):
    """HF retriever + LeanSATP's startup cache-shape validation.

    Validation reads the npy header via mmap BEFORE the upstream loader runs,
    so a malformed cache can neither trip upstream's bare assert, pay a device
    transfer, nor leave the retriever in a half-populated state."""

    def load(self) -> None:
        emb_path = os.path.join(self.cache_dir, "premise_embeddings.npy")
        txt_path = os.path.join(self.cache_dir, "mathlib4_premises.txt")
        emb = _INFER.np.load(emb_path, mmap_mode="r")
        if emb.ndim != 2 or emb.shape[1] != self.hidden_size:
            raise ValueError(
                f"premise cache shape {tuple(emb.shape)} != (N, {self.hidden_size}) "
                "— wrong-encoder cache; regenerate premise_embeddings.npy with "
                "this checkpoint's encoder"
            )
        with open(txt_path, encoding="utf-8") as f:
            n_names = sum(1 for ln in f if ln.strip())
        if n_names != emb.shape[0]:
            raise ValueError(
                f"premise/embedding count mismatch: {n_names} names vs "
                f"{emb.shape[0]} embeddings — cache is inconsistent"
            )
        super().load()


def to_lean4_string(
    tactic_decisions,
    lemma_decisions=None,
    lemma_premises=None,
    config_level=None,
    config_binary=None,
    tactic_name: str = "aesop",
) -> str:
    """HF renderer + the LeanSATP-only ``tactic_name`` head override."""
    name = (tactic_name or "").strip()
    if not name:
        raise ValueError("tactic_name must be a non-empty string")
    s = _INFER.to_lean4_string(
        tactic_decisions, lemma_decisions, lemma_premises, config_level, config_binary
    )
    if name != "aesop":
        if not s.startswith("  aesop"):
            raise RuntimeError(
                "pinned renderer no longer emits the '  aesop' head — "
                "cannot apply the tactic_name override; re-verify the pin"
            )
        s = "  " + name + s[len("  aesop") :]
    return s


def build_policy_v2(ckpt: dict, cache_dir: str, device: str):
    """Construct the HF AesopPolicy and strict-load a v2 checkpoint dict."""
    sd = ckpt.get("model_state_dict", ckpt)

    # order/semantics drifts render wrong rules while tensor shapes still load;
    # ckpt and infer.py live in the same HF repo, but the pin means they can
    # point at different revisions — keep the metadata cross-check.
    meta = ckpt.get("tactic_pool") if isinstance(ckpt, dict) else None
    for key, want in (
        ("pool", TACTIC_POOL),
        ("lemma_host_pool", LEMMA_HOST_POOL),
        ("type_priority_values", TYPE_PRIORITY_VALUES),
    ):
        got = (meta or {}).get(key)
        if got is None:
            continue
        if isinstance(want, dict):
            got = {k: list(v) for k, v in got.items()}
        else:
            got = list(got)
        if got != want:
            raise RuntimeError(
                f"checkpoint {key} != pinned infer.py decode constants — "
                "would emit wrong rules"
            )

    model = _INFER.AesopPolicy(device=device, cache_dir=cache_dir)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    real_missing = [
        k for k in missing if not k.startswith(("premise_encoder.", "_retriever."))
    ]
    if real_missing or unexpected:
        raise RuntimeError(
            "v2 checkpoint architecture mismatch: "
            f"missing_keys={real_missing[:3]!r} ({len(real_missing)} total), "
            f"unexpected_keys={list(unexpected)[:3]!r} ({len(unexpected)} total)"
        )
    # give the resident retriever the width check without rebuilding its wiring
    model._retriever.__class__ = PremiseRetriever
    model.arch = "v2"
    return model


def policy_tactic_v2(
    model,
    device,
    formal_statement: str,
    *,
    retrieval_enabled: bool = True,
    tactic_name: str = "aesop",
    strip_retrieval: bool = False,
) -> str:
    """Greedy decode via the pinned HF infer.py (v2 factored heads)."""
    if not retrieval_enabled:
        raise RuntimeError(
            # No directory name here: this function never sees cache_dir, and
            # naming one era's default (it used to say cache_v2/) is wrong the
            # moment the pin moves. ./setup.sh knows where the cache belongs.
            "the satp policy requires the premise cache "
            "(premise_embeddings.npy + mathlib4_premises.txt) in the "
            "service's --cache-dir — run ./setup.sh; serving without "
            "retrieval would emit configs that do not reproduce the model card"
        )
    # ablation/strip calls pay a (discarded) retrieve so the decode path stays
    # single-sourced — one infer.py call for every mode
    tactic, lemma, premises, level, binary = _INFER.policy_decisions(
        model, formal_statement, formal_statement
    )
    if strip_retrieval or os.environ.get("SATP_ABLATE_RETRIEVAL") == "1":
        lemma = [0] * len(lemma)
    if os.environ.get("SATP_ABLATE_TACTIC_PRIO") == "1":
        tactic = [0] * len(tactic)
    if os.environ.get("SATP_ABLATE_BUDGET") == "1":
        level = None
    return to_lean4_string(
        tactic, lemma, premises, level, binary, tactic_name=tactic_name
    )
