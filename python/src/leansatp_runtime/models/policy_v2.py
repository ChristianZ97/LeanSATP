"""satp-policy-v2 runtime, vendored from the v2 HF bundle's infer.py.

Deltas from that file: Premise comes from ..core; AesopPolicy is renamed
AesopPolicyV2 (arch="v2", + has_premise_cache); to_lean4_string takes
tactic_name; policy_tactic_v2 adds the SATP_ABLATE_* gates and decode-time
strip_retrieval (v2 lemma lines never match the v1 text-strip regex).
Self-contained on purpose: v1 components lack lora_disabled + mean-pool.
"""

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForTextEncoding, AutoTokenizer

from ..core import Premise
from ..core.premise import premise_name

# ============================================================================
# HARDCODED CONFIG  (verbatim from src/aesop/config/hyperparameters.py)
# ============================================================================
SENTENCE_ENCODER = "kaiyuy/leandojo-lean4-retriever-byt5-small"
MAX_SEQUENCE_LENGTH = 1024
DROPOUT_RATE = 0.2
HEAD_DROPOUT_RATE = 0.5  # eval() disables dropout → value irrelevant at inference

# LoRA (must match the trained adapter so base.*.lora_A/lora_B keys load)
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ("q", "k", "v", "o")

# Heads / action space
LEMMA_K = 32
N_TACTICS = 28
N_TYPE = 3
N_PRIORITY = 3
N_DECISION = 10  # off + N_TYPE×N_PRIORITY
N_LEMMA_HOST = 20
N_LEMMA_DECISION = 181  # off + host×type×priority = 1 + 20·3·3
N_CONFIG_BINARY = 5
PREMISE_BATCH_SIZE = 16

TACTIC_TYPES_ON = ("norm", "safe", "unsafe")
TYPE_PRIORITY_VALUES = {
    "norm": [100, 1, 0],
    "safe": [100, 1, 0],
    "unsafe": [10, 50, 90],
}

# Positional tactic names (decode index i → TACTIC_POOL[i]); asserted == ckpt['tactic_pool']
TACTIC_POOL = [
    "ring",
    "field_simp",
    "norm_num",
    "norm_cast",
    "linarith",
    "nlinarith",
    "positivity",
    "omega",
    "abel",
    "push_neg",
    "zify",
    "gcongr",
    "bound",
    "interval_cases",
    "ring_nf",
    "ext",
    "split",
    "exfalso",
    "simp",
    "simp_all",
    "ring_nf at *",
    "field_simp [*] at *",
    "norm_num [*] at *",
    "norm_cast at *",
    "rfl",
    "decide",
    "push_cast",
    "assumption_mod_cast",
]

LEMMA_HOST_POOL = [
    "apply {L}",
    "rw [{L}]",
    "rw [{L}] at *",
    "rw [← {L}]",
    "rw [← {L}] at *",
    "simp_rw [{L}]",
    "simp only [{L}]",
    "simp only [{L}] at *",
    "simp [{L}] at *",
    "simp_all [{L}]",
    "simp_all only [{L}]",
    "norm_num [{L}] at *",
    "field_simp [{L}] at *",
    "push_cast [{L}] at *",
    "unfold {L}",
    "solve_by_elim [{L}]",
    "grind only [{L}]",
    "linarith [{L}]",
    "nlinarith [{L}]",
    "positivity [{L}]",
]

CONFIG_LEVEL_KEYS = (
    "maxRuleApplications",
    "maxRuleApplicationDepth",
    "maxNormIterations",
    "maxGoals",
)
CONFIG_LEVEL_VALUES = {
    "maxRuleApplications": [40, 100, 200, 400, 700, 1100, 1600, 2400],
    "maxRuleApplicationDepth": [10, 20, 30, 60, 120, 220, 380, 600],
    "maxNormIterations": [30, 60, 100, 180, 320, 520, 800, 1200],
    "maxGoals": [None, 1024, 512, 256, 128, 64, 32, 16],
}
CONFIG_LEVEL_DEFAULTS = {
    "maxRuleApplications": 200,
    "maxRuleApplicationDepth": 30,
    "maxNormIterations": 100,
    "maxGoals": None,
}
CONFIG_LEVEL_CARD = [
    len(CONFIG_LEVEL_VALUES[k]) for k in CONFIG_LEVEL_KEYS
]  # [8,8,8,8]

CONFIG_BINARY_KEYS = (
    "enableSimp",
    "useSimpAll",
    "enableUnfold",
    "useDefaultSimpSet",
    "enableBuiltin",
)
DEFAULT_CONFIG_BINARY = {
    "enableSimp": True,
    "useSimpAll": True,
    "enableUnfold": True,
    "useDefaultSimpSet": True,
    "enableBuiltin": True,
}


# ============================================================================
# VENDORED: src/aesop/models/components/lora_adapter.py
# (not v1's: query encoding needs the `disabled` switch)
# ============================================================================
@dataclass
class LoRAConfig:
    r: int = LORA_R
    lora_alpha: int = LORA_ALPHA
    lora_dropout: float = LORA_DROPOUT
    target_modules: Tuple[str, ...] = LORA_TARGET_MODULES
    bias: str = "none"
    modules_to_exclude: Tuple[str, ...] = ()


@contextmanager
def lora_disabled(model: nn.Module):
    layers = [m for m in model.modules() if isinstance(m, LoRALayer)]
    prev = [layer.disabled for layer in layers]
    for layer in layers:
        layer.disabled = True
    try:
        yield
    finally:
        for layer, p in zip(layers, prev):
            layer.disabled = p


class LoRALayer(nn.Module):
    def __init__(self, original_layer: nn.Linear, r=8, lora_alpha=16, lora_dropout=0.1):
        super().__init__()
        self.original_layer = original_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        for param in self.original_layer.parameters():
            param.requires_grad = False
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.lora_dropout = (
            nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        )
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
        self.disabled = False

    @property
    def weight(self):
        return self.original_layer.weight

    @property
    def bias(self):
        return self.original_layer.bias

    @property
    def in_features(self):
        return self.original_layer.in_features

    @property
    def out_features(self):
        return self.original_layer.out_features

    def forward(self, x):
        original_output = self.original_layer(x)
        if self.disabled:
            return original_output
        lora_output = self.lora_dropout(x)
        lora_output = lora_output @ self.lora_A.T @ self.lora_B.T
        lora_output = lora_output * self.scaling
        return original_output + lora_output


def apply_lora_to_model(model: nn.Module, cfg: LoRAConfig):
    adapted_modules = []
    target_modules_lower = tuple(t.lower() for t in cfg.target_modules)

    def _should_apply_lora(name: str) -> bool:
        return name.lower() in target_modules_lower

    def _apply_lora_recursive(module: nn.Module, prefix: str = ""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if any(excl in full_name for excl in cfg.modules_to_exclude):
                continue
            is_linear = isinstance(child, nn.Linear) or (
                hasattr(child, "weight")
                and hasattr(child, "in_features")
                and hasattr(child, "out_features")
                and not list(child.children())
            )
            if is_linear:
                if _should_apply_lora(name):
                    lora_layer = LoRALayer(
                        child,
                        r=cfg.r,
                        lora_alpha=cfg.lora_alpha,
                        lora_dropout=cfg.lora_dropout,
                    )
                    setattr(module, name, lora_layer)
                    adapted_modules.append(full_name)
            else:
                _apply_lora_recursive(child, full_name)

    _apply_lora_recursive(model)
    return model, adapted_modules


# ============================================================================
# VENDORED: src/aesop/models/components/heads.py
# ============================================================================
class TacticHeads(nn.Module):
    def __init__(
        self, hidden_size: int, n_tactics: int = N_TACTICS, dropout_rate: float = 0.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_tactics = n_tactics
        self.dropout_rate = dropout_rate
        self.tactic_emb = nn.Parameter(torch.randn(self.n_tactics, hidden_size) * 0.02)
        self.goal_proj = nn.Linear(hidden_size, hidden_size)
        self.decision_head = nn.Linear(hidden_size, N_DECISION)

    def forward(self, features):
        g_p = self.goal_proj(features)
        z = g_p.unsqueeze(1) * self.tactic_emb.unsqueeze(0)
        z = F.dropout(z, p=self.dropout_rate, training=self.training)
        return self.decision_head(z)


class LemmaHeads(nn.Module):
    def __init__(self, hidden_size: int, lemma_k: int, dropout_rate: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.lemma_k = lemma_k
        self.num_decisions = N_LEMMA_DECISION
        self.dropout_rate = dropout_rate
        self.shared_head = nn.Linear(hidden_size * 2, self.num_decisions)
        self.calibration = nn.Sequential(
            nn.Linear(1, lemma_k),
            nn.ReLU(),
            nn.Linear(lemma_k, self.num_decisions),
        )
        film_d = 32
        self.embed_proj = nn.Linear(hidden_size, film_d)
        self.film_gamma = nn.Linear(film_d, self.num_decisions)
        self.film_beta = nn.Linear(film_d, self.num_decisions)
        for layer in (self.film_gamma, self.film_beta):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, features, lemma_embeddings=None):
        batch_size = features.shape[0]
        if lemma_embeddings is None:
            lemma_embeddings = features.new_zeros(
                batch_size, self.lemma_k, self.hidden_size
            )
        expanded = features.unsqueeze(1).expand(-1, lemma_embeddings.shape[1], -1)
        combined = torch.cat([expanded, lemma_embeddings], dim=-1)
        combined = F.dropout(combined, p=self.dropout_rate, training=self.training)
        return self.shared_head(combined)

    def compute_residual(self, retrieved_scores, retrieved_embeddings=None):
        batch_size, lemma_k = retrieved_scores.shape
        residual = self.calibration(retrieved_scores.view(-1, 1))
        if retrieved_embeddings is not None:
            emb_flat = retrieved_embeddings.reshape(-1, self.hidden_size)
            h = F.gelu(self.embed_proj(emb_flat))
            gamma = self.film_gamma(h)
            beta = self.film_beta(h)
            residual = (1 + gamma) * residual + beta
        return residual.view(batch_size, lemma_k, self.num_decisions)


class ConfigHeads(nn.Module):
    def __init__(self, hidden_size: int, dropout_rate: float = 0.0):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.level_heads = nn.ModuleList(
            nn.Linear(hidden_size, c) for c in CONFIG_LEVEL_CARD
        )
        self.binary_head = nn.Linear(hidden_size, N_CONFIG_BINARY * 2)

    def _drop(self, features):
        return F.dropout(features, p=self.dropout_rate, training=self.training)

    def forward(self, features):
        B = features.shape[0]
        return {
            "level": [h(self._drop(features)) for h in self.level_heads],
            "binary": self.binary_head(self._drop(features)).view(
                B, N_CONFIG_BINARY, 2
            ),
        }


# ============================================================================
# VENDORED: src/aesop/models/components/premise_encoder.py  (mean-pool)
# ============================================================================
class PremiseEncoder:
    def __init__(self, base_model, tokenizer, cache_dir="./cache/"):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir

    def encode_premises(
        self, premises: List[str], batch_size: int = None
    ) -> torch.Tensor:
        if batch_size is None:
            batch_size = PREMISE_BATCH_SIZE
        all_embeddings = []
        for i in range(0, len(premises), batch_size):
            batch_premises = premises[i : i + batch_size]
            inputs = self.tokenizer(
                batch_premises,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_SEQUENCE_LENGTH,
            )
            inputs = {k: v.to(self.base_model.device) for k, v in inputs.items()}
            with torch.no_grad():
                was_training = self.base_model.training
                self.base_model.eval()
                try:
                    with lora_disabled(self.base_model):
                        outputs = self.base_model(**inputs)
                finally:
                    self.base_model.train(was_training)
                hidden = outputs.last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                lens = mask.sum(dim=1).clamp(min=1)
                embeddings = (hidden * mask).sum(dim=1) / lens
                embeddings = F.normalize(embeddings, dim=1)
            all_embeddings.append(embeddings.cpu())
            if torch.cuda.is_available():
                del inputs, outputs
                torch.cuda.empty_cache()
        return torch.cat(all_embeddings, dim=0)


# ============================================================================
# VENDORED: src/aesop/models/components/retriever.py  (dense-only, txt names)
# ============================================================================
def _minmax_normalize_rows(scores: torch.Tensor) -> torch.Tensor:
    if scores.numel() == 0:
        return scores
    if scores.dim() == 1:
        lo, hi = scores.min(), scores.max()
        return (scores - lo) / (hi - lo).clamp(min=1e-8)
    lo = scores.min(dim=1, keepdim=True).values
    hi = scores.max(dim=1, keepdim=True).values
    return (scores - lo) / (hi - lo).clamp(min=1e-8)


class PremiseRetriever:
    def __init__(
        self, cache_dir, encode_fn, hidden_size, device, embeddings_on_device=True
    ):
        self.cache_dir = cache_dir
        self.encode_fn = encode_fn
        self.hidden_size = hidden_size
        self.device = device
        self._embeddings_on_device = embeddings_on_device
        self._premises = None
        self._cached_embs = None

    @property
    def ready(self) -> bool:
        return self._premises is not None and self._cached_embs is not None

    def load(self) -> None:
        # txt premise names are index-aligned with the embeddings by construction
        emb_path = os.path.join(self.cache_dir, "premise_embeddings.npy")
        txt_path = os.path.join(self.cache_dir, "mathlib4_premises.txt")
        missing = [p for p in (emb_path, txt_path) if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                "missing v2 retrieval assets: " + ", ".join(missing)
            )
        embeddings = torch.from_numpy(np.load(emb_path)).float()
        if embeddings.ndim != 2 or embeddings.shape[1] != self.hidden_size:
            raise ValueError(
                f"premise embeddings shape {tuple(embeddings.shape)} != (N, {self.hidden_size}) — wrong-encoder cache"
            )
        if self._embeddings_on_device:
            embeddings = embeddings.to(self.device)
        with open(txt_path, encoding="utf-8") as f:
            self._premises = [
                Premise.from_leandojo_format(ln.strip()) for ln in f if ln.strip()
            ]
        if len(self._premises) != embeddings.shape[0]:
            raise ValueError(
                f"premise/embedding count mismatch: {len(self._premises)} names vs "
                f"{embeddings.shape[0]} embeddings — cache is inconsistent"
            )
        self._cached_embs = embeddings

    def retrieve_batch(self, queries, k=None, normalize_scores=True):
        if k is None:
            k = LEMMA_K
        self._require_cache()
        query_embs = self.encode_fn(queries).to(self._cached_embs.device)
        q_norm = F.normalize(query_embs, dim=1)
        c_norm = F.normalize(self._cached_embs, dim=1)
        similarities = torch.mm(q_norm, c_norm.t())
        return self._dense_topk(similarities, k, normalize_scores)

    def retrieve(self, query, k=None, normalize_scores=True):
        prems_batch, scores_batch, embs_batch = self.retrieve_batch(
            [query], k=k, normalize_scores=normalize_scores
        )
        return prems_batch[0], scores_batch[0], embs_batch[0]

    def _dense_topk(self, similarities, k, normalize_scores):
        batch_size = similarities.shape[0]
        n_docs = similarities.shape[1]
        actual_k = min(k, n_docs)
        top_scores, top_indices = torch.topk(similarities, actual_k, dim=1)
        top_embeddings = self._cached_embs[top_indices.view(-1)].view(
            batch_size, actual_k, -1
        )
        premises_batch = []
        for b in range(batch_size):
            row = [self._premises[i] for i in top_indices[b].tolist()]
            if actual_k < k:
                row.extend([None] * (k - actual_k))
            premises_batch.append(row)
        top_scores, top_embeddings = self._pad_scores_embs(
            top_scores, top_embeddings, k, actual_k, batch_size
        )
        if normalize_scores:
            top_scores = _minmax_normalize_rows(top_scores)
        return premises_batch, top_scores, top_embeddings

    def _pad_scores_embs(self, scores, embs, k, actual_k, batch_size):
        if actual_k >= k:
            return scores, embs
        pad = k - actual_k
        pad_scores = torch.zeros(batch_size, pad, device=scores.device)
        pad_embs = torch.zeros(batch_size, pad, embs.shape[-1], device=embs.device)
        return torch.cat([scores, pad_scores], dim=1), torch.cat(
            [embs, pad_embs], dim=1
        )

    def _require_cache(self) -> None:
        if not self.ready:
            raise ValueError(
                "Premise cache not loaded. Need cache/premise_embeddings.npy + cache/mathlib4_premises.txt."
            )


# ============================================================================
# VENDORED: src/aesop/models/policy.py  (AesopPolicy, eval subset)
# ============================================================================
class AesopPolicyV2(nn.Module):
    arch = "v2"

    def __init__(self, device: Optional[str] = None, cache_dir: str = "./cache/"):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = os.path.abspath(cache_dir or "./cache")

        self.tokenizer = AutoTokenizer.from_pretrained(
            SENTENCE_ENCODER, cache_dir=cache_dir
        )
        self.tokenizer.padding_side = "right"
        self.base = AutoModelForTextEncoding.from_pretrained(
            SENTENCE_ENCODER, cache_dir=cache_dir, dtype=torch.float32
        )

        # LoRA must be applied so base.*.lora_A/lora_B keys exist for state_dict load.
        for param in self.base.parameters():
            param.requires_grad = False
        self.base, _ = apply_lora_to_model(self.base, LoRAConfig())

        self.hidden_size = self.base.config.d_model
        self.expanded_size = self.hidden_size * 2

        self.dropout = nn.Dropout(p=DROPOUT_RATE)
        self.shared_mlp = nn.Sequential(
            self.dropout,
            nn.Linear(self.hidden_size, self.expanded_size),
            nn.LayerNorm(self.expanded_size),
            nn.ReLU(),
            self.dropout,
            nn.Linear(self.expanded_size, self.expanded_size),
            nn.LayerNorm(self.expanded_size),
            nn.ReLU(),
            self.dropout,
            nn.Linear(self.expanded_size, self.expanded_size),
            nn.LayerNorm(self.expanded_size),
            nn.ReLU(),
            nn.Linear(self.expanded_size, self.hidden_size),
        )

        self.tactic_heads = TacticHeads(
            self.hidden_size, dropout_rate=HEAD_DROPOUT_RATE
        )
        self.lemma_heads = LemmaHeads(
            self.hidden_size, LEMMA_K, dropout_rate=HEAD_DROPOUT_RATE
        )
        self.config_heads = ConfigHeads(
            self.hidden_size, dropout_rate=HEAD_DROPOUT_RATE
        )
        self.premise_encoder = PremiseEncoder(self.base, self.tokenizer, cache_dir)

        self.to(self.device)
        self._retriever = PremiseRetriever(
            cache_dir=self.cache_dir,
            encode_fn=lambda texts: self.premise_encoder.encode_premises(texts),
            hidden_size=self.hidden_size,
            device=torch.device(self.device),
        )

    @staticmethod
    def _mean_pool(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        lens = mask.sum(dim=1).clamp(min=1)
        return F.normalize(summed / lens, dim=1)

    def forward(
        self,
        input_ids,
        attention_mask,
        lemma_scores=None,
        lemma_embeddings=None,
        **kwargs,
    ):
        base_outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._mean_pool(base_outputs.last_hidden_state, attention_mask)
        shared_features = self.shared_mlp(pooled)

        lemma_features = None
        if lemma_embeddings is not None:
            B, K, D = lemma_embeddings.shape
            lemma_features = self.shared_mlp(
                lemma_embeddings.reshape(B * K, D)
            ).reshape(B, K, D)

        tactic_logits = self.tactic_heads(shared_features)
        lemma_logits = self.lemma_heads(shared_features, lemma_features)
        config_logits = self.config_heads(shared_features)

        residual_logits = None
        if lemma_scores is not None:
            residual_logits = self.lemma_heads.compute_residual(
                lemma_scores, lemma_features
            )

        return {
            "tactic_logits": tactic_logits,
            "lemma_logits": lemma_logits,
            "config_logits": config_logits,
            "residual_logits": residual_logits,
            "shared_features": shared_features,
        }

    # ---- Retrieval API (same surface as v1 AesopPolicy) -------------------

    def load_premise_embeddings(self) -> None:
        self._retriever.load()

    def has_premise_cache(self) -> bool:
        return self._retriever.ready

    def retrieve(self, query, k=None, normalize_scores=True):
        return self._retriever.retrieve(query, k=k, normalize_scores=normalize_scores)


# ============================================================================
# VENDORED: src/aesop/models/tactic_string.py  (render direction only)
# ============================================================================
def decode_decision(v: int):
    v = int(v)
    if v <= 0:
        return None
    t, p = divmod(v - 1, N_PRIORITY)
    if t >= N_TYPE:
        return None
    return TACTIC_TYPES_ON[t], p


def decode_lemma_decision(v: int):
    v = int(v)
    if v <= 0:
        return None
    host_idx, rem = divmod(v - 1, N_TYPE * N_PRIORITY)
    if host_idx >= N_LEMMA_HOST:
        return None
    t, p = divmod(rem, N_PRIORITY)
    return host_idx, TACTIC_TYPES_ON[t], p


def _to_list(x):
    if x is None:
        return None
    return x.tolist() if hasattr(x, "tolist") else list(x)


def _rule_line(type_name: str, pval: int, body: str) -> str:
    inner = f"(by {body})"
    if type_name == "unsafe":
        return f"    (add unsafe {pval}% {inner})"
    return f"    (add {type_name} {pval} {inner})"


def to_lean4_string(
    tactic_decisions,
    lemma_decisions=None,
    lemma_premises=None,
    config_level=None,
    config_binary=None,
    tactic_name: str = "aesop",
) -> str:
    if not tactic_name or not tactic_name.strip():
        raise ValueError("tactic_name must be a non-empty string")
    tactic_name = tactic_name.strip()

    tactic_decisions = _to_list(tactic_decisions) or []
    lemma_decisions = _to_list(lemma_decisions)
    config_binary = _to_list(config_binary)

    config_lines = []
    cl = _to_list(config_level)
    if cl is not None:
        for i, key in enumerate(CONFIG_LEVEL_KEYS):
            if i >= len(cl):
                continue
            val = CONFIG_LEVEL_VALUES[key][int(cl[i])]
            if val != CONFIG_LEVEL_DEFAULTS[key]:
                config_lines.append(f"    {key} := {val}")
    binmap = {}
    if config_binary is not None:
        for i, key in enumerate(CONFIG_BINARY_KEYS):
            if i < len(config_binary):
                binmap[key] = bool(config_binary[i])
    for key in CONFIG_BINARY_KEYS:
        if key == "enableBuiltin":
            continue
        v = binmap.get(key, DEFAULT_CONFIG_BINARY[key])
        if v == DEFAULT_CONFIG_BINARY[key]:
            continue
        config_lines.append(f"    {key} := {'true' if v else 'false'}")

    clauses = []
    if config_lines:
        clauses.append("(config := {\n" + "\n".join(config_lines) + "\n  })")
    if not binmap.get("enableBuiltin", DEFAULT_CONFIG_BINARY["enableBuiltin"]):
        clauses.append("(rule_sets := [-builtin])")
    head = f"  {tactic_name}" + ("" if not clauses else " " + " ".join(clauses))

    entries = []
    type_order = {"norm": 0, "safe": 1, "unsafe": 2}
    for i, v in enumerate(tactic_decisions):
        if i >= N_TACTICS:
            break
        dec = decode_decision(v)
        if dec is None:
            continue
        type_name, p = dec
        pval = TYPE_PRIORITY_VALUES[type_name][p]
        entries.append(
            (type_order[type_name], i, _rule_line(type_name, pval, TACTIC_POOL[i]))
        )

    if lemma_decisions and lemma_premises:
        for j, v in enumerate(lemma_decisions):
            dec = decode_lemma_decision(v)
            if dec is None or j >= len(lemma_premises) or lemma_premises[j] is None:
                continue
            name = premise_name(lemma_premises[j])
            if not name.strip():
                continue
            host_idx, type_name, p = dec
            pval = TYPE_PRIORITY_VALUES[type_name][p]
            body = LEMMA_HOST_POOL[host_idx].format(L=name)
            entries.append(
                (3 + type_order[type_name], j, _rule_line(type_name, pval, body))
            )

    if not entries:
        return head
    entries.sort(key=lambda x: (x[0], x[1]))
    return head + "\n" + "\n".join(e[2] for e in entries)


# ============================================================================
# Service-facing entry points
# ============================================================================
def build_policy_v2(ckpt: dict, cache_dir: str, device: str) -> AesopPolicyV2:
    """Construct AesopPolicyV2 and strict-load a v2 checkpoint dict."""
    sd = ckpt.get("model_state_dict", ckpt)

    # order/semantics drifts render wrong rules while tensor shapes still load;
    # cardinality drifts are already caught by the strict state-dict load below
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
                f"checkpoint {key} != vendored decode constants — would emit wrong rules"
            )

    model = AesopPolicyV2(device=device, cache_dir=cache_dir)
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
    return model


def policy_tactic_v2(
    model: AesopPolicyV2,
    device: str,
    formal_statement: str,
    *,
    retrieval_enabled: bool = True,
    tactic_name: str = "aesop",
    strip_retrieval: bool = False,
) -> str:
    """Greedy decode the 69-head joint action → aesop string (v2 factored heads)."""
    top_k_premises = None
    lemma_scores = None
    lemma_embs = None
    # stripped/ablated calls zero the lemma actions anyway — skip the dense
    # top-k (and its ByT5 query encode) instead of paying it inside the lock
    skip_retrieval = strip_retrieval or os.environ.get("SATP_ABLATE_RETRIEVAL") == "1"
    if retrieval_enabled and not skip_retrieval and model.has_premise_cache():
        top_k_premises, lemma_scores, lemma_embs = model.retrieve(
            formal_statement, k=LEMMA_K
        )

    inputs = model.tokenizer(
        [formal_statement],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    with torch.no_grad():
        out = model(
            input_ids,
            attention_mask,
            lemma_scores=lemma_scores.unsqueeze(0).to(device)
            if lemma_scores is not None
            else None,
            lemma_embeddings=lemma_embs.unsqueeze(0).to(device)
            if lemma_embs is not None
            else None,
        )

    tactic = out["tactic_logits"][0].argmax(dim=-1).tolist()
    lemma_logits = out["lemma_logits"][0]
    if out.get("residual_logits") is not None:
        lemma_logits = lemma_logits + out["residual_logits"][0]
    lemma = lemma_logits.argmax(dim=-1).tolist()
    level = [int(t[0].argmax(dim=-1).item()) for t in out["config_logits"]["level"]]
    binary = out["config_logits"]["binary"][0].argmax(dim=-1).tolist()

    if skip_retrieval:
        lemma = [0] * len(lemma)
    if os.environ.get("SATP_ABLATE_TACTIC_PRIO") == "1":
        tactic = [0] * len(tactic)
    if os.environ.get("SATP_ABLATE_BUDGET") == "1":
        level = None

    return to_lean4_string(
        tactic, lemma, top_k_premises, level, binary, tactic_name=tactic_name
    )
