# leansatp_runtime/models/policy.py
"""
Aesop Policy Model v1.1.0 (Simplified Single GPU)

Refactored main policy model.
Removed all distributed training code for simplicity.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForTextEncoding
from typing import List, Optional
from huggingface_hub import PyTorchModelHubMixin

from ..config import config
from ..core import Premise
from ..core.bm25 import BM25Index, HybridRetriever
from .components import (
    TacticHeads,
    LemmaHeads,
    ConfigHeads,
    PremiseEncoder,
    LoRAConfig,
    apply_lora_to_model,
)


def to_lean4_string(
    safe_actions,
    unsafe_actions,
    lemma_actions=None,
    lemma_premises: Optional[List[Premise]] = None,
    config_level_actions=None,
    config_binary_actions=None,
    tactic_name: str = "aesop",
) -> str:
    """
    Convert model outputs to Lean4 aesop tactic string.

    Args:
        safe_actions: Priority levels for safe tactics
        unsafe_actions: Priority levels for unsafe tactics
        lemma_actions: Priority levels for retrieved lemmas
        lemma_premises: List of Premise objects for lemmas
        config_level_actions: Priority levels for config (3 values)
        config_binary_actions: Binary actions for config (2 values)
        tactic_name: Lean tactic prefix to emit (default: "aesop")

    Returns:
        str: Lean4 aesop tactic string
    """
    if not tactic_name or not tactic_name.strip():
        raise ValueError("tactic_name must be a non-empty string")
    tactic_name = tactic_name.strip()

    # Convert tensors to lists
    if hasattr(safe_actions, "tolist"):
        safe_actions = safe_actions.tolist()
    if hasattr(unsafe_actions, "tolist"):
        unsafe_actions = unsafe_actions.tolist()
    if lemma_actions is not None and hasattr(lemma_actions, "tolist"):
        lemma_actions = lemma_actions.tolist()
    if config_level_actions is not None and hasattr(config_level_actions, "tolist"):
        config_level_actions = config_level_actions.tolist()
    if config_binary_actions is not None and hasattr(config_binary_actions, "tolist"):
        config_binary_actions = config_binary_actions.tolist()

    # Build config values (level * 20 for 5-level system)
    if config_level_actions is not None and len(config_level_actions) >= 3:
        max_depth = (
            config.DEFAULT_AESOP_CONFIG["maxRuleApplicationDepth"]
            + 20 * config_level_actions[0]
        )
        max_apps = (
            config.DEFAULT_AESOP_CONFIG["maxRuleApplications"]
            + 20 * config_level_actions[1]
        )
        max_norm = (
            config.DEFAULT_AESOP_CONFIG["maxNormIterations"]
            + 20 * config_level_actions[2]
        )
    else:
        max_depth = config.DEFAULT_AESOP_CONFIG["maxRuleApplicationDepth"]
        max_apps = config.DEFAULT_AESOP_CONFIG["maxRuleApplications"]
        max_norm = config.DEFAULT_AESOP_CONFIG["maxNormIterations"]

    if config_binary_actions is not None and len(config_binary_actions) >= 2:
        enable_simp = bool(config_binary_actions[0])
        use_simp_all = bool(config_binary_actions[1])
    else:
        enable_simp = config.DEFAULT_AESOP_CONFIG["enableSimp"]
        use_simp_all = config.DEFAULT_AESOP_CONFIG["useSimpAll"]

    enable_simp_str = "true" if enable_simp else "false"
    use_simp_all_str = "true" if use_simp_all else "false"

    aesop_config = f"""  {tactic_name} (config := {{
    maxRuleApplicationDepth := {max_depth}
    maxRuleApplications     := {max_apps}
    maxNormIterations       := {max_norm}
    enableSimp              := {enable_simp_str}
    useSimpAll              := {use_simp_all_str}
  }})"""

    # Rule entries: (sort_key, rule_string)
    rule_entries = []

    # 1. Safe Rules (priority 0 = disabled, 1-4 → Lean priority 4,3,2,1)
    for idx, priority in enumerate(safe_actions):
        if priority > 0:
            tactic_name = config.SAFE_TACTICS[idx]
            aesop_priority = 5 - priority  # level 4→1 (highest), level 1→4
            rule_str = f"    (add safe {aesop_priority} (by {tactic_name}))"
            rule_entries.append((0, -priority, tactic_name, rule_str))

    # 2. Unsafe Rules (priority 0 = disabled, 1-4 → 70%,80%,90%,100%)
    UNSAFE_PROB = {1: 70, 2: 80, 3: 90, 4: 100}
    for idx, priority in enumerate(unsafe_actions):
        if priority > 0:
            tactic_name = config.UNSAFE_TACTICS[idx]
            prob_pct = UNSAFE_PROB[priority]
            rule_str = f"    (add unsafe {prob_pct}% (by {tactic_name}))"
            rule_entries.append((1, -priority, tactic_name, rule_str))

    # 3. Lemma Rules (priority 0 = disabled, 1-4 → 10%,20%,30%,40%)
    if lemma_actions and lemma_premises:
        for idx, (priority, premise) in enumerate(zip(lemma_actions, lemma_premises)):
            if priority <= 0 or premise is None:
                continue

            if hasattr(premise, "full_name"):
                lemma_full_name = premise.full_name
            else:
                lemma_full_name = str(premise)

            # Skip empty lemma names (malformed premise entry)
            if not lemma_full_name or not lemma_full_name.strip():
                continue

            prob_pct = priority * 10  # 1→10%, 2→20%, 3→30%, 4→40%
            rule_str = f"    (add unsafe {prob_pct}% (by first | apply {lemma_full_name} | rw [{lemma_full_name}] | simp only [{lemma_full_name}]))"
            rule_entries.append((2, -priority, lemma_full_name, rule_str))

    if not rule_entries:
        return aesop_config

    # Sort by (category, -priority, name) and extract rule strings
    rule_entries.sort(key=lambda x: (x[0], x[1], x[2]))
    final_rules = [entry[3] for entry in rule_entries]

    return aesop_config + "\n" + "\n".join(final_rules)


def to_satp_string(*args, **kwargs) -> str:
    """Alias for to_lean4_string(..., tactic_name='satp')."""
    kwargs["tactic_name"] = "satp"
    return to_lean4_string(*args, **kwargs)


class AesopPolicy(nn.Module, PyTorchModelHubMixin):
    """
    Aesop Tactic Configuration Policy with Lemma Retrieval (Single GPU)

    Supports optional LoRA fine-tuning of the base encoder.
    """

    def __init__(
        self,
        freeze_base: bool = True,
        use_lora: bool = False,
        lora_config: LoRAConfig = None,
        device: str = None,
        cache_dir: str = "./cache/",
        dropout_rate: float = config.DROPOUT_RATE,
    ):
        super().__init__()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.freeze_base = freeze_base
        self.use_lora = use_lora
        self.lora_config = lora_config or LoRAConfig()

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.SENTENCE_ENCODER, cache_dir=cache_dir
        )
        self.tokenizer.padding_side = "right"

        self.cache_dir = os.path.abspath(cache_dir or "./cache")

        self.base = AutoModelForTextEncoding.from_pretrained(
            config.SENTENCE_ENCODER,
            cache_dir=cache_dir,
            dtype=torch.float32,
        )

        # Enable gradient checkpointing if configured
        # Note: PyTorch warns "None of the inputs have requires_grad=True" because
        # base encoder inputs are frozen. This is harmless — LoRA parameters inside
        # each block still have requires_grad=True and receive correct gradients.
        # The checkpoint() call falls back to a normal forward when no input has grad.
        if config.USE_GRADIENT_CHECKPOINTING:
            import warnings

            warnings.filterwarnings(
                "ignore",
                message="None of the inputs have requires_grad=True",
                module="torch.utils.checkpoint",
            )
            self.base.gradient_checkpointing_enable()

        # Apply LoRA if enabled
        # LoRA will automatically freeze original weights and only train adapters
        if use_lora:
            self.base, _adapted_modules = apply_lora_to_model(
                self.base, self.lora_config
            )
            # LoRA layers handle freezing internally, no need to freeze_base
        elif freeze_base:
            # Only freeze if not using LoRA and freeze_base is explicitly True
            self.base.eval()
            for param in self.base.parameters():
                param.requires_grad = False

        self.hidden_size = self.base.config.d_model
        self.expanded_size = self.hidden_size * 2

        self.dropout = nn.Dropout(p=dropout_rate)

        # Shared MLP — LayerNorm + moderate dropout for regularization.
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

        # Initialize components with per-head dropout for Bayesian TS
        head_dropout = getattr(config, "HEAD_DROPOUT_RATE", 0.0)
        self.tactic_heads = TacticHeads(self.hidden_size, dropout_rate=head_dropout)
        self.lemma_heads = LemmaHeads(
            self.hidden_size, config.LEMMA_K, dropout_rate=head_dropout
        )
        self.config_heads = ConfigHeads(
            self.hidden_size, config.NUM_PRIORITY_LEVELS, dropout_rate=head_dropout
        )
        self.premise_encoder = PremiseEncoder(self.base, self.tokenizer, cache_dir)

        # Move to device
        self.to(self.device)

        # Premise caching
        self._premise_cache = None

        # Hybrid retrieval (optional)
        self._hybrid_retriever = None
        self._use_hybrid = getattr(config, "USE_HYBRID_RETRIEVAL", False)

    def forward(
        self,
        input_ids,
        attention_mask,
        lemma_scores=None,
        lemma_embeddings=None,
        **kwargs,
    ):
        """Forward pass through the model.

        Args:
            input_ids: Tokenized input [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            lemma_scores: Retrieved premise scores [batch, LEMMA_K] (optional).
            lemma_embeddings: Precomputed premise embeddings [batch, LEMMA_K, hidden_size] (optional).
                              Used by shared_head for content-based logits and
                              FiLM conditioning in calibration.
        """
        # Base encoding
        base_outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = base_outputs.last_hidden_state[:, 0, :]

        # Shared MLP (statement path)
        shared_features = self.shared_mlp(hidden_states)

        # Process lemma embeddings through SAME shared_mlp to match abstraction level.
        # Both statement CLS and lemma CLS come from ByT5 encoder — same source,
        # same processing, same representation space before concatenation.
        lemma_features = None
        if lemma_embeddings is not None:
            B, K, D = lemma_embeddings.shape
            lemma_features = self.shared_mlp(
                lemma_embeddings.reshape(B * K, D)
            ).reshape(B, K, D)

        # Get predictions from all heads
        tactic_logits = self.tactic_heads(shared_features)
        lemma_logits = self.lemma_heads(shared_features, lemma_features)
        config_logits = self.config_heads(shared_features)

        # Compute residual logits for lemma calibration (FiLM-modulated)
        if lemma_scores is not None:
            residual_logits = self.lemma_heads.compute_residual(
                lemma_scores, lemma_features
            )
        else:
            residual_logits = None

        return {
            "tactic_logits": tactic_logits,
            "lemma_logits": lemma_logits,
            "config_logits": config_logits,
            "residual_logits": residual_logits,
        }

    def encode_premises(self, premises, verbose: bool = True):
        """Encode premises for retrieval.

        Args:
            premises: List of premise strings to encode
            verbose: If True, show progress bar. Set to False during training.
        """
        return self.premise_encoder.encode_premises(premises, verbose=verbose)

    def load_premise_embeddings(self) -> None:
        """Load pre-computed premise embeddings from cache."""
        emb_path = os.path.join(self.cache_dir, "premise_embeddings.npy")
        raw_path = os.path.join(self.cache_dir, "premises_raw.npy")

        if not os.path.exists(emb_path) or not os.path.exists(raw_path):
            missing = [
                path for path in (emb_path, raw_path) if not os.path.exists(path)
            ]
            raise FileNotFoundError("missing retrieval assets: " + ", ".join(missing))

        embeddings = np.load(emb_path)
        raw_premises = np.load(raw_path, allow_pickle=True)

        # Keep precomputed premise embeddings on the policy's device so the
        # per-query cosine similarity + top-k runs in one GPU kernel instead of
        # round-tripping the query back to CPU.  On GPU this measured ~40%
        # faster /infer (1.04s -> 0.63s) in exchange for ~4GB of extra VRAM.
        # Falls back to CPU automatically when self.device == "cpu".
        embeddings = torch.from_numpy(embeddings).float().to(self.device)

        loaded_premises = [Premise.from_leandojo_format(r) for r in raw_premises]
        self._premise_cache = (loaded_premises, embeddings)

        # Load BM25 index for hybrid retrieval if enabled
        if self._use_hybrid:
            self._load_hybrid_retriever()

    def has_premise_cache(self) -> bool:
        """Return whether retrieval assets are loaded and available."""
        return self._premise_cache is not None

    def clear_premise_cache(self) -> None:
        """Clear premise cache."""
        self._premise_cache = None
        if hasattr(self, "_premise_name_to_idx"):
            del self._premise_name_to_idx
        self._hybrid_retriever = None

    def get_premise_embeddings_by_names(
        self, names_batch: List[List[str]]
    ) -> torch.Tensor:
        """Look up precomputed premise embeddings by full_name for BC.

        Args:
            names_batch: List of lists of premise full_names, shape [B][K]

        Returns:
            Tensor [B, K, hidden_size] of cached embeddings (zeros for unknown names)
        """
        if self._premise_cache is None:
            raise ValueError("Premise cache not initialized")

        # Lazily build name → index lookup
        if not hasattr(self, "_premise_name_to_idx"):
            premises, _ = self._premise_cache
            self._premise_name_to_idx = {}
            for idx, p in enumerate(premises):
                key = getattr(p, "full_name", str(p))
                self._premise_name_to_idx[key] = idx

        _, cached_embs = self._premise_cache
        batch_size = len(names_batch)
        k = len(names_batch[0]) if names_batch else 0
        result = torch.zeros(batch_size, k, cached_embs.shape[1])

        for b, names in enumerate(names_batch):
            for j, name in enumerate(names):
                idx = self._premise_name_to_idx.get(name, -1)
                if idx >= 0:
                    result[b, j] = cached_embs[idx]

        return result

    def _load_hybrid_retriever(self) -> None:
        """Load BM25 index and initialize hybrid retriever."""
        bm25_path = os.path.join(self.cache_dir, "bm25_index.pkl")
        if not os.path.exists(bm25_path):
            self._use_hybrid = False
            return

        bm25_index = BM25Index()
        bm25_index.load(bm25_path)

        self._hybrid_retriever = HybridRetriever(
            fusion_method=getattr(config, "HYBRID_FUSION_METHOD", "rrf"),
            dense_weight=getattr(config, "HYBRID_DENSE_WEIGHT", 0.7),
            bm25_weight=getattr(config, "HYBRID_BM25_WEIGHT", 0.3),
            rrf_k=getattr(config, "HYBRID_RRF_K", 60),
        )
        self._hybrid_retriever.set_bm25_index(bm25_index)

    def retrieve(self, query: str, k: int = None, normalize_scores: bool = True):
        """
        Retrieve top-k relevant premises for a query.

        Args:
            query: The theorem statement to find relevant premises for
            k: Number of premises to retrieve (default: config.LEMMA_K)
            normalize_scores: Whether to normalize scores to [0, 1] range

        Returns:
            Tuple of (top_k_premises, top_k_scores, top_k_embeddings) with exactly k elements
        """
        if k is None:
            k = config.LEMMA_K

        if self._premise_cache is None:
            raise ValueError("Premise cache not initialized.")

        cached_premises, cached_embs = self._premise_cache

        # Encode query (verbose=False to suppress progress bar during training)
        query_emb = self.encode_premises([query], verbose=False)
        query_emb = query_emb.to(cached_embs.device)
        similarities = torch.cosine_similarity(query_emb, cached_embs, dim=1)

        # Use hybrid retrieval if enabled
        if self._use_hybrid and self._hybrid_retriever is not None:
            # Convert to numpy for hybrid retrieval
            dense_scores = similarities.cpu().numpy()
            top_k_indices, top_k_scores_np = self._hybrid_retriever.retrieve(
                query, dense_scores, k
            )
            top_k_premises = [cached_premises[i] for i in top_k_indices]
            top_k_scores = torch.tensor(top_k_scores_np, dtype=torch.float32)
            top_k_embeddings = cached_embs[top_k_indices]  # [k, hidden]
        else:
            # Original dense-only retrieval
            actual_k = min(k, len(cached_premises))
            top_k_scores, top_k_indices = torch.topk(similarities, actual_k)
            top_k_premises = [cached_premises[i] for i in top_k_indices.tolist()]
            top_k_embeddings = cached_embs[top_k_indices]  # [actual_k, hidden]

            # Pad to exactly k if necessary
            if actual_k < k:
                pad_size = k - actual_k
                top_k_scores = torch.cat(
                    [top_k_scores, torch.zeros(pad_size, device=top_k_scores.device)]
                )
                top_k_premises.extend([None] * pad_size)
                top_k_embeddings = torch.cat(
                    [
                        top_k_embeddings,
                        torch.zeros(pad_size, cached_embs.shape[1]),
                    ]
                )

        # Normalize scores to [0, 1] range for consistent calibration
        if normalize_scores and len(top_k_scores) > 0:
            scores_min = top_k_scores.min()
            scores_max = top_k_scores.max()
            if scores_max > scores_min:
                top_k_scores = (top_k_scores - scores_min) / (scores_max - scores_min)

        return top_k_premises, top_k_scores, top_k_embeddings

    def retrieve_batch(
        self,
        queries: List[str],
        k: int = None,
        normalize_scores: bool = True,
    ):
        """
        Batch retrieve top-k premises for multiple queries at once.
        Encodes all queries in a single forward pass — 12x faster than
        calling retrieve() in a loop for batch_size=12.

        Args:
            queries: List of theorem statements
            k: Number of premises per query (default: config.LEMMA_K)
            normalize_scores: Whether to normalize scores to [0, 1] per query

        Returns:
            Tuple of (list_of_top_k_premises, batched_scores [B, k], batched_embeddings [B, k, hidden])
        """
        if k is None:
            k = config.LEMMA_K

        if self._premise_cache is None:
            raise ValueError("Premise cache not initialized.")

        cached_premises, cached_embs = self._premise_cache

        # Encode ALL queries in one forward pass
        query_embs = self.encode_premises(queries, verbose=False)
        # query_embs: [B, hidden], cached_embs: [N, hidden]

        # Ensure same device
        query_embs = query_embs.to(cached_embs.device)

        # Batch cosine similarity: [B, N]
        query_norm = torch.nn.functional.normalize(query_embs, dim=1)
        cache_norm = torch.nn.functional.normalize(cached_embs, dim=1)
        similarities = torch.mm(query_norm, cache_norm.t())  # [B, N]

        actual_k = min(k, len(cached_premises))
        all_top_k_scores, all_top_k_indices = torch.topk(similarities, actual_k, dim=1)
        # [B, k]

        # Gather top-k embeddings: [B, k, hidden]
        all_top_k_embeddings = cached_embs[all_top_k_indices.view(-1)].view(
            len(queries), actual_k, -1
        )

        all_premises_list = []
        for b in range(len(queries)):
            top_k_premises = [cached_premises[i] for i in all_top_k_indices[b].tolist()]
            if actual_k < k:
                top_k_premises.extend([None] * (k - actual_k))
            all_premises_list.append(top_k_premises)

        # Pad scores and embeddings if needed
        if actual_k < k:
            pad_size = k - actual_k
            pad_scores = torch.zeros(len(queries), pad_size)
            all_top_k_scores = torch.cat([all_top_k_scores, pad_scores], dim=1)
            pad_embs = torch.zeros(len(queries), pad_size, cached_embs.shape[1])
            all_top_k_embeddings = torch.cat([all_top_k_embeddings, pad_embs], dim=1)

        # Per-query normalization to [0, 1]
        if normalize_scores and all_top_k_scores.numel() > 0:
            scores_min = all_top_k_scores.min(dim=1, keepdim=True).values
            scores_max = all_top_k_scores.max(dim=1, keepdim=True).values
            denom = scores_max - scores_min
            denom = denom.clamp(min=1e-8)  # avoid div-by-zero
            all_top_k_scores = (all_top_k_scores - scores_min) / denom

        return all_premises_list, all_top_k_scores, all_top_k_embeddings
