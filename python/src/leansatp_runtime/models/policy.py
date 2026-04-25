"""LeanSATP inference policy: frozen ByT5 + LoRA + shared MLP + decision heads.

Pure greedy / deterministic forward pass. No sampling, no exploration, no
buffer. Retrieval is delegated to `PremiseRetriever`.

Architecture constants (encoder name, sequence length, head dimensions, LoRA
rank, …) are baked into the trained checkpoint and hard-coded here.
"""

import os
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForTextEncoding, AutoTokenizer

from .components import (
    ConfigHeads,
    LemmaHeads,
    LoRAConfig,
    PremiseEncoder,
    PremiseRetriever,
    TacticHeads,
    apply_lora_to_model,
)
from .components.heads import DEFAULT_LEMMA_K, NUM_PRIORITY_LEVELS

_SENTENCE_ENCODER = "kaiyuy/leandojo-lean4-retriever-byt5-small"
_USE_GRADIENT_CHECKPOINTING = os.getenv("USE_GRADIENT_CHECKPOINTING", "0") == "1"


class AesopPolicy(nn.Module):
    """Inference-time aesop policy with cached premise retrieval."""

    def __init__(
        self,
        use_lora: bool = True,
        lora_config: Optional[LoRAConfig] = None,
        device: Optional[str] = None,
        cache_dir: str = "./cache/",
    ):
        super().__init__()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = os.path.abspath(cache_dir or "./cache")

        self.tokenizer = AutoTokenizer.from_pretrained(
            _SENTENCE_ENCODER, cache_dir=cache_dir
        )
        self.tokenizer.padding_side = "right"

        self.base = AutoModelForTextEncoding.from_pretrained(
            _SENTENCE_ENCODER, cache_dir=cache_dir, dtype=torch.float32
        )

        if _USE_GRADIENT_CHECKPOINTING:
            self.base.gradient_checkpointing_enable()

        if use_lora:
            self.base, _ = apply_lora_to_model(self.base, lora_config or LoRAConfig())
        else:
            self.base.eval()
            for param in self.base.parameters():
                param.requires_grad = False

        self.hidden_size = self.base.config.d_model
        self.expanded_size = self.hidden_size * 2

        # nn.Dropout positions are preserved for checkpoint state_dict alignment;
        # p=0 makes the no-op explicit (greedy / deterministic inference).
        dropout = nn.Dropout(p=0.0)
        self.shared_mlp = nn.Sequential(
            dropout,
            nn.Linear(self.hidden_size, self.expanded_size),
            nn.LayerNorm(self.expanded_size),
            nn.ReLU(),
            dropout,
            nn.Linear(self.expanded_size, self.expanded_size),
            nn.LayerNorm(self.expanded_size),
            nn.ReLU(),
            dropout,
            nn.Linear(self.expanded_size, self.expanded_size),
            nn.LayerNorm(self.expanded_size),
            nn.ReLU(),
            nn.Linear(self.expanded_size, self.hidden_size),
        )

        self.tactic_heads = TacticHeads(self.hidden_size)
        self.lemma_heads = LemmaHeads(self.hidden_size, DEFAULT_LEMMA_K)
        self.config_heads = ConfigHeads(self.hidden_size, NUM_PRIORITY_LEVELS)
        self.premise_encoder = PremiseEncoder(self.base, self.tokenizer, cache_dir)

        self.to(self.device)

        self._retriever = PremiseRetriever(
            cache_dir=self.cache_dir,
            encode_fn=lambda texts: self.premise_encoder.encode(texts),
            hidden_size=self.hidden_size,
            device=torch.device(self.device),
            use_hybrid=True,
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        lemma_scores=None,
        lemma_embeddings=None,
        **_kwargs,
    ):
        """Base encoder → shared MLP → per-head logits + lemma residual."""
        base_outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = base_outputs.last_hidden_state[:, 0, :]
        shared_features = self.shared_mlp(hidden_states)

        # Lemma embeddings share the same shared_mlp so statement CLS and lemma CLS
        # live in the same representation space before concatenation.
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
        }

    # ---- Retrieval API (delegates to PremiseRetriever) -------------------

    def load_premise_embeddings(self) -> None:
        self._retriever.load()

    def has_premise_cache(self) -> bool:
        return self._retriever.ready

    def retrieve(self, query: str, k: Optional[int] = None):
        return self._retriever.retrieve(query, k=k)
