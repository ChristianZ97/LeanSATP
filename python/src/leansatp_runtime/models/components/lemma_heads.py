# src/aesop/models/components/lemma_heads.py
"""
Lemma heads component for premise priority prediction.

LEMMA_K is the fixed number of premises retrieved for each problem (default: 8).
Per-head independent dropout on combined features enables Bayesian TS.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import config


class LemmaHeads(nn.Module):
    """
    Predicts priority levels for each retrieved lemma.

    Architecture (Shared Head):
    - Single shared head takes concat [statement_features ‖ lemma_embedding]
      → priority logits.  Content-based (not positional): the same head
      evaluates every (statement, lemma) pair, so the output depends on
      *which* lemma occupies a slot, not *which rank* it has.
    - Calibration + FiLM: retrieval score → residual logits, modulated by
      lemma embedding.  Additive correction on top of base logits.
    """

    def __init__(self, hidden_size: int, lemma_k: int, dropout_rate: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.lemma_k = lemma_k
        self.num_levels = config.NUM_PRIORITY_LEVELS
        self.dropout_rate = dropout_rate

        # Shared head: [statement_features ‖ lemma_embedding] → priority logits
        # Permutation-equivariant: swapping two lemmas swaps their outputs
        self.shared_head = nn.Linear(hidden_size * 2, self.num_levels)

        # Calibration: retrieval score (scalar) -> priority logits
        self.calibration = nn.Sequential(
            nn.Linear(1, lemma_k),
            nn.ReLU(),
            nn.Linear(lemma_k, self.num_levels),
        )

        # FiLM conditioning: lemma embedding modulates score-based calibration
        film_d = 32  # bottleneck dimension
        self.embed_proj = nn.Linear(hidden_size, film_d)
        self.film_gamma = nn.Linear(film_d, self.num_levels)  # scale
        self.film_beta = nn.Linear(film_d, self.num_levels)  # shift

        self._init_film()

    def _init_film(self):
        """FiLM zero-init: γ=0, β=0 → (1+0)*calibration + 0 = calibration.
        Safe warm-start: FiLM has no effect until learned."""
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

    def forward(self, features, lemma_embeddings=None):
        """
        Compute priority logits for each lemma position.

        Args:
            features: Shared MLP output [batch_size, hidden_size]
            lemma_embeddings: Precomputed premise embeddings
                              [batch_size, lemma_k, hidden_size] (optional).

        Returns:
            logits: [batch_size, lemma_k, num_priority_levels]
        """
        batch_size = features.shape[0]

        if lemma_embeddings is None:
            lemma_embeddings = features.new_zeros(
                batch_size, self.lemma_k, self.hidden_size
            )

        # Expand statement features to match lemma dim: [B, K, hidden]
        features_expanded = features.unsqueeze(1).expand(
            -1, lemma_embeddings.shape[1], -1
        )
        # Concat: [B, K, 2*hidden]
        combined = torch.cat([features_expanded, lemma_embeddings], dim=-1)
        # Per-head dropout on combined features before shared head
        combined = F.dropout(combined, p=self.dropout_rate, training=self.training)
        # Shared head: [B, K, num_levels]
        return self.shared_head(combined)

    def sample(
        self,
        logits,
        temperature=1.0,
        retrieved_scores=None,
        retrieved_embeddings=None,
    ):
        """
        Sample priority levels for each lemma.

        Args:
            logits: [batch_size, lemma_k, num_levels] from forward()
            temperature: Sampling temperature (0 = greedy)
            retrieved_scores: [batch_size, lemma_k] retrieval similarities (optional)
            retrieved_embeddings: [batch_size, lemma_k, hidden_size] (optional)

        Returns:
            List of priority lists, one per batch element
        """
        # Add retrieval score guidance if available
        if retrieved_scores is not None:
            logits = logits + self.compute_residual(
                retrieved_scores, retrieved_embeddings
            )

        # Sample or argmax
        if temperature > 0:
            flat_logits = logits.view(-1, self.num_levels)
            probs = torch.softmax(flat_logits / temperature, dim=-1)
            sampled = torch.multinomial(probs, 1).squeeze(-1)
        else:
            sampled = logits.argmax(dim=-1).view(-1)

        # Reshape back to [batch_size, lemma_k] and convert to list
        return sampled.view(logits.size(0), logits.size(1)).tolist()

    def compute_residual(self, retrieved_scores, retrieved_embeddings=None):
        """
        Compute residual logits from retrieval scores, optionally
        modulated by lemma embeddings via FiLM conditioning.

        Hierarchy: see embedding → see score → embedding adjusts score.
        FiLM: residual = (1 + γ(embed)) * calibration(score) + β(embed)
        At init (γ=0, β=0): degrades to pure score calibration.
        """
        batch_size, lemma_k = retrieved_scores.shape
        # Score-based calibration (existing path)
        # [batch * lemma_k, 1] -> [batch * lemma_k, num_levels]
        residual = self.calibration(retrieved_scores.view(-1, 1))

        # FiLM: embedding modulates score-based residual
        if retrieved_embeddings is not None:
            emb_flat = retrieved_embeddings.reshape(-1, self.hidden_size)
            h = torch.nn.functional.gelu(self.embed_proj(emb_flat))  # [B*K, 32]
            gamma = self.film_gamma(h)  # [B*K, M]
            beta = self.film_beta(h)  # [B*K, M]
            residual = (1 + gamma) * residual + beta

        return residual.view(batch_size, lemma_k, self.num_levels)
