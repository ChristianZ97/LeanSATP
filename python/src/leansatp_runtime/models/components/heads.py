"""Inference-time decision heads for AesopPolicy.

Architecture constants below are checkpoint-locked: they must match the
trained model exactly or `load_state_dict` would either fail or silently
fill the heads with random weights.

Greedy / deterministic only — no Thompson Sampling, no per-head dropout.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Tactic vocabulary (head ordering must match the trained checkpoint).
SAFE_TACTICS = (
    "abel",
    "push_neg",
    "zify",
    "ring",
    "field_simp",
    "norm_num",
    "norm_cast",
)

UNSAFE_TACTICS = (
    "gcongr",
    "interval_cases",
    "ext",
    "exfalso",
    "split",
    "linarith",
    "nlinarith",
    "positivity",
    "omega",
    "ring_nf",
    "ring_nf at *",
    "simp",
    "simp_all",
    "field_simp [*] at *",
    "norm_num [*] at *",
    "norm_cast at *",
    "bound",
)

NUM_PRIORITY_LEVELS = 5
HEAD_ATTN_HEADS = 4
DEFAULT_LEMMA_K = 8

CONFIG_LEVEL_KEYS = (
    "maxRuleApplicationDepth",
    "maxRuleApplications",
    "maxNormIterations",
    "maxGoals",
)

CONFIG_BINARY_KEYS = (
    "enableSimp",
    "useSimpAll",
    "enableUnfold",
    "useDefaultSimpSet",
)

DEFAULT_AESOP_CONFIG = {
    "maxRuleApplicationDepth": 30,
    "maxRuleApplications": 200,
    "maxNormIterations": 100,
    "maxGoals": None,
    "enableSimp": True,
    "useSimpAll": True,
    "enableUnfold": True,
    "useDefaultSimpSet": True,
}

LEVEL_VALUES = {
    "maxRuleApplicationDepth": [30, 50, 70, 90, 110],
    "maxRuleApplications": [200, 220, 240, 260, 280],
    "maxNormIterations": [100, 120, 140, 160, 180],
    "maxGoals": [None, 256, 128, 64, 32],
}


class GroupHeadAttention(nn.Module):
    """Self-attention within a group of tactic heads for coordinated priority."""

    def __init__(
        self,
        num_heads: int,
        hidden_size: int,
        num_levels: int = NUM_PRIORITY_LEVELS,
        attn_heads: int = HEAD_ATTN_HEADS,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        self.head_embeddings = nn.Parameter(torch.randn(num_heads, hidden_size) * 0.02)
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads=attn_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.output_heads = nn.ModuleList(
            [nn.Linear(hidden_size, num_levels) for _ in range(num_heads)]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """[B, hidden] → [B, num_heads, num_levels]."""
        x = features.unsqueeze(1).expand(-1, self.num_heads, -1)
        x = x + self.head_embeddings.unsqueeze(0)
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm(x + attn_out)
        return torch.stack(
            [head(x[:, i, :]) for i, head in enumerate(self.output_heads)],
            dim=1,
        )


class TacticHeads(nn.Module):
    """Priority heads for safe + unsafe tactics, each with within-group attention."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.safe_group = GroupHeadAttention(
            num_heads=len(SAFE_TACTICS), hidden_size=hidden_size
        )
        self.unsafe_group = GroupHeadAttention(
            num_heads=len(UNSAFE_TACTICS), hidden_size=hidden_size
        )

    def forward(self, features):
        return {
            "safe": self.safe_group(features),
            "unsafe": self.unsafe_group(features),
        }


class ConfigHeads(nn.Module):
    """Aesop configuration heads: 4 level heads + 4 binary heads.

    Plan F (2026-04-27 in trainer): level covers
    ``CONFIG_LEVEL_KEYS = (maxRuleApplicationDepth, maxRuleApplications,
    maxNormIterations, maxGoals)``; binary covers
    ``CONFIG_BINARY_KEYS = (enableSimp, useSimpAll, enableUnfold,
    useDefaultSimpSet)``.
    """

    def __init__(
        self, hidden_size: int, num_priority_levels: int = NUM_PRIORITY_LEVELS
    ):
        super().__init__()
        n_level = len(CONFIG_LEVEL_KEYS)
        n_binary = len(CONFIG_BINARY_KEYS)
        self.heads = nn.ModuleDict(
            {
                "level": nn.ModuleList(
                    [
                        nn.Linear(hidden_size, num_priority_levels)
                        for _ in range(n_level)
                    ]
                ),
                "binary": nn.ModuleList(
                    [nn.Linear(hidden_size, 1) for _ in range(n_binary)]
                ),
            }
        )

    def forward(self, features):
        level_logits = torch.stack(
            [head(features) for head in self.heads["level"]], dim=1
        )
        binary_logits = torch.stack(
            [head(features) for head in self.heads["binary"]], dim=1
        )
        return {"level": level_logits, "binary": binary_logits}


class LemmaHeads(nn.Module):
    """Shared content head + FiLM-calibrated residual on retrieval scores."""

    def __init__(self, hidden_size: int, lemma_k: int = DEFAULT_LEMMA_K):
        super().__init__()
        self.hidden_size = hidden_size
        self.lemma_k = lemma_k
        self.num_levels = NUM_PRIORITY_LEVELS

        self.shared_head = nn.Linear(hidden_size * 2, self.num_levels)

        self.calibration = nn.Sequential(
            nn.Linear(1, lemma_k),
            nn.ReLU(),
            nn.Linear(lemma_k, self.num_levels),
        )

        film_d = 32
        self.embed_proj = nn.Linear(hidden_size, film_d)
        self.film_gamma = nn.Linear(film_d, self.num_levels)
        self.film_beta = nn.Linear(film_d, self.num_levels)

    def forward(self, features, lemma_embeddings=None):
        """features [B, hidden], lemma_embeddings [B, K, hidden] → [B, K, num_levels]."""
        batch_size = features.shape[0]
        if lemma_embeddings is None:
            lemma_embeddings = features.new_zeros(
                batch_size, self.lemma_k, self.hidden_size
            )
        expanded = features.unsqueeze(1).expand(-1, lemma_embeddings.shape[1], -1)
        combined = torch.cat([expanded, lemma_embeddings], dim=-1)
        return self.shared_head(combined)

    def compute_residual(self, retrieved_scores, retrieved_embeddings=None):
        """(1 + γ(embed)) · calibration(score) + β(embed)."""
        batch_size, lemma_k = retrieved_scores.shape
        residual = self.calibration(retrieved_scores.view(-1, 1))
        if retrieved_embeddings is not None:
            emb_flat = retrieved_embeddings.reshape(-1, self.hidden_size)
            h = F.gelu(self.embed_proj(emb_flat))
            gamma = self.film_gamma(h)
            beta = self.film_beta(h)
            residual = (1 + gamma) * residual + beta
        return residual.view(batch_size, lemma_k, self.num_levels)
