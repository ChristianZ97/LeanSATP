# src/aesop/models/components/config_heads.py
"""
Config heads component for Aesop configuration.

Per-head independent dropout enables Bayesian Thompson Sampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfigHeads(nn.Module):
    """Handles Aesop configuration heads."""

    def __init__(
        self, hidden_size: int, num_priority_levels: int, dropout_rate: float = 0.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_priority_levels = num_priority_levels
        self.dropout_rate = dropout_rate

        # Config Heads — default Kaiming uniform init
        self.heads = nn.ModuleDict(
            {
                "level": nn.ModuleList(
                    [nn.Linear(hidden_size, num_priority_levels) for _ in range(3)]
                ),
                "binary": nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(2)]),
            }
        )

    def forward(self, features):
        """Forward pass with per-head independent dropout."""
        level_logits = torch.stack(
            [
                head(F.dropout(features, p=self.dropout_rate, training=self.training))
                for head in self.heads["level"]
            ],
            dim=1,
        )
        binary_logits = torch.stack(
            [
                head(F.dropout(features, p=self.dropout_rate, training=self.training))
                for head in self.heads["binary"]
            ],
            dim=1,
        )

        return {
            "level": level_logits,
            "binary": binary_logits,
        }

    def sample(self, logits_dict, temperature=1.0):
        """Sample configuration actions from logits."""
        # Sample level configs
        level_configs = {}
        level_names = [
            "maxRuleApplicationDepth",
            "maxRuleApplications",
            "maxNormIterations",
        ]

        for i, name in enumerate(level_names):
            logits = logits_dict["level"][:, i].squeeze(0)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                level_idx = torch.multinomial(probs, 1).item()
            else:
                level_idx = torch.argmax(logits, dim=-1).item()
            level_configs[name] = level_idx

        # Sample binary configs
        binary_configs = {}
        binary_names = ["enableSimp", "useSimpAll"]

        for i, name in enumerate(binary_names):
            logit = logits_dict["binary"][:, i].squeeze()
            if temperature > 0:
                binary_configs[name] = bool(
                    torch.bernoulli(torch.sigmoid(logit / temperature)).item()
                )
            else:
                binary_configs[name] = bool((logit > 0).item())

        return {"level": level_configs, "binary": binary_configs}
