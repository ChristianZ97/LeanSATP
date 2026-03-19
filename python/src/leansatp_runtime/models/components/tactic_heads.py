# src/aesop/models/components/tactic_heads.py
"""
Tactic heads component.

Per-head independent dropout enables Bayesian Thompson Sampling:
each head applies its own dropout mask to shared features,
producing independent posterior samples across decision dimensions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import config


class TacticHeads(nn.Module):
    """Handles tactic selection heads."""

    def __init__(self, hidden_size: int, dropout_rate: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        # Tactic Heads — default Kaiming uniform init (input-responsive)
        self.heads = nn.ModuleDict(
            {
                "safe": nn.ModuleList(
                    [
                        nn.Linear(hidden_size, config.NUM_PRIORITY_LEVELS)
                        for _ in range(len(config.SAFE_TACTICS))
                    ]
                ),
                "unsafe": nn.ModuleList(
                    [
                        nn.Linear(hidden_size, config.NUM_PRIORITY_LEVELS)
                        for _ in range(len(config.UNSAFE_TACTICS))
                    ]
                ),
            }
        )

    def forward(self, features):
        """Forward pass with per-head independent dropout."""
        safe_logits = torch.stack(
            [
                head(F.dropout(features, p=self.dropout_rate, training=self.training))
                for head in self.heads["safe"]
            ],
            dim=1,
        )
        unsafe_logits = torch.stack(
            [
                head(F.dropout(features, p=self.dropout_rate, training=self.training))
                for head in self.heads["unsafe"]
            ],
            dim=1,
        )

        return {
            "safe": safe_logits,
            "unsafe": unsafe_logits,
        }
