"""LoRA adapter: minimal load-and-forward subset for inference.

Original LoRA weights live in the trained checkpoint; here we only need to
reconstruct the same module hierarchy so `model.load_state_dict` finds the
expected `lora_A` / `lora_B` keys, then forward-pass through them.

Defaults below match the trained `ChristianZ97/satp-policy-goal` checkpoint.
"""

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: Tuple[str, ...] = ("q", "k", "v", "o")


class LoRALayer(nn.Module):
    """Wraps an `nn.Linear` with a low-rank residual update."""

    def __init__(
        self,
        original_layer: nn.Linear,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
    ):
        super().__init__()

        self.original_layer = original_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        for param in self.original_layer.parameters():
            param.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(r, original_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(original_layer.out_features, r))

        self.lora_dropout = (
            nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        )

        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    # T5 internals occasionally read these directly off the layer.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original = self.original_layer(x)
        lora = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return original + lora * self.scaling


def apply_lora_to_model(
    model: nn.Module,
    config: LoRAConfig,
) -> Tuple[nn.Module, List[str]]:
    """Replace target Linear layers in `model` with `LoRALayer` wrappers.

    Match on the immediate attribute name only — substring on the full
    path over-matches single-letter T5 targets like "o"/"k" against
    "enc*o*der"/"bloc*k*", which silently wraps every FFN linear too.
    See SATP-Training commit 8e18ad1 (`fix(model): LoRA over-config`)
    where the same bug was fixed on the training side.
    """
    adapted: List[str] = []
    targets = tuple(t.lower() for t in config.target_modules)

    def _hit(name: str) -> bool:
        return name.lower() in targets

    def _walk(module: nn.Module, prefix: str = "") -> None:
        for name, child in module.named_children():
            full = f"{prefix}.{name}" if prefix else name
            is_linear = isinstance(child, nn.Linear) or (
                hasattr(child, "weight")
                and hasattr(child, "in_features")
                and hasattr(child, "out_features")
                and not list(child.children())
            )
            if is_linear:
                if _hit(name):
                    setattr(
                        module,
                        name,
                        LoRALayer(
                            original_layer=child,
                            r=config.r,
                            lora_alpha=config.lora_alpha,
                            lora_dropout=config.lora_dropout,
                        ),
                    )
                    adapted.append(full)
            else:
                _walk(child, full)

    _walk(model)
    return model, adapted
