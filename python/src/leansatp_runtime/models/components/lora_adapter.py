# src/aesop/models/components/lora_adapter.py
"""
LoRA (Low-Rank Adaptation) adapter for fine-tuning base model without destroying features.

This module provides LoRA integration for the Aesop policy model, allowing
fine-tuning of the base encoder while preserving most of the original features.
"""

import torch
import torch.nn as nn
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapter."""

    # LoRA rank (smaller = less parameters, larger = more capacity)
    r: int = 8

    # Alpha parameter for scaling (effective scaling = alpha / r)
    lora_alpha: int = 16

    # Dropout for LoRA layers
    lora_dropout: float = 0.1

    # Which modules to apply LoRA to
    # For T5/ByT5: ["q", "k", "v", "o"] (attention) or ["wi", "wo"] (FFN)
    # For BERT-style: ["query", "key", "value", "dense"]
    target_modules: Tuple[str, ...] = ("q", "k", "v", "o")

    # Whether to use bias in LoRA layers
    bias: str = "none"  # "none", "all", or "lora_only"

    # Modules to exclude from LoRA
    modules_to_exclude: Tuple[str, ...] = ()


class LoRALayer(nn.Module):
    """A single LoRA layer that wraps a linear layer."""

    def __init__(
        self,
        original_layer: nn.Linear,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
    ):
        super().__init__()

        self.original_layer = original_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False

        # LoRA matrices A and B
        # A: (r, in_features) - projects down
        # B: (out_features, r) - projects up
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # Dropout
        self.lora_dropout = (
            nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        )

        # Initialize A with Kaiming, B with zeros (so initial output = original)
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

        self.merged = False

    # Proxy attributes to original layer (required for T5 compatibility)
    @property
    def weight(self):
        """Proxy weight attribute to original layer."""
        return self.original_layer.weight

    @property
    def bias(self):
        """Proxy bias attribute to original layer."""
        return self.original_layer.bias

    @property
    def in_features(self):
        """Proxy in_features attribute to original layer."""
        return self.original_layer.in_features

    @property
    def out_features(self):
        """Proxy out_features attribute to original layer."""
        return self.original_layer.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original output
        original_output = self.original_layer(x)

        if self.merged:
            return original_output

        # LoRA output: x @ A^T @ B^T * scaling
        lora_output = self.lora_dropout(x)
        lora_output = lora_output @ self.lora_A.T @ self.lora_B.T
        lora_output = lora_output * self.scaling

        return original_output + lora_output

    def merge_weights(self):
        """Merge LoRA weights into original layer for faster inference."""
        if self.merged:
            return

        with torch.no_grad():
            # W' = W + B @ A * scaling
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.original_layer.weight.data += delta_w

        self.merged = True

    def unmerge_weights(self):
        """Unmerge LoRA weights (for continued training)."""
        if not self.merged:
            return

        with torch.no_grad():
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.original_layer.weight.data -= delta_w

        self.merged = False


def apply_lora_to_model(
    model: nn.Module,
    config: LoRAConfig,
    debug: bool = False,
) -> Tuple[nn.Module, List[str]]:
    """
    Apply LoRA adapters to a model.

    Args:
        model: The model to apply LoRA to (typically the base encoder)
        config: LoRA configuration
        debug: If True, print layer names for debugging

    Returns:
        Tuple of (modified model, list of adapted module names)
    """
    adapted_modules = []
    all_linear_layers = []  # For debugging

    # Convert target modules to lowercase for case-insensitive matching
    target_modules_lower = tuple(t.lower() for t in config.target_modules)

    def _should_apply_lora(name: str, full_name: str) -> bool:
        """Check if LoRA should be applied to this module."""
        name_lower = name.lower()
        full_name_lower = full_name.lower()

        for target in target_modules_lower:
            # Exact match on module name (for T5-style: q, k, v, o)
            if name_lower == target:
                return True
            # Substring match on full path (for BERT-style: query, key, value)
            if target in full_name_lower:
                return True
        return False

    def _apply_lora_recursive(module: nn.Module, prefix: str = ""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            # Check if this module should be excluded
            if any(excl in full_name for excl in config.modules_to_exclude):
                continue

            # Check if this is a target module (Linear layer with matching name)
            # Support both nn.Linear and any module with weight attribute (for custom Linear layers)
            is_linear = isinstance(child, nn.Linear) or (
                hasattr(child, "weight")
                and hasattr(child, "in_features")
                and hasattr(child, "out_features")
                and not list(child.children())  # No child modules (leaf node)
            )

            if is_linear:
                all_linear_layers.append(full_name)
                if _should_apply_lora(name, full_name):
                    # Replace with LoRA layer
                    lora_layer = LoRALayer(
                        original_layer=child,
                        r=config.r,
                        lora_alpha=config.lora_alpha,
                        lora_dropout=config.lora_dropout,
                    )
                    setattr(module, name, lora_layer)
                    adapted_modules.append(full_name)
            else:
                # Recurse into child modules
                _apply_lora_recursive(child, full_name)

    _apply_lora_recursive(model)

    if debug or len(adapted_modules) == 0:
        print(f"[LoRA Debug] Target modules: {config.target_modules}")
        print(f"[LoRA Debug] Found {len(all_linear_layers)} Linear layers")
        if all_linear_layers:
            print(f"[LoRA Debug] Sample layer names: {all_linear_layers[:15]}")
        else:
            # Print all module names and types for debugging
            print("[LoRA Debug] No Linear layers found! Listing all modules:")
            for name, module in model.named_modules():
                if name and not list(module.children()):  # Leaf modules only
                    print(f"  {name}: {type(module).__name__}")
                    if len(name.split(".")) > 5:  # Stop after a few to avoid spam
                        break

    return model, adapted_modules


def count_lora_parameters(model: nn.Module) -> Tuple[int, int, float]:
    """
    Count trainable and total parameters with LoRA.

    Returns:
        Tuple of (trainable_params, total_params, percentage_trainable)
    """
    trainable_params = 0
    total_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    percentage = 100 * trainable_params / total_params if total_params > 0 else 0

    return trainable_params, total_params, percentage
