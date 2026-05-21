"""Regression tests for the flat constant layout.

The runtime deliberately deleted `leansatp_runtime.config`: every constant
now lives in exactly one module next to the code that uses it. These tests
pin that layout so any future refactor that moves or renames a hot constant
fails fast in CI instead of silently shipping a randomly-init checkpoint.
"""

from __future__ import annotations

import unittest


class FlatConstantHomesTests(unittest.TestCase):
    def test_head_architecture_constants(self) -> None:
        from leansatp_runtime.models.components.heads import (
            DEFAULT_LEMMA_K,
            HEAD_ATTN_HEADS,
            NUM_PRIORITY_LEVELS,
            SAFE_TACTICS,
            UNSAFE_TACTICS,
        )

        # Head counts are checkpoint-locked: changing them requires retraining.
        # Aligned with ChristianZ97/satp-policy-goal post-2026-05 schema:
        # UNSAFE pool widened 9→17, HEAD_ATTN_HEADS bumped 1→4 (Plan F bump
        # in SATP-Training/src/aesop/config/hyperparameters.py:60).
        self.assertEqual(len(SAFE_TACTICS), 7)
        self.assertEqual(len(UNSAFE_TACTICS), 17)
        self.assertEqual(NUM_PRIORITY_LEVELS, 5)
        self.assertEqual(HEAD_ATTN_HEADS, 4)
        self.assertEqual(DEFAULT_LEMMA_K, 8)

    def test_premise_encoder_max_sequence_length(self) -> None:
        from leansatp_runtime.models.components.premise_encoder import (
            MAX_SEQUENCE_LENGTH,
        )

        self.assertEqual(MAX_SEQUENCE_LENGTH, 1024)

    def test_lora_defaults_match_trained_checkpoint(self) -> None:
        from leansatp_runtime.models.components.lora_adapter import LoRAConfig

        cfg = LoRAConfig()
        self.assertEqual(cfg.r, 16)
        self.assertEqual(cfg.lora_alpha, 32)
        self.assertEqual(cfg.lora_dropout, 0.1)
        self.assertEqual(cfg.target_modules, ("q", "k", "v", "o"))

    def test_legacy_config_module_is_gone(self) -> None:
        """Codify the deliberate removal — no `leansatp_runtime.config` shim."""
        with self.assertRaises(ImportError):
            import leansatp_runtime.config  # noqa: F401


class LoRATargetAutoDetectTests(unittest.TestCase):
    """Pin auto-detection of LoRA target_modules from the checkpoint layout.

    Older `satp-policy-goal` checkpoints (pre 2026-05-21) trained LoRA on
    both attention and FF; the `only_DPO_RL` regime folds FF weights back
    into plain Linears. The runtime must reproduce the same wrapper layout
    the checkpoint was trained against, or `load_state_dict` mismatches.
    """

    def _detect(self, state_dict):
        from leansatp_runtime.service import _detect_lora_targets, _ensure_imports

        _ensure_imports()
        return _detect_lora_targets(state_dict)

    def test_new_format_returns_attention_only(self) -> None:
        sd = {
            "base.encoder.block.0.layer.0.SelfAttention.q.lora_A": None,
            "base.encoder.block.0.layer.0.SelfAttention.q.lora_B": None,
            "base.encoder.block.0.layer.0.SelfAttention.q.original_layer.weight": None,
            "base.encoder.block.0.layer.1.DenseReluDense.wi_0.weight": None,
            "base.encoder.block.0.layer.1.DenseReluDense.wi_1.weight": None,
            "base.encoder.block.0.layer.1.DenseReluDense.wo.weight": None,
        }
        self.assertEqual(self._detect(sd), ("q", "k", "v", "o"))

    def test_legacy_ff_lora_format_extends_targets(self) -> None:
        sd = {
            "base.encoder.block.0.layer.0.SelfAttention.q.lora_A": None,
            "base.encoder.block.0.layer.1.DenseReluDense.wi_0.lora_A": None,
            "base.encoder.block.0.layer.1.DenseReluDense.wi_0.lora_B": None,
            "base.encoder.block.0.layer.1.DenseReluDense.wi_0.original_layer.weight": None,
            "base.encoder.block.0.layer.1.DenseReluDense.wi_1.lora_A": None,
            "base.encoder.block.0.layer.1.DenseReluDense.wo.lora_A": None,
        }
        self.assertEqual(
            self._detect(sd),
            ("q", "k", "v", "o", "wi_0", "wi_1", "wo"),
        )

    def test_partial_ff_lora_still_extends_targets(self) -> None:
        """One `wi_0.lora_A` anywhere is enough to flip into legacy mode.

        A partial-wrap checkpoint is malformed, but the loader's job is to
        reproduce whatever wrapper layout the keys imply; ``strict`` load
        validation will catch the residual key mismatch downstream.
        """
        sd = {"base.encoder.block.7.layer.1.DenseReluDense.wi_0.lora_A": None}
        self.assertIn("wi_0", self._detect(sd))


if __name__ == "__main__":
    unittest.main()
