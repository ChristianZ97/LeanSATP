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
        self.assertEqual(len(SAFE_TACTICS), 7)
        self.assertEqual(len(UNSAFE_TACTICS), 9)
        self.assertEqual(NUM_PRIORITY_LEVELS, 5)
        self.assertEqual(HEAD_ATTN_HEADS, 1)
        self.assertEqual(DEFAULT_LEMMA_K, 8)

    def test_premise_encoder_max_sequence_length(self) -> None:
        from leansatp_runtime.models.components.premise_encoder import (
            MAX_SEQUENCE_LENGTH,
        )

        self.assertEqual(MAX_SEQUENCE_LENGTH, 4096)

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


if __name__ == "__main__":
    unittest.main()
