"""Minimal inference-time configuration for LeanSATP."""

from __future__ import annotations

from types import SimpleNamespace


config = SimpleNamespace(
    SAFE_TACTICS=[
        "ring",
        "abel",
        "norm_num",
        "norm_cast",
        "push_neg",
        "field_simp",
        "zify",
    ],
    UNSAFE_TACTICS=[
        "linarith",
        "nlinarith",
        "omega",
        "gcongr",
        "positivity",
        "interval_cases",
        "ext",
        "exfalso",
        "split",
    ],
    NUM_PRIORITY_LEVELS=5,
    LEMMA_K=8,
    SENTENCE_ENCODER="kaiyuy/leandojo-lean4-retriever-byt5-small",
    MAX_SEQUENCE_LENGTH=4096,
    PREMISE_BATCH_SIZE=16,
    DROPOUT_RATE=0.2,
    DEFAULT_AESOP_CONFIG={
        "maxRuleApplicationDepth": 30,
        "maxRuleApplications": 200,
        "maxNormIterations": 100,
        "enableSimp": True,
        "useSimpAll": True,
    },
    USE_LORA=True,
    LORA_R=16,
    LORA_ALPHA=32,
    LORA_DROPOUT=0.1,
    LORA_TARGET_MODULES=("q", "k", "v", "o"),
    USE_GRADIENT_CHECKPOINTING=False,
    USE_HYBRID_RETRIEVAL=True,
    HYBRID_FUSION_METHOD="rrf",
    HYBRID_DENSE_WEIGHT=0.7,
    HYBRID_BM25_WEIGHT=0.3,
    HYBRID_RRF_K=60,
)
