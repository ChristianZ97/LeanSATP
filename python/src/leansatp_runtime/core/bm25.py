# src/aesop/core/bm25.py
"""
BM25 Retrieval Module for Hybrid Search

Provides BM25 sparse retrieval to complement dense embedding retrieval.
Uses rank-biased centroid (RBC) fusion or simple score fusion.

Implementation is backed by the `bm25s` library (vectorised numpy
scoring) for two to three orders of magnitude lower per-query latency
than the previous pure-Python loop — this matters because the DSP eval
fires N /infer calls per sketch and BM25 scoring was the biggest CPU
hotspot inside /infer on a 180k-premise corpus.  The external API
(BM25Index.build / score / search / save / load and HybridRetriever)
is preserved so callers do not change.
"""

import re
import pickle
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

import bm25s
import numpy as np


@dataclass
class BM25Config:
    """BM25 hyperparameters."""

    k1: float = 1.5  # Term frequency saturation parameter
    b: float = 0.75  # Length normalization parameter


class BM25Index:
    """
    BM25 index for premise retrieval.

    Implements Okapi BM25 scoring for sparse retrieval via ``bm25s``.
    """

    _PICKLE_VERSION = "bm25s-v1"

    def __init__(self, config: BM25Config = None):
        self.config = config or BM25Config()
        self.documents: List[str] = []
        self._tokens: List[List[str]] = []
        self._bm25: Optional[bm25s.BM25] = None
        self._is_built = False

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 indexing.

        Lean4-aware: splits on common delimiters while keeping meaningful
        alphanumeric+underscore tokens, filters stopwords.  Kept identical
        to the pre-bm25s implementation for deterministic score parity.
        """
        text = text.lower()
        tokens = re.findall(r"[a-z0-9_]+", text)
        stopwords = {"a", "an", "the", "is", "are", "be", "to", "of", "in", "for", "by"}
        return [t for t in tokens if len(t) > 1 and t not in stopwords]

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def build(self, documents: List[str]) -> None:
        """Build BM25 index from documents.

        Args:
            documents: List of premise strings (raw format)
        """
        self.documents = list(documents)
        self._tokens = [self.tokenize(doc) for doc in self.documents]
        self._build_engine()

    def _build_engine(self) -> None:
        self._bm25 = bm25s.BM25(k1=self.config.k1, b=self.config.b)
        self._bm25.index(self._tokens, show_progress=False)
        self._is_built = True

    # ------------------------------------------------------------------
    # Scoring / retrieval
    # ------------------------------------------------------------------

    def score(self, query: str, doc_idx: int) -> float:
        """
        Compute BM25 score for a query-document pair.

        Kept for backward compatibility with any direct caller; internally
        bm25s.get_scores computes the whole row, so this is O(N) — prefer
        ``search`` when ranking over the whole corpus.
        """
        if not self._is_built or self._bm25 is None:
            raise ValueError("Index not built. Call build() first.")
        q_tokens = self.tokenize(query)
        scores = self._bm25.get_scores(q_tokens)
        return float(scores[doc_idx])

    def search(self, query: str, k: int = 10) -> Tuple[List[int], np.ndarray]:
        """
        Search for top-k documents matching the query.

        Returns:
            Tuple of (document indices, scores), length min(k, N).
        """
        if not self._is_built or self._bm25 is None:
            raise ValueError("Index not built. Call build() first.")

        q_tokens = self.tokenize(query)
        actual_k = min(k, len(self.documents))
        if actual_k == 0:
            return [], np.zeros(0, dtype=np.float32)

        docs, scores = self._bm25.retrieve(
            [q_tokens],
            k=actual_k,
            return_as="tuple",
            show_progress=False,
        )
        # bm25s returns shape [n_queries, k]; we always pass one query.
        return docs[0].tolist(), scores[0]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save BM25 index to disk.

        We persist {config, documents, pre-tokenised docs} and rebuild the
        bm25s engine on load (fast: single-digit seconds on 180k premises).
        This keeps save/load a single pickle file (matches the legacy
        interface) instead of pulling in bm25s's multi-file save format.
        """
        data: Dict[str, Any] = {
            "version": self._PICKLE_VERSION,
            "config": {"k1": self.config.k1, "b": self.config.b},
            "documents": self.documents,
            "tokens": self._tokens,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str) -> None:
        """Load BM25 index from disk.

        Accepts both the new bm25s-v1 pickle shape and the pre-bm25s
        legacy shape (dict with doc_freqs / term_freqs / idf_cache etc.).
        Legacy pickles get auto-migrated by retokenising from their
        documents list.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.config = BM25Config(**data["config"])
        self.documents = list(data.get("documents", []))

        if data.get("version") == self._PICKLE_VERSION and "tokens" in data:
            self._tokens = data["tokens"]
        else:
            # Legacy pickle (doc_freqs/term_freqs/idf_cache keys) —
            # the pre-tokenised stream wasn't stored, so retokenise
            # from the documents to feed bm25s.
            self._tokens = [self.tokenize(doc) for doc in self.documents]

        self._build_engine()


class HybridRetriever:
    """
    Hybrid retriever combining dense (sentence encoder) and sparse (BM25) retrieval.

    Supports multiple fusion strategies:
    - "rrf": Reciprocal Rank Fusion (recommended)
    - "linear": Linear combination of normalized scores
    - "dense_only": Use only dense retrieval (original behavior)
    - "bm25_only": Use only BM25 retrieval
    """

    def __init__(
        self,
        fusion_method: str = "rrf",
        dense_weight: float = 0.7,
        bm25_weight: float = 0.3,
        rrf_k: int = 60,
    ):
        """
        Initialize hybrid retriever.

        Args:
            fusion_method: One of "rrf", "linear", "dense_only", "bm25_only"
            dense_weight: Weight for dense scores in linear fusion
            bm25_weight: Weight for BM25 scores in linear fusion
            rrf_k: Constant k for RRF formula (default 60)
        """
        self.fusion_method = fusion_method
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k
        self.bm25_index: Optional[BM25Index] = None

    def set_bm25_index(self, index: BM25Index) -> None:
        """Attach a pre-built BM25 index."""
        self.bm25_index = index

    def retrieve(
        self, query: str, dense_scores: np.ndarray, k: int
    ) -> Tuple[List[int], np.ndarray]:
        """
        Perform hybrid retrieval.

        Args:
            query: Query string for BM25
            dense_scores: Dense similarity scores (shape: [num_docs])
            k: Number of top results

        Returns:
            Tuple of (top_k_indices, top_k_scores)
        """
        if self.fusion_method == "dense_only" or self.bm25_index is None:
            # Fall back to dense-only
            actual_k = min(k, len(dense_scores))
            top_k_indices = np.argsort(dense_scores)[-actual_k:][::-1]
            top_k_scores = dense_scores[top_k_indices]
            return top_k_indices.tolist(), top_k_scores

        # Get BM25 scores for all documents
        bm25_indices, bm25_scores_top = self.bm25_index.search(query, len(dense_scores))

        # Build full BM25 score array
        bm25_scores = np.zeros(len(dense_scores), dtype=np.float32)
        for idx, score in zip(bm25_indices, bm25_scores_top):
            bm25_scores[idx] = score

        if self.fusion_method == "bm25_only":
            actual_k = min(k, len(bm25_scores))
            top_k_indices = np.argsort(bm25_scores)[-actual_k:][::-1]
            top_k_scores = bm25_scores[top_k_indices]
            return top_k_indices.tolist(), top_k_scores

        elif self.fusion_method == "rrf":
            return self._rrf_fusion(dense_scores, bm25_scores, k)

        elif self.fusion_method == "linear":
            return self._linear_fusion(dense_scores, bm25_scores, k)

        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

    def _rrf_fusion(
        self, dense_scores: np.ndarray, bm25_scores: np.ndarray, k: int
    ) -> Tuple[List[int], np.ndarray]:
        """Reciprocal Rank Fusion."""
        # Get rankings (higher score = better rank)
        dense_ranks = np.argsort(np.argsort(-dense_scores))
        bm25_ranks = np.argsort(np.argsort(-bm25_scores))

        # RRF score: sum of 1/(rank + k) for each retriever
        rrf_scores = (
            1.0 / (dense_ranks + self.rrf_k + 1)
            + 1.0 / (bm25_ranks + self.rrf_k + 1)
        )

        actual_k = min(k, len(rrf_scores))
        top_k_indices = np.argsort(rrf_scores)[-actual_k:][::-1]
        top_k_scores = rrf_scores[top_k_indices]

        return top_k_indices.tolist(), top_k_scores

    def _linear_fusion(
        self, dense_scores: np.ndarray, bm25_scores: np.ndarray, k: int
    ) -> Tuple[List[int], np.ndarray]:
        """Linear combination of normalized scores."""
        # Min-max normalize both score arrays
        def normalize(scores):
            min_s, max_s = scores.min(), scores.max()
            if max_s - min_s > 1e-9:
                return (scores - min_s) / (max_s - min_s)
            return np.zeros_like(scores)

        norm_dense = normalize(dense_scores)
        norm_bm25 = normalize(bm25_scores)

        combined = self.dense_weight * norm_dense + self.bm25_weight * norm_bm25

        actual_k = min(k, len(combined))
        top_k_indices = np.argsort(combined)[-actual_k:][::-1]
        top_k_scores = combined[top_k_indices]

        return top_k_indices.tolist(), top_k_scores
