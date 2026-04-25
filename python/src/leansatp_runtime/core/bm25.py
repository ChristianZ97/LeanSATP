"""BM25 sparse retrieval (via `bm25s`) + hybrid dense/sparse fusion.

Inference-only subset of the training repo's BM25:
- Load a trained index (with on-the-fly rebuild from cached `documents` if
  the bm25s native directory was not shipped alongside the pickle).
- Search top-k for a query.
- Fuse with precomputed dense scores (RRF / linear / single-stream).
"""

import os
import pickle
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import bm25s
import numpy as np


@dataclass
class BM25Config:
    """BM25 hyperparameters."""

    k1: float = 1.5
    b: float = 0.75


_STOPWORDS = frozenset(
    {"a", "an", "the", "is", "are", "be", "to", "of", "in", "for", "by"}
)


def tokenize_lean(text: str) -> List[str]:
    """Lowercase, split on non-alphanumeric, drop stopwords + single chars."""
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    return [t for t in tokens if len(t) > 1 and t not in _STOPWORDS]


class BM25Index:
    """BM25 index for premise retrieval, backed by bm25s. Load + search only."""

    def __init__(self, config: Optional[BM25Config] = None):
        self.config = config or BM25Config()
        self.documents: List[str] = []
        self._bm25: Optional[bm25s.BM25] = None

    def search(self, query: str, k: int = 10) -> Tuple[List[int], np.ndarray]:
        """Return (top-k indices, scores) for the query."""
        if self._bm25 is None:
            raise ValueError("Index not loaded.")
        if not self.documents:
            raise ValueError(
                "BM25 index has no documents list; refusing to search silently."
            )

        actual_k = min(k, len(self.documents))
        query_tokens = [tokenize_lean(query)]
        if not query_tokens[0]:
            return [], np.zeros(0, dtype=np.float32)

        docs, scores = self._bm25.retrieve(
            query_tokens, k=actual_k, show_progress=False
        )
        return docs[0].tolist(), scores[0]

    def load(self, path: str) -> None:
        """Load a BM25 pickle written by the training pipeline.

        Two formats supported:
        1. `bm25s` native dir referenced by `meta["dir"]` — fastest reload.
        2. Pickle carrying `documents` only — rebuild the bm25s engine in
           memory from the documents list.
        """
        with open(path, "rb") as f:
            meta = pickle.load(f)

        self.config = self._extract_config(meta)
        self.documents = self._extract_documents(meta)

        save_dir = (
            meta.get("dir", path + ".bm25s")
            if isinstance(meta, dict)
            else (path + ".bm25s")
        )
        if os.path.isdir(save_dir):
            self._bm25 = bm25s.BM25.load(save_dir)
            if not self.documents:
                raise ValueError(
                    f"BM25 native index loaded from {save_dir} but its sidecar pickle "
                    f"{path} carries no `documents` list; refusing to serve a "
                    "silently-empty BM25 stream."
                )
            return

        if not self.documents:
            raise ValueError(
                "BM25 pickle has neither a bm25s directory nor a documents list."
            )

        corpus_tokens = [tokenize_lean(doc) for doc in self.documents]
        self._bm25 = bm25s.BM25(k1=self.config.k1, b=self.config.b)
        self._bm25.index(corpus_tokens, show_progress=False)

    @staticmethod
    def _extract_config(meta: Any) -> BM25Config:
        default = BM25Config()
        if isinstance(meta, dict):
            if "k1" in meta or "b" in meta:
                return BM25Config(
                    k1=float(meta.get("k1", default.k1)),
                    b=float(meta.get("b", default.b)),
                )
            legacy = meta.get("config")
            if isinstance(legacy, BM25Config):
                return legacy
            if isinstance(legacy, dict):
                return BM25Config(
                    k1=float(legacy.get("k1", default.k1)),
                    b=float(legacy.get("b", default.b)),
                )
        return default

    @staticmethod
    def _extract_documents(meta: Any) -> List[str]:
        if isinstance(meta, dict):
            documents = meta.get("documents")
            if isinstance(documents, list):
                return documents
        return []


class HybridRetriever:
    """Dense + sparse fusion: RRF (default), linear, or single-stream."""

    def __init__(
        self,
        fusion_method: str = "rrf",
        dense_weight: float = 0.7,
        bm25_weight: float = 0.3,
        rrf_k: int = 60,
    ):
        self.fusion_method = fusion_method
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k
        self.bm25_index: Optional[BM25Index] = None

    def set_bm25_index(self, index: BM25Index) -> None:
        self.bm25_index = index

    def retrieve(
        self,
        query: str,
        dense_scores: np.ndarray,
        k: int,
    ) -> Tuple[List[int], np.ndarray]:
        n_docs = len(dense_scores)
        dense_k = min(k * 2, n_docs)
        dense_top_indices = np.argsort(dense_scores)[-dense_k:][::-1]
        dense_top_scores = dense_scores[dense_top_indices]

        if self.fusion_method == "dense_only" or self.bm25_index is None:
            return dense_top_indices[:k].tolist(), dense_top_scores[:k]
        if self.fusion_method == "bm25_only":
            return self.bm25_index.search(query, k)

        bm25_indices, bm25_scores = self.bm25_index.search(query, dense_k)

        if self.fusion_method == "rrf":
            return self._fuse_rrf(
                dense_top_indices.tolist(),
                bm25_indices,
                k,
            )
        if self.fusion_method == "linear":
            return self._fuse_linear(
                dense_top_indices.tolist(),
                dense_top_scores,
                bm25_indices,
                bm25_scores,
                k,
                n_docs,
            )
        raise ValueError(f"Unknown fusion method: {self.fusion_method}")

    def _fuse_rrf(
        self,
        dense_indices: List[int],
        bm25_indices: List[int],
        k: int,
    ) -> Tuple[List[int], np.ndarray]:
        dense_ranks = {idx: rank for rank, idx in enumerate(dense_indices)}
        bm25_ranks = {idx: rank for rank, idx in enumerate(bm25_indices)}

        scores: Dict[int, float] = {}
        for idx in set(dense_indices) | set(bm25_indices):
            s = 0.0
            if idx in dense_ranks:
                s += self.dense_weight / (self.rrf_k + dense_ranks[idx])
            if idx in bm25_ranks:
                s += self.bm25_weight / (self.rrf_k + bm25_ranks[idx])
            scores[idx] = s

        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [i for i, _ in top], np.array([s for _, s in top])

    def _fuse_linear(
        self,
        dense_indices: List[int],
        dense_scores: np.ndarray,
        bm25_indices: List[int],
        bm25_scores: np.ndarray,
        k: int,
        n_docs: int,
    ) -> Tuple[List[int], np.ndarray]:
        def normalize(scores: np.ndarray) -> np.ndarray:
            if len(scores) == 0 or scores.max() == scores.min():
                return scores
            return (scores - scores.min()) / (scores.max() - scores.min())

        full_dense = np.zeros(n_docs)
        full_bm25 = np.zeros(n_docs)
        for idx, s in zip(dense_indices, normalize(dense_scores)):
            full_dense[idx] = s
        for idx, s in zip(bm25_indices, normalize(bm25_scores)):
            full_bm25[idx] = s

        combined = self.dense_weight * full_dense + self.bm25_weight * full_bm25
        top_indices = np.argsort(combined)[-k:][::-1]
        return top_indices.tolist(), combined[top_indices]
