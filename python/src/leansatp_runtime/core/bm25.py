# src/aesop/core/bm25.py
"""
BM25 Retrieval Module for Hybrid Search

Provides BM25 sparse retrieval to complement dense embedding retrieval.
Uses rank-biased centroid (RBC) fusion or simple score fusion.
"""

import re
import math
import pickle
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from collections import Counter

import numpy as np


@dataclass
class BM25Config:
    """BM25 hyperparameters."""

    k1: float = 1.5  # Term frequency saturation parameter
    b: float = 0.75  # Length normalization parameter


class BM25Index:
    """
    BM25 index for premise retrieval.

    Implements Okapi BM25 scoring for sparse retrieval.
    """

    def __init__(self, config: BM25Config = None):
        self.config = config or BM25Config()
        self.documents: List[str] = []
        self.doc_freqs: Dict[str, int] = {}  # Document frequency per term
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.term_freqs: List[Dict[str, int]] = []  # Term frequencies per document
        self.idf_cache: Dict[str, float] = {}
        self._is_built = False

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 indexing.

        Handles Lean4 syntax by splitting on common delimiters
        while preserving meaningful tokens.
        """
        # Lowercase and split on whitespace and common Lean delimiters
        text = text.lower()

        # Split on various delimiters while keeping alphanumeric tokens
        tokens = re.findall(r"[a-z0-9_]+", text)

        # Filter out very short tokens and common stopwords
        stopwords = {"a", "an", "the", "is", "are", "be", "to", "of", "in", "for", "by"}
        tokens = [t for t in tokens if len(t) > 1 and t not in stopwords]

        return tokens

    def build(self, documents: List[str]) -> None:
        """
        Build BM25 index from documents.

        Args:
            documents: List of premise strings (raw format)
        """
        self.documents = documents
        self.doc_freqs = {}
        self.doc_lengths = []
        self.term_freqs = []

        total_length = 0

        for doc in documents:
            tokens = self.tokenize(doc)
            self.doc_lengths.append(len(tokens))
            total_length += len(tokens)

            # Count term frequencies in this document
            tf = Counter(tokens)
            self.term_freqs.append(dict(tf))

            # Update document frequencies
            for term in set(tokens):
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        self.avg_doc_length = total_length / len(documents) if documents else 0

        # Precompute IDF values
        n_docs = len(documents)
        for term, df in self.doc_freqs.items():
            # Standard BM25 IDF formula
            self.idf_cache[term] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)

        self._is_built = True

    def score(self, query: str, doc_idx: int) -> float:
        """
        Compute BM25 score for a query-document pair.

        Args:
            query: Query string
            doc_idx: Index of the document

        Returns:
            BM25 score
        """
        if not self._is_built:
            raise ValueError("Index not built. Call build() first.")

        query_tokens = self.tokenize(query)
        doc_tf = self.term_freqs[doc_idx]
        doc_len = self.doc_lengths[doc_idx]

        score = 0.0
        k1, b = self.config.k1, self.config.b

        for term in query_tokens:
            if term not in doc_tf:
                continue

            tf = doc_tf[term]
            idf = self.idf_cache.get(term, 0)

            # BM25 scoring formula
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_len / self.avg_doc_length)
            score += idf * numerator / denominator

        return score

    def search(self, query: str, k: int = 10) -> Tuple[List[int], np.ndarray]:
        """
        Search for top-k documents matching the query.

        Args:
            query: Query string
            k: Number of results to return

        Returns:
            Tuple of (document indices, scores)
        """
        if not self._is_built:
            raise ValueError("Index not built. Call build() first.")

        # Compute scores for all documents
        scores = np.array([self.score(query, i) for i in range(len(self.documents))])

        # Get top-k
        actual_k = min(k, len(self.documents))
        top_indices = np.argsort(scores)[-actual_k:][::-1]
        top_scores = scores[top_indices]

        return top_indices.tolist(), top_scores

    def save(self, path: str) -> None:
        """Save BM25 index to disk."""
        data = {
            "config": {"k1": self.config.k1, "b": self.config.b},
            "documents": self.documents,
            "doc_freqs": self.doc_freqs,
            "doc_lengths": self.doc_lengths,
            "term_freqs": self.term_freqs,
            "avg_doc_length": self.avg_doc_length,
            "idf_cache": self.idf_cache,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str) -> None:
        """Load BM25 index from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.config = BM25Config(**data["config"])
        self.documents = data["documents"]
        self.doc_freqs = data["doc_freqs"]
        self.doc_lengths = data["doc_lengths"]
        self.term_freqs = data["term_freqs"]
        self.avg_doc_length = data["avg_doc_length"]
        self.idf_cache = data["idf_cache"]
        self._is_built = True


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
        """Set the BM25 index."""
        self.bm25_index = index

    def fuse_rrf(
        self,
        dense_indices: List[int],
        dense_scores: np.ndarray,
        bm25_indices: List[int],
        bm25_scores: np.ndarray,
        k: int,
    ) -> Tuple[List[int], np.ndarray]:
        """
        Reciprocal Rank Fusion of dense and BM25 results.

        RRF score = sum(1 / (k + rank_i)) for each result list
        """
        # Build rank maps
        dense_ranks = {idx: rank for rank, idx in enumerate(dense_indices)}
        bm25_ranks = {idx: rank for rank, idx in enumerate(bm25_indices)}

        # Get all unique document indices
        all_indices = set(dense_indices) | set(bm25_indices)

        # Compute RRF scores
        rrf_scores = {}
        for idx in all_indices:
            score = 0.0
            if idx in dense_ranks:
                score += self.dense_weight / (self.rrf_k + dense_ranks[idx])
            if idx in bm25_ranks:
                score += self.bm25_weight / (self.rrf_k + bm25_ranks[idx])
            rrf_scores[idx] = score

        # Sort by RRF score and take top-k
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        final_indices = [item[0] for item in sorted_items]
        final_scores = np.array([item[1] for item in sorted_items])

        return final_indices, final_scores

    def fuse_linear(
        self,
        dense_indices: List[int],
        dense_scores: np.ndarray,
        bm25_indices: List[int],
        bm25_scores: np.ndarray,
        k: int,
        n_docs: int,
    ) -> Tuple[List[int], np.ndarray]:
        """
        Linear fusion of normalized dense and BM25 scores.
        """

        # Normalize scores to [0, 1]
        def normalize(scores):
            if len(scores) == 0 or scores.max() == scores.min():
                return scores
            return (scores - scores.min()) / (scores.max() - scores.min())

        # Create full score arrays
        full_dense = np.zeros(n_docs)
        full_bm25 = np.zeros(n_docs)

        for idx, score in zip(dense_indices, normalize(dense_scores)):
            full_dense[idx] = score
        for idx, score in zip(bm25_indices, normalize(bm25_scores)):
            full_bm25[idx] = score

        # Linear combination
        combined = self.dense_weight * full_dense + self.bm25_weight * full_bm25

        # Get top-k
        top_indices = np.argsort(combined)[-k:][::-1]
        top_scores = combined[top_indices]

        return top_indices.tolist(), top_scores

    def retrieve(
        self,
        query: str,
        dense_scores: np.ndarray,
        k: int,
    ) -> Tuple[List[int], np.ndarray]:
        """
        Perform hybrid retrieval.

        Args:
            query: Query string for BM25
            dense_scores: Precomputed dense similarity scores for all premises
            k: Number of results to return

        Returns:
            Tuple of (premise indices, hybrid scores)
        """
        n_docs = len(dense_scores)

        # Get dense top-k
        dense_k = min(k * 2, n_docs)  # Retrieve more for fusion
        dense_top_indices = np.argsort(dense_scores)[-dense_k:][::-1]
        dense_top_scores = dense_scores[dense_top_indices]

        if self.fusion_method == "dense_only" or self.bm25_index is None:
            # Original behavior
            final_indices = dense_top_indices[:k].tolist()
            final_scores = dense_top_scores[:k]
            return final_indices, final_scores

        if self.fusion_method == "bm25_only":
            bm25_indices, bm25_scores = self.bm25_index.search(query, k)
            return bm25_indices, bm25_scores

        # Get BM25 results
        bm25_indices, bm25_scores = self.bm25_index.search(query, dense_k)

        if self.fusion_method == "rrf":
            return self.fuse_rrf(
                dense_top_indices.tolist(),
                dense_top_scores,
                bm25_indices,
                bm25_scores,
                k,
            )
        elif self.fusion_method == "linear":
            return self.fuse_linear(
                dense_top_indices.tolist(),
                dense_top_scores,
                bm25_indices,
                bm25_scores,
                k,
                n_docs,
            )
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
