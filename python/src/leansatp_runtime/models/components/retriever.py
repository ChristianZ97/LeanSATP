"""Premise retrieval: cached dense embeddings + optional BM25 hybrid."""

from __future__ import annotations

import os
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ...core import Premise
from ...core.bm25 import BM25Index, HybridRetriever
from .heads import DEFAULT_LEMMA_K

# Hybrid retrieval defaults baked into the trained checkpoint.
_HYBRID_FUSION_METHOD = "rrf"
_HYBRID_DENSE_WEIGHT = 0.7
_HYBRID_BM25_WEIGHT = 0.3
_HYBRID_RRF_K = 60


class PremiseRetriever:
    """Cached premise embeddings + optional BM25 hybrid retriever (read-only)."""

    def __init__(
        self,
        cache_dir: str,
        encode_fn: Callable[[List[str]], torch.Tensor],
        hidden_size: int,
        device: torch.device,
        use_hybrid: bool = True,
    ):
        self.cache_dir = cache_dir
        self.encode_fn = encode_fn
        self.hidden_size = hidden_size
        self.device = device
        self.use_hybrid = use_hybrid

        self._premises: Optional[List[Premise]] = None
        self._cached_embs: Optional[torch.Tensor] = None
        self._hybrid: Optional[HybridRetriever] = None

    @property
    def ready(self) -> bool:
        return self._premises is not None and self._cached_embs is not None

    def load(self) -> None:
        emb_path = os.path.join(self.cache_dir, "premise_embeddings.npy")
        raw_path = os.path.join(self.cache_dir, "premises_raw.npy")

        if not os.path.exists(emb_path) or not os.path.exists(raw_path):
            missing = [p for p in (emb_path, raw_path) if not os.path.exists(p)]
            raise FileNotFoundError("missing retrieval assets: " + ", ".join(missing))

        embeddings = torch.from_numpy(np.load(emb_path)).float().to(self.device)
        raw = np.load(raw_path, allow_pickle=True)
        self._premises = [Premise.from_leandojo_format(r) for r in raw]
        self._cached_embs = embeddings

        if self.use_hybrid:
            self._load_hybrid()

    def _load_hybrid(self) -> None:
        bm25_path = os.path.join(self.cache_dir, "bm25_index.pkl")
        if not os.path.exists(bm25_path):
            self.use_hybrid = False
            return

        bm25_index = BM25Index()
        bm25_index.load(bm25_path)
        self._hybrid = HybridRetriever(
            fusion_method=_HYBRID_FUSION_METHOD,
            dense_weight=_HYBRID_DENSE_WEIGHT,
            bm25_weight=_HYBRID_BM25_WEIGHT,
            rrf_k=_HYBRID_RRF_K,
        )
        self._hybrid.set_bm25_index(bm25_index)

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        normalize_scores: bool = True,
    ) -> Tuple[List[Optional[Premise]], torch.Tensor, torch.Tensor]:
        """Single-query top-k retrieval."""
        if k is None:
            k = DEFAULT_LEMMA_K
        if not self.ready:
            raise ValueError("Premise cache not initialized.")
        assert self._premises is not None and self._cached_embs is not None

        query_emb = self.encode_fn([query]).to(self._cached_embs.device)
        q_norm = F.normalize(query_emb, dim=1)
        c_norm = F.normalize(self._cached_embs, dim=1)
        similarities = torch.mm(q_norm, c_norm.t())[0]

        if self.use_hybrid and self._hybrid is not None:
            dense_np = similarities.detach().cpu().numpy()
            indices, scores_np = self._hybrid.retrieve(query, dense_np, k)
            premises = [self._premises[i] for i in indices]
            scores = torch.tensor(
                scores_np, dtype=torch.float32, device=self._cached_embs.device
            )
            embeddings = self._cached_embs[
                torch.tensor(indices, device=self._cached_embs.device, dtype=torch.long)
            ]
        else:
            actual_k = min(k, len(self._premises))
            scores, top_indices = torch.topk(similarities, actual_k)
            premises = [self._premises[i] for i in top_indices.tolist()]
            embeddings = self._cached_embs[top_indices]
            if actual_k < k:
                pad = k - actual_k
                scores = torch.cat([scores, scores.new_zeros(pad)])
                premises.extend([None] * pad)
                embeddings = torch.cat(
                    [embeddings, embeddings.new_zeros(pad, embeddings.shape[1])]
                )

        if normalize_scores and scores.numel() > 0:
            lo, hi = scores.min(), scores.max()
            denom = (hi - lo).clamp(min=1e-8)
            scores = (scores - lo) / denom

        return premises, scores, embeddings
