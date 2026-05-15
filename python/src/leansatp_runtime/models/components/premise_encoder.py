"""ByT5 premise/query encoder (CLS-token, no grad)."""

from typing import List

import torch

# ByT5 byte-level encoder; cap aligned with SATP-Training's
# `MAX_SEQUENCE_LENGTH = 1024` (hyperparameters.py:65 — itself aligned with
# ReProver upstream's LeanDojo retriever pretraining geometry). The training
# distribution is byte-truncated at 1024; inference must match or the model
# sees out-of-distribution longer sequences where the per-byte positional
# encoding was never optimised, silently degrading both `policy_tactic`
# tokenisation (service.py:562) and premise CLS encoding (encode below).
MAX_SEQUENCE_LENGTH = 1024
_DEFAULT_BATCH_SIZE = 16


class PremiseEncoder:
    """Encodes premise / query strings into ByT5 CLS embeddings."""

    def __init__(self, base_model, tokenizer, cache_dir: str = "./cache/"):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir

    def encode(
        self, texts: List[str], batch_size: int = _DEFAULT_BATCH_SIZE
    ) -> torch.Tensor:
        """Encode `texts` into [N, hidden] embeddings."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_SEQUENCE_LENGTH,
            )
            inputs = {k: v.to(self.base_model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.base_model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(embeddings.cpu())

            if torch.cuda.is_available():
                del inputs, outputs
                torch.cuda.empty_cache()

        return torch.cat(all_embeddings, dim=0)
