# src/aesop/models/components/premise_encoder.py
"""
Premise encoder component for embedding premises.
"""

import torch
from typing import List
from tqdm import tqdm

from ...config import config


class PremiseEncoder:
    """Handles premise encoding and caching."""

    def __init__(self, base_model, tokenizer, cache_dir: str = "./cache/"):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir

    def encode_premises(
        self,
        premises: List[str],
        batch_size: int = None,
        verbose: bool = True,
    ) -> torch.Tensor:
        """Encode premises into embeddings in batches.

        Args:
            premises: List of premise strings to encode
            batch_size: Batch size for encoding (default: config.PREMISE_BATCH_SIZE)
            verbose: If True, show progress bar. Set to False during training.
        """
        if batch_size is None:
            batch_size = getattr(config, "PREMISE_BATCH_SIZE", 128)

        all_embeddings = []

        # Only show progress bar for large batches and when verbose is True
        iterator = range(0, len(premises), batch_size)
        if verbose and len(premises) > batch_size:
            iterator = tqdm(iterator, desc="Encoding premises")

        for i in iterator:
            batch_premises = premises[i : i + batch_size]

            # Tokenize batch
            # ByT5 is byte-level: 1 char ≈ 1-4 bytes, needs longer max_length
            inputs = self.tokenizer(
                batch_premises,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.MAX_SEQUENCE_LENGTH,
            )

            # Move to device
            inputs = {k: v.to(self.base_model.device) for k, v in inputs.items()}

            # Encode
            with torch.no_grad():
                outputs = self.base_model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token

            all_embeddings.append(embeddings.cpu())  # Move to CPU to save GPU memory

            # Clear GPU cache periodically to prevent memory accumulation
            if torch.cuda.is_available():
                del inputs
                del outputs
                torch.cuda.empty_cache()

        # Concatenate all embeddings
        final_embeddings = torch.cat(all_embeddings, dim=0)

        return final_embeddings
