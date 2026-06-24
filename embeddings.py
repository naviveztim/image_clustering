from __future__ import annotations

import math
from typing import List, Sequence

from tqdm import tqdm


def generate_embeddings_real(texts: Sequence[str], model_name: str) -> List[List[float]]:
    """Convert captions into normalized embedding vectors with SentenceTransformers."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    vectors = model.encode(list(texts), normalize_embeddings=True, show_progress_bar=True)
    return vectors.tolist()


def generate_embeddings_mock(texts: Sequence[str], dims: int = 24) -> List[List[float]]:
    """Create deterministic mock embeddings without loading external ML models."""
    vectors: List[List[float]] = []
    for text in tqdm(texts, desc="Generating mock embeddings"):
        seed = abs(hash(text))
        vec = [((seed >> i) & 255) / 255.0 for i in range(dims)]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        vectors.append([v / norm for v in vec])
    return vectors

