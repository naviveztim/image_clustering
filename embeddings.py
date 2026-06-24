from __future__ import annotations

from typing import List, Sequence


def generate_embeddings_real(texts: Sequence[str], model_name: str) -> List[List[float]]:
    """Convert captions into normalized embedding vectors with SentenceTransformers."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    vectors = model.encode(list(texts), normalize_embeddings=True, show_progress_bar=True)
    return vectors.tolist()
