from __future__ import annotations

from typing import List, Sequence


def generate_embeddings_real(texts: Sequence[str], model_name: str) -> List[List[float]]:
    """Convert captions into normalized embedding vectors with SentenceTransformers."""
    # Keep dependency import local to reduce module import cost when this
    # function is not used.
    from sentence_transformers import SentenceTransformer

    # Load the embedding model from a local path or HF model id.
    model = SentenceTransformer(model_name)

    # Encode all input texts in one batch call and L2-normalize vectors so
    # cosine similarity can be computed directly and consistently.
    vectors = model.encode(list(texts), normalize_embeddings=True, show_progress_bar=True)

    # Convert NumPy array/tensor output to plain Python lists for JSON
    # serialization and easier downstream handling.
    return vectors.tolist()
