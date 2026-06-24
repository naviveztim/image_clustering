from __future__ import annotations

import logging
import math
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, cast

logger = logging.getLogger(__name__)


def cluster_embeddings(
    embeddings: Sequence[Sequence[float]],
    method: str,
    n_clusters: int | None,
    distance_threshold: float,
) -> List[int]:
    """Assign cluster labels to embedding vectors using hierarchical clustering or KMeans."""
    if not embeddings:
        return []
    if len(embeddings) == 1:
        return [0]

    import numpy as np
    from sklearn.cluster import AgglomerativeClustering, KMeans

    array = np.array(embeddings, dtype=float)
    if method == "hierarchical":
        if n_clusters is not None:
            model = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric="cosine",
                linkage="average",
            )
        else:
            model = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                metric="cosine",
                linkage="average",
            )
    else:
        if n_clusters is None:
            n_clusters = max(2, min(8, int(math.sqrt(len(embeddings)))))
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(array)
    return [int(label) for label in labels]


def _top_terms_from_captions(captions: Sequence[str], top_k: int = 3) -> List[str]:
    """Extract representative caption terms to help build readable cluster names."""
    if not captions:
        return []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=1000)
        matrix = vectorizer.fit_transform(captions)
        weights = matrix.sum(axis=0)
        terms = vectorizer.get_feature_names_out()
        ranked = sorted(
            ((terms[i], float(weights[0, i])) for i in range(len(terms))),
            key=lambda item: item[1],
            reverse=True,
        )
        return [term for term, _ in ranked[:top_k] if term]
    except Exception:
        words = re.findall(r"[a-zA-Z]{3,}", " ".join(captions).lower())
        common = Counter(words).most_common(top_k)
        return [word for word, _ in common]


def sanitize_cluster_name(raw_name: str, max_len: int = 20) -> str:
    """Normalize a cluster name into a short filesystem-friendly folder name."""
    value = re.sub(r"[^a-zA-Z0-9\-_ ]+", "", raw_name.lower()).strip()
    value = re.sub(r"\s+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    if not value:
        value = "cluster"
    return value[:max_len]


def generate_cluster_names(label_to_captions: Dict[int, List[str]]) -> Dict[int, str]:
    """Create unique human-readable names for each cluster label from its captions."""
    names: Dict[int, str] = {}
    used: set[str] = set()
    for label in sorted(label_to_captions):
        terms = _top_terms_from_captions(label_to_captions[label])
        candidate: str = sanitize_cluster_name("-".join(terms) if terms else f"cluster-{label}")
        if not candidate:
            candidate = f"cluster-{label}"
        if candidate in used:
            suffix = 2
            base = candidate[:16] if len(candidate) > 16 else candidate
            while f"{base}-{suffix}" in used:
                suffix += 1
            candidate = f"{base}-{suffix}"[:20]
        names[label] = candidate
        used.add(candidate)
    return names


def summarize_clusters(records: Sequence[Dict[str, object]], cluster_names: Dict[int, str]) -> List[Dict[str, object]]:
    """Build a stable cluster summary containing numeric labels, names, and counts."""
    counts: Counter[int] = Counter()
    for record in records:
        label = record.get("cluster_label")
        if isinstance(label, int):
            counts[label] += 1

    summaries: List[Dict[str, object]] = []
    for label in sorted(counts):
        summaries.append(
            {
                "label": label,
                "name": cluster_names.get(label, f"cluster-{label}"),
                "count": counts[label],
            }
        )
    return summaries


def _resolve_record_source_path(record: Dict[str, object]) -> Path | None:
    """Resolve the best available source file path for an image record."""
    for key in ("image_path", "cluster_file_path"):
        value = record.get(key)
        if isinstance(value, str) and value:
            path = Path(value)
            if path.exists():
                return path
    return None


def organize_cluster_files(
    records: List[Dict[str, object]],
    cluster_root: Path,
    cluster_names: Dict[int, str],
) -> Dict[str, int]:
    """Copy images into the latest cluster folders and record their new locations."""
    counts: Dict[str, int] = Counter()
    cluster_root.mkdir(parents=True, exist_ok=True)

    for record in records:
        label = int(cast(int, record["cluster_label"]))
        cluster_name = cluster_names[label]

        source = _resolve_record_source_path(record)
        if source is None:
            logger.warning("Skipping file sync for %s because no source file exists.", record.get("image_path"))
            record["cluster_name"] = cluster_name
            continue

        destination_dir = cluster_root / cluster_name
        destination_dir.mkdir(parents=True, exist_ok=True)

        destination = destination_dir / source.name
        if destination.exists() and destination.resolve() != source.resolve():
            try:
                destination.unlink()
            except OSError as exc:
                logger.warning("Unable to replace existing destination %s: %s", destination, exc)
                continue

        if source.resolve() != destination.resolve():
            shutil.copy2(str(source), str(destination))

        existing_cluster_file = record.get("cluster_file_path")
        if isinstance(existing_cluster_file, str) and existing_cluster_file:
            previous_path = Path(existing_cluster_file)
            if previous_path.exists() and previous_path.resolve() not in {source.resolve(), destination.resolve()}:
                try:
                    previous_path.unlink()
                except OSError as exc:
                    logger.warning("Unable to remove stale cluster file %s: %s", previous_path, exc)

        record["cluster_name"] = cluster_name
        record["cluster_file_path"] = str(destination)
        counts[cluster_name] += 1
    return dict(counts)

