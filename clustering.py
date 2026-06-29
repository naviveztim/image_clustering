from __future__ import annotations

import logging
import math
import re
import shutil
from datetime import datetime, timezone
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
    # Short-circuit tiny inputs so downstream clustering APIs are not called
    # with degenerate cases.
    if not embeddings:
        return []
    if len(embeddings) == 1:
        return [0]

    # Import heavy numerical dependencies only when clustering is requested.
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering, KMeans

    # Convert to a dense float array expected by scikit-learn estimators.
    array = np.array(embeddings, dtype=float)
    if method == "hierarchical":
        # Use cosine distance + average linkage to group semantically similar
        # caption embeddings.
        if n_clusters is not None:
            model = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric="cosine",
                linkage="average",
            )
        else:
            # If cluster count is unknown, stop merges based on distance.
            model = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                metric="cosine",
                linkage="average",
            )
    else:
        # Pick a conservative default cluster count for KMeans when not given.
        if n_clusters is None:
            n_clusters = max(2, min(8, int(math.sqrt(len(embeddings)))))
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    # Fit model and normalize labels to plain Python ints.
    labels = model.fit_predict(array)
    return [int(label) for label in labels]


def _top_terms_from_captions(captions: Sequence[str], top_k: int = 3) -> List[str]:
    """Extract representative caption terms to help build readable cluster names."""
    # Empty caption groups should produce empty term lists.
    if not captions:
        return []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        # Use TF-IDF over unigrams/bigrams to get descriptive cluster keywords.
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
        # Fall back to simple term frequency if TF-IDF fails for any reason.
        words = re.findall(r"[a-zA-Z]{3,}", " ".join(captions).lower())
        common = Counter(words).most_common(top_k)
        return [word for word, _ in common]


def sanitize_cluster_name(raw_name: str, max_len: int = 20) -> str:
    """Normalize a cluster name into a short filesystem-friendly folder name."""
    # Keep only safe characters and normalize whitespace to dash separators.
    value = re.sub(r"[^a-zA-Z0-9\-_ ]+", "", raw_name.lower()).strip()
    value = re.sub(r"\s+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    # Guarantee a non-empty folder name.
    if not value:
        value = "cluster"
    # Keep names short for readability and filesystem friendliness.
    return value[:max_len]


def generate_cluster_names(label_to_captions: Dict[int, List[str]]) -> Dict[int, str]:
    """Create unique human-readable names for each cluster label from its captions."""
    names: Dict[int, str] = {}
    used: set[str] = set()
    for label in sorted(label_to_captions):
        # Build a candidate name from top caption terms (or numeric fallback).
        terms = _top_terms_from_captions(label_to_captions[label])
        candidate: str = sanitize_cluster_name("-".join(terms) if terms else f"cluster-{label}")
        if not candidate:
            candidate = f"cluster-{label}"
        # Ensure uniqueness by appending an incrementing suffix when needed.
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
    # Count how many records belong to each cluster label.
    counts: Counter[int] = Counter()
    for record in records:
        label = record.get("cluster_label")
        if isinstance(label, int):
            counts[label] += 1

    # Emit deterministic summaries sorted by numeric label.
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
    # Prefer original image path; fall back to prior cluster file path for
    # cached/re-run scenarios.
    for key in ("image_path", "cluster_file_path"):
        value = record.get(key)
        if isinstance(value, str) and value:
            path = Path(value)
            if path.exists():
                return path
    return None


def _parse_datetime_for_sort(value: str) -> float | None:
    """Parse supported datetime strings and return a UTC timestamp for ordering."""
    try:
        # EXIF timestamps are typically formatted as YYYY:MM:DD HH:MM:SS.
        parsed = datetime.strptime(value, "%Y:%m:%d %H:%M:%S").replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except ValueError:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.timestamp()
        except ValueError:
            return None


def _record_sort_key(record: Dict[str, object], source: Path) -> tuple[float, str]:
    """Build a stable sort key using date-taken first, then modified/creation time."""
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        date_taken = metadata.get("date_taken")
        if isinstance(date_taken, str) and date_taken.strip():
            parsed = _parse_datetime_for_sort(date_taken.strip())
            if parsed is not None:
                return (parsed, source.name.lower())

        modified_utc = metadata.get("modified_utc")
        if isinstance(modified_utc, str) and modified_utc.strip():
            parsed = _parse_datetime_for_sort(modified_utc.strip())
            if parsed is not None:
                return (parsed, source.name.lower())

    try:
        stat = source.stat()
        return (float(stat.st_ctime), source.name.lower())
    except OSError:
        # Fall back to epoch when metadata cannot be read.
        return (0.0, source.name.lower())


def organize_cluster_files(
    records: List[Dict[str, object]],
    cluster_root: Path,
    cluster_names: Dict[int, str],
    order_by_date_with_prefix: bool = False,
) -> Dict[str, int]:
    """Copy images into the latest cluster folders and record their new locations."""
    # Track per-cluster file totals while syncing files.
    counts: Dict[str, int] = Counter()
    cluster_root.mkdir(parents=True, exist_ok=True)

    # Resolve valid sources first so each cluster can be copied in creation-date order.
    cluster_entries: Dict[str, List[tuple[Dict[str, object], Path, tuple[float, str]]]] = {}

    for record in records:
        label = int(cast(int, record["cluster_label"]))
        cluster_name = cluster_names[label]
        source = _resolve_record_source_path(record)
        if source is None:
            logger.warning("Skipping file sync for %s because no source file exists.", record.get("image_path"))
            record["cluster_name"] = cluster_name
            continue

        cluster_entries.setdefault(cluster_name, []).append((record, source, _record_sort_key(record, source)))

    for cluster_name, entries in cluster_entries.items():
        # Optionally enforce date ordering and rank prefixes within each cluster.
        ordered_entries = (
            sorted(entries, key=lambda item: item[2])
            if order_by_date_with_prefix
            else entries
        )
        destination_dir = cluster_root / cluster_name
        destination_dir.mkdir(parents=True, exist_ok=True)

        for rank, (record, source, _sort_key) in enumerate(ordered_entries, start=1):
            destination_name = f"{rank}_{source.name}" if order_by_date_with_prefix else source.name
            destination = destination_dir / destination_name
            # Replace stale destination files when they point to different sources.
            if destination.exists() and destination.resolve() != source.resolve():
                try:
                    destination.unlink()
                except OSError as exc:
                    logger.warning("Unable to replace existing destination %s: %s", destination, exc)
                    continue

            # Copy only when source and destination are not already the same file.
            if source.resolve() != destination.resolve():
                shutil.copy2(str(source), str(destination))

            # Remove stale previous cluster copies to avoid orphaned files.
            existing_cluster_file = record.get("cluster_file_path")
            if isinstance(existing_cluster_file, str) and existing_cluster_file:
                previous_path = Path(existing_cluster_file)
                if previous_path.exists() and previous_path.resolve() not in {source.resolve(), destination.resolve()}:
                    try:
                        previous_path.unlink()
                    except OSError as exc:
                        logger.warning("Unable to remove stale cluster file %s: %s", previous_path, exc)

            # Persist latest cluster metadata for downstream reporting/export.
            record["cluster_name"] = cluster_name
            record["cluster_file_path"] = str(destination)
            counts[cluster_name] += 1
    return dict(counts)

