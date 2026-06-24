from __future__ import annotations

import datetime as dt
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, cast

logger = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


def discover_images(input_dir: Path, extensions: Iterable[str]) -> List[Path]:
    """Recursively collect image files under the input directory for allowed extensions."""
    allowed = {ext.lower().strip() for ext in extensions if ext.strip()}
    images: List[Path] = []
    for path in input_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in allowed:
            images.append(path)
    return sorted(images)


def read_image_metadata(image_path: Path) -> Dict[str, object]:
    """Read filesystem and image properties for a single image file."""
    from PIL import Image

    stat = image_path.stat()
    with Image.open(image_path) as img:
        width, height = img.size
        metadata = {
            "filename": image_path.name,
            "suffix": image_path.suffix.lower(),
            "file_size_bytes": stat.st_size,
            "modified_utc": dt.datetime.fromtimestamp(stat.st_mtime, dt.timezone.utc).isoformat(),
            "width": width,
            "height": height,
            "mode": img.mode,
            "format": img.format,
        }
    return metadata


def remove_path_if_exists(path: Path) -> None:
    """Remove a file or directory if it already exists."""
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _coerce_embedding(value: Any) -> List[float] | None:
    """Convert cached embedding data into a numeric vector when possible."""
    if not isinstance(value, list) or not value:
        return None
    vector: List[float] = []
    for item in value:
        if not isinstance(item, (int, float)):
            return None
        vector.append(float(item))
    return vector


def build_image_records(
    image_paths: Sequence[Path],
    existing_cache: Dict[str, Dict[str, object]],
) -> List[Dict[str, object]]:
    """Build a complete record list from current images plus any cached-but-missing records."""
    from clustering import _resolve_record_source_path

    records: List[Dict[str, object]] = []
    seen_image_paths: set[str] = set()

    for image_path in image_paths:
        record: Dict[str, object] = {
            "image_path": str(image_path),
            "metadata": read_image_metadata(image_path),
        }
        cached_record = existing_cache.get(str(image_path))
        if cached_record:
            if isinstance(cached_record.get("caption"), str):
                record["caption"] = cached_record["caption"]
            cached_embedding = _coerce_embedding(cached_record.get("embedding"))
            if cached_embedding is not None:
                record["embedding"] = cached_embedding
            cached_cluster_file = cached_record.get("cluster_file_path")
            if isinstance(cached_cluster_file, str) and cached_cluster_file:
                record["cluster_file_path"] = cached_cluster_file
        records.append(record)
        seen_image_paths.add(str(image_path))

    for cached_image_path, cached_record in existing_cache.items():
        if cached_image_path in seen_image_paths:
            continue

        source_path = _resolve_record_source_path(cached_record)
        if source_path is None:
            continue

        try:
            metadata = read_image_metadata(source_path)
        except (OSError, ValueError) as exc:
            logger.warning("Skipping cached image %s because metadata could not be read: %s", source_path, exc)
            continue

        record = {
            "image_path": cached_image_path,
            "metadata": metadata,
        }
        cached_caption = cached_record.get("caption")
        if isinstance(cached_caption, str):
            record["caption"] = cached_caption
        cached_embedding = _coerce_embedding(cached_record.get("embedding"))
        if cached_embedding is not None:
            record["embedding"] = cached_embedding
        cached_cluster_file = cached_record.get("cluster_file_path")
        if isinstance(cached_cluster_file, str) and cached_cluster_file:
            record["cluster_file_path"] = cached_cluster_file
        records.append(record)

    return records


def replace_directory(staging_dir: Path, target_dir: Path) -> None:
    """Replace a target directory with a fully prepared staging directory."""
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.move(str(staging_dir), str(target_dir))




def write_json_output(
    json_path: Path,
    records: Sequence[Dict[str, object]],
    config: Dict[str, object],
    cluster_summaries: Sequence[Dict[str, object]],
) -> None:
    """Write pipeline configuration, summary data, and image records to JSON."""
    cluster_counts = {
        str(cast(str, summary["name"])): int(cast(int, summary["count"]))
        for summary in cluster_summaries
    }
    clusters_payload = [
        {
            "label": int(cast(int, summary["label"])),
            "name": str(cast(str, summary["name"])),
            "count": int(cast(int, summary["count"])),
        }
        for summary in cluster_summaries
    ]
    payload = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": config,
        "summary": {
            "total_images": len(records),
            "total_clusters": len(cluster_summaries),
            "cluster_counts": cluster_counts,
            "clusters": clusters_payload,
        },
        "images": records,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_existing_image_cache(json_path: Path) -> Dict[str, Dict[str, object]]:
    """Load prior JSON output and index image entries by absolute image path."""
    if not json_path.exists():
        return {}
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Unable to read existing JSON cache from %s: %s", json_path, exc)
        return {}

    images = payload.get("images") if isinstance(payload, dict) else None
    if not isinstance(images, list):
        return {}

    cached: Dict[str, Dict[str, object]] = {}
    for item in images:
        if not isinstance(item, dict):
            continue
        image_path = item.get("image_path")
        if isinstance(image_path, str) and image_path:
            cached[image_path] = item
    return cached


def write_text_report(
    report_path: Path,
    *,
    input_dir: Path,
    action: str,
    method: str,
    total_images: int,
    cluster_summaries: Sequence[Dict[str, object]],
) -> None:
    """Write a plain-text report summarizing the clustering run and file distribution."""
    lines = [
        "Image Clustering Report",
        "=" * 24,
        f"Generated (UTC): {dt.datetime.now(dt.timezone.utc).isoformat()}",
        f"Input folder: {input_dir}",
        f"File action: {action}",
        f"Clustering method: {method}",
        "",
        "Method justification:",
        (
            "- Hierarchical clustering was chosen by default because it does not require a fixed number "
            "of clusters and can split groups based on semantic distance thresholds."
            if method == "hierarchical"
            else "- KMeans was chosen for faster execution when a fixed number of groups is preferred."
        ),
        "",
        f"Total images processed: {total_images}",
        f"Total clusters generated: {len(cluster_summaries)}",
        "",
        "Cluster distribution:",
    ]
    action_past_tense = "moved" if action == "move" else "copied"
    if cluster_summaries:
        for summary in sorted(
            cluster_summaries,
            key=lambda item: (-int(item["count"]), int(item["label"])),
        ):
            label = int(cast(int, summary["label"]))
            name = str(cast(str, summary["name"]))
            count = int(cast(int, summary["count"]))
            lines.append(f"- Cluster {label} ({name}): {count} file(s) {action_past_tense}")
    else:
        lines.append("- No clusters generated")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

