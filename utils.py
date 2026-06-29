from __future__ import annotations

import datetime as dt
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, cast

logger = logging.getLogger(__name__)


def confirm_directory_deletion(path: Path, reason: str, *, skip_prompt: bool = False) -> bool:
    """Prompt for confirmation before deleting an existing directory."""
    if skip_prompt:
        return True

    prompt = (
        "\nPath deletion requested:\n"
        f"- Path: {path}\n"
        f"- Reason: {reason}\n"
        "Proceed with deletion? [y/N]: "
    )
    response = input(prompt).strip().lower()
    return response in {"y", "yes"}


def _rmtree_with_retry(path: Path, max_retries: int = 3) -> None:
    """Remove a directory tree with retries for Windows file-locking issues."""
    for attempt in range(max_retries):
        try:
            # On Windows, recursively remove read-only files before deletion
            def handle_remove_readonly(func, path, exc):
                if os.name == 'nt' and exc[0] == PermissionError:
                    os.chmod(path, 0o777)
                    func(path)
                else:
                    raise
            
            shutil.rmtree(str(path), onerror=handle_remove_readonly)
            return
        except OSError as e:
            if attempt < max_retries - 1:
                logger.warning(f"Attempt {attempt + 1} to delete {path} failed: {e}. Retrying...")
                time.sleep(0.5)  # Brief delay to allow Windows file handles to close
            else:
                raise SystemExit(
                    f"Failed to delete directory after {max_retries} attempts: {path}\n"
                    f"Error: {e}\n"
                    "Check that files are not locked by another process."
                )


def discover_images(
    input_dir: Path,
    extensions: Iterable[str],
    *,
    excluded_dir_names: Iterable[str] | None = None,
) -> List[Path]:
    """Recursively collect image files while skipping configured directory names."""
    allowed = {ext.lower().strip() for ext in extensions if ext.strip()}
    excluded = {
        name.lower().strip()
        for name in (excluded_dir_names or ())
        if name and name.strip()
    }

    images: List[Path] = []
    for root, dirnames, filenames in os.walk(input_dir):
        # Prune ignored folders in-place so os.walk does not recurse into them.
        dirnames[:] = [name for name in dirnames if name.lower() not in excluded]
        root_path = Path(root)
        for filename in filenames:
            path = root_path / filename
            if path.suffix.lower() in allowed:
                images.append(path)
    return sorted(images)


def read_image_metadata(image_path: Path) -> Dict[str, object]:
    """Read filesystem and image properties for a single image file."""
    from PIL import Image

    stat = image_path.stat()
    with Image.open(image_path) as img:
        date_taken: str | None = None
        exif = img.getexif()
        if exif:
            # Prefer EXIF capture timestamps for chronological ordering.
            raw_date_taken = exif.get(36867) or exif.get(36868) or exif.get(306)
            if isinstance(raw_date_taken, bytes):
                raw_date_taken = raw_date_taken.decode("utf-8", errors="ignore")
            if isinstance(raw_date_taken, str) and raw_date_taken.strip():
                date_taken = raw_date_taken.strip()

        width, height = img.size
        metadata = {
            "filename": image_path.name,
            "suffix": image_path.suffix.lower(),
            "file_size_bytes": stat.st_size,
            "modified_utc": dt.datetime.fromtimestamp(stat.st_mtime, dt.timezone.utc).isoformat(),
            "date_taken": date_taken,
            "width": width,
            "height": height,
            "mode": img.mode,
            "format": img.format,
        }
    return metadata


def remove_path_if_exists(
    path: Path,
    *,
    deletion_reason: str = "Clearing previous output before writing new results.",
    skip_prompt: bool = False,
) -> None:
    """Remove a file or directory if it already exists."""
    if not path.exists():
       return
    if not confirm_directory_deletion(path, deletion_reason, skip_prompt=skip_prompt):
        raise SystemExit(f"Path deletion cancelled by user: {path}")
    if path.is_dir():
        _rmtree_with_retry(path)
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
    seen_source_paths: set[str] = set()

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
        seen_source_paths.add(str(image_path.resolve()))

    for cached_image_path, cached_record in existing_cache.items():
        if cached_image_path in seen_image_paths:
            continue

        source_path = _resolve_record_source_path(cached_record)
        if source_path is None:
            continue

        resolved_source_path = str(source_path.resolve())
        if resolved_source_path in seen_source_paths:
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
        seen_source_paths.add(resolved_source_path)

    return records


def replace_directory(staging_dir: Path, target_dir: Path, *, skip_prompt: bool = False) -> None:
    """Replace a target directory with a fully prepared staging directory."""
    if target_dir.exists():
        if not confirm_directory_deletion(
            target_dir,
            "Replacing existing cluster output with freshly generated clusters.",
            skip_prompt=skip_prompt,
        ):
            raise SystemExit(f"Directory replacement cancelled by user: {target_dir}")
        _rmtree_with_retry(target_dir)
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
    action_past_tense = "copied"
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

