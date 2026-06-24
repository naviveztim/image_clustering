#!/usr/bin/env python3
"""CLI entrypoint for image clustering pipeline."""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from captions import caption_images_real
from clustering import (
    cluster_embeddings,
    generate_cluster_names,
    organize_cluster_files,
    summarize_clusters,
)
from embeddings import generate_embeddings_real
from utils import (
    SUPPORTED_EXTENSIONS,
    _coerce_embedding,
    build_image_records,
    discover_images,
    load_existing_image_cache,
    remove_path_if_exists,
    replace_directory,
    write_json_output,
    write_text_report,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default model paths – change these to point at local directories if needed
# ---------------------------------------------------------------------------
DEFAULT_CAPTION_MODEL = "microsoft/Florence-2-base"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def parse_args() -> argparse.Namespace:
    """Parse CLI options for image captioning, embedding, clustering, and export."""
    parser = argparse.ArgumentParser(
        description="Generate captions, embeddings, cluster images, and organize them by cluster."
    )
    parser.add_argument("--input-dir", required=True, help="Directory with source images.")
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory where JSON/report/cluster folders are written.",
    )
    parser.add_argument(
        "--json-path",
        default=None,
        help="Path for JSON output. Defaults to <output-dir>/image_data.json",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="Optional path for report output. Defaults to <output-dir>/report.txt",
    )
    parser.add_argument(
        "--caption-model",
        default=DEFAULT_CAPTION_MODEL,
        help="Hugging Face image captioning model (local path or HF repo id).",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="SentenceTransformer model name.",
    )
    parser.add_argument(
        "--cluster-method",
        choices=["hierarchical", "kmeans"],
        default="hierarchical",
        help="Clustering algorithm to use for embeddings.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="Fixed number of clusters. If omitted for hierarchical, distance-threshold is used.",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=0.65,
        help="Hierarchical clustering threshold (used when --n-clusters is not set).",
    )
    parser.add_argument(
        "--file-action",
        choices=["copy", "move"],
        default="copy",
        help="Deprecated. Accepted for backward compatibility; files are always copied.",
    )
    parser.add_argument(
        "--extensions",
        default=",".join(sorted(SUPPORTED_EXTENSIONS)),
        help="Comma-separated allowed extensions (example: .jpg,.png,.webp).",
    )
    parser.add_argument(
        "--prompt-deletion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prompt before deleting directories (use --no-prompt-deletion to skip prompts).",
    )
    return parser.parse_args()



def main() -> None:
    """Run the end-to-end image captioning, embedding, clustering, and export workflow."""
    args = parse_args()
    if getattr(args, "file_action", "copy") == "move":
        logger.warning("--file-action move is no longer supported; proceeding with copy behavior.")
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    json_path = Path(args.json_path).resolve() if args.json_path else output_dir / "image_data.json"
    report_path = Path(args.report_path).resolve() if args.report_path else output_dir / "report.txt"
    cluster_root = output_dir / "clusters"

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory not found or not a directory: {input_dir}")

    extensions = [item.strip() for item in args.extensions.split(",") if item.strip()]
    image_paths = discover_images(input_dir, extensions)
    if not image_paths:
        logger.info("No images found in %s; cached records will be used if available.", input_dir)

    print(f"Discovered {len(image_paths)} image(s).")
    existing_cache = load_existing_image_cache(json_path)
    remove_path_if_exists(report_path, skip_prompt=not args.prompt_deletion)
    remove_path_if_exists(json_path, skip_prompt=not args.prompt_deletion)

    staging_cluster_root = output_dir / f".{cluster_root.name}__staging"
    remove_path_if_exists(
        staging_cluster_root,
        deletion_reason="Removing previous staging cluster directory from an earlier run.",
        skip_prompt=not args.prompt_deletion,
    )

    records = build_image_records(image_paths, existing_cache)

    if not records:
        raise SystemExit("No readable images were found after metadata extraction.")

    captions: List[str] = [""] * len(records)
    embeddings: List[List[float] | None] = [None] * len(records)
    uncached_indices: List[int] = []
    uncached_paths: List[Path] = []
    for idx, record in enumerate(records):
        image_path = Path(str(record["image_path"]))
        cached_record = existing_cache.get(str(image_path))
        if not cached_record:
            uncached_indices.append(idx)
            uncached_paths.append(image_path)
            continue

        cached_embedding = _coerce_embedding(cached_record.get("embedding"))
        if cached_embedding is None:
            uncached_indices.append(idx)
            uncached_paths.append(image_path)
            continue

        embeddings[idx] = cached_embedding
        cached_caption = cached_record.get("caption")
        if isinstance(cached_caption, str) and cached_caption.strip():
            captions[idx] = cached_caption
        else:
            captions[idx] = f"Image file named {image_path.stem}"

    if uncached_paths:
        new_captions = caption_images_real(uncached_paths, args.caption_model)
        new_embeddings = generate_embeddings_real(new_captions, args.embedding_model)

        for idx, caption, embedding in zip(uncached_indices, new_captions, new_embeddings):
            captions[idx] = caption
            embeddings[idx] = embedding

    final_embeddings: List[List[float]] = []
    for idx, embedding in enumerate(embeddings):
        if embedding is None:
            raise SystemExit(f"Missing embedding for image: {records[idx]['image_path']}")
        final_embeddings.append(embedding)

    labels = cluster_embeddings(
        embeddings=final_embeddings,
        method=args.cluster_method,
        n_clusters=args.n_clusters,
        distance_threshold=args.distance_threshold,
    )

    label_to_captions: Dict[int, List[str]] = defaultdict(list)
    for idx, record in enumerate(records):
        record["caption"] = captions[idx]
        record["embedding"] = final_embeddings[idx]
        record["cluster_label"] = labels[idx]
        label_to_captions[labels[idx]].append(captions[idx])

    cluster_names = generate_cluster_names(label_to_captions)
    cluster_summaries = summarize_clusters(records, cluster_names)
    _cluster_counts = organize_cluster_files(
        records=records,
        cluster_root=staging_cluster_root,
        cluster_names=cluster_names,
    )

    replace_directory(staging_cluster_root, cluster_root, skip_prompt=not args.prompt_deletion)

    config = {
        "caption_model": args.caption_model,
        "embedding_model": args.embedding_model,
        "cluster_method": args.cluster_method,
        "n_clusters": args.n_clusters,
        "distance_threshold": args.distance_threshold,
        "file_action": "copy",
    }
    write_json_output(json_path, records, config, cluster_summaries)
    write_text_report(
        report_path,
        input_dir=input_dir,
        action="copy",
        method=args.cluster_method,
        total_images=len(records),
        cluster_summaries=cluster_summaries,
    )

    print(f"JSON output:   {json_path}")
    print(f"Text report:   {report_path}")
    print(f"Cluster root:  {cluster_root}")


if __name__ == "__main__":
    main()

