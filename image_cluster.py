#!/usr/bin/env python3
"""Cluster images by caption embeddings and organize files by cluster."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import shutil
import logging
from tqdm import tqdm
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, cast

logger = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


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
        default="microsoft/Florence-2-base",
        help="Hugging Face image captioning model.",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
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
        default=0.5,
        help="Hierarchical clustering threshold (used when --n-clusters is not set).",
    )
    parser.add_argument(
        "--file-action",
        choices=["copy", "move"],
        default="copy",
        help="Whether clustered files are copied or moved.",
    )
    parser.add_argument(
        "--extensions",
        default=",".join(sorted(SUPPORTED_EXTENSIONS)),
        help="Comma-separated allowed extensions (example: .jpg,.png,.webp).",
    )
    parser.add_argument(
        "--use-mock-models",
        action="store_true",
        help="Use deterministic mock caption/embedding generators (fast/offline smoke testing).",
    )
    return parser.parse_args()


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


def caption_images_real(image_paths: Sequence[Path], model_name: str) -> List[str]:
    """Generate captions for images using the smaller Florence-2 captioning model."""
    import random

    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    captions: List[str] = []
    from PIL import Image, UnidentifiedImageError
    import torch
    from tqdm import tqdm

    captions = []

    for image_path in tqdm(image_paths, desc="Generating image captions"):

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")

                inputs = processor(
                    text="<MORE_DETAILED_CAPTION>",
                    images=image,
                    return_tensors="pt",
                )

            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
            }

            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=256,
                    do_sample=False,
                )

            output_text = processor.batch_decode(
                generated_ids,
                skip_special_tokens=False,
            )

            caption = (
                output_text[0].strip()
                if output_text and output_text[0].strip()
                else f"Image file named {image_path.stem}"
            )

            captions.append(caption)

        except UnidentifiedImageError:
            print(f"WARNING: Cannot read image: {image_path}")
            captions.append(f"Corrupted image {image_path.stem}")

        except FileNotFoundError:
            print(f"WARNING: File not found: {image_path}")
            captions.append(f"Missing image {image_path.stem}")

        except Exception as ex:
            print(f"WARNING: Failed processing {image_path}")
            print(f"Reason: {ex}")
            captions.append(f"Failed image {image_path.stem}")

    return captions


def caption_images_mock(image_paths: Sequence[Path]) -> List[str]:
    """Generate deterministic placeholder captions for offline or smoke-test runs."""
    captions: List[str] = []
    for image_path in tqdm(image_paths, desc="Generating mock captions"):
        stem = image_path.stem.replace("_", " ").replace("-", " ")
        captions.append(f"Mock caption describing {stem}".strip())
    return captions


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


def unique_destination(destination_dir: Path, filename: str) -> Path:
    """Return a non-conflicting output path for a file inside a destination directory."""
    candidate = destination_dir / filename
    if not candidate.exists():
        return candidate
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    idx = 2
    while True:
        candidate = destination_dir / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def organize_cluster_files(
    records: List[Dict[str, object]],
    cluster_root: Path,
    action: str,
    cluster_names: Dict[int, str],
) -> Dict[str, int]:
    """Copy or move images into cluster folders and record their new locations."""
    counts: Dict[str, int] = Counter()
    cluster_root.mkdir(parents=True, exist_ok=True)

    for record in records:
        label = int(cast(int, record["cluster_label"]))
        source = Path(str(record["image_path"]))
        cluster_name = cluster_names[label]
        destination_dir = cluster_root / cluster_name
        destination_dir.mkdir(parents=True, exist_ok=True)

        destination = unique_destination(destination_dir, source.name)
        if action == "move":
            shutil.move(str(source), str(destination))
        else:
            shutil.copy2(str(source), str(destination))

        record["cluster_name"] = cluster_name
        record["cluster_file_path"] = str(destination)
        counts[cluster_name] += 1
    return dict(counts)


def write_json_output(
    json_path: Path,
    records: Sequence[Dict[str, object]],
    config: Dict[str, object],
    cluster_counts: Dict[str, int],
) -> None:
    """Write pipeline configuration, summary data, and image records to JSON."""
    payload = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": config,
        "summary": {
            "total_images": len(records),
            "total_clusters": len(cluster_counts),
            "cluster_counts": cluster_counts,
        },
        "images": records,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_text_report(
    report_path: Path,
    *,
    input_dir: Path,
    action: str,
    method: str,
    total_images: int,
    cluster_counts: Dict[str, int],
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
        f"Total clusters generated: {len(cluster_counts)}",
        "",
        "Cluster distribution:",
    ]
    if cluster_counts:
        for cluster_name, count in sorted(cluster_counts.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- {cluster_name}: {count} file(s) {action}d")
    else:
        lines.append("- No clusters generated")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the end-to-end image captioning, embedding, clustering, and export workflow."""
    args = parse_args()
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
        raise SystemExit("No images found with the provided extensions.")

    print(f"Discovered {len(image_paths)} image(s).")
    records: List[Dict[str, object]] = []
    for image_path in tqdm(image_paths, desc="Reading image metadata"):
        try:
            records.append(
                {
                    "image_path": str(image_path),
                    "metadata": read_image_metadata(image_path),
                }
            )
        except (OSError, ValueError) as e:
            logger.warning(f"Failed to read image metadata for {image_path}: {type(e).__name__}: {e}")


    if args.use_mock_models:
        captions = caption_images_mock(image_paths)
        embeddings = generate_embeddings_mock(captions)
    else:
        captions = caption_images_real(image_paths, args.caption_model)
        embeddings = generate_embeddings_real(captions, args.embedding_model)

    labels = cluster_embeddings(
        embeddings=embeddings,
        method=args.cluster_method,
        n_clusters=args.n_clusters,
        distance_threshold=args.distance_threshold,
    )

    label_to_captions: Dict[int, List[str]] = defaultdict(list)
    for idx, record in enumerate(records):
        record["caption"] = captions[idx]
        record["embedding"] = embeddings[idx]
        record["cluster_label"] = labels[idx]
        label_to_captions[labels[idx]].append(captions[idx])

    cluster_names = generate_cluster_names(label_to_captions)
    cluster_counts = organize_cluster_files(
        records=records,
        cluster_root=cluster_root,
        action=args.file_action,
        cluster_names=cluster_names,
    )

    config = {
        "caption_model": args.caption_model,
        "embedding_model": args.embedding_model,
        "cluster_method": args.cluster_method,
        "n_clusters": args.n_clusters,
        "distance_threshold": args.distance_threshold,
        "file_action": args.file_action,
        "use_mock_models": args.use_mock_models,
    }
    write_json_output(json_path, records, config, cluster_counts)
    write_text_report(
        report_path,
        input_dir=input_dir,
        action=args.file_action,
        method=args.cluster_method,
        total_images=len(records),
        cluster_counts=cluster_counts,
    )

    print(f"JSON output:   {json_path}")
    print(f"Text report:   {report_path}")
    print(f"Cluster root:  {cluster_root}")


if __name__ == "__main__":
    main()

