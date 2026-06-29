# image_clustering

End-to-end image organization pipeline that:

1. Reads all images in a folder.
2. Generates text captions for each image.
3. Produces text embeddings from captions.
4. Clusters images by semantic similarity.
5. Copies images into cluster folders.
6. Exports a JSON dataset and plain text report.

## Features

- Image-to-text generation (`transformers` image captioning model).
- Embedding generation (`sentence-transformers`).
- Hierarchical clustering by default (or KMeans).
- Cluster folder naming based on top cluster terms (max 20 characters).
- JSON output includes image path, metadata, caption, embedding, and cluster info.
- Text report includes cluster counts and file copy summary.

## Project files

- `image_cluster.py` - CLI entrypoint (argument parsing + pipeline orchestration).
- `captions.py` - Caption generation logic.
- `embeddings.py` - Embedding generation logic.
- `clustering.py` - Clustering, cluster naming, cluster summaries, and cluster file organization.
- `utils.py` - Discovery, metadata/cache management, and JSON/report helpers.
- `requirements.txt` - Python dependencies.

## Install

```bash
python -m pip install -r requirements.txt
```

## Run (real models)

```bash
python image_cluster.py --input-dir "path/to/images" --output-dir "output"
```

### Optional arguments

- `--cluster-method hierarchical|kmeans` (default: `hierarchical`)
- `--n-clusters 8` (fixed cluster count)
- `--distance-threshold 0.8` (hierarchical split threshold)
- `--json-path output/image_data.json`
- `--report-path output/report.txt`
- `--prompt-deletion` / `--no-prompt-deletion` (boolean flag; default is `--prompt-deletion`)
- `--order-prefix none|date-taken` (default: `none`)

`--prompt-deletion` is a boolean parameter using paired flags:

- Use `--prompt-deletion` to require a confirmation prompt before deleting any directory.
- Use `--no-prompt-deletion` to skip prompts (useful for scheduled/non-interactive runs).

Examples:

```bash
# Prompt before any directory deletion (default behavior)
python image_cluster.py --input-dir "path/to/images" --output-dir "output" --prompt-deletion

# Skip deletion prompts
python image_cluster.py --input-dir "path/to/images" --output-dir "output" --no-prompt-deletion

# Order files in each cluster by date taken and prefix with rank (1_, 2_, 3_, ...)
python image_cluster.py --input-dir "path/to/images" --output-dir "output" --order-prefix date-taken
```


## Output

- `output/image_data.json`: full structured data with captions, metadata, embeddings, and clusters.
- `output/report.txt`: plain text report with total cluster count and files per cluster.
- `output/clusters/<cluster-name>/...`: copied image files.
