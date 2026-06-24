import tempfile
import unittest
import json
import argparse
from pathlib import Path

import clustering
import image_cluster
import utils


class ImageClusterUtilityTests(unittest.TestCase):
    def test_sanitize_cluster_name(self) -> None:
        value = clustering.sanitize_cluster_name("Sunny Beach & Ocean Views!!!")
        self.assertEqual(value, "sunny-beach-ocean-vi")

    def test_generate_cluster_names_are_unique(self) -> None:
        names = clustering.generate_cluster_names(
            {
                0: ["cat on grass", "cat playing"],
                1: ["cat on grass", "cat sleeping"],
            }
        )
        self.assertEqual(len(set(names.values())), 2)

    def test_unique_destination(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "photo.jpg"
            first.write_text("x", encoding="utf-8")
            new_path = clustering.unique_destination(root, "photo.jpg")
            self.assertEqual(new_path.name, "photo_2.jpg")

    def test_coerce_embedding(self) -> None:
        self.assertEqual(utils._coerce_embedding([1, 2.5, 3]), [1.0, 2.5, 3.0])
        self.assertIsNone(utils._coerce_embedding([]))
        self.assertIsNone(utils._coerce_embedding([1, "x"]))

    def test_load_existing_image_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            json_path = Path(tmp) / "image_data.json"
            payload = {
                "images": [
                    {"image_path": "C:/img/a.jpg", "caption": "a", "embedding": [0.1, 0.2]},
                    {"image_path": "C:/img/b.jpg", "caption": "b", "embedding": [0.3, 0.4]},
                ]
            }
            json_path.write_text(json.dumps(payload), encoding="utf-8")

            cache = utils.load_existing_image_cache(json_path)
            self.assertIn("C:/img/a.jpg", cache)
            self.assertIn("C:/img/b.jpg", cache)

    def test_write_json_output_includes_cluster_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            json_path = Path(tmp) / "image_data.json"
            utils.write_json_output(
                json_path,
                records=[{"image_path": "C:/img/a.jpg", "cluster_label": 1}],
                config={"file_action": "copy"},
                cluster_summaries=[{"label": 1, "name": "new-name", "count": 1}],
            )

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["total_clusters"], 1)
            self.assertEqual(payload["summary"]["cluster_counts"], {"new-name": 1})
            self.assertEqual(payload["summary"]["clusters"], [{"label": 1, "name": "new-name", "count": 1}])

    def test_write_text_report_includes_cluster_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "report.txt"
            utils.write_text_report(
                report_path,
                input_dir=Path(tmp),
                action="copy",
                method="hierarchical",
                total_images=1,
                cluster_summaries=[{"label": 2, "name": "sunny-beach", "count": 1}],
            )

            report = report_path.read_text(encoding="utf-8")
            self.assertIn("Cluster 2 (sunny-beach): 1 file(s) copied", report)

    def test_organize_cluster_files_resyncs_to_new_cluster(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            old_cluster_dir = root / "clusters" / "old-name"
            input_dir.mkdir(parents=True)
            old_cluster_dir.mkdir(parents=True)

            source = input_dir / "photo.jpg"
            source.write_text("source-content", encoding="utf-8")
            stale_copy = old_cluster_dir / "photo.jpg"
            stale_copy.write_text("stale-content", encoding="utf-8")

            records = [
                {
                    "image_path": str(source),
                    "cluster_label": 1,
                    "cluster_file_path": str(stale_copy),
                }
            ]

            counts = clustering.organize_cluster_files(
                records=records,
                cluster_root=root / "clusters",
                action="copy",
                cluster_names={1: "new-name"},
            )

            new_path = root / "clusters" / "new-name" / "photo.jpg"
            self.assertTrue(new_path.exists())
            self.assertEqual(new_path.read_text(encoding="utf-8"), "source-content")
            self.assertFalse(stale_copy.exists())
            self.assertEqual(records[0]["cluster_name"], "new-name")
            self.assertEqual(records[0]["cluster_file_path"], str(new_path))
            self.assertEqual(counts, {"new-name": 1})

    def test_main_removes_previous_outputs_before_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir(parents=True)
            output_dir.mkdir(parents=True)

            image_path = input_dir / "photo.jpg"
            image_path.write_text("source-content", encoding="utf-8")

            old_json = output_dir / "image_data.json"
            old_json.write_text(json.dumps({"images": [{"image_path": str(image_path), "embedding": [1.0]}]}), encoding="utf-8")

            old_report = output_dir / "report.txt"
            old_report.write_text("old report", encoding="utf-8")

            old_cluster_dir = output_dir / "clusters" / "legacy"
            old_cluster_dir.mkdir(parents=True)
            (old_cluster_dir / "photo.jpg").write_text("legacy-content", encoding="utf-8")

            original_parse_args = image_cluster.parse_args
            image_cluster.parse_args = lambda: argparse.Namespace(
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                json_path=None,
                report_path=None,
                caption_model="test-caption-model",
                embedding_model="test-embedding-model",
                cluster_method="hierarchical",
                n_clusters=None,
                distance_threshold=0.65,
                file_action="copy",
                extensions=".jpg",
            )
            try:
                image_cluster.main()
            finally:
                image_cluster.parse_args = original_parse_args

            self.assertTrue(old_json.exists())
            self.assertTrue(old_report.exists())
            self.assertFalse((output_dir / "clusters" / "legacy").exists())
            self.assertTrue((output_dir / "clusters").exists())

            payload = json.loads(old_json.read_text(encoding="utf-8"))
            self.assertIn("clusters", payload["summary"])
            self.assertEqual(payload["summary"]["total_images"], 1)
            self.assertEqual(payload["summary"]["total_clusters"], len(payload["summary"]["clusters"]))
            self.assertIn("Cluster ", old_report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

