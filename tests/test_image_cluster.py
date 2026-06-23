import tempfile
import unittest
from pathlib import Path

import image_cluster


class ImageClusterUtilityTests(unittest.TestCase):
    def test_sanitize_cluster_name(self) -> None:
        value = image_cluster.sanitize_cluster_name("Sunny Beach & Ocean Views!!!")
        self.assertEqual(value, "sunny-beach-ocean-vi")

    def test_generate_cluster_names_are_unique(self) -> None:
        names = image_cluster.generate_cluster_names(
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
            new_path = image_cluster.unique_destination(root, "photo.jpg")
            self.assertEqual(new_path.name, "photo_2.jpg")


if __name__ == "__main__":
    unittest.main()

