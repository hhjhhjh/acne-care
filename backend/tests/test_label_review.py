import tempfile
import unittest
from pathlib import Path

from acne_care.dataset.label_review import build_label_review


class LabelReviewTest(unittest.TestCase):
    def test_collects_annotations_and_review_hints(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for split in ("train", "valid", "test"):
                image_dir = root / split / "images"
                label_dir = root / split / "labels"
                image_dir.mkdir(parents=True)
                label_dir.mkdir(parents=True)
                (image_dir / f"{split}.jpg").write_bytes(split.encode())
                label = "0 0.5 0.5 0.2 0.2\n"
                if split == "valid":
                    label += "0 0.5 0.5 0.2 0.2\n"
                (label_dir / f"{split}.txt").write_text(label, encoding="utf-8")

            result = build_label_review(root)

            self.assertEqual(result["summary"]["images"], 3)
            self.assertEqual(result["summary"]["annotations"], 4)
            valid_item = next(
                item for item in result["items"] if item["split"] == "valid"
            )
            self.assertEqual(valid_item["annotation_count"], 2)
            self.assertIn("duplicate_box", valid_item["review_hints"])

    def test_flags_empty_labels_and_unexpected_classes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for split, label in (
                ("train", ""),
                ("valid", "1 0.5 0.5 0.2 0.2\n"),
                ("test", "0 0.5 0.5 0.2 0.2\n"),
            ):
                image_dir = root / split / "images"
                label_dir = root / split / "labels"
                image_dir.mkdir(parents=True)
                label_dir.mkdir(parents=True)
                (image_dir / f"{split}.jpg").write_bytes(split.encode())
                (label_dir / f"{split}.txt").write_text(label, encoding="utf-8")

            result = build_label_review(root, expected_class_ids=(0,))

            hints = {
                item["split"]: item["review_hints"] for item in result["items"]
            }
            self.assertIn("empty_label", hints["train"])
            self.assertIn("unexpected_class", hints["valid"])


if __name__ == "__main__":
    unittest.main()
