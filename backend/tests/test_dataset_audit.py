import tempfile
import unittest
from pathlib import Path

from acne_care.dataset.audit import audit_yolo_dataset


class YoloDatasetAuditTest(unittest.TestCase):
    def test_reports_valid_dataset_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_sample(root, "train", "face-a", b"train-image")
            self._write_sample(root, "valid", "face-b", b"valid-image")
            self._write_sample(root, "test", "face-c", b"test-image")

            report = audit_yolo_dataset(root)

            self.assertTrue(report.is_valid)
            self.assertEqual(report.image_counts, {"train": 1, "valid": 1, "test": 1})
            self.assertEqual(report.label_counts, {"train": 1, "valid": 1, "test": 1})
            self.assertEqual(report.class_ids, (0,))

    def test_reports_missing_label(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "train" / "images"
            image_dir.mkdir(parents=True)
            (image_dir / "face-a.jpg").write_bytes(b"image")

            report = audit_yolo_dataset(root)

            self.assertFalse(report.is_valid)
            self.assertIn("train/images/face-a.jpg has no matching label", report.errors)

    def test_reports_invalid_normalized_box(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_sample(
                root,
                "train",
                "face-a",
                b"image",
                label="0 1.2 0.5 0.1 0.1\n",
            )

            report = audit_yolo_dataset(root)

            self.assertFalse(report.is_valid)
            self.assertTrue(any("normalized coordinates" in error for error in report.errors))

    def test_reports_exact_duplicate_across_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_sample(root, "train", "face-a", b"same-image")
            self._write_sample(root, "test", "face-b", b"same-image")

            report = audit_yolo_dataset(root)

            self.assertFalse(report.is_valid)
            self.assertEqual(
                report.cross_split_duplicates,
                (("test/images/face-b.jpg", "train/images/face-a.jpg"),),
            )

    @staticmethod
    def _write_sample(
        root: Path,
        split: str,
        stem: str,
        image_content: bytes,
        label: str = "0 0.5 0.5 0.2 0.2\n",
    ) -> None:
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        (image_dir / f"{stem}.jpg").write_bytes(image_content)
        (label_dir / f"{stem}.txt").write_text(label, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
