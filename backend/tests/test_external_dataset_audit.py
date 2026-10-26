import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from acne_care.dataset.external_candidate import (
    audit_roboflow_csv_dataset,
    build_acne_pimple_review_sample,
    find_reference_image_matches,
)


class ExternalDatasetAuditTest(unittest.TestCase):
    def test_reports_annotation_counts_and_cross_split_source_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_image(root / "train" / "face-a_jpg.rf.11111111111111111111111111111111.jpg")
            self._write_image(root / "valid" / "face-a_jpg.rf.22222222222222222222222222222222.jpg")
            self._write_annotations(
                root / "train" / "_annotations.csv",
                [
                    ("face-a_jpg.rf.11111111111111111111111111111111.jpg", "acne", 1, 2, 6, 8),
                    ("face-a_jpg.rf.11111111111111111111111111111111.jpg", "pimple", 3, 4, 7, 9),
                ],
            )
            self._write_annotations(
                root / "valid" / "_annotations.csv",
                [("face-a_jpg.rf.22222222222222222222222222222222.jpg", "spot", 2, 2, 5, 5)],
            )

            report = audit_roboflow_csv_dataset(root)

            self.assertTrue(report.is_valid)
            self.assertEqual(report.image_counts, {"train": 1, "valid": 1, "test": 0})
            self.assertEqual(report.annotation_counts, {"train": 2, "valid": 1, "test": 0})
            self.assertEqual(report.class_counts, {"acne": 1, "pimple": 1, "spot": 1})
            self.assertEqual(
                report.cross_split_source_collisions,
                (("face-a_jpg", ("train", "valid")),),
            )

    def test_reports_missing_images_and_invalid_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "train").mkdir(parents=True)
            self._write_annotations(
                root / "train" / "_annotations.csv",
                [("missing.jpg", "acne", 7, 2, 6, 8)],
            )

            report = audit_roboflow_csv_dataset(root)

            self.assertFalse(report.is_valid)
            self.assertIn("train/missing.jpg is referenced but missing", report.errors)
            self.assertTrue(any("xmin must be less than xmax" in error for error in report.errors))

    def test_finds_reencoded_reference_image_as_near_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            candidate = root / "candidate" / "train" / "candidate.jpg"
            reference = root / "reference" / "train" / "images" / "reference.jpg"
            self._write_gradient(candidate, size=(32, 32), quality=95)
            self._write_gradient(reference, size=(64, 64), quality=70)

            report = find_reference_image_matches(
                candidate_root=root / "candidate",
                reference_root=root / "reference",
                max_hamming_distance=4,
            )

            self.assertEqual(report.exact_matches, ())
            self.assertEqual(len(report.near_matches), 1)
            self.assertEqual(report.near_matches[0].candidate_path, "train/candidate.jpg")
            self.assertEqual(
                report.near_matches[0].reference_path,
                "train/images/reference.jpg",
            )

    def test_builds_deterministic_single_class_review_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            output = root / "output"
            rows = [
                ("pimple-a.jpg", "pimple", 1, 1, 5, 5),
                ("pimple-a.jpg", "spot", 5, 5, 8, 8),
                ("pimple-b.jpg", "pimple", 2, 2, 7, 7),
                ("acne-a.jpg", "acne", 1, 2, 6, 8),
                ("acne-b.jpg", "acne", 3, 3, 9, 9),
                ("spot-only.jpg", "spot", 1, 1, 4, 4),
            ]
            for filename in {row[0] for row in rows}:
                self._write_image(source / "train" / filename)
            self._write_annotations(source / "train" / "_annotations.csv", rows)

            result = build_acne_pimple_review_sample(
                source_root=source,
                output_root=output,
                sample_size=4,
                seed=42,
            )

            self.assertEqual(result.selected_images, 4)
            self.assertEqual(result.pimple_group_images, 2)
            self.assertEqual(result.acne_only_group_images, 2)
            labels = sorted((output / "valid" / "labels").glob("*.txt"))
            self.assertEqual(len(labels), 4)
            self.assertTrue(all(" 0." in path.read_text() for path in labels))
            self.assertTrue(all("spot" not in path.read_text() for path in labels))
            self.assertFalse(any("spot-only" in path.name for path in labels))

            stale = output / "valid" / "labels" / "stale.txt"
            stale.write_text("stale", encoding="utf-8")
            repeated = build_acne_pimple_review_sample(
                source_root=source,
                output_root=output,
                sample_size=4,
                seed=42,
            )
            self.assertEqual(repeated, result)
            self.assertFalse(stale.exists())

    @staticmethod
    def _write_image(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (10, 10), color=(120, 80, 60)).save(path)

    @staticmethod
    def _write_gradient(path: Path, size: tuple[int, int], quality: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", size)
        width, height = size
        pixels = image.load()
        for y in range(height):
            for x in range(width):
                pixels[x, y] = (
                    int(255 * x / max(width - 1, 1)),
                    int(255 * y / max(height - 1, 1)),
                    int(255 * (x + y) / max(width + height - 2, 1)),
                )
        image.save(path, quality=quality)

    @staticmethod
    def _write_annotations(
        path: Path,
        rows: list[tuple[str, str, int, int, int, int]],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(
                ["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"]
            )
            for filename, class_name, xmin, ymin, xmax, ymax in rows:
                writer.writerow([filename, 10, 10, class_name, xmin, ymin, xmax, ymax])


if __name__ == "__main__":
    unittest.main()
