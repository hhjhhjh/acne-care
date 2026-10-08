import tempfile
import unittest
from pathlib import Path

from acne_care.dataset.label_edits import build_dataset_v2
from acne_care.dataset.label_review_dashboard import review_dataset_fingerprint


class LabelEditsTest(unittest.TestCase):
    def test_builds_copy_with_edited_labels_and_excluded_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            output = root / "dataset-v2"
            for split in ("train", "valid", "test"):
                self._write_sample(source, split, f"{split}-keep")
            self._write_sample(source, "valid", "valid-exclude")
            self._write_sample(source, "test", "test-known-duplicate")
            audit = self._audit_document(source)
            fingerprint = review_dataset_fingerprint(audit)
            edits = {
                "schema_version": 1,
                "dataset_fingerprint": fingerprint,
                "edits": {
                    "valid/images/valid-keep.jpg": {
                        "annotations": [
                            {
                                "class_id": 0,
                                "x_center": 0.12345678901234566,
                                "y_center": 0.3,
                                "width": 0.1,
                                "height": 0.2,
                                "source": "added",
                            }
                        ],
                        "excluded": False,
                        "completed": True,
                    },
                    "valid/images/valid-exclude.jpg": {
                        "annotations": [],
                        "excluded": True,
                        "completed": True,
                    },
                },
            }

            result = build_dataset_v2(
                source,
                output,
                audit,
                edits,
                preexisting_exclusions={
                    "test/images/test-known-duplicate.jpg"
                },
            )

            self.assertEqual(
                result.image_counts,
                {"train": 1, "valid": 1, "test": 1},
            )
            self.assertEqual(result.edited_images, 1)
            self.assertEqual(result.excluded_images, 2)
            self.assertEqual(
                (output / "valid" / "labels" / "valid-keep.txt").read_text(
                    encoding="utf-8"
                ),
                "0 0.12345678901234566 0.3 0.1 0.2\n",
            )
            self.assertFalse(
                (output / "valid" / "images" / "valid-exclude.jpg").exists()
            )
            self.assertFalse(
                (output / "valid" / "labels" / "valid-exclude.txt").exists()
            )
            self.assertFalse(
                (output / "test" / "images" / "test-known-duplicate.jpg").exists()
            )
            self.assertEqual(
                (source / "valid" / "labels" / "valid-keep.txt").read_text(
                    encoding="utf-8"
                ),
                "0 0.5 0.5 0.2 0.2\n",
            )
            yaml_text = (output / "data.yaml").read_text(encoding="utf-8")
            self.assertIn(f'path: "{output.resolve().as_posix()}"', yaml_text)
            self.assertIn("val: valid/images", yaml_text)

    def test_rejects_incomplete_edits_before_writing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            for split in ("train", "valid", "test"):
                self._write_sample(source, split, f"{split}-keep")
            audit = self._audit_document(source)
            edits = {
                "schema_version": 1,
                "dataset_fingerprint": review_dataset_fingerprint(audit),
                "edits": {
                    "valid/images/valid-keep.jpg": {
                        "annotations": [],
                        "excluded": False,
                        "completed": False,
                    }
                },
            }
            output = root / "dataset-v2"

            with self.assertRaisesRegex(ValueError, "not completed"):
                build_dataset_v2(source, output, audit, edits)

            self.assertFalse(output.exists())

    def test_repeated_build_removes_stale_generated_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            for split in ("train", "valid", "test"):
                self._write_sample(source, split, f"{split}-keep")
            audit = self._audit_document(source)
            edits = {
                "schema_version": 1,
                "dataset_fingerprint": review_dataset_fingerprint(audit),
                "edits": {},
            }
            output = root / "dataset-v2"

            build_dataset_v2(source, output, audit, edits)
            (output / "stale.txt").write_text("stale", encoding="utf-8")
            build_dataset_v2(source, output, audit, edits)

            self.assertFalse((output / "stale.txt").exists())

    @staticmethod
    def _write_sample(root: Path, split: str, stem: str) -> None:
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        (image_dir / f"{stem}.jpg").write_bytes(stem.encode("utf-8"))
        (label_dir / f"{stem}.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n",
            encoding="utf-8",
        )

    @staticmethod
    def _audit_document(source: Path) -> dict[str, object]:
        items = []
        for split in ("train", "valid", "test"):
            for image_path in sorted((source / split / "images").glob("*.jpg")):
                items.append(
                    {
                        "split": split,
                        "image_path": image_path.relative_to(source).as_posix(),
                        "label_path": (
                            Path(split) / "labels" / f"{image_path.stem}.txt"
                        ).as_posix(),
                        "annotation_count": 1,
                        "annotations": [],
                        "review_hints": [],
                        "details": [],
                    }
                )
        return {"schema_version": 1, "dataset_root": str(source), "items": items}


if __name__ == "__main__":
    unittest.main()
