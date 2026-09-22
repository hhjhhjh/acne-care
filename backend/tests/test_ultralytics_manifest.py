import tempfile
import unittest
from pathlib import Path

from acne_care.dataset.ultralytics_manifest import build_ultralytics_manifest


class UltralyticsManifestTest(unittest.TestCase):
    def test_builds_clean_split_manifests_and_dataset_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            output = root / "output"
            self._write_sample(source, "train", "train-a")
            self._write_sample(source, "valid", "valid-a")
            self._write_sample(source, "test", "test-a")
            self._write_sample(source, "test", "test-b")

            result = build_ultralytics_manifest(
                source_root=source,
                output_dir=output,
                excluded_paths={"test/images/test-b.jpg"},
                class_names=("Acne",),
            )

            self.assertEqual(result.image_counts, {"train": 1, "val": 1, "test": 1})
            self.assertEqual(
                (output / "test.txt").read_text(encoding="utf-8").splitlines(),
                [(source / "test" / "images" / "test-a.jpg").resolve().as_posix()],
            )
            yaml_text = result.dataset_yaml.read_text(encoding="utf-8")
            self.assertIn(
                f'train: "{(output / "train.txt").resolve().as_posix()}"',
                yaml_text,
            )
            self.assertIn('  0: "Acne"', yaml_text)

    def test_rejects_unknown_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            self._write_sample(source, "train", "train-a")
            self._write_sample(source, "valid", "valid-a")
            self._write_sample(source, "test", "test-a")

            with self.assertRaisesRegex(ValueError, "unknown exclusion"):
                build_ultralytics_manifest(
                    source_root=source,
                    output_dir=root / "output",
                    excluded_paths={"test/images/missing.jpg"},
                    class_names=("Acne",),
                )

    def test_repeated_build_produces_the_same_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            output = root / "output"
            for split in ("train", "valid", "test"):
                self._write_sample(source, split, f"{split}-a")

            build_ultralytics_manifest(source, output, set(), ("Acne",))
            first_contents = {
                path.name: path.read_text(encoding="utf-8")
                for path in output.iterdir()
            }
            build_ultralytics_manifest(source, output, set(), ("Acne",))
            second_contents = {
                path.name: path.read_text(encoding="utf-8")
                for path in output.iterdir()
            }

            self.assertEqual(second_contents, first_contents)

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


if __name__ == "__main__":
    unittest.main()
