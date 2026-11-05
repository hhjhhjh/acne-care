import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from acne_care.dataset.external_label_editor import prepare_external_label_editor


class ExternalLabelEditorTest(unittest.TestCase):
    def make_source(self, root):
        for split in ("train", "valid", "test"):
            directory = root / split
            directory.mkdir(parents=True)
            with (directory / "_annotations.csv").open("w", newline="") as output:
                writer = csv.writer(output)
                writer.writerow(["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"])
                if split == "train":
                    for name in ("face.jpg", "spot.jpg", "empty.jpg"):
                        Image.new("RGB", (20, 10)).save(directory / name)
                    writer.writerows([
                        ["face.jpg", 20, 10, "acne", 1, 1, 5, 5],
                        ["face.jpg", 20, 10, "pimple", 6, 1, 10, 5],
                        ["spot.jpg", 20, 10, "spot", 1, 1, 5, 5],
                    ])

    def test_includes_all_images_but_only_acne_and_pimple_boxes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_source(root / "source")
            result = prepare_external_label_editor(root / "source", root / "editor")
            self.assertEqual(result["summary"]["images"], 3)
            self.assertEqual(result["summary"]["annotations"], 2)
            self.assertTrue((root / "editor/edit.html").is_file())
            self.assertEqual((root / "editor/dataset/train/labels/spot.txt").read_text(), "")
            self.assertEqual((root / "editor/dataset/train/labels/empty.txt").read_text(), "")
            self.assertEqual(len((root / "editor/dataset/train/labels/face.txt").read_text().splitlines()), 2)
            self.assertEqual(result["items"][2]["source_class_counts"], {"spot": 1})

    def test_rerun_preserves_saved_edits_and_rejects_changed_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_source(root / "source")
            prepare_external_label_editor(root / "source", root / "editor")
            saved = root / "editor/user-edits.json"
            saved.write_text('{"keep":true}')
            prepare_external_label_editor(root / "source", root / "editor")
            self.assertEqual(json.loads(saved.read_text()), {"keep": True})
            Image.new("RGB", (20, 10), "red").save(root / "source/train/face.jpg")
            with self.assertRaisesRegex(ValueError, "source.*changed"):
                prepare_external_label_editor(root / "source", root / "editor")

    def test_refuses_output_inside_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_source(root)
            with self.assertRaises(ValueError):
                prepare_external_label_editor(root, root / "editor")

    def test_rejects_csv_dimensions_that_do_not_match_the_image(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_source(root / "source")
            Image.new("RGB", (40, 20)).save(root / "source/train/face.jpg")
            with self.assertRaisesRegex(ValueError, "dimensions"):
                prepare_external_label_editor(root / "source", root / "editor")
            self.assertFalse((root / "editor").exists())
