import json
import tempfile
import unittest
from pathlib import Path

from acne_care.dataset.external_label_edits import apply_completed_external_edits
from acne_care.dataset.label_review_dashboard import review_dataset_fingerprint
from backend.tests import test_external_label_editor as editor_fixtures
from acne_care.dataset.external_label_editor import prepare_external_label_editor


class ExternalLabelEditsTest(unittest.TestCase):
    def prepare(self, root):
        editor_fixtures.ExternalLabelEditorTest().make_source(root / "source")
        audit = prepare_external_label_editor(root / "source", root / "editor")
        edits = {
            "schema_version": 1,
            "dataset_fingerprint": review_dataset_fingerprint(audit, splits=("train", "valid", "test")),
            "review_splits": ["train", "valid", "test"],
            "edits": {
                "train/images/face.jpg": {"completed": True, "excluded": False,
                    "annotations": [{"class_id": 0, "x_center": 0.5, "y_center": 0.5, "width": 0.2, "height": 0.2}]},
                "train/images/spot.jpg": {"completed": False, "excluded": True, "annotations": []},
            },
        }
        path = root / "edits.json"
        path.write_text(json.dumps(edits))
        return path, edits

    def test_applies_completed_only_and_preserves_every_other_sample(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, _ = self.prepare(root)
            source_labels = {p.name: p.read_bytes() for p in (root / "editor/dataset/train/labels").iterdir()}
            result = apply_completed_external_edits(root / "editor", path, root / "output")
            self.assertEqual(result["completed_images"], 1)
            self.assertEqual(result["unreviewed_images_preserved"], 2)
            self.assertEqual(result["changed_label_images"], 1)
            self.assertEqual(result["added_boxes"], 1)
            self.assertEqual(result["removed_boxes"], 2)
            self.assertEqual(result["excluded_images"], 0)
            self.assertTrue((root / "output/train/images/spot.jpg").exists())
            self.assertEqual((root / "output/train/labels/spot.txt").read_bytes(), source_labels["spot.txt"])
            self.assertEqual((root / "output/train/labels/empty.txt").read_bytes(), source_labels["empty.txt"])
            for name, content in source_labels.items():
                self.assertEqual((root / "editor/dataset/train/labels" / name).read_bytes(), content)
            self.assertEqual((root / "output/label-edits-source.json").read_bytes(), path.read_bytes())
            self.assertEqual(apply_completed_external_edits(root / "editor", path, root / "output"), result)

    def test_applies_completed_exclusion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, edits = self.prepare(root)
            edits["edits"]["train/images/face.jpg"]["excluded"] = True
            path.write_text(json.dumps(edits))
            result = apply_completed_external_edits(root / "editor", path, root / "output")
            self.assertEqual(result["excluded_images"], 1)
            self.assertFalse((root / "output/train/images/face.jpg").exists())

    def test_rejects_wrong_dataset_and_does_not_overwrite_existing_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, edits = self.prepare(root)
            edits["dataset_fingerprint"] = "wrong"
            path.write_text(json.dumps(edits))
            with self.assertRaisesRegex(ValueError, "different dataset"):
                apply_completed_external_edits(root / "editor", path, root / "output")
            self.assertFalse((root / "output").exists())

    def test_rejects_nested_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, _ = self.prepare(root)
            with self.assertRaises(ValueError):
                apply_completed_external_edits(root / "editor", path, root / "editor/dataset/output")
