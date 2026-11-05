import unittest

from acne_care.dataset.label_editor_dashboard import render_label_editor_html


class LabelEditorDashboardTest(unittest.TestCase):
    def test_all_image_editor_includes_train_and_empty_labels_without_prior_review(self) -> None:
        document = {"items": [
            {"split": "train", "image_path": "train/images/empty.jpg", "annotations": [],
             "annotation_count": 0, "asset_path": "dataset/train/images/empty.jpg"},
            {"split": "test", "image_path": "test/images/face.jpg", "annotations": [],
             "annotation_count": 0},
        ]}
        html = render_label_editor_html(document, include_all=True)
        self.assertIn("train/images/empty.jpg", html)
        self.assertIn("dataset/train/images/empty.jpg", html)
        self.assertIn("test/images/face.jpg", html)
        self.assertIn('id="import-file"', html)
        self.assertIn('id="split-filter"', html)
        self.assertIn('id="zoom"', html)

    def test_train_label_changes_produce_a_different_full_editor_fingerprint(self) -> None:
        import re
        document = {"items": [{"split": "train", "image_path": "train/images/face.jpg",
                               "annotations": [], "annotation_count": 0}]}
        first = render_label_editor_html(document, include_all=True)
        document["items"][0]["annotation_count"] = 1
        second = render_label_editor_html(document, include_all=True)
        pattern = r'const datasetFingerprint = "([a-f0-9]+)"'
        self.assertNotEqual(re.search(pattern, first).group(1), re.search(pattern, second).group(1))

    def test_embeds_only_actionable_review_items(self) -> None:
        document = {
            "schema_version": 1,
            "items": [
                {
                    "split": "valid",
                    "image_path": "valid/images/ok.jpg",
                    "annotation_count": 1,
                    "annotations": [],
                    "review_hints": [],
                    "details": [],
                },
                {
                    "split": "valid",
                    "image_path": "valid/images/fix.jpg",
                    "annotation_count": 1,
                    "annotations": [],
                    "review_hints": [],
                    "details": [],
                },
            ],
        }
        reviews = {
            "dataset_fingerprint": "test-fingerprint",
            "reviews": {
                "valid/images/ok.jpg": {"status": "ok", "note": ""},
                "valid/images/fix.jpg": {
                    "status": "needs_correction",
                    "note": "missing lesion",
                },
            },
        }

        html = render_label_editor_html(document, reviews)

        self.assertIn("valid/images/fix.jpg", html)
        self.assertNotIn("valid/images/ok.jpg", html)
        self.assertIn("상자 추가", html)
        self.assertIn("편집 결과 내보내기", html)


if __name__ == "__main__":
    unittest.main()
