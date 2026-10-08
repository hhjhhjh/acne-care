import unittest

from acne_care.dataset.label_editor_dashboard import render_label_editor_html


class LabelEditorDashboardTest(unittest.TestCase):
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
