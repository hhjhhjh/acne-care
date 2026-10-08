import unittest

from acne_care.dataset.label_review_dashboard import render_dashboard_html


class LabelReviewDashboardTest(unittest.TestCase):
    def test_embeds_only_requested_splits_and_review_controls(self) -> None:
        document = {
            "schema_version": 1,
            "summary": {"images": 2},
            "items": [
                {
                    "split": "train",
                    "image_path": "train/images/train.jpg",
                    "annotation_count": 1,
                    "annotations": [],
                    "review_hints": [],
                    "details": [],
                },
                {
                    "split": "valid",
                    "image_path": "valid/images/valid.jpg",
                    "annotation_count": 1,
                    "annotations": [],
                    "review_hints": ["very_small_box"],
                    "details": [],
                },
            ],
        }

        html = render_dashboard_html(document, splits=("valid", "test"))

        self.assertIn("valid/images/valid.jpg", html)
        self.assertNotIn("train/images/train.jpg", html)
        self.assertIn("검토 결과 내보내기", html)
        self.assertIn("수정 필요", html)


if __name__ == "__main__":
    unittest.main()
