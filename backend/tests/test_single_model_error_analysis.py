import unittest

from acne_care.evaluation.single_model_error_analysis import (
    analyze_prediction_document,
    rank_error_images,
    summarize_condition_groups,
)


class SingleModelErrorAnalysisTest(unittest.TestCase):
    def test_filters_predictions_and_reports_errors_by_size(self) -> None:
        document = self._document()

        result = analyze_prediction_document(
            document,
            confidence_threshold=0.3,
            iou_threshold=0.5,
        )

        self.assertEqual(result["counts"]["true_positives"], 1)
        self.assertEqual(result["counts"]["false_positives"], 1)
        self.assertEqual(result["counts"]["false_negatives"], 1)
        self.assertEqual(result["by_size"]["small"]["recall"], 1.0)
        self.assertEqual(result["by_size"]["medium"]["recall"], 0.0)
        self.assertEqual(result["per_image"][0]["prediction_count"], 2)

    def test_rejects_threshold_below_saved_prediction_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "below saved prediction threshold"):
            analyze_prediction_document(
                self._document(),
                confidence_threshold=0.1,
                iou_threshold=0.5,
            )

    def test_summarizes_ranked_condition_groups(self) -> None:
        images = [
            self._image_summary("low.jpg", brightness=0.1, true_positives=1),
            self._image_summary("middle.jpg", brightness=0.5, true_positives=2),
            self._image_summary(
                "high.jpg",
                brightness=0.9,
                true_positives=3,
                false_negatives=0,
            ),
        ]

        result = summarize_condition_groups(images, "brightness")

        self.assertEqual(result["low"]["images"], 1)
        self.assertEqual(result["low"]["recall"], 0.5)
        self.assertEqual(result["high"]["recall"], 1.0)

    def test_ranks_images_by_requested_error(self) -> None:
        images = [
            self._image_summary("first.jpg", false_negatives=1),
            self._image_summary("second.jpg", false_negatives=3),
        ]

        result = rank_error_images(images, "false_negatives", limit=1)

        self.assertEqual(result[0]["image_path"], "second.jpg")
        self.assertEqual(result[0]["false_negatives"], 3)

    @staticmethod
    def _document() -> dict[str, object]:
        return {
            "schema_version": 1,
            "model_id": "candidate",
            "split": "val",
            "confidence_threshold": 0.2,
            "iou_threshold": 0.5,
            "samples": [
                {
                    "image_path": "/server/valid/images/image.jpg",
                    "image_width": 640,
                    "image_height": 640,
                    "ground_truths": [
                        {"box": [0, 0, 10, 10], "class_id": 0},
                        {"box": [100, 100, 140, 140], "class_id": 0},
                    ],
                    "predictions": [
                        {"box": [0, 0, 10, 10], "confidence": 0.9, "class_id": 0},
                        {
                            "box": [100, 100, 140, 140],
                            "confidence": 0.25,
                            "class_id": 0,
                        },
                        {
                            "box": [200, 200, 210, 210],
                            "confidence": 0.8,
                            "class_id": 0,
                        },
                    ],
                }
            ],
        }

    @staticmethod
    def _image_summary(
        image_path: str,
        *,
        brightness: float = 0.5,
        true_positives: int = 1,
        false_positives: int = 0,
        false_negatives: int = 1,
    ) -> dict[str, object]:
        return {
            "image_path": image_path,
            "ground_truth_count": true_positives + false_negatives,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "brightness": brightness,
            "sharpness": 1.0,
        }


if __name__ == "__main__":
    unittest.main()
