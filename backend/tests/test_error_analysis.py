import unittest

from acne_care.evaluation.error_analysis import compare_prediction_documents


class ErrorAnalysisTest(unittest.TestCase):
    def test_compares_detection_outcomes_and_size_recall(self) -> None:
        ground_truths = [
            {"box": [0, 0, 10, 10], "class_id": 0},
            {"box": [100, 100, 140, 140], "class_id": 0},
        ]
        shared = {
            "schema_version": 1,
            "split": "val",
            "confidence_threshold": 0.2,
            "iou_threshold": 0.5,
        }
        legacy = {
            **shared,
            "model_id": "legacy",
            "samples": [
                {
                    "image_path": "/data/image.jpg",
                    "image_width": 640,
                    "image_height": 640,
                    "ground_truths": ground_truths,
                    "predictions": [
                        {"box": [100, 100, 140, 140], "confidence": 0.9, "class_id": 0}
                    ],
                }
            ],
        }
        yolo = {
            **shared,
            "model_id": "yolo11n",
            "samples": [
                {
                    "image_path": "/data/image.jpg",
                    "image_width": 640,
                    "image_height": 640,
                    "ground_truths": ground_truths,
                    "predictions": [
                        {"box": [0, 0, 10, 10], "confidence": 0.8, "class_id": 0},
                        {"box": [200, 200, 210, 210], "confidence": 0.7, "class_id": 0},
                    ],
                }
            ],
        }

        result = compare_prediction_documents(legacy, yolo)

        self.assertEqual(result["outcomes"]["yolo11_only"], 1)
        self.assertEqual(result["outcomes"]["legacy_only"], 1)
        self.assertEqual(result["false_positives"]["legacy"], 0)
        self.assertEqual(result["false_positives"]["yolo11"], 1)
        self.assertEqual(result["by_size"]["small"]["yolo11_recall"], 1.0)
        self.assertEqual(result["by_size"]["medium"]["legacy_recall"], 1.0)

    def test_reports_localization_sensitivity_at_relaxed_iou(self) -> None:
        sample = {
            "image_path": "/data/image.jpg",
            "image_width": 640,
            "image_height": 640,
            "ground_truths": [{"box": [0, 0, 10, 10], "class_id": 0}],
            "predictions": [
                {"box": [4, 0, 14, 10], "confidence": 0.9, "class_id": 0}
            ],
        }
        document = {
            "schema_version": 1,
            "model_id": "model",
            "split": "val",
            "confidence_threshold": 0.2,
            "iou_threshold": 0.5,
            "samples": [sample],
        }

        result = compare_prediction_documents(document, document)

        self.assertEqual(result["iou_sensitivity"]["0.30"]["legacy_recall"], 1.0)
        self.assertEqual(result["iou_sensitivity"]["0.50"]["legacy_recall"], 0.0)


if __name__ == "__main__":
    unittest.main()
