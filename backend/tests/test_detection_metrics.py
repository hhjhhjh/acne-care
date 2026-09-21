import unittest

from acne_care.domain import BoundingBox, Detection
from acne_care.evaluation.detection_metrics import evaluate_detections


class DetectionMetricsTest(unittest.TestCase):
    def test_counts_true_positive_false_positive_and_false_negative(self) -> None:
        ground_truths = (
            (BoundingBox(0, 0, 10, 10), 0),
            (BoundingBox(20, 20, 30, 30), 0),
        )
        predictions = (
            Detection(BoundingBox(0, 0, 10, 10), confidence=0.9, class_id=0),
            Detection(BoundingBox(40, 40, 50, 50), confidence=0.8, class_id=0),
        )

        result = evaluate_detections(
            ground_truths=ground_truths,
            predictions=predictions,
            iou_threshold=0.5,
        )

        self.assertEqual(result.true_positives, 1)
        self.assertEqual(result.false_positives, 1)
        self.assertEqual(result.false_negatives, 1)
        self.assertEqual(result.precision, 0.5)
        self.assertEqual(result.recall, 0.5)
        self.assertEqual(result.f1, 0.5)

    def test_counts_second_prediction_for_same_target_as_false_positive(self) -> None:
        ground_truths = ((BoundingBox(0, 0, 10, 10), 0),)
        predictions = (
            Detection(BoundingBox(0, 0, 10, 10), confidence=0.9, class_id=0),
            Detection(BoundingBox(1, 1, 9, 9), confidence=0.8, class_id=0),
        )

        result = evaluate_detections(ground_truths, predictions, iou_threshold=0.5)

        self.assertEqual(result.true_positives, 1)
        self.assertEqual(result.false_positives, 1)
        self.assertEqual(result.false_negatives, 0)

    def test_handles_no_predictions(self) -> None:
        ground_truths = ((BoundingBox(0, 0, 10, 10), 0),)

        result = evaluate_detections(ground_truths, (), iou_threshold=0.5)

        self.assertEqual(result.precision, 0.0)
        self.assertEqual(result.recall, 0.0)
        self.assertEqual(result.f1, 0.0)


if __name__ == "__main__":
    unittest.main()
