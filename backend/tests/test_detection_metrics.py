import unittest

from acne_care.domain import BoundingBox, Detection
from acne_care.evaluation.detection_metrics import (
    evaluate_detections,
    evaluate_thresholds,
    match_detections,
    select_best_threshold,
)


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

    def test_reports_original_indexes_for_matches_and_errors(self) -> None:
        ground_truths = (
            (BoundingBox(0, 0, 10, 10), 0),
            (BoundingBox(20, 20, 30, 30), 0),
        )
        predictions = (
            Detection(BoundingBox(40, 40, 50, 50), confidence=0.7, class_id=0),
            Detection(BoundingBox(20, 20, 30, 30), confidence=0.9, class_id=0),
        )

        result = match_detections(ground_truths, predictions, iou_threshold=0.5)

        self.assertEqual(result.matched_pairs, ((1, 1, 1.0),))
        self.assertEqual(result.false_positive_indices, (0,))
        self.assertEqual(result.false_negative_indices, (0,))

    def test_reports_duplicate_prediction_as_false_positive(self) -> None:
        ground_truths = ((BoundingBox(0, 0, 10, 10), 0),)
        predictions = (
            Detection(BoundingBox(1, 1, 9, 9), confidence=0.8, class_id=0),
            Detection(BoundingBox(0, 0, 10, 10), confidence=0.9, class_id=0),
        )

        result = match_detections(ground_truths, predictions, iou_threshold=0.5)

        self.assertEqual(result.matched_pairs, ((0, 1, 1.0),))
        self.assertEqual(result.false_positive_indices, (0,))
        self.assertEqual(result.false_negative_indices, ())

    def test_handles_no_predictions(self) -> None:
        ground_truths = ((BoundingBox(0, 0, 10, 10), 0),)

        result = evaluate_detections(ground_truths, (), iou_threshold=0.5)

        self.assertEqual(result.precision, 0.0)
        self.assertEqual(result.recall, 0.0)
        self.assertEqual(result.f1, 0.0)

    def test_evaluates_thresholds_and_count_mae(self) -> None:
        ground_truths = ((BoundingBox(0, 0, 10, 10), 0),)
        predictions = (
            Detection(BoundingBox(0, 0, 10, 10), confidence=0.8, class_id=0),
            Detection(BoundingBox(20, 20, 30, 30), confidence=0.3, class_id=0),
        )

        results = evaluate_thresholds(
            ((ground_truths, predictions),),
            thresholds=(0.2, 0.5),
            iou_threshold=0.5,
        )

        self.assertEqual(results[0].false_positives, 1)
        self.assertEqual(results[0].count_mae, 1.0)
        self.assertEqual(results[1].false_positives, 0)
        self.assertEqual(results[1].count_mae, 0.0)

    def test_selects_highest_f1_then_lowest_count_mae(self) -> None:
        ground_truths = ((BoundingBox(0, 0, 10, 10), 0),)
        predictions = (
            Detection(BoundingBox(0, 0, 10, 10), confidence=0.9, class_id=0),
            Detection(BoundingBox(20, 20, 30, 30), confidence=0.4, class_id=0),
        )
        results = evaluate_thresholds(
            ((ground_truths, predictions),),
            thresholds=(0.3, 0.5),
            iou_threshold=0.5,
        )

        best = select_best_threshold(results)

        self.assertEqual(best.threshold, 0.5)


if __name__ == "__main__":
    unittest.main()
