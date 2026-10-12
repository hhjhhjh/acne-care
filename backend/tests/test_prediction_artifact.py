import unittest
from pathlib import Path

from acne_care.domain import BoundingBox, Detection
from acne_care.evaluation.prediction_artifact import serialize_sample


class PredictionArtifactTest(unittest.TestCase):
    def test_serializes_boxes_and_filters_by_confidence(self) -> None:
        ground_truths = ((BoundingBox(1, 2, 3, 4), 0),)
        predictions = (
            Detection(BoundingBox(5, 6, 7, 8), confidence=0.8, class_id=0),
            Detection(BoundingBox(9, 10, 11, 12), confidence=0.2, class_id=0),
        )

        result = serialize_sample(
            image_path=Path("image.jpg"),
            image_width=640,
            image_height=640,
            ground_truths=ground_truths,
            predictions=predictions,
            confidence_threshold=0.3,
        )

        self.assertEqual(result["image_path"], "image.jpg")
        self.assertEqual(result["ground_truths"][0]["box"], [1, 2, 3, 4])
        self.assertEqual(len(result["predictions"]), 1)
        self.assertEqual(result["predictions"][0]["confidence"], 0.8)


if __name__ == "__main__":
    unittest.main()
