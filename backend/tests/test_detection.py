import unittest

from acne_care.domain.detection import BoundingBox, Detection


class BoundingBoxTest(unittest.TestCase):
    def test_calculates_center_and_area(self) -> None:
        box = BoundingBox(x_min=10.0, y_min=20.0, x_max=30.0, y_max=50.0)

        self.assertEqual(box.center, (20.0, 35.0))
        self.assertEqual(box.area, 600.0)

    def test_rejects_reversed_coordinates(self) -> None:
        with self.assertRaisesRegex(ValueError, "x_max"):
            BoundingBox(x_min=30.0, y_min=20.0, x_max=10.0, y_max=50.0)

    def test_rejects_box_outside_image(self) -> None:
        box = BoundingBox(x_min=0.0, y_min=0.0, x_max=101.0, y_max=50.0)

        with self.assertRaisesRegex(ValueError, "image bounds"):
            box.validate_image_bounds(width=100, height=100)


class DetectionTest(unittest.TestCase):
    def test_rejects_confidence_above_one(self) -> None:
        box = BoundingBox(x_min=0.0, y_min=0.0, x_max=10.0, y_max=10.0)

        with self.assertRaisesRegex(ValueError, "confidence"):
            Detection(box=box, confidence=1.1, class_id=0)

    def test_rejects_negative_class_id(self) -> None:
        box = BoundingBox(x_min=0.0, y_min=0.0, x_max=10.0, y_max=10.0)

        with self.assertRaisesRegex(ValueError, "class_id"):
            Detection(box=box, confidence=0.7, class_id=-1)


if __name__ == "__main__":
    unittest.main()

