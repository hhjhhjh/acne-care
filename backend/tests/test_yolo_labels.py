import tempfile
import unittest
from pathlib import Path

from acne_care.dataset.yolo import read_yolo_ground_truths


class YoloLabelTest(unittest.TestCase):
    def test_converts_normalized_coordinates_to_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            label_path = Path(temp_dir) / "sample.txt"
            label_path.write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")

            ground_truths = read_yolo_ground_truths(
                label_path,
                image_width=100,
                image_height=200,
            )

            box, class_id = ground_truths[0]
            self.assertEqual(class_id, 0)
            self.assertAlmostEqual(box.x_min, 40.0)
            self.assertAlmostEqual(box.y_min, 60.0)
            self.assertAlmostEqual(box.x_max, 60.0)
            self.assertAlmostEqual(box.y_max, 140.0)

    def test_rejects_malformed_label(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            label_path = Path(temp_dir) / "sample.txt"
            label_path.write_text("0 0.5 0.5 0.2\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "sample.txt:1"):
                read_yolo_ground_truths(
                    label_path,
                    image_width=100,
                    image_height=100,
                )


if __name__ == "__main__":
    unittest.main()
