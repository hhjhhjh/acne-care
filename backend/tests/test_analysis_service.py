import io
import tempfile
import unittest
from datetime import date
from pathlib import Path

from PIL import Image

from acne_care.adapters.sqlite_repository import SQLiteAnalysisRepository
from acne_care.application.analysis import AnalysisService, InvalidImage, ModelInfo
from acne_care.domain import BoundingBox, Detection


class FakeDetector:
    model_id = "test-detector"

    def detect(self, image):
        return (Detection(BoundingBox(10, 10, 30, 30), 0.8, 0),)


def image_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (64, 80), "beige").save(buffer, format="PNG")
    return buffer.getvalue()


class AnalysisServiceTest(unittest.TestCase):
    def make_service(self, root, detector=None):
        repository = SQLiteAnalysisRepository(root / "records.sqlite3")
        repository.initialize()
        info = ModelInfo("test-detector", "a" * 64, 0.3, 640, "cpu")
        return AnalysisService(detector or FakeDetector(), info, repository, root / "media"), repository

    def test_analysis_survives_reopening_database_and_retains_model_settings(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            service, _ = self.make_service(root)
            record = service.analyze(image_bytes(), date(2026, 10, 1))
            self.assertEqual(record["count"], 1)
            self.assertEqual(record["captured_on"], "2026-10-01")
            self.assertEqual(record["model_id"], "test-detector")
            self.assertEqual(record["confidence_threshold"], 0.3)
            reopened = SQLiteAnalysisRepository(root / "records.sqlite3")
            reopened.initialize()
            detail = reopened.get(record["id"])
            self.assertEqual(detail["detections"][0]["box"]["x_min"], 10)
            self.assertEqual(reopened.list(20, 0)["total"], 1)
            self.assertTrue((root / "media" / record["id"] / "annotated.jpg").is_file())

    def test_invalid_image_does_not_create_a_record(self):
        with tempfile.TemporaryDirectory() as temp:
            service, repo = self.make_service(Path(temp))
            with self.assertRaises(InvalidImage):
                service.analyze(b"not an image", date(2026, 10, 1))
            self.assertEqual(repo.list(20, 0)["total"], 0)

    def test_out_of_bounds_detection_is_not_saved(self):
        class BadDetector(FakeDetector):
            def detect(self, image):
                return (Detection(BoundingBox(0, 0, 200, 200), 0.8, 0),)
        with tempfile.TemporaryDirectory() as temp:
            service, repo = self.make_service(Path(temp), BadDetector())
            with self.assertRaises(ValueError):
                service.analyze(image_bytes(), date(2026, 10, 1))
            self.assertEqual(repo.list(20, 0)["total"], 0)

    def test_zero_detections_are_saved_as_zero(self):
        class EmptyDetector(FakeDetector):
            def detect(self, image):
                return ()
        with tempfile.TemporaryDirectory() as temp:
            service, repo = self.make_service(Path(temp), EmptyDetector())
            record = service.analyze(image_bytes(), date(2026, 10, 1))
            self.assertEqual(record["count"], 0)
            self.assertEqual(repo.get(record["id"])["detections"], [])

    def test_database_failure_cleans_up_new_images(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            service, repo = self.make_service(root)
            def fail(record, detections):
                raise RuntimeError("database unavailable")
            repo.insert = fail
            with self.assertRaises(RuntimeError):
                service.analyze(image_bytes(), date(2026, 10, 1))
            self.assertEqual(list((root / "media").iterdir()), [])

    def test_image_pixel_limit_is_enforced(self):
        with tempfile.TemporaryDirectory() as temp:
            service, _ = self.make_service(Path(temp))
            service.max_pixels = 100
            with self.assertRaises(InvalidImage):
                service.analyze(image_bytes(), date(2026, 10, 1))

    def test_exif_orientation_is_applied_and_metadata_is_not_saved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            service, _ = self.make_service(root)
            source = Image.new("RGB", (64, 80), "beige")
            exif = Image.Exif()
            exif[274] = 6
            exif[315] = "should not be retained"
            data = io.BytesIO()
            source.save(data, format="JPEG", exif=exif)
            result = service.analyze(data.getvalue(), date(2026, 10, 1))
            self.assertEqual((result["width"], result["height"]), (80, 64))
            with Image.open(root / "media" / result["id"] / "original.jpg") as saved:
                self.assertEqual(dict(saved.getexif()), {})

    def test_history_orders_by_capture_date_and_supports_pagination(self):
        with tempfile.TemporaryDirectory() as temp:
            service, repo = self.make_service(Path(temp))
            newer = service.analyze(image_bytes(), date(2026, 10, 3))
            older = service.analyze(image_bytes(), date(2026, 10, 1))
            self.assertEqual(repo.list(1, 0)["items"][0]["id"], newer["id"])
            self.assertEqual(repo.list(1, 1)["items"][0]["id"], older["id"])

    def test_missing_or_wrong_weights_fail_before_loading_a_model(self):
        from acne_care.adapters.yolo_detector import YOLOAcneDetector
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "missing.pt"
            with self.assertRaises(FileNotFoundError):
                YOLOAcneDetector(path, model_id="test", expected_sha256=None)
            path.write_bytes(b"not the expected model")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                YOLOAcneDetector(path, model_id="test", expected_sha256="a" * 64)
