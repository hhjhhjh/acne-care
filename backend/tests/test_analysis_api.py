import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from acne_care.api.app import create_app
from acne_care.api.settings import Settings
from acne_care.application.analysis import ModelInfo
from backend.tests import test_analysis_service as fixtures


class AnalysisAPITest(unittest.TestCase):
    def client(self, root, **options):
        settings = Settings(storage_dir=root / "media", database_path=root / "app.sqlite3", **options)
        return TestClient(create_app(settings, detector=fixtures.FakeDetector(),
                          model_info=ModelInfo("test-detector", "a" * 64, 0.3, 640, "cpu")))

    def test_upload_list_detail_and_images(self):
        with tempfile.TemporaryDirectory() as temp, self.client(Path(temp)) as client:
            response = client.post("/api/analyses", files={"file": ("../../photo.png", fixtures.image_bytes(), "image/png")}, data={"captured_on": "2026-10-01"})
            self.assertEqual(response.status_code, 201, response.text)
            record = response.json()
            self.assertEqual(record["count"], 1)
            self.assertEqual(client.get("/api/analyses").json()["total"], 1)
            self.assertEqual(client.get(f"/api/analyses/{record['id']}").json()["id"], record["id"])
            self.assertEqual(client.get(record["original_image_url"]).headers["content-type"], "image/jpeg")
            self.assertEqual(client.get(record["annotated_image_url"]).status_code, 200)
            self.assertEqual(client.get("/").status_code, 200)

    def test_bad_file_and_future_date_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp, self.client(Path(temp)) as client:
            bad = client.post("/api/analyses", files={"file": ("bad.jpg", b"bad", "image/jpeg")}, data={"captured_on": "2026-10-01"})
            self.assertEqual(bad.status_code, 422)
            future = client.post("/api/analyses", files={"file": ("photo.png", fixtures.image_bytes(), "image/png")}, data={"captured_on": "2999-01-01"})
            self.assertEqual(future.status_code, 422)
            self.assertEqual(client.get("/api/analyses").json()["total"], 0)

    def test_file_size_limit_and_unknown_record(self):
        with tempfile.TemporaryDirectory() as temp, self.client(Path(temp), max_upload_bytes=10) as client:
            response = client.post("/api/analyses", files={"file": ("photo.png", fixtures.image_bytes(), "image/png")}, data={"captured_on": "2026-10-01"})
            self.assertEqual(response.status_code, 413)
            self.assertEqual(client.get("/api/analyses/not-a-real-id").status_code, 404)

    def test_cross_origin_upload_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp, self.client(Path(temp)) as client:
            response = client.post("/api/analyses", headers={"Origin": "https://unrelated.example"}, files={"file": ("photo.png", fixtures.image_bytes(), "image/png")}, data={"captured_on": "2026-10-01"})
            self.assertEqual(response.status_code, 403)
