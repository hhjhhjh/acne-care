import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from acne_care.api.app import create_app
from acne_care.api.settings import Settings
from acne_care.application.analysis import ModelInfo
from backend.tests import test_analysis_service as fixtures


class AnalysisAPITest(unittest.TestCase):
    def client(self, root, *, preview_address=None, **options):
        settings = Settings(storage_dir=root / "media", database_path=root / "app.sqlite3", **options)
        app = create_app(settings, detector=fixtures.FakeDetector(),
                         model_info=ModelInfo("test-detector", "a" * 64, 0.3, 640, "cpu"),
                         preview_address=preview_address)
        base_url = f"http://{preview_address}:8443" if preview_address else "http://testserver"
        return TestClient(app, base_url=base_url)

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

    def test_mobile_preview_accepts_local_browser_uploads(self):
        with tempfile.TemporaryDirectory() as temp, self.client(Path(temp), preview_address="192.168.1.10") as client:
            self.assertEqual(client.get("/").status_code, 200)
            self.assertEqual(client.get("/", headers={"Host": "unrelated.example"}).status_code, 400)
            self.assertEqual(client.get("/api/analyses").json()["total"], 0)
            photo = {"file": ("camera.png", fixtures.image_bytes(), "image/png")}
            date = {"captured_on": "2026-10-01"}
            self.assertEqual(client.post("/api/analyses", files=photo, data=date).status_code, 403)
            response = client.post("/api/analyses", files=photo, data=date,
                                   headers={"Origin": "http://192.168.1.10:8443"})
            self.assertEqual(response.status_code, 201, response.text)
            image = response.json()["original_image_url"]
            self.assertEqual(client.get(image).status_code, 200)

    def test_capture_details_are_persisted_and_returned_with_comparison(self):
        with tempfile.TemporaryDirectory() as temp, self.client(Path(temp)) as client:
            previous = self.upload(client, "2026-10-01")
            response = client.post(
                "/api/analyses",
                files={"file": ("photo.png", fixtures.image_bytes(), "image/png")},
                data={
                    "captured_on": "2026-10-02", "capture_lighting": "밝고 고른 조명",
                    "capture_angle": "정면", "capture_distance": "가까이",
                    "capture_filter": "사용 안 함", "capture_note": "  세안제 변경  ",
                },
            )
            self.assertEqual(response.status_code, 201, response.text)
            current = response.json()
            self.assertEqual(current["capture_note"], "세안제 변경")
            detail = client.get(f"/api/analyses/{current['id']}").json()
            self.assertEqual(detail["capture_lighting"], "밝고 고른 조명")
            self.assertEqual(detail["capture_angle"], "정면")
            self.assertEqual(detail["capture_distance"], "가까이")
            self.assertEqual(detail["capture_filter"], "사용 안 함")
            comparison = client.get("/api/comparisons", params={"previous_id": previous["id"], "current_id": current["id"]}).json()
            self.assertIsNone(comparison["previous"]["capture_note"])
            self.assertEqual(comparison["current"]["capture_note"], "세안제 변경")
            self.assertTrue(comparison["comparison"]["analysis_settings_match"])

    def test_invalid_capture_details_do_not_create_a_record(self):
        with tempfile.TemporaryDirectory() as temp, self.client(Path(temp)) as client:
            for invalid in ({"capture_angle": "알 수 없는 값"}, {"capture_note": "메" * 161}, {"capture_note": "두 줄\n메모"}):
                with self.subTest(invalid=invalid):
                    response = client.post(
                        "/api/analyses", files={"file": ("photo.png", fixtures.image_bytes(), "image/png")},
                        data={"captured_on": "2026-10-01", **invalid},
                    )
                    self.assertEqual(response.status_code, 422, response.text)
            self.assertEqual(client.get("/api/analyses").json()["total"], 0)

    def upload(self, client, captured_on):
        response = client.post("/api/analyses", files={"file": ("photo.png", fixtures.image_bytes(), "image/png")},
                               data={"captured_on": captured_on})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def test_comparison_uses_saved_records_without_new_analysis(self):
        with tempfile.TemporaryDirectory() as temp, self.client(Path(temp)) as client:
            previous = self.upload(client, "2026-10-01")
            current = self.upload(client, "2026-10-06")
            response = client.get("/api/comparisons", params={"previous_id": previous["id"], "current_id": current["id"]})
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            self.assertEqual(result["comparison"]["days_between"], 5)
            self.assertEqual(result["comparison"]["count_delta"], 0)
            self.assertTrue(result["comparison"]["analysis_settings_match"])
            for side in ("previous", "current"):
                self.assertNotIn("original_path", result[side])
                self.assertEqual(client.get(result[side]["annotated_image_url"]).status_code, 200)
            self.assertEqual(response.headers["cache-control"], "no-store")
            self.assertEqual(client.get("/api/analyses").json()["total"], 2)

    def test_comparison_rejects_same_record_and_reversed_dates(self):
        with tempfile.TemporaryDirectory() as temp, self.client(Path(temp)) as client:
            previous = self.upload(client, "2026-10-01")
            current = self.upload(client, "2026-10-06")
            for ids in [(previous["id"], previous["id"]), (current["id"], previous["id"])]:
                response = client.get("/api/comparisons", params={"previous_id": ids[0], "current_id": ids[1]})
                self.assertEqual(response.status_code, 422, response.text)

    def test_comparison_requires_two_existing_records(self):
        with tempfile.TemporaryDirectory() as temp, self.client(Path(temp)) as client:
            previous = self.upload(client, "2026-10-01")
            for unknown in ["invalid-id", "00000000-0000-0000-0000-000000000000"]:
                response = client.get("/api/comparisons", params={"previous_id": previous["id"], "current_id": unknown})
                self.assertEqual(response.status_code, 404)
            self.assertEqual(client.get("/api/comparisons").status_code, 422)
