"""
Module 6 Unit Tests: Flask Backend API Endpoints & Request Handling
File: tests/test_api.py
"""

import io
import json
import os
import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure test environment config
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "reports" / ".mplconfig"))
os.environ.setdefault("KERAS_HOME", str(PROJECT_ROOT / ".keras"))

from backend.app import create_app
from backend.config import TestingConfig

TEST_DIR = PROJECT_ROOT / "datasets" / "processed" / "plantdoc" / "test"


class TestFlaskBackendAPI(unittest.TestCase):
    """Test suite for Module 6 Flask backend endpoints, error handling, and inference."""

    @classmethod
    def setUpClass(cls):
        """Initializes Flask test client and locates sample test images."""
        cls.app = create_app(TestingConfig)
        cls.client = cls.app.test_client()

        sample_imgs = list((TEST_DIR / "Apple_leaf").glob("*.jpg"))
        cls.sample_image_path = sample_imgs[0] if sample_imgs else None

    def test_01_root_endpoint(self):
        """Verify GET / returns 200 OK with service description and route index."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertIn("service", data)
        self.assertIn("endpoints", data)
        self.assertEqual(data["endpoints"]["health"], "/api/health")
        self.assertEqual(data["endpoints"]["predict"], "/api/predict")

    def test_02_health_endpoint(self):
        """Verify GET /api/health returns 200 OK with model readiness status."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("model_status", data)
        self.assertTrue(data["model_status"]["model_ready"])
        self.assertEqual(data["model_status"]["supported_classes_count"], 28)

        # Verify no sensitive filesystem paths are leaked
        data_str = json.dumps(data)
        self.assertNotIn("c:\\", data_str.lower())
        self.assertNotIn("desktop", data_str.lower())

    def test_03_predict_valid_image(self):
        """Verify POST /api/predict with valid image file returns 200 OK with structured prediction."""
        self.assertIsNotNone(self.sample_image_path, "Sample test image not found")

        with open(self.sample_image_path, "rb") as f:
            image_bytes = f.read()

        data = {
            "file": (io.BytesIO(image_bytes), self.sample_image_path.name)
        }
        response = self.client.post(
            "/api/predict",
            data=data,
            content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 200)

        res = response.get_json()
        self.assertIn(res["status"], ["SUCCESS", "UNCERTAIN_LOW_CONFIDENCE"])
        self.assertEqual(res["predicted_class"], "Apple_leaf")
        self.assertIsInstance(res["confidence"], float)
        self.assertTrue(0.0 <= res["confidence"] <= 1.0)
        self.assertEqual(len(res["top_three"]), 3)
        self.assertIn("visual_health_category", res)
        self.assertIn("plucking_recommendation", res)
        self.assertIn("disclaimer", res)

    def test_04_predict_valid_image_image_field_name(self):
        """Verify POST /api/predict accepts field name 'image' as well as 'file'."""
        with open(self.sample_image_path, "rb") as f:
            image_bytes = f.read()

        data = {
            "image": (io.BytesIO(image_bytes), "sample_leaf.jpg")
        }
        response = self.client.post(
            "/api/predict",
            data=data,
            content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 200)
        res = response.get_json()
        self.assertIsNotNone(res["predicted_class"])

    def test_05_predict_missing_file_payload(self):
        """Verify POST /api/predict with empty payload returns 400 Bad Request."""
        response = self.client.post(
            "/api/predict",
            data={},
            content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data["status"], "ERROR")
        self.assertEqual(data["error"], "BAD_REQUEST")

    def test_06_predict_empty_filename(self):
        """Verify POST /api/predict with empty filename returns 400 Bad Request."""
        data = {
            "file": (io.BytesIO(b"dummy data"), "")
        }
        response = self.client.post(
            "/api/predict",
            data=data,
            content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data["status"], "ERROR")

    def test_07_predict_unsupported_file_extension(self):
        """Verify POST /api/predict with unsupported file (.txt) returns 400 Bad Request."""
        data = {
            "file": (io.BytesIO(b"Hello world"), "notes.txt")
        }
        response = self.client.post(
            "/api/predict",
            data=data,
            content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data["status"], "ERROR")
        self.assertIn("not supported", data["message"])

    def test_08_predict_empty_zero_byte_file(self):
        """Verify POST /api/predict with empty 0-byte file returns 400 Bad Request."""
        data = {
            "file": (io.BytesIO(b""), "empty.jpg")
        }
        response = self.client.post(
            "/api/predict",
            data=data,
            content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data["status"], "ERROR")
        self.assertIn("empty", data["message"].lower())

    def test_09_predict_corrupted_image_bytes(self):
        """Verify POST /api/predict with corrupted non-image bytes returns 422 Unprocessable Entity."""
        corrupted_bytes = b"NOT_A_VALID_JPEG_HEADER_RANDOM_GARBAGE_BYTES"
        data = {
            "file": (io.BytesIO(corrupted_bytes), "corrupt.jpg")
        }
        response = self.client.post(
            "/api/predict",
            data=data,
            content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 422)
        data = response.get_json()
        self.assertEqual(data["status"], "ERROR")
        self.assertEqual(data["error"], "UNPROCESSABLE_ENTITY")

    def test_10_predict_custom_confidence_threshold_uncertain(self):
        """Verify custom confidence_threshold query parameter triggers UNCERTAIN_LOW_CONFIDENCE."""
        with open(self.sample_image_path, "rb") as f:
            image_bytes = f.read()

        data = {
            "file": (io.BytesIO(image_bytes), self.sample_image_path.name)
        }
        response = self.client.post(
            "/api/predict?confidence_threshold=0.999",
            data=data,
            content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 200)
        res = response.get_json()
        self.assertEqual(res["status"], "UNCERTAIN_LOW_CONFIDENCE")
        self.assertFalse(res["is_above_threshold"])
        self.assertIsNotNone(res["confidence_warning"])

    def test_11_predict_invalid_confidence_threshold(self):
        """Verify invalid confidence_threshold returns 400 Bad Request."""
        with open(self.sample_image_path, "rb") as f:
            image_bytes = f.read()

        data = {
            "file": (io.BytesIO(image_bytes), self.sample_image_path.name)
        }
        response = self.client.post(
            "/api/predict?confidence_threshold=2.5",
            data=data,
            content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data["status"], "ERROR")

    def test_12_not_found_endpoint(self):
        """Verify non-existent route returns standardized 404 JSON."""
        response = self.client.get("/api/non_existent_route")
        self.assertEqual(response.status_code, 404)
        data = response.get_json()
        self.assertEqual(data["status"], "ERROR")
        self.assertEqual(data["error"], "NOT_FOUND")

    def test_13_method_not_allowed(self):
        """Verify invalid HTTP method returns 405 Method Not Allowed JSON."""
        response = self.client.get("/api/predict")
        self.assertEqual(response.status_code, 405)
        data = response.get_json()
        self.assertEqual(data["status"], "ERROR")
        self.assertEqual(data["error"], "METHOD_NOT_ALLOWED")

    def test_14_cors_headers_present(self):
        """Verify CORS headers are returned for API routes."""
        response = self.client.options("/api/health")
        self.assertIn("Access-Control-Allow-Origin", response.headers)

    def test_15_predict_saves_to_database(self):
        """Verify POST /api/predict saves the analysis record and returns analysis_id."""
        with open(self.sample_image_path, "rb") as f:
            image_bytes = f.read()

        data = {
            "file": (io.BytesIO(image_bytes), "sample_apple_save.jpg")
        }
        response = self.client.post(
            "/api/predict",
            data=data,
            content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 200)
        res = response.get_json()
        self.assertIn("analysis_id", res)
        self.assertTrue(res.get("history_saved"))
        self.assertIsInstance(res["analysis_id"], int)
        self.assertGreater(res["analysis_id"], 0)

    def test_16_get_history_endpoint(self):
        """Verify GET /api/history returns paginated analysis records."""
        response = self.client.get("/api/history?page=1&limit=5")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertIn("records", data)
        self.assertIn("pagination", data)
        self.assertEqual(data["pagination"]["page"], 1)
        self.assertEqual(data["pagination"]["limit"], 5)

    def test_17_get_history_stats_endpoint(self):
        """Verify GET /api/history/stats returns aggregate analytics."""
        response = self.client.get("/api/history/stats")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertIn("statistics", data)
        stats = data["statistics"]
        self.assertIn("total_records", stats)
        self.assertIn("healthy_count", stats)
        self.assertIn("diseased_count", stats)
        self.assertIn("average_confidence", stats)

    def test_18_get_history_record_by_id(self):
        """Verify GET /api/history/<id> retrieves full diagnostic details."""
        # Insert one record via predict
        with open(self.sample_image_path, "rb") as f:
            image_bytes = f.read()
        res = self.client.post(
            "/api/predict",
            data={"file": (io.BytesIO(image_bytes), "detail_test.jpg")},
            content_type="multipart/form-data"
        ).get_json()
        rec_id = res["analysis_id"]

        response = self.client.get(f"/api/history/{rec_id}")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "SUCCESS")
        record = data["record"]
        self.assertEqual(record["id"], rec_id)
        self.assertEqual(record["image_filename"], "detail_test.jpg")
        self.assertIn("top_three", record)
        self.assertIn("quality_assessment", record)

    def test_19_get_history_record_not_found(self):
        """Verify GET /api/history/<id> returns 404 for non-existent ID."""
        response = self.client.get("/api/history/999999")
        self.assertEqual(response.status_code, 404)
        data = response.get_json()
        self.assertEqual(data["status"], "ERROR")
        self.assertEqual(data["error"], "NOT_FOUND")

    def test_20_delete_history_record(self):
        """Verify DELETE /api/history/<id> deletes a single record."""
        # Create a record first
        with open(self.sample_image_path, "rb") as f:
            image_bytes = f.read()
        res = self.client.post(
            "/api/predict",
            data={"file": (io.BytesIO(image_bytes), "delete_me.jpg")},
            content_type="multipart/form-data"
        ).get_json()
        rec_id = res["analysis_id"]

        del_resp = self.client.delete(f"/api/history/{rec_id}")
        self.assertEqual(del_resp.status_code, 200)
        del_data = del_resp.get_json()
        self.assertEqual(del_data["status"], "SUCCESS")
        self.assertEqual(del_data["deleted_id"], rec_id)

        # Confirm 404 after deletion
        get_resp = self.client.get(f"/api/history/{rec_id}")
        self.assertEqual(get_resp.status_code, 404)

    def test_21_delete_history_record_not_found(self):
        """Verify DELETE /api/history/<id> returns 404 for non-existent ID."""
        del_resp = self.client.delete("/api/history/999999")
        self.assertEqual(del_resp.status_code, 404)
        del_data = del_resp.get_json()
        self.assertEqual(del_data["status"], "ERROR")

    def test_22_clear_history_requires_confirmation(self):
        """Verify DELETE /api/history without confirmation query returns 400 Bad Request."""
        response = self.client.delete("/api/history")
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data["status"], "ERROR")
        self.assertEqual(data["error"], "BAD_REQUEST")
        self.assertIn("Confirmation required", data["message"])

    def test_23_clear_history_with_confirmation(self):
        """Verify DELETE /api/history?confirm=true successfully clears records."""
        # Ensure at least 1 record exists
        with open(self.sample_image_path, "rb") as f:
            image_bytes = f.read()
        self.client.post(
            "/api/predict",
            data={"file": (io.BytesIO(image_bytes), "clear_test.jpg")},
            content_type="multipart/form-data"
        )

        response = self.client.delete("/api/history?confirm=true")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertIn("deleted_count", data)
        self.assertGreaterEqual(data["deleted_count"], 1)


if __name__ == "__main__":
    unittest.main()
