"""
Leaf Quality Classification and Plucking Standard Monitoring System
Module 9: Automated Edge Case & Security Hardening Test Suite
File: tests/test_edge_cases.py
"""

import io
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sqlite3

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure Matplotlib and Keras offline cache
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "reports" / ".mplconfig"))
os.environ.setdefault("KERAS_HOME", str(PROJECT_ROOT / ".keras"))

from backend.app import create_app
from backend.config import TestingConfig
from database.db import init_db, insert_analysis, get_history

TEST_DIR = PROJECT_ROOT / "datasets" / "processed" / "plantdoc" / "test"


class TestEdgeCasesAndReliability(unittest.TestCase):
    """Test suite covering edge cases, failure modes, boundaries, and security."""

    @classmethod
    def setUpClass(cls):
        """Initializes Flask client and locates sample test image."""
        cls.app = create_app(TestingConfig)
        cls.client = cls.app.test_client()

        sample_imgs = list((TEST_DIR / "Apple_leaf").glob("*.jpg"))
        cls.sample_image_path = sample_imgs[0] if sample_imgs else None
        with open(cls.sample_image_path, "rb") as f:
            cls.valid_image_bytes = f.read()

    def setUp(self):
        """Creates an isolated temporary database for edge case testing."""
        self.test_dir = tempfile.mkdtemp(prefix="edge_case_db_")
        self.test_db = Path(self.test_dir) / "edge_test.db"
        init_db(self.test_db)

    def tearDown(self):
        """Cleans up temporary directory."""
        if Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_edge_01_text_file_upload_rejected(self):
        """Verify uploaded .txt file is rejected with 400 Bad Request."""
        data = {"file": (io.BytesIO(b"Simple plain text"), "notes.txt")}
        resp = self.client.post("/api/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 400)
        res = resp.get_json()
        self.assertEqual(res["status"], "ERROR")
        self.assertIn("not supported", res["message"])

    def test_edge_02_corrupted_image_header_rejected(self):
        """Verify file with invalid image header returns 422 Unprocessable Entity."""
        corrupt_bytes = b"\xFF\xD8\xFF\xE0" + b"\x00" * 20 + b"RANDOM_CORRUPT_BYTES"
        data = {"file": (io.BytesIO(corrupt_bytes), "corrupt.jpg")}
        resp = self.client.post("/api/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 422)
        res = resp.get_json()
        self.assertEqual(res["status"], "ERROR")
        self.assertEqual(res["error"], "UNPROCESSABLE_ENTITY")

    def test_edge_03_truncated_binary_bytes(self):
        """Verify truncated image stream returns 422."""
        truncated_bytes = self.valid_image_bytes[:30]  # Incomplete header
        data = {"file": (io.BytesIO(truncated_bytes), "truncated.jpg")}
        resp = self.client.post("/api/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 422)

    def test_edge_04_unsupported_format_gif(self):
        """Verify animated GIF format is rejected with 400 Bad Request."""
        gif_bytes = b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;"
        data = {"file": (io.BytesIO(gif_bytes), "animation.gif")}
        resp = self.client.post("/api/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 400)

    def test_edge_05_empty_zero_byte_file(self):
        """Verify empty 0-byte file returns 400 Bad Request."""
        data = {"file": (io.BytesIO(b""), "zero_byte.jpg")}
        resp = self.client.post("/api/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 400)
        res = resp.get_json()
        self.assertIn("empty", res["message"].lower())

    def test_edge_06_missing_file_payload(self):
        """Verify POST /api/predict without multipart file field returns 400 Bad Request."""
        resp = self.client.post("/api/predict", data={}, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 400)

    def test_edge_07_blurry_photograph_detection(self):
        """Verify heavily blurred image triggers blur alert and rating."""
        from PIL import Image, ImageFilter
        # Blur the genuine leaf image to test blur detection on actual foliage
        img = Image.open(io.BytesIO(self.valid_image_bytes))
        blurred = img.filter(ImageFilter.GaussianBlur(radius=15))
        buf = io.BytesIO()
        blurred.save(buf, format="JPEG")
        buf.seek(0)

        data = {"file": (buf, "apple_leaf_blurry.jpg")}
        resp = self.client.post("/api/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 200)
        res = resp.get_json()

        self.assertIsNotNone(res.get("blur_warning"))
        quality = res.get("quality_assessment", {})
        self.assertTrue(quality.get("is_blurry"))
        self.assertIn("Blurry", quality.get("quality_rating"))

    def test_edge_07b_blank_solid_image_rejected(self):
        """Verify solid flat blank image is rejected by OOD validation with 422."""
        from PIL import Image
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        data = {"file": (buf, "flat_grey_blank.jpg")}
        resp = self.client.post("/api/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 422)
        res = resp.get_json()
        self.assertEqual(res.get("status"), "REJECTED_NOT_LEAF")
        self.assertIn("No valid plant leaf detected", res.get("message"))

    def test_edge_08_low_confidence_uncertainty_threshold(self):
        """Verify forcing a high confidence threshold (0.999) flags UNCERTAIN_LOW_CONFIDENCE."""
        data = {"file": (io.BytesIO(self.valid_image_bytes), "apple_leaf.jpg")}
        resp = self.client.post("/api/predict?confidence_threshold=0.999", data=data, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 200)
        res = resp.get_json()

        self.assertEqual(res["status"], "UNCERTAIN_LOW_CONFIDENCE")
        self.assertFalse(res["is_above_threshold"])
        self.assertIsNotNone(res["confidence_warning"])
        self.assertIn("do not make harvesting decisions", res["plucking_recommendation"].lower())

    def test_edge_09_database_write_failure_containment(self):
        """Verify that a database write exception does NOT break the prediction response."""
        with patch("backend.routes.prediction.insert_analysis", side_effect=sqlite3.OperationalError("database is locked")):
            data = {"file": (io.BytesIO(self.valid_image_bytes), "apple_leaf.jpg")}
            resp = self.client.post("/api/predict", data=data, content_type="multipart/form-data")

            # Must still return 200 OK to the user
            self.assertEqual(resp.status_code, 200)
            res = resp.get_json()
            self.assertIn(res["status"], ["SUCCESS", "UNCERTAIN_LOW_CONFIDENCE"])
            self.assertFalse(res.get("history_saved"))
            self.assertIsNotNone(res.get("history_warning"))
            self.assertIn("Database storage warning", res["history_warning"])

    def test_edge_10_invalid_history_pagination_negative_page(self):
        """Verify invalid negative or 0 page parameter returns 400 Bad Request."""
        resp = self.client.get("/api/history?page=-1&limit=10")
        self.assertEqual(resp.status_code, 400)
        res = resp.get_json()
        self.assertEqual(res["status"], "ERROR")

    def test_edge_11_invalid_history_pagination_string_page(self):
        """Verify non-integer page parameter returns 400 Bad Request."""
        resp = self.client.get("/api/history?page=not_a_number&limit=10")
        self.assertEqual(resp.status_code, 400)

    def test_edge_12_large_page_number_beyond_range(self):
        """Verify requesting page 9999 returns empty list with valid pagination metadata."""
        resp = self.client.get("/api/history?page=9999&limit=10")
        self.assertEqual(resp.status_code, 200)
        res = resp.get_json()
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(len(res["records"]), 0)
        self.assertEqual(res["pagination"]["page"], 9999)

    def test_edge_13_delete_history_without_confirmation_safeguard(self):
        """Verify DELETE /api/history without ?confirm=true is safely rejected with 400."""
        resp = self.client.delete("/api/history")
        self.assertEqual(resp.status_code, 400)
        res = resp.get_json()
        self.assertIn("Confirmation required", res["message"])

    def test_edge_14_delete_non_existent_history_id(self):
        """Verify deleting non-existent ID returns 404 Not Found."""
        resp = self.client.delete("/api/history/888888")
        self.assertEqual(resp.status_code, 404)

    def test_edge_15_history_search_sql_injection_attempt(self):
        """Verify SQL injection payloads in search query are safely parameterized."""
        sql_payload = "Apple' OR '1'='1"
        resp = self.client.get(f"/api/history?search={sql_payload}")
        self.assertEqual(resp.status_code, 200)
        res = resp.get_json()
        self.assertEqual(res["status"], "SUCCESS")
        # Should not dump all records
        self.assertEqual(len(res["records"]), 0)

    def test_edge_16_filename_path_sanitization_in_db(self):
        """Verify inserting Windows or Unix absolute paths stores only the sanitized basename."""
        record = {
            "source": "C:\\SecretProject\\ConfidentialPhotos\\farm_leaf_01.jpg",
            "species": "Apple",
            "predicted_class": "Apple_scab_healthy",
            "confidence": 0.95,
            "visual_health": "Healthy",
            "is_visually_healthy": True,
            "status": "SUCCESS"
        }
        rec_id = insert_analysis(record, db_path=self.test_db)
        records, _ = get_history(db_path=self.test_db)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["image_filename"], "farm_leaf_01.jpg")
        self.assertNotIn("SecretProject", records[0]["image_filename"])
        self.assertNotIn("C:", records[0]["image_filename"])


if __name__ == "__main__":
    unittest.main()
