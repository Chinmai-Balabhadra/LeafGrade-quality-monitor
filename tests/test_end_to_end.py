"""
Leaf Quality Classification and Plucking Standard Monitoring System
Module 9: Full End-to-End Workflow Verification Suite
File: tests/test_end_to_end.py
"""

import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure Matplotlib and Keras offline cache
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "reports" / ".mplconfig"))
os.environ.setdefault("KERAS_HOME", str(PROJECT_ROOT / ".keras"))

from backend.app import create_app
from backend.config import TestingConfig
from database.db import init_db, get_statistics

TEST_DIR = PROJECT_ROOT / "datasets" / "processed" / "plantdoc" / "test"


class TestEndToEndWorkflow(unittest.TestCase):
    """Verifies the complete 15-step lifecycle from ingestion to deletion."""

    @classmethod
    def setUpClass(cls):
        """Initializes application and locates real test image."""
        cls.test_dir = tempfile.mkdtemp(prefix="e2e_test_db_")
        cls.e2e_db = Path(cls.test_dir) / "e2e_analysis.db"
        init_db(cls.e2e_db)

        # Set environment database path for isolated e2e testing
        os.environ["DATABASE_PATH"] = str(cls.e2e_db)

        cls.app = create_app(TestingConfig)
        cls.client = cls.app.test_client()

        sample_imgs = list((TEST_DIR / "Apple_leaf").glob("*.jpg"))
        cls.sample_image_path = sample_imgs[0] if sample_imgs else None
        with open(cls.sample_image_path, "rb") as f:
            cls.image_bytes = f.read()

    @classmethod
    def tearDownClass(cls):
        """Clean up isolated database directory and environment variable."""
        if "DATABASE_PATH" in os.environ:
            del os.environ["DATABASE_PATH"]
        if Path(cls.test_dir).exists():
            shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_complete_15_step_workflow(self):
        """Executes the full 15-step lifecycle in sequence."""
        # 1. Start application / test client
        self.assertIsNotNone(self.client)

        # 2. Verify model initialization & health check
        health_resp = self.client.get("/api/health")
        self.assertEqual(health_resp.status_code, 200)
        health_data = health_resp.get_json()
        self.assertEqual(health_data["status"], "healthy")
        self.assertTrue(health_data["model_status"]["model_ready"])
        self.assertEqual(health_data["model_status"]["supported_classes_count"], 28)

        # 3. Open frontend index.html
        fe_resp = self.client.get("/index.html")
        self.assertEqual(fe_resp.status_code, 200)
        self.assertIn("AgriLeaf", fe_resp.get_data(as_text=True))
        self.assertIn("historyDetailModal", fe_resp.get_data(as_text=True))

        # Initial baseline stats check
        stats_before = self.client.get("/api/history/stats").get_json()["statistics"]
        init_count = stats_before["total_records"]

        # 4 & 5. Upload leaf photograph & send to prediction API
        predict_payload = {
            "file": (io.BytesIO(self.image_bytes), "e2e_field_leaf.jpg")
        }
        pred_resp = self.client.post("/api/predict", data=predict_payload, content_type="multipart/form-data")
        self.assertEqual(pred_resp.status_code, 200)
        result = pred_resp.get_json()

        # 6. Receive classification results
        self.assertIn(result["status"], ["SUCCESS", "UNCERTAIN_LOW_CONFIDENCE"])
        self.assertEqual(result["predicted_class"], "Apple_leaf")
        self.assertEqual(result["species"], "Apple")

        # 7. Display prediction confidence
        self.assertIsInstance(result["confidence"], float)
        self.assertIsInstance(result["confidence_percent"], float)
        self.assertTrue(0.0 <= result["confidence_percent"] <= 100.0)

        # 8. Display top-three predictions
        self.assertEqual(len(result["top_three"]), 3)
        for cand in result["top_three"]:
            self.assertIn("rank", cand)
            self.assertIn("class_name", cand)
            self.assertIn("confidence_percent", cand)
            self.assertIn("visual_health_type", cand)

        # 9. Display visual health interpretation
        self.assertIn("Healthy", result["visual_health_category"])
        self.assertIsInstance(result["is_visually_healthy"], bool)
        self.assertTrue(result["is_visually_healthy"])

        # 10. Display plucking recommendation
        self.assertIsNotNone(result["plucking_recommendation"])
        self.assertGreater(len(result["plucking_recommendation"]), 10)

        # 11. Save the prediction to SQLite
        self.assertTrue(result.get("history_saved"))
        analysis_id = result.get("analysis_id")
        self.assertIsInstance(analysis_id, int)
        self.assertGreater(analysis_id, 0)

        # 12. Verify that the record appears in history
        hist_resp = self.client.get("/api/history?page=1&limit=10")
        self.assertEqual(hist_resp.status_code, 200)
        hist_data = hist_resp.get_json()
        record_ids = [r["id"] for r in hist_data["records"]]
        self.assertIn(analysis_id, record_ids)

        # 13. Open the historical record
        rec_resp = self.client.get(f"/api/history/{analysis_id}")
        self.assertEqual(rec_resp.status_code, 200)
        record = rec_resp.get_json()["record"]
        self.assertEqual(record["id"], analysis_id)
        self.assertEqual(record["image_filename"], "e2e_field_leaf.jpg")
        self.assertEqual(record["species"], "Apple")
        self.assertEqual(len(record["top_three"]), 3)
        self.assertIn("laplacian_variance", record["quality_assessment"])

        # Check stats incremented
        stats_mid = self.client.get("/api/history/stats").get_json()["statistics"]
        self.assertEqual(stats_mid["total_records"], init_count + 1)

        # 14. Delete the selected record
        del_resp = self.client.delete(f"/api/history/{analysis_id}")
        self.assertEqual(del_resp.status_code, 200)
        self.assertEqual(del_resp.get_json()["deleted_id"], analysis_id)

        # Confirm 404 after deletion
        del_check = self.client.get(f"/api/history/{analysis_id}")
        self.assertEqual(del_check.status_code, 404)

        # 15. Verify that statistics update correctly after deletion
        stats_after = self.client.get("/api/history/stats").get_json()["statistics"]
        self.assertEqual(stats_after["total_records"], init_count)


if __name__ == "__main__":
    unittest.main()
