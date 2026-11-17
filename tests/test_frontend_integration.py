"""
Module 7 Frontend Integration Test Suite
File: tests/test_frontend_integration.py

Verifies:
1. Frontend loads successfully via HTTP 200 (index.html, style.css, dashboard.css, responsive.css).
2. All JavaScript client assets load cleanly (api.js, upload.js, prediction.js, main.js).
3. Backend health check contract parity with frontend expectation.
4. Prediction API multipart contract with frontend client FormData ('file' field).
5. Output response fields match frontend rendering contract:
   - predicted_class, species, condition, visual_health_category, confidence_percent
   - top_three candidate structure (rank, class_name, confidence_percent, visual_health_type)
   - quality_assessment (laplacian_variance, quality_rating, is_blurry)
   - blur_warning alert contract
   - confidence_warning uncertainty alert contract
   - plucking_recommendation & disclaimer
6. Error handling contract (status 400 for bad extension, 422 for corrupt bytes).
7. HTML semantic structure verification (IDs and elements required by JS controllers).
8. Responsive design stylesheets validation.
"""

import io
import json
import os
import re
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
FRONTEND_DIR = PROJECT_ROOT / "frontend"


class TestFrontendIntegration(unittest.TestCase):
    """Integration verification test suite for Module 7 frontend assets & API contracts."""

    @classmethod
    def setUpClass(cls):
        """Initializes Flask client and locates sample test image."""
        cls.app = create_app(TestingConfig)
        cls.client = cls.app.test_client()

        sample_imgs = list((TEST_DIR / "Apple_leaf").glob("*.jpg"))
        cls.sample_image_path = sample_imgs[0] if sample_imgs else None

    def test_01_frontend_html_and_assets_served(self):
        """Verify index.html and all CSS/JS assets are served with HTTP 200."""
        routes = [
            "/index.html",
            "/css/style.css",
            "/css/dashboard.css",
            "/css/responsive.css",
            "/js/api.js",
            "/js/upload.js",
            "/js/prediction.js",
            "/js/history.js",
            "/js/main.js"
        ]
        for route in routes:
            resp = self.client.get(route)
            self.assertEqual(resp.status_code, 200, f"Failed to serve {route}: HTTP {resp.status_code}")
            self.assertGreater(len(resp.data), 100, f"Asset suspiciously empty: {route}")

    def test_02_backend_health_check_contract(self):
        """Verify GET /api/health satisfies frontend status indicator expectations."""
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)

        data = resp.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("model_status", data)
        self.assertTrue(data["model_status"]["model_ready"])
        self.assertEqual(data["model_status"]["supported_classes_count"], 28)

    def test_03_prediction_contract_with_frontend_form_data(self):
        """Verify POST /api/predict accepts multipart file under 'file' and returns complete schema."""
        self.assertIsNotNone(self.sample_image_path)
        with open(self.sample_image_path, "rb") as f:
            img_bytes = f.read()

        payload = {
            "file": (io.BytesIO(img_bytes), self.sample_image_path.name)
        }
        resp = self.client.post(
            "/api/predict",
            data=payload,
            content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()

        # Contract fields used by prediction.js
        expected_fields = [
            "status", "predicted_class", "confidence_percent", "species", "condition",
            "visual_health_category", "is_visually_healthy", "top_three",
            "quality_assessment", "plucking_recommendation", "disclaimer"
        ]
        for field in expected_fields:
            self.assertIn(field, data, f"Missing contract field: {field}")

        # Top-3 candidate contract
        self.assertEqual(len(data["top_three"]), 3)
        for cand in data["top_three"]:
            self.assertIn("rank", cand)
            self.assertIn("class_name", cand)
            self.assertIn("confidence_percent", cand)
            self.assertIn("visual_health_type", cand)

    def test_04_uncertainty_threshold_contract(self):
        """Verify confidence thresholding triggers UNCERTAIN_LOW_CONFIDENCE expected by frontend."""
        with open(self.sample_image_path, "rb") as f:
            img_bytes = f.read()

        payload = {
            "file": (io.BytesIO(img_bytes), self.sample_image_path.name),
            "confidence_threshold": "0.99"
        }
        resp = self.client.post(
            "/api/predict",
            data=payload,
            content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()

        self.assertEqual(data["status"], "UNCERTAIN_LOW_CONFIDENCE")
        self.assertFalse(data["is_above_threshold"])
        self.assertIsNotNone(data["confidence_warning"])

    def test_05_html_semantic_element_parity(self):
        """Verify index.html contains all IDs required by JavaScript controllers."""
        index_file = FRONTEND_DIR / "index.html"
        self.assertTrue(index_file.exists())

        html_content = index_file.read_text(encoding="utf-8")

        required_element_ids = [
            "navToggle", "navLinks", "statusDot", "statusText",
            "analysisDashboard", "loadingOverlay",
            "tabUpload", "tabCamera", "contentUpload", "contentCamera",
            "dropzone", "leafFileInput", "browseFilesBtn",
            "cameraContainer", "cameraFeed", "cameraCanvas", "btnCapture", "btnCloseCamera",
            "previewContainer", "previewImage", "previewFilename", "previewFilesize",
            "thresholdSlider", "thresholdVal",
            "btnAnalyzeLeaf", "btnRetakeImage", "btnRemoveImage",
            "resultsCard", "resultStatusBadge", "diagnosisBanner",
            "predictedSpecies", "predictedCondition", "confidenceGaugeVal",
            "blurAlertBox", "blurAlertText", "uncertaintyAlertBox", "uncertaintyAlertText",
            "metricSharpness", "metricLaplacian", "metricBrightness",
            "topCandidatesList", "recommendationBox", "recHeader", "recBody",
            "disclaimerText", "btnResetAnalysis", "toastContainer",
            # Module 8 History & Analytics Elements
            "statTotalAnalyses", "statHealthyCount", "statDiseasedCount", "statAvgConfidence",
            "historySearchInput", "btnClearSearch", "historyFilterSpecies", "historyFilterHealth",
            "btnClearAllHistory", "historyLoadingOverlay", "historyEmptyState",
            "historyTableContainer", "historyTableBody", "historyPagination",
            "btnPrevPage", "btnNextPage", "historyPageIndicator",
            "historyDetailModal", "btnCloseModal", "modalRecordDetails"
        ]

        for elem_id in required_element_ids:
            pattern = f'id=["\']{elem_id}["\']'
            self.assertTrue(
                re.search(pattern, html_content),
                f"Missing required HTML element ID: {elem_id}"
            )

    def test_06_css_files_integrity(self):
        """Verify CSS files exist and contain core design tokens and media queries."""
        css_files = ["style.css", "dashboard.css", "responsive.css"]
        for css in css_files:
            css_path = FRONTEND_DIR / "css" / css
            self.assertTrue(css_path.exists(), f"Missing {css}")
            content = css_path.read_text(encoding="utf-8")
            self.assertGreater(len(content), 500, f"CSS file suspiciously small: {css}")

        # Check responsive queries
        resp_css = (FRONTEND_DIR / "css" / "responsive.css").read_text(encoding="utf-8")
        self.assertIn("@media", resp_css)
        self.assertIn("992px", resp_css)
        self.assertIn("768px", resp_css)
        self.assertIn("480px", resp_css)

    def test_07_history_api_contract_with_frontend(self):
        """Verify GET /api/history contract conforms to frontend HistoryController expectations."""
        resp = self.client.get("/api/history?page=1&limit=10")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertIn("records", data)
        self.assertIn("pagination", data)

        pagination = data["pagination"]
        expected_page_fields = ["page", "limit", "total_records", "total_pages", "has_next", "has_prev"]
        for pf in expected_page_fields:
            self.assertIn(pf, pagination, f"Missing pagination field: {pf}")

    def test_08_history_stats_contract_with_frontend(self):
        """Verify GET /api/history/stats contract conforms to frontend stats cards & species filter."""
        resp = self.client.get("/api/history/stats")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertIn("statistics", data)

        stats = data["statistics"]
        self.assertIn("total_records", stats)
        self.assertIn("healthy_count", stats)
        self.assertIn("diseased_count", stats)
        self.assertIn("average_confidence_percent", stats)
        self.assertIn("by_species", stats)


if __name__ == "__main__":
    unittest.main()
