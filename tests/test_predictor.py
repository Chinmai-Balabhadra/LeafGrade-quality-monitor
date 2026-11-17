"""
Module 5 Unit Tests: Leaf Quality Predictor & Inference Pipeline
File: tests/test_predictor.py
"""

import json
import os
import sys
from pathlib import Path
import unittest

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Avoid Matplotlib font cache warnings
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "reports" / ".mplconfig"))
os.environ.setdefault("KERAS_HOME", str(PROJECT_ROOT / ".keras"))

from utils.predictor import LeafPredictor, get_default_predictor, predict_leaf

TEST_DIR = PROJECT_ROOT / "datasets" / "processed" / "plantdoc" / "test"
TEMP_DIR = PROJECT_ROOT / "reports" / ".test_temp"


class TestLeafPredictorPipeline(unittest.TestCase):
    """Comprehensive test suite for Module 5 single-image prediction pipeline."""

    @classmethod
    def setUpClass(cls):
        """Initializes default predictor and locates sample test images."""
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        cls.predictor = get_default_predictor()

        # Find sample healthy and diseased images
        apple_healthy_imgs = list((TEST_DIR / "Apple_leaf").glob("*.jpg"))
        apple_rust_imgs = list((TEST_DIR / "Apple_rust_leaf").glob("*.jpg"))

        cls.sample_healthy_path = apple_healthy_imgs[0] if apple_healthy_imgs else None
        cls.sample_diseased_path = apple_rust_imgs[0] if apple_rust_imgs else None

    @classmethod
    def tearDownClass(cls):
        """Cleans up temporary test files."""
        import shutil
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)

    def test_01_predictor_initialization_and_model(self):
        """Verify LeafPredictor initializes properly and loads MobileNetV2."""
        self.assertIsNotNone(self.predictor.model, "Model was not loaded")
        self.assertEqual(len(self.predictor.idx_to_class), 28, "Expected 28 classes in idx_to_class")
        self.assertEqual(len(self.predictor.class_to_idx), 28, "Expected 28 classes in class_to_idx")
        self.assertEqual(self.predictor.model.input_shape, (None, 224, 224, 3))
        self.assertEqual(self.predictor.model.output_shape, (None, 28))

    def test_02_predict_real_image_structure(self):
        """Verify prediction dictionary structure and value constraints on real image."""
        self.assertIsNotNone(self.sample_healthy_path, "Sample test image not found")
        result = self.predictor.predict(self.sample_healthy_path)

        # Check required fields
        required_keys = [
            "status", "source", "predicted_class", "predicted_index",
            "confidence", "confidence_percent", "confidence_threshold_applied",
            "is_above_threshold", "confidence_warning", "blur_warning",
            "visual_health_category", "is_visually_healthy", "plucking_recommendation",
            "species", "condition", "plucking_suitability", "top_three",
            "quality_assessment", "disclaimer"
        ]
        for key in required_keys:
            self.assertIn(key, result, f"Missing key in prediction result: {key}")

        self.assertIn(result["status"], ["SUCCESS", "UNCERTAIN_LOW_CONFIDENCE"])
        self.assertIsInstance(result["predicted_class"], str)
        self.assertIsInstance(result["predicted_index"], int)
        self.assertTrue(0 <= result["predicted_index"] < 28)
        self.assertTrue(0.0 <= result["confidence"] <= 1.0)
        self.assertEqual(result["confidence_percent"], round(result["confidence"] * 100.0, 2))

    def test_03_top_three_ranking_and_prob_consistency(self):
        """Verify Top-3 predictions are sorted in descending confidence order."""
        result = self.predictor.predict(self.sample_healthy_path)
        top_three = result["top_three"]

        self.assertEqual(len(top_three), 3, "Expected exactly 3 candidates in top_three")

        # Monotonically descending order
        c1, c2, c3 = top_three[0]["confidence"], top_three[1]["confidence"], top_three[2]["confidence"]
        self.assertGreaterEqual(c1, c2, "Top-1 confidence should be >= Top-2")
        self.assertGreaterEqual(c2, c3, "Top-2 confidence should be >= Top-3")

        # Top-1 matches top prediction
        self.assertEqual(top_three[0]["class_name"], result["predicted_class"])
        self.assertEqual(top_three[0]["class_index"], result["predicted_index"])

    def test_04_json_serializability(self):
        """Verify entire prediction result is 100% JSON-serializable for Flask APIs."""
        result = self.predictor.predict(self.sample_healthy_path)
        try:
            serialized = json.dumps(result, indent=2)
            deserialized = json.loads(serialized)
            self.assertEqual(deserialized["predicted_class"], result["predicted_class"])
        except TypeError as e:
            self.fail(f"Prediction result is not JSON serializable: {e}")

    def test_05_confidence_threshold_uncertainty_trigger(self):
        """Verify low-confidence threshold triggers UNCERTAIN_LOW_CONFIDENCE status."""
        # Set impossible threshold (0.999) to force uncertainty
        result = self.predictor.predict(self.sample_healthy_path, confidence_threshold=0.999)

        self.assertEqual(result["status"], "UNCERTAIN_LOW_CONFIDENCE")
        self.assertFalse(result["is_above_threshold"])
        self.assertIsNotNone(result["confidence_warning"])
        self.assertIn("Uncertain visual classification", result["plucking_recommendation"])

    def test_06_non_existent_file_handling(self):
        """Verify graceful error handling for missing files."""
        fake_path = TEMP_DIR / "non_existent_leaf_image.jpg"
        result = self.predictor.predict(fake_path)

        self.assertEqual(result["status"], "ERROR")
        self.assertIsNone(result["predicted_class"])
        self.assertEqual(result["confidence"], 0.0)
        self.assertIn("File does not exist", result["message"])

    def test_07_unsupported_file_extension(self):
        """Verify graceful rejection of unsupported file extensions."""
        txt_path = TEMP_DIR / "sample_doc.txt"
        txt_path.write_text("Plain text content")
        result = self.predictor.predict(txt_path)

        self.assertEqual(result["status"], "ERROR")
        self.assertIn("Unsupported file extension", result["message"])

    def test_08_empty_zero_byte_file(self):
        """Verify handling of 0-byte corrupt image files."""
        empty_file = TEMP_DIR / "empty_corrupt.png"
        empty_file.write_bytes(b"")
        result = self.predictor.predict(empty_file)

        self.assertEqual(result["status"], "ERROR")
        self.assertIn("empty", result["message"].lower())

    def test_09_in_memory_bytes_support(self):
        """Verify prediction directly from raw byte streams (simulating Flask file uploads)."""
        with open(self.sample_healthy_path, "rb") as f:
            raw_bytes = f.read()

        result = self.predictor.predict(raw_bytes)
        self.assertIn(result["status"], ["SUCCESS", "UNCERTAIN_LOW_CONFIDENCE"])
        self.assertEqual(result["source"], "in_memory_bytes")
        self.assertIsNotNone(result["predicted_class"])

    def test_10_numpy_array_input_support(self):
        """Verify prediction directly from NumPy uint8 image array (e.g. from OpenCV/camera feed)."""
        bgr = cv2.imread(str(self.sample_healthy_path))
        result = self.predictor.predict(bgr)

        self.assertIn(result["status"], ["SUCCESS", "UNCERTAIN_LOW_CONFIDENCE"])
        self.assertEqual(result["source"], "numpy_array")
        self.assertIsNotNone(result["predicted_class"])

    def test_11_blur_detection(self):
        """Verify Laplacian sharpness detector flags heavily blurred captures."""
        bgr = cv2.imread(str(self.sample_healthy_path))
        # Apply heavy Gaussian blur
        blurred = cv2.GaussianBlur(bgr, (35, 35), 15)

        result = self.predictor.predict(blurred)
        qa = result["quality_assessment"]

        self.assertTrue(qa["is_blurry"], "Blurry image was not flagged as blurry")
        self.assertLess(qa["laplacian_variance"], 50.0)
        self.assertIsNotNone(result["blur_warning"])

    def test_12_plucking_readiness_distinction(self):
        """Verify visual health does NOT automatically grant harvest permission without maturity check."""
        result = self.predictor.predict(self.sample_healthy_path)
        if result["is_visually_healthy"] and result["status"] == "SUCCESS":
            rec = result["plucking_recommendation"].lower()
            # Must explicitly warn that visual health alone is insufficient
            self.assertTrue(
                "maturity" in rec or "inspection" in rec or "tactile" in rec or "field" in rec,
                f"Harvest recommendation failed to state maturity requirement: {result['plucking_recommendation']}"
            )


if __name__ == "__main__":
    unittest.main()
