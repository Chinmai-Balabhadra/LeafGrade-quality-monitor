"""
Module 4 Unit Tests: Deep Learning Model Architecture, Inference, and Artifact Integrity
File: tests/test_model.py
"""

import json
import os
import sys
from pathlib import Path
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Avoid Matplotlib font cache warnings
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "reports" / ".mplconfig"))
os.environ.setdefault("KERAS_HOME", str(PROJECT_ROOT / ".keras"))

MODEL_PATH = PROJECT_ROOT / "model" / "leaf_classifier.keras"
CLASS_INDICES_PATH = PROJECT_ROOT / "model" / "class_indices.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
CLEANED_DIR = PROJECT_ROOT / "datasets" / "cleaned" / "plantdoc" / "original_classes"


class TestMobileNetV2Model(unittest.TestCase):
    """Test suite for Module 4 MobileNetV2 leaf quality classification model."""

    @classmethod
    def setUpClass(cls):
        """Load class indices and determine model availability."""
        cls.has_model_file = MODEL_PATH.exists()
        cls.has_class_indices = CLASS_INDICES_PATH.exists()
        cls.model = None

        if cls.has_class_indices:
            with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
                cls.class_indices_data = json.load(f)
        else:
            cls.class_indices_data = {}

        if cls.has_model_file:
            try:
                import keras
                cls.model = keras.models.load_model(str(MODEL_PATH))
            except Exception as e:
                print(f"Warning: Failed to load model in setUpClass: {e}")

    def test_01_class_indices_existence_and_structure(self):
        """Verify model/class_indices.json exists and contains 28 valid classes."""
        self.assertTrue(CLASS_INDICES_PATH.exists(), f"Missing {CLASS_INDICES_PATH}")
        with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("class_to_idx", data)
        self.assertIn("idx_to_class", data)
        self.assertIn("metadata", data)
        self.assertIn("healthy_classes", data)
        self.assertIn("diseased_classes", data)

        self.assertEqual(len(data["class_to_idx"]), 28, "Expected 28 classes in class_to_idx")
        self.assertEqual(len(data["idx_to_class"]), 28, "Expected 28 classes in idx_to_class")
        self.assertEqual(len(data["healthy_classes"]), 10, "Expected 10 healthy classes")
        self.assertEqual(len(data["diseased_classes"]), 18, "Expected 18 diseased classes")

    def test_02_model_file_exists(self):
        """Verify that model/leaf_classifier.keras has been generated."""
        self.assertTrue(MODEL_PATH.exists(), f"Model file not found at {MODEL_PATH}")
        file_size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
        self.assertGreater(file_size_mb, 5.0, f"Model file suspiciously small: {file_size_mb:.2f} MB")

    def test_03_model_architecture_and_tensors(self):
        """Verify model input shape (None, 224, 224, 3) and output shape (None, 28)."""
        if self.model is None:
            import keras
            self.model = keras.models.load_model(str(MODEL_PATH))

        input_shape = self.model.input_shape
        output_shape = self.model.output_shape

        self.assertEqual(input_shape, (None, 224, 224, 3), f"Unexpected input shape: {input_shape}")
        self.assertEqual(output_shape, (None, 28), f"Unexpected output shape: {output_shape}")

    def test_04_dummy_tensor_inference_and_softmax(self):
        """Verify model processes raw [0, 255] RGB float tensors and produces valid probabilities."""
        if self.model is None:
            import keras
            self.model = keras.models.load_model(str(MODEL_PATH))

        # Create dummy batch of 2 images with [0, 255] pixel range
        dummy_batch = np.random.uniform(0.0, 255.0, size=(2, 224, 224, 3)).astype(np.float32)
        predictions = self.model.predict(dummy_batch, verbose=0)

        self.assertEqual(predictions.shape, (2, 28))
        self.assertTrue(np.all(predictions >= 0.0), "Negative probability found")
        self.assertTrue(np.all(predictions <= 1.0001), "Probability > 1.0 found")

        # Softmax sum should be 1.0 per sample
        sums = np.sum(predictions, axis=1)
        np.testing.assert_allclose(sums, [1.0, 1.0], atol=1e-5, err_msg="Softmax probabilities do not sum to 1.0")

    def test_05_real_image_inference(self):
        """Verify inference on an actual cleaned PlantDoc leaf image."""
        if self.model is None:
            import keras
            self.model = keras.models.load_model(str(MODEL_PATH))

        sample_imgs = list(CLEANED_DIR.glob("*/*.jpg"))
        self.assertGreater(len(sample_imgs), 0, "No cleaned PlantDoc images found")
        sample_path = sample_imgs[0]

        from utils.preprocessing import load_image, bgr_to_rgb, resize_image

        bgr = load_image(sample_path)
        rgb = bgr_to_rgb(bgr)
        resized = resize_image(rgb, (224, 224))
        batch_input = np.expand_dims(resized.astype(np.float32), axis=0)

        preds = self.model.predict(batch_input, verbose=0)
        self.assertEqual(preds.shape, (1, 28))

        pred_idx = int(np.argmax(preds[0]))
        confidence = float(preds[0, pred_idx])

        self.assertIn(pred_idx, range(28))
        self.assertGreater(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_06_evaluation_artifacts_exist(self):
        """Verify that curves, confusion matrix, and classification report files exist."""
        curves_path = REPORTS_DIR / "training_curves.png"
        cm_path = REPORTS_DIR / "confusion_matrix.png"
        report_csv_path = REPORTS_DIR / "classification_report.csv"
        report_md_path = REPORTS_DIR / "module4_model_report.md"

        self.assertTrue(curves_path.exists(), f"Missing {curves_path}")
        self.assertTrue(cm_path.exists(), f"Missing {cm_path}")
        self.assertTrue(report_csv_path.exists(), f"Missing {report_csv_path}")
        self.assertTrue(report_md_path.exists(), f"Missing {report_md_path}")

        # Ensure image files are valid sizes
        self.assertGreater(curves_path.stat().st_size, 20_000, "Curves image is empty or corrupted")
        self.assertGreater(cm_path.stat().st_size, 20_000, "Confusion matrix image is empty or corrupted")

    def test_07_classification_report_metrics(self):
        """Verify that classification report CSV contains valid empirical metrics."""
        import pandas as pd
        report_csv_path = REPORTS_DIR / "classification_report.csv"
        self.assertTrue(report_csv_path.exists())

        df = pd.read_csv(report_csv_path)
        col_names = [c.lower() for c in df.columns]
        self.assertTrue(any("precision" in c for c in col_names), "Precision column missing")
        self.assertTrue(any("recall" in c for c in col_names), "Recall column missing")
        self.assertTrue(any("f1" in c for c in col_names), "F1-Score column missing")
        self.assertTrue(any("support" in c for c in col_names), "Support column missing")

        # Total support across classes should match test split size (465 images)
        class_col = "Class Name" if "Class Name" in df.columns else "class"
        support_col = "Support" if "Support" in df.columns else "support"
        class_rows = df[~df[class_col].isin(["accuracy", "macro avg", "weighted avg"])]
        self.assertEqual(len(class_rows), 28, "Expected 28 class rows in classification report")
        total_support = int(class_rows[support_col].sum())
        self.assertEqual(total_support, 465, f"Expected 465 test samples, found {total_support}")


if __name__ == "__main__":
    unittest.main()
