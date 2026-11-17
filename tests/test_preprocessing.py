"""
Unit Tests for Module 3: Image Preprocessing Utilities
File: tests/test_preprocessing.py
"""

import unittest
from pathlib import Path
import numpy as np

from utils.preprocessing import (
    load_image,
    bgr_to_rgb,
    resize_image,
    mobilenet_v2_preprocess,
    denormalize_for_display,
    apply_training_augmentation,
    preprocess_single_image,
    preprocess_batch,
    TARGET_IMAGE_SIZE,
    HAS_TENSORFLOW
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEANED_DIR = PROJECT_ROOT / "datasets" / "cleaned" / "plantdoc" / "original_classes"


class TestImagePreprocessing(unittest.TestCase):
    """Test suite covering preprocessing functions for MobileNetV2."""

    @classmethod
    def setUpClass(cls):
        # Locate a valid test image from cleaned dataset
        all_imgs = list(CLEANED_DIR.glob("*/*.jpg"))
        cls.has_real_data = len(all_imgs) > 0
        if cls.has_real_data:
            cls.sample_path = all_imgs[0]
        else:
            cls.sample_path = None

        # Create synthetic test array (100x150x3 uint8)
        rng = np.random.RandomState(42)
        cls.synthetic_bgr = rng.randint(0, 256, (100, 150, 3), dtype=np.uint8)

    def test_load_image_invalid_path(self):
        """Verifies safe handling of non-existent files."""
        result = load_image("non_existent_image_12345.jpg")
        self.assertIsNone(result)

    def test_load_image_real_file(self):
        """Verifies loading real dataset images."""
        if not self.has_real_data:
            self.skipTest("No cleaned dataset images found")
        img = load_image(self.sample_path)
        self.assertIsNotNone(img)
        self.assertEqual(len(img.shape), 3)
        self.assertEqual(img.shape[2], 3)
        self.assertEqual(img.dtype, np.uint8)

    def test_bgr_to_rgb(self):
        """Verifies BGR to RGB color conversion."""
        bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        bgr[:, :, 0] = 255  # Blue channel in BGR
        rgb = bgr_to_rgb(bgr)
        # Blue should now be at index 2
        self.assertEqual(rgb[0, 0, 0], 0)
        self.assertEqual(rgb[0, 0, 2], 255)

    def test_resize_image(self):
        """Verifies resizing dimensions match MobileNetV2 standard."""
        resized = resize_image(self.synthetic_bgr, target_size=(224, 224))
        self.assertEqual(resized.shape, (224, 224, 3))
        self.assertEqual(resized.dtype, np.uint8)

    def test_mobilenet_v2_preprocess_range(self):
        """Verifies pixel normalization maps strictly to [-1.0, 1.0]."""
        dummy_rgb = np.array([[[0, 127, 255]]], dtype=np.uint8)
        norm = mobilenet_v2_preprocess(dummy_rgb)

        # 0 maps to -1.0
        self.assertAlmostEqual(norm[0, 0, 0], -1.0, places=3)
        # 127.5 maps to 0.0 -> 127 maps to -0.00392
        self.assertTrue(-0.01 <= norm[0, 0, 1] <= 0.01)
        # 255 maps to +1.0
        self.assertAlmostEqual(norm[0, 0, 2], 1.0, places=3)
        self.assertEqual(norm.dtype, np.float32)

    def test_denormalize_for_display(self):
        """Verifies inverse transformation restores [0, 255] uint8 array."""
        norm_tensor = np.array([[[-1.0, 0.0, 1.0]]], dtype=np.float32)
        disp = denormalize_for_display(norm_tensor)
        self.assertEqual(disp.dtype, np.uint8)
        self.assertEqual(disp[0, 0, 0], 0)
        self.assertTrue(126 <= disp[0, 0, 1] <= 128)
        self.assertEqual(disp[0, 0, 2], 255)

    def test_apply_training_augmentation(self):
        """Verifies training augmentation preserves shape and dtype."""
        aug = apply_training_augmentation(self.synthetic_bgr)
        self.assertEqual(aug.shape, self.synthetic_bgr.shape)
        self.assertEqual(aug.dtype, np.uint8)

    def test_preprocess_single_image(self):
        """Verifies end-to-end single image processing."""
        if not self.has_real_data:
            self.skipTest("No cleaned dataset images found")

        tensor, meta = preprocess_single_image(self.sample_path, is_training=False)
        self.assertIsNotNone(tensor)
        self.assertEqual(meta["status"], "success")
        self.assertEqual(tensor.shape, (224, 224, 3))
        self.assertTrue(-1.0 <= meta["min_val"] <= 1.0)
        self.assertTrue(-1.0 <= meta["max_val"] <= 1.0)
        self.assertTrue(meta["min_val"] <= meta["max_val"])

    def test_preprocess_batch(self):
        """Verifies batch tensor generation across multiple images."""
        if not self.has_real_data:
            self.skipTest("No cleaned dataset images found")

        paths = [self.sample_path, self.sample_path]
        batch, metas = preprocess_batch(paths, is_training=False)
        self.assertEqual(batch.shape, (2, 224, 224, 3))
        self.assertEqual(batch.dtype, np.float32)
        self.assertEqual(len(metas), 2)


if __name__ == "__main__":
    unittest.main()
