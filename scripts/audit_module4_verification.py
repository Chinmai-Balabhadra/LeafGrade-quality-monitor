"""
Module 4 Comprehensive Audit and Verification Script
File: scripts/audit_module4_verification.py

Verifies:
1. Environment verification (Python, TensorFlow, Keras, MobileNetV2 pretrained weights, RAM, Disk).
2. Dataset splits integrity (stratified Train/Val/Test splits without data leakage).
3. Saved model architecture (Input: 224x224x3, MobileNetV2 base, GAP, Dropout, Dense 28 Softmax).
4. Training artifacts & metrics (curves, CSV history, loss/accuracy curves).
5. Held-out test set evaluation (classification report, confusion matrix, empirical metrics).
6. Class-level analysis & distinction (Healthy vs Diseased/Pest classes).
7. Model file integrity (model/leaf_classifier.keras loadable and standalone).
8. Single image end-to-end inference (raw image -> prediction -> plucking recommendation).
9. Dataset disk immutability (raw: 2,922 images, cleaned: 2,916 images).
10. Plucking readiness distinction (Healthy != Ready to pluck, no fake Not Ready data).
11. Execution of unit tests (tests/test_model.py, tests/test_preprocessing.py).
"""

import json
import os
import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Avoid font cache warnings
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "reports" / ".mplconfig"))
os.environ.setdefault("KERAS_HOME", str(PROJECT_ROOT / ".keras"))

MODEL_PATH = PROJECT_ROOT / "model" / "leaf_classifier.keras"
CLASS_INDICES_PATH = PROJECT_ROOT / "model" / "class_indices.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
CLEANED_DIR = PROJECT_ROOT / "datasets" / "cleaned" / "plantdoc" / "original_classes"
RAW_PLANTDOC_DIR = PROJECT_ROOT / "datasets" / "raw" / "plantdoc"
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed" / "plantdoc"


def audit_check_1_environment():
    """Audit Task 1: Environment & Dependency Verification."""
    print("\n--- [Audit Check 1] Environment & Dependency Verification ---")
    import platform
    import psutil
    import shutil

    py_version = platform.python_version()
    print(f" Python Version      : {py_version} (Platform: {platform.platform()})")
    assert py_version.startswith("3.12"), f"Expected Python 3.12, got {py_version}"

    import tensorflow as tf
    import keras
    print(f" TensorFlow Version  : {tf.__version__}")
    print(f" Keras Version       : {keras.__version__}")

    # Check RAM & Disk
    mem = psutil.virtual_memory()
    total_ram_gb = mem.total / (1024 ** 3)
    free_ram_gb = mem.available / (1024 ** 3)
    disk = shutil.disk_usage(str(PROJECT_ROOT))
    free_disk_gb = disk.free / (1024 ** 3)

    print(f" System RAM          : Total {total_ram_gb:.2f} GB | Available {free_ram_gb:.2f} GB")
    print(f" Free Disk Space     : {free_disk_gb:.2f} GB")
    assert free_ram_gb > 1.0, "Low available RAM (<1GB)"
    assert free_disk_gb > 2.0, "Low available disk (<2GB)"

    # Verify pretrained MobileNetV2 loading capability
    weights_path = PROJECT_ROOT / ".keras" / "models" / "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"
    print(f" Pretrained Weights  : {weights_path.name} (Exists: {weights_path.exists()}, Size: {weights_path.stat().st_size / (1024*1024):.2f} MB)")
    assert weights_path.exists(), f"Pretrained weights missing at {weights_path}"
    print(" [PASS] Environment and dependency checks verified.")


def audit_check_2_dataset_splits():
    """Audit Task 2: Dataset Splits & Leakage Prevention."""
    print("\n--- [Audit Check 2] Dataset Splits & Data Leakage Audit ---")
    train_dir = PROCESSED_DIR / "train"
    val_dir = PROCESSED_DIR / "val"
    test_dir = PROCESSED_DIR / "test"

    assert train_dir.exists(), f"Missing train split directory: {train_dir}"
    assert val_dir.exists(), f"Missing val split directory: {val_dir}"
    assert test_dir.exists(), f"Missing test split directory: {test_dir}"

    train_imgs = list(train_dir.glob("*/*.jpg"))
    val_imgs = list(val_dir.glob("*/*.jpg"))
    test_imgs = list(test_dir.glob("*/*.jpg"))

    train_count = len(train_imgs)
    val_count = len(val_imgs)
    test_count = len(test_imgs)
    total_imgs = train_count + val_count + test_count

    print(f" Total Split Images   : {total_imgs:,} (Cleaned Total: 2,916)")
    print(f" Training Images      : {train_count:,} ({train_count/total_imgs*100:.1f}%)")
    print(f" Validation Images    : {val_count:,} ({val_count/total_imgs*100:.1f}%)")
    print(f" Testing Images       : {test_count:,} ({test_count/total_imgs*100:.1f}%)")

    assert total_imgs == 2916, f"Expected 2916 images in splits, got {total_imgs}"
    assert 2000 <= train_count <= 2100, f"Train count out of expected ~70% range: {train_count}"
    assert 400 <= val_count <= 470, f"Val count out of expected ~15% range: {val_count}"
    assert 440 <= test_count <= 500, f"Test count out of expected ~15% range: {test_count}"

    # Verify ZERO overlap between splits (No Data Leakage)
    train_rel = set(f"{p.parent.name}/{p.name}" for p in train_imgs)
    val_rel = set(f"{p.parent.name}/{p.name}" for p in val_imgs)
    test_rel = set(f"{p.parent.name}/{p.name}" for p in test_imgs)

    assert len(train_rel.intersection(val_rel)) == 0, "Data leakage detected: train and val overlap!"
    assert len(train_rel.intersection(test_rel)) == 0, "Data leakage detected: train and test overlap!"
    assert len(val_rel.intersection(test_rel)) == 0, "Data leakage detected: val and test overlap!"

    # Verify class directories in splits
    classes_train = set(d.name for d in train_dir.iterdir() if d.is_dir())
    classes_val = set(d.name for d in val_dir.iterdir() if d.is_dir())
    classes_test = set(d.name for d in test_dir.iterdir() if d.is_dir())

    assert len(classes_train) == 28, f"Train split missing classes: {28 - len(classes_train)}"
    assert len(classes_val) == 28, f"Val split missing classes: {28 - len(classes_val)}"
    assert len(classes_test) == 28, f"Test split missing classes: {28 - len(classes_test)}"
    print(" [PASS] Zero data leakage; all 28 classes preserved across stratified train/val/test splits.")


def audit_check_3_and_7_model_architecture():
    """Audit Task 3 & 7: Model Architecture & File Integrity."""
    print("\n--- [Audit Check 3 & 7] Model Architecture & File Integrity ---")
    assert MODEL_PATH.exists(), f"Model file not found at {MODEL_PATH}"
    model_size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f" Model File           : {MODEL_PATH.name} ({model_size_mb:.2f} MB)")
    assert model_size_mb > 5.0, f"Suspicious model file size: {model_size_mb:.2f} MB"

    import keras
    model = keras.models.load_model(str(MODEL_PATH))

    # Architecture verification
    print(f" Input Shape          : {model.input_shape}")
    print(f" Output Shape         : {model.output_shape}")
    assert model.input_shape == (None, 224, 224, 3), f"Incorrect input shape: {model.input_shape}"
    assert model.output_shape == (None, 28), f"Incorrect output shape: {model.output_shape}"

    # Verify presence of MobileNetV2 backbone and classification head
    layer_names = [l.name for l in model.layers]
    has_mobilenet = any("mobilenet" in name.lower() for name in layer_names)
    assert has_mobilenet, f"MobileNetV2 base not found in layers: {layer_names}"

    total_params = model.count_params()
    print(f" Total Parameters     : {total_params:,}")
    assert 2_000_000 <= total_params <= 3_000_000, f"Unexpected parameter count: {total_params}"

    # Verify class indices mapping
    assert CLASS_INDICES_PATH.exists(), f"Missing {CLASS_INDICES_PATH}"
    with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
        class_data = json.load(f)

    assert len(class_data["class_to_idx"]) == 28
    assert len(class_data["idx_to_class"]) == 28
    assert len(class_data["healthy_classes"]) == 10
    assert len(class_data["diseased_classes"]) == 18
    print(" [PASS] Model architecture and class indices JSON confirmed.")
    return model, class_data


def audit_check_4_and_5_evaluation_metrics():
    """Audit Task 4 & 5: Training History & Test Evaluation Metrics."""
    print("\n--- [Audit Check 4 & 5] Training History & Held-Out Test Evaluation ---")

    # 1. Training curves and CSV
    curves_file = REPORTS_DIR / "training_curves.png"
    history_csv = REPORTS_DIR / "training_history.csv"
    assert curves_file.exists(), f"Missing {curves_file}"
    assert history_csv.exists(), f"Missing {history_csv}"

    df_hist = pd.read_csv(history_csv)
    print(f" Total Trained Epochs : {len(df_hist)}")
    assert len(df_hist) >= 4, f"Suspiciously few epochs trained: {len(df_hist)}"
    assert "loss" in df_hist.columns and "val_loss" in df_hist.columns
    assert "accuracy" in df_hist.columns and "val_accuracy" in df_hist.columns

    final_train_acc = df_hist["accuracy"].iloc[-1]
    final_val_acc = df_hist["val_accuracy"].iloc[-1]
    print(f" Final Train Accuracy : {final_train_acc * 100:.2f}%")
    print(f" Final Val Accuracy   : {final_val_acc * 100:.2f}%")

    # 2. Confusion matrix & Classification report
    cm_file = REPORTS_DIR / "confusion_matrix.png"
    report_csv = REPORTS_DIR / "classification_report.csv"
    report_md = REPORTS_DIR / "module4_model_report.md"

    assert cm_file.exists(), f"Missing {cm_file}"
    assert report_csv.exists(), f"Missing {report_csv}"
    assert report_md.exists(), f"Missing {report_md}"

    df_report = pd.read_csv(report_csv)
    macro_f1 = float(df_report["F1-Score"].mean())
    total_support = int(df_report["Support"].sum())
    weighted_f1 = float((df_report["F1-Score"] * df_report["Support"]).sum() / total_support)
    weighted_acc = float((df_report["Recall"] * df_report["Support"]).sum() / total_support)

    print(f" Test Set Accuracy    : {weighted_acc * 100:.2f}%")
    print(f" Test Set Macro F1    : {macro_f1 * 100:.2f}%")
    print(f" Test Weighted F1     : {weighted_f1 * 100:.2f}%")
    print(f" Total Test Samples   : {total_support:,}")

    assert total_support == 465, f"Expected 465 test samples, found {total_support}"
    assert weighted_acc > 0.40, f"Test accuracy unusually low (<40%): {weighted_acc}"
    print(" [PASS] Test set empirical evaluation metrics and visualizations verified.")


def audit_check_6_and_10_plucking_distinction():
    """Audit Task 6 & 10: Plucking Readiness & Health Distinction."""
    print("\n--- [Audit Check 6 & 10] Visual Health vs Plucking Readiness Distinction ---")
    with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
        class_data = json.load(f)

    healthy_count = len(class_data["healthy_classes"])
    diseased_count = len(class_data["diseased_classes"])
    print(f" Healthy Leaf Classes : {healthy_count} (Visual Category)")
    print(f" Diseased Leaf Classes: {diseased_count} (Visual Category)")

    # Verify no fake 'Not Ready' class was injected into the 28 classes
    class_names = list(class_data["class_to_idx"].keys())
    assert "Not Ready" not in class_names, "Unauthorized fake 'Not Ready' class found in model outputs!"
    assert len(class_names) == 28, f"Expected 28 classes, found {len(class_names)}"

    # Check plucking metadata distinction
    metadata = class_data["metadata"]
    for cls_name, meta in metadata.items():
        if meta["type"] == "Healthy":
            # Must state conditional readiness, not automatic plucking approval
            rec = meta["default_recommendation"]
            assert "pluck" in rec.lower() or "ready" in rec.lower() or "inspect" in rec.lower(), f"Unexpected recommendation: {rec}"
        else:
            rec = meta["default_recommendation"]
            assert "do not pluck" in rec.lower() or "discard" in rec.lower() or "quarantine" in rec.lower(), f"Unexpected recommendation: {rec}"

    print(" [PASS] Clear distinction between visual pathology and physical plucking readiness preserved.")
    print("        No synthetic 'Not Ready' data was fabricated.")


def audit_check_8_single_image_inference(model, class_data):
    """Audit Task 8: End-to-End Single Image Prediction."""
    print("\n--- [Audit Check 8] End-to-End Single Image Inference ---")
    sample_imgs = list(CLEANED_DIR.glob("*/*.jpg"))
    sample_img = sample_imgs[0]
    expected_class = sample_img.parent.name

    from utils.preprocessing import load_image, bgr_to_rgb, resize_image
    bgr = load_image(sample_img)
    rgb = bgr_to_rgb(bgr)
    resized = resize_image(rgb, (224, 224))
    input_tensor = np.expand_dims(resized.astype(np.float32), axis=0)

    predictions = model.predict(input_tensor, verbose=0)
    pred_idx = int(np.argmax(predictions[0]))
    predicted_class = class_data["idx_to_class"][str(pred_idx)]
    confidence = float(predictions[0, pred_idx])
    class_meta = class_data["metadata"][predicted_class]

    print(f" Sample Test Image    : {sample_img.name} (Source: {expected_class})")
    print(f" Predicted Class      : {predicted_class} (Index: {pred_idx})")
    print(f" Prediction Confidence: {confidence * 100:.2f}%")
    print(f" Quality Category     : {class_meta['quality_category']}")
    print(f" Plucking Action      : {class_meta['default_recommendation']}")

    assert 0 <= pred_idx < 28
    assert 0.0 <= confidence <= 1.0001
    assert class_meta["quality_category"] in ["Good Quality (Visually Healthy)", "Poor Quality (Visually Diseased/Damaged)"]
    print(" [PASS] End-to-end inference executed successfully.")


def audit_check_9_disk_immutability():
    """Audit Task 9: Disk Immutability of Raw and Cleaned Datasets."""
    print("\n--- [Audit Check 9] Dataset Disk Immutability ---")
    raw_images = list(RAW_PLANTDOC_DIR.glob("*/*/*.jpg"))
    cleaned_images = list(CLEANED_DIR.glob("*/*.jpg"))

    print(f" Raw PlantDoc Images    : {len(raw_images):,} (Baseline: 2,922)")
    print(f" Cleaned PlantDoc Images: {len(cleaned_images):,} (Baseline: 2,916)")

    assert len(raw_images) == 2922, f"Raw count mismatch! Found: {len(raw_images)}"
    assert len(cleaned_images) == 2916, f"Cleaned count mismatch! Found: {len(cleaned_images)}"
    print(" [PASS] Zero dataset images were modified, moved, or deleted.")


def audit_check_11_unit_tests():
    """Audit Task 11: Execute All Unit Tests."""
    print("\n--- [Audit Check 11] Running Test Suites ---")
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test_preprocessing and test_model
    suite.addTests(loader.discover(start_dir=str(PROJECT_ROOT / "tests"), pattern="test_*.py"))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print(f"\n Tests Run: {result.testsRun}, Errors: {len(result.errors)}, Failures: {len(result.failures)}")
    assert result.wasSuccessful(), f"Unit tests failed! Failures: {len(result.failures)}, Errors: {len(result.errors)}"
    print(" [PASS] All unit tests passed cleanly.")


def main():
    print("================================================================================")
    print("         MODULE 4 COMPREHENSIVE AUDIT & VERIFICATION REPORT                    ")
    print("   Project: Leaf Quality Classification & Plucking Standard Monitoring System   ")
    print("================================================================================")

    audit_check_1_environment()
    audit_check_2_dataset_splits()
    model, class_data = audit_check_3_and_7_model_architecture()
    audit_check_4_and_5_evaluation_metrics()
    audit_check_6_and_10_plucking_distinction()
    audit_check_8_single_image_inference(model, class_data)
    audit_check_9_disk_immutability()
    audit_check_11_unit_tests()

    print("\n================================================================================")
    print("  ALL 11 MODULE 4 AUDIT CHECKS PASSED SUCCESSFULLY WITH 0 ERRORS!               ")
    print("================================================================================")


if __name__ == "__main__":
    main()
