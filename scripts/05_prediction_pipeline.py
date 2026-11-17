"""
Leaf Quality Classification and Plucking Standard Monitoring System
Module 5: Prediction Pipeline Demonstration, Testing & Visualization
File: scripts/05_prediction_pipeline.py

Description:
    Demonstrates, tests, and visualizes the end-to-end single-image prediction pipeline:
    1. Tests on held-out test images across diverse agricultural crop species.
    2. Tests healthy foliage vs diseased foliage.
    3. Tests confidence thresholding behavior and uncertainty handling.
    4. Tests image quality and blur detection (Laplacian variance).
    5. Tests edge cases and error handling (unsupported formats, empty bytes, corrupt files).
    6. Generates publication-quality diagnostic visualization figures saved in reports/prediction_samples/.
    7. Exports test results summary to reports/prediction_test_results.json.
"""

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Avoid Matplotlib font cache warnings
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "reports" / ".mplconfig"))
os.environ.setdefault("KERAS_HOME", str(PROJECT_ROOT / ".keras"))

from utils.predictor import LeafPredictor, get_default_predictor

TEST_SPLIT_DIR = PROJECT_ROOT / "datasets" / "processed" / "plantdoc" / "test"
REPORTS_DIR = PROJECT_ROOT / "reports"
SAMPLES_DIR = REPORTS_DIR / "prediction_samples"
SCRATCH_DIR = REPORTS_DIR / ".temp_test_inputs"


def setup_directories() -> None:
    """Creates required output directories."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)


def plot_prediction_sample(
    image_bgr: np.ndarray,
    prediction_result: Dict[str, Any],
    true_class: str,
    output_filename: str,
    case_title: str
) -> Path:
    """
    Renders and saves a diagnostic visualization containing:
    - Input image with true and predicted labels
    - Horizontal bar chart of Top-3 predictions with confidence
    - Agronomic plucking recommendation and quality badge
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1, 1.2]})

    # 1. Left Panel: Input Image
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    axes[0].imshow(img_rgb)
    axes[0].axis("off")

    status = prediction_result.get("status", "UNKNOWN")
    status_color = "#2ca02c" if status == "SUCCESS" else ("#ff7f0e" if "UNCERTAIN" in status else "#d62728")
    axes[0].set_title(
        f"True Class: {true_class}\nSource: {prediction_result.get('source', 'Unknown')}",
        fontsize=11, fontweight="bold", pad=8
    )

    # 2. Right Panel: Top-3 Confidence Distribution
    top_three = prediction_result.get("top_three", [])
    if top_three:
        candidates = [c["class_name"] for c in reversed(top_three)]
        confs = [c["confidence_percent"] for c in reversed(top_three)]
        types = [c.get("visual_health_type", "Unknown") for c in reversed(top_three)]

        bar_colors = ["#2ca02c" if t == "Healthy" else "#d62728" for t in types]

        bars = axes[1].barh(candidates, confs, color=bar_colors, height=0.55, edgecolor="black", alpha=0.85)
        axes[1].set_xlim(0, 105)
        axes[1].set_xlabel("Confidence (%)", fontsize=10, fontweight="bold")
        axes[1].set_title(f"Top Candidate Predictions (Status: {status})", fontsize=11, fontweight="bold", color=status_color)
        axes[1].grid(axis="x", linestyle="--", alpha=0.6)

        # Label bars with confidence values
        for bar in bars:
            width = bar.get_width()
            axes[1].text(
                width + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{width:.1f}%",
                ha="left", va="center", fontsize=9, fontweight="bold"
            )

        # Add threshold reference line
        thresh_pct = prediction_result.get("confidence_threshold_applied", 0.40) * 100.0
        axes[1].axvline(x=thresh_pct, color="gray", linestyle=":", linewidth=1.5, label=f"Certainty Threshold ({thresh_pct:.0f}%)")
        axes[1].legend(loc="lower right", fontsize=8)
    else:
        axes[1].text(0.5, 0.5, "No Predictions Generated (Error State)", ha="center", va="center", color="red")
        axes[1].axis("off")

    # Bottom Agronomic Interpretation Text Box
    category = prediction_result.get("visual_health_category", "N/A")
    rec_text = prediction_result.get("plucking_recommendation", "N/A")
    quality = prediction_result.get("quality_assessment", {})
    lap_var = quality.get("laplacian_variance", "N/A")
    sharpness = quality.get("quality_rating", "N/A")

    footer_text = (
        f"Visual Health Category: {category}  |  Sharpness: {sharpness} (Laplacian Var: {lap_var})\n"
        f"Harvest Guidance: {rec_text}"
    )

    fig.text(
        0.5, 0.02, footer_text,
        ha="center", va="bottom", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#ced4da", linewidth=1.2),
        wrap=True
    )

    plt.suptitle(case_title, fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.10, 1, 0.95])

    save_path = SAMPLES_DIR / output_filename
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    return save_path


def run_pipeline_demonstration() -> Dict[str, Any]:
    """Runs test suite covering real held-out images, synthetic blur, and edge cases."""
    print("=" * 80)
    print(" CSE471: Leaf Quality Monitoring System - Module 5 Prediction Pipeline ")
    print("=" * 80)

    predictor = get_default_predictor()
    print(f"[1/5] Loaded Predictor with model: {predictor.model_path.name}")
    print(f"      Total Classes in Mapping : {len(predictor.idx_to_class)}")
    print(f"      Default Threshold        : {predictor.confidence_threshold * 100:.1f}%\n")

    test_results: List[Dict[str, Any]] = []

    # --------------------------------------------------------------------------
    # Case 1: Healthy Apple Leaf (High Confidence)
    # --------------------------------------------------------------------------
    print("[2/5] Testing standard held-out leaf samples...")
    sample1_path = list((TEST_SPLIT_DIR / "Apple_leaf").glob("*.jpg"))[0]
    res1 = predictor.predict(sample1_path)
    img1_bgr = cv2.imread(str(sample1_path))
    fig1 = plot_prediction_sample(
        img1_bgr, res1, "Apple_leaf (Healthy)",
        "sample1_healthy_apple.png",
        "Case 1: Visually Healthy Foliage (Apple_leaf)"
    )
    print(f"  [Case 1] Healthy Apple Leaf -> Pred: {res1['predicted_class']} ({res1['confidence_percent']}%) | Status: {res1['status']}")
    test_results.append({"case": "Healthy Apple Leaf", "path": str(fig1), "result": res1})

    # --------------------------------------------------------------------------
    # Case 2: Diseased Apple Rust Leaf (High Confidence)
    # --------------------------------------------------------------------------
    sample2_path = list((TEST_SPLIT_DIR / "Apple_rust_leaf").glob("*.jpg"))[0]
    res2 = predictor.predict(sample2_path)
    img2_bgr = cv2.imread(str(sample2_path))
    fig2 = plot_prediction_sample(
        img2_bgr, res2, "Apple_rust_leaf (Diseased)",
        "sample2_diseased_apple_rust.png",
        "Case 2: Diseased Foliage (Apple_rust_leaf)"
    )
    print(f"  [Case 2] Diseased Apple Rust -> Pred: {res2['predicted_class']} ({res2['confidence_percent']}%) | Status: {res2['status']}")
    test_results.append({"case": "Diseased Apple Rust", "path": str(fig2), "result": res2})

    # --------------------------------------------------------------------------
    # Case 3: Diseased Corn Leaf Blight
    # --------------------------------------------------------------------------
    sample3_path = list((TEST_SPLIT_DIR / "Corn_leaf_blight").glob("*.jpg"))[0]
    res3 = predictor.predict(sample3_path)
    img3_bgr = cv2.imread(str(sample3_path))
    fig3 = plot_prediction_sample(
        img3_bgr, res3, "Corn_leaf_blight (Diseased)",
        "sample3_diseased_corn_blight.png",
        "Case 3: Cereal Crop Pathology (Corn_leaf_blight)"
    )
    print(f"  [Case 3] Corn Leaf Blight    -> Pred: {res3['predicted_class']} ({res3['confidence_percent']}%) | Status: {res3['status']}")
    test_results.append({"case": "Corn Leaf Blight", "path": str(fig3), "result": res3})

    # --------------------------------------------------------------------------
    # Case 4: Diseased Tomato Yellow Leaf Curl Virus
    # --------------------------------------------------------------------------
    sample4_path = list((TEST_SPLIT_DIR / "Tomato_leaf_yellow_virus").glob("*.jpg"))[0]
    res4 = predictor.predict(sample4_path)
    img4_bgr = cv2.imread(str(sample4_path))
    fig4 = plot_prediction_sample(
        img4_bgr, res4, "Tomato_leaf_yellow_virus (Diseased)",
        "sample4_diseased_tomato_virus.png",
        "Case 4: Viral Infection (Tomato_leaf_yellow_virus)"
    )
    print(f"  [Case 4] Tomato Yellow Virus -> Pred: {res4['predicted_class']} ({res4['confidence_percent']}%) | Status: {res4['status']}")
    test_results.append({"case": "Tomato Yellow Virus", "path": str(fig4), "result": res4})

    # --------------------------------------------------------------------------
    # Case 5: Low-Confidence / Borderline Leaf (Triggering Uncertainty)
    # --------------------------------------------------------------------------
    print("\n[3/5] Testing confidence thresholding & uncertainty handling...")
    # Test with strict 85% threshold to demonstrate UNCERTAIN_LOW_CONFIDENCE behavior
    res5 = predictor.predict(sample3_path, confidence_threshold=0.85)
    fig5 = plot_prediction_sample(
        img3_bgr, res5, "Corn_leaf_blight (Evaluated at Strict 85% Threshold)",
        "sample5_low_confidence_uncertain.png",
        "Case 5: Ambiguous / Low-Confidence Leaf (Uncertainty Flagged)"
    )
    print(f"  [Case 5] Strict Threshold (85%) -> Status: {res5['status']} (Pred: {res5['predicted_class']} at {res5['confidence_percent']}%)")
    test_results.append({"case": "Low Confidence Ambiguity", "path": str(fig5), "result": res5})

    # --------------------------------------------------------------------------
    # Case 6: Blurred Handheld Simulation (Laplacian Sharpness Warning)
    # --------------------------------------------------------------------------
    print("\n[4/5] Testing motion blur & image degradation...")
    blurred_bgr = cv2.GaussianBlur(img1_bgr, (25, 25), 10)
    blurred_path = SCRATCH_DIR / "synthetic_blurred_leaf.jpg"
    cv2.imwrite(str(blurred_path), blurred_bgr)

    res6 = predictor.predict(blurred_path)
    fig6 = plot_prediction_sample(
        blurred_bgr, res6, "Apple_leaf (Artificially Blurred Handheld Capture)",
        "sample6_blurred_leaf_warning.png",
        "Case 6: Blurred Field Capture (Sharpness Warning Triggered)"
    )
    print(f"  [Case 6] Blurred Leaf        -> Sharpness: {res6['quality_assessment']['quality_rating']} (Var: {res6['quality_assessment']['laplacian_variance']}) | Warning: {res6['blur_warning'] is not None}")
    test_results.append({"case": "Blurred Handheld Image", "path": str(fig6), "result": res6})

    # --------------------------------------------------------------------------
    # Edge Cases: Error Handling
    # --------------------------------------------------------------------------
    print("\n[5/5] Testing input edge cases & defensive error handling...")

    # Edge Case A: Unsupported file extension (.txt)
    txt_path = SCRATCH_DIR / "not_an_image.txt"
    txt_path.write_text("Hello, this is a plain text file, not a leaf image.")
    res_txt = predictor.predict(txt_path)
    print(f"  [Edge A] Unsupported Format (.txt)  -> Status: {res_txt['status']} | Message: {res_txt['message']}")
    test_results.append({"case": "Unsupported Format (.txt)", "result": res_txt})

    # Edge Case B: Zero-byte empty file
    empty_path = SCRATCH_DIR / "empty_corrupt.jpg"
    empty_path.write_bytes(b"")
    res_empty = predictor.predict(empty_path)
    print(f"  [Edge B] Zero-Byte Empty Image File -> Status: {res_empty['status']} | Message: {res_empty['message']}")
    test_results.append({"case": "Zero-Byte File", "result": res_empty})

    # Edge Case C: Non-existent file path
    non_existent = SCRATCH_DIR / "does_not_exist.png"
    res_missing = predictor.predict(non_existent)
    print(f"  [Edge C] Non-Existent File Path     -> Status: {res_missing['status']} | Message: {res_missing['message']}")
    test_results.append({"case": "Non-Existent File", "result": res_missing})

    # Edge Case D: In-memory byte array (Valid)
    with open(sample1_path, "rb") as f:
        valid_bytes = f.read()
    res_bytes = predictor.predict(valid_bytes)
    print(f"  [Edge D] Raw Byte Stream (In-Memory)-> Status: {res_bytes['status']} | Pred: {res_bytes['predicted_class']} ({res_bytes['confidence_percent']}%)")
    test_results.append({"case": "In-Memory Byte Stream", "result": res_bytes})

    # Edge Case E: Empty byte array
    res_empty_bytes = predictor.predict(b"")
    print(f"  [Edge E] Empty Byte Array           -> Status: {res_empty_bytes['status']} | Message: {res_empty_bytes['message']}")
    test_results.append({"case": "Empty Byte Stream", "result": res_empty_bytes})

    # Cleanup temporary scratch inputs
    if SCRATCH_DIR.exists():
        shutil.rmtree(SCRATCH_DIR)

    # Save summary report
    summary_path = REPORTS_DIR / "prediction_test_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=4)
    print(f"\nSaved test results summary to: {summary_path}")
    print(f"Saved 6 diagnostic visualization figures to: {SAMPLES_DIR}")

    return {"summary_path": str(summary_path), "samples_count": len(test_results)}


if __name__ == "__main__":
    setup_directories()
    run_pipeline_demonstration()
