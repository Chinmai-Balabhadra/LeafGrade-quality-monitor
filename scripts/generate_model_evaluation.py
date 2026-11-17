"""
Comprehensive evaluation script for MobileNetV2 on the held-out PlantDoc test set (465 images).
Computes exact ground-truth metrics using bilinear interpolation (cv2.INTER_LINEAR).
Exports:
- reports/model_evaluation_metrics.json
- reports/classification_report.csv
- reports/confusion_matrix.png
"""
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

mpl_dir = PROJECT_ROOT / "reports" / ".mplconfig"
mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(mpl_dir)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import keras

from utils.preprocessing import bgr_to_rgb, resize_image, TARGET_IMAGE_SIZE

def run_evaluation():
    print("=" * 80)
    print("STARTING EMPIRICAL MODEL EVALUATION ON HELD-OUT TEST SET (465 IMAGES)")
    print("=" * 80)

    test_dir = PROJECT_ROOT / "datasets" / "processed" / "plantdoc" / "test"
    train_dir = PROJECT_ROOT / "datasets" / "processed" / "plantdoc" / "train"
    val_dir = PROJECT_ROOT / "datasets" / "processed" / "plantdoc" / "val"
    model_path = PROJECT_ROOT / "model" / "leaf_classifier.keras"
    indices_path = PROJECT_ROOT / "model" / "class_indices.json"
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    with open(indices_path, "r", encoding="utf-8") as f:
        class_data = json.load(f)

    class_to_idx = class_data["class_to_idx"]
    idx_to_class = {int(k): v for k, v in class_data["idx_to_class"].items()}
    num_classes = len(class_to_idx)
    assert num_classes == 28, f"Expected 28 classes, found {num_classes}"

    # Load model
    print(f"[*] Loading trained MobileNetV2 model: {model_path}...")
    model = keras.models.load_model(str(model_path), compile=False)

    # Collect all test images deterministically
    print(f"[*] Loading and preprocessing 465 test images using bilinear interpolation...")
    y_true_indices = []
    y_pred_probs = []
    file_records = []

    # Count dataset splits for dashboard
    split_counts = {
        "train": sum(len(list((train_dir / cls).glob("*.jpg"))) for cls in class_to_idx),
        "val": sum(len(list((val_dir / cls).glob("*.jpg"))) for cls in class_to_idx),
        "test": sum(len(list((test_dir / cls).glob("*.jpg"))) for cls in class_to_idx),
    }
    split_counts["total"] = split_counts["train"] + split_counts["val"] + split_counts["test"]
    print(f"      Dataset distribution: Train={split_counts['train']}, Val={split_counts['val']}, Test={split_counts['test']}, Total={split_counts['total']}")

    t0 = time.time()
    for cls_name, cls_idx in sorted(class_to_idx.items(), key=lambda x: x[1]):
        cls_folder = test_dir / cls_name
        cls_files = sorted(list(cls_folder.glob("*.jpg")))
        for img_p in cls_files:
            bgr = cv2.imread(str(img_p))
            if bgr is None:
                continue
            rgb = bgr_to_rgb(bgr)
            # Consistent bilinear resizing
            resized = resize_image(rgb, target_size=TARGET_IMAGE_SIZE, interpolation=cv2.INTER_LINEAR).astype(np.float32)
            inp = np.expand_dims(resized, axis=0)
            
            probs = model.predict(inp, verbose=0)[0]
            # Defensively normalize probabilities
            prob_sum = float(np.sum(probs))
            if prob_sum > 0:
                probs = probs / prob_sum

            y_true_indices.append(cls_idx)
            y_pred_probs.append(probs)
            pred_idx = int(np.argmax(probs))
            file_records.append({
                "path": str(img_p),
                "name": img_p.name,
                "true_class": cls_name,
                "true_idx": cls_idx,
                "pred_class": idx_to_class[pred_idx],
                "pred_idx": pred_idx,
                "confidence": float(probs[pred_idx]),
                "correct": (pred_idx == cls_idx)
            })

    elapsed = time.time() - t0
    y_true = np.array(y_true_indices, dtype=np.int32)
    y_pred_probs = np.array(y_pred_probs, dtype=np.float32)
    y_pred = np.argmax(y_pred_probs, axis=1)
    n_samples = len(y_true)
    print(f"[*] Processed {n_samples} test samples in {elapsed:.2f}s ({elapsed/n_samples*1000:.1f}ms/img).")
    assert n_samples == 465, f"Expected 465 test samples, found {n_samples}"

    # 1. Compute 28x28 Confusion Matrix
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    # 2. Compute Overall Cross-Entropy Test Loss
    epsilon = 1e-15
    clipped_probs = np.clip(y_pred_probs, epsilon, 1.0 - epsilon)
    # Categorical crossentropy loss: -sum(log(p_true)) / N
    sample_losses = -np.log(clipped_probs[np.arange(n_samples), y_true])
    test_loss = float(np.mean(sample_losses))

    # 3. Compute Per-Class Metrics (Precision, Recall, F1, Support)
    per_class_metrics = []
    for i in range(num_classes):
        cls_name = idx_to_class[i]
        support = int(np.sum(cm[i, :]))
        tp = int(cm[i, i])
        fp = int(np.sum(cm[:, i]) - tp)
        fn = int(np.sum(cm[i, :]) - tp)

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / support) if support > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        meta = class_data.get("metadata", {}).get(cls_name, {})
        per_class_metrics.append({
            "class_index": i,
            "class_name": cls_name,
            "species": meta.get("species", "Unknown"),
            "condition": meta.get("condition", "Unknown"),
            "visual_health": meta.get("type", "Unknown"),
            "quality_category": meta.get("quality_category", "Unknown"),
            "support": support,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "precision_pct": round(precision * 100.0, 2),
            "recall_pct": round(recall * 100.0, 2),
            "f1_pct": round(f1 * 100.0, 2)
        })

    # 4. Overall & Macro & Weighted Metrics
    total_tp = int(np.trace(cm))
    overall_accuracy = float(total_tp / n_samples)

    macro_precision = float(np.mean([m["precision"] for m in per_class_metrics]))
    macro_recall = float(np.mean([m["recall"] for m in per_class_metrics]))
    macro_f1 = float(np.mean([m["f1_score"] for m in per_class_metrics]))

    total_support = sum(m["support"] for m in per_class_metrics)
    weighted_precision = float(sum(m["precision"] * m["support"] for m in per_class_metrics) / total_support)
    weighted_recall = float(sum(m["recall"] * m["support"] for m in per_class_metrics) / total_support)
    weighted_f1 = float(sum(m["f1_score"] * m["support"] for m in per_class_metrics) / total_support)

    print("\n" + "=" * 80)
    print("EMPIRICAL TEST METRICS SUMMARY")
    print("=" * 80)
    print(f"  Overall Test Accuracy : {overall_accuracy * 100:.2f}% ({total_tp}/{n_samples})")
    print(f"  Overall Test Loss     : {test_loss:.4f}")
    print(f"  Macro Precision       : {macro_precision * 100:.2f}%")
    print(f"  Macro Recall          : {macro_recall * 100:.2f}%")
    print(f"  Macro F1-Score        : {macro_f1 * 100:.2f}%")
    print(f"  Weighted Precision    : {weighted_precision * 100:.2f}%")
    print(f"  Weighted Recall       : {weighted_recall * 100:.2f}%")
    print(f"  Weighted F1-Score     : {weighted_f1 * 100:.2f}%")

    apple_scab_metrics = per_class_metrics[class_to_idx["Apple_Scab_Leaf"]]
    print(f"\n  Apple_Scab_Leaf : Precision={apple_scab_metrics['precision_pct']}%, Recall={apple_scab_metrics['recall_pct']}%, F1={apple_scab_metrics['f1_pct']}% (Support: {apple_scab_metrics['support']})")

    # 5. Export Updated reports/classification_report.csv
    csv_rows = []
    for m in per_class_metrics:
        csv_rows.append({
            "Class Name": m["class_name"],
            "Support": m["support"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1-Score": m["f1_score"]
        })
    csv_df = pd.DataFrame(csv_rows)
    csv_path = reports_dir / "classification_report.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"[*] Exported updated: {csv_path}")

    # 6. Generate High-Resolution 28x28 Confusion Matrix Plot
    print("[*] Generating high-resolution 28x28 Confusion Matrix heatmap...")
    plt.figure(figsize=(18, 16))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("MobileNetV2 Confusion Matrix (PlantDoc 28 Classes, Held-Out Test Set)", fontsize=16, pad=20)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(num_classes)
    class_labels = [idx_to_class[i].replace("_", " ") for i in range(num_classes)]
    plt.xticks(tick_marks, class_labels, rotation=90, fontsize=9)
    plt.yticks(tick_marks, class_labels, fontsize=9)

    # Add numeric text to each cell
    thresh = cm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            val = cm[i, j]
            if val > 0:
                plt.text(j, i, format(val, "d"),
                         ha="center", va="center",
                         color="white" if val > thresh else "black",
                         fontsize=7, fontweight="bold")

    plt.ylabel("True Class Label", fontsize=12, labelpad=10)
    plt.xlabel("Predicted Class Label", fontsize=12, labelpad=10)
    plt.tight_layout()
    cm_plot_path = reports_dir / "confusion_matrix.png"
    plt.savefig(cm_plot_path, dpi=200)
    plt.close()
    print(f"[*] Exported updated: {cm_plot_path}")

    # 7. Export Full Structured JSON for Frontend Dashboard
    eval_payload = {
        "status": "SUCCESS",
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "model_info": {
            "name": "MobileNetV2 Transfer Learning",
            "weights": "Pretrained ImageNet Base (Frozen Feature Extractor)",
            "fine_tuning": "Top 34 Layers Fine-Tuned (Learning Rate: 1e-4)",
            "input_resolution": "224 x 224 x 3",
            "preprocessing": "Bilinear Rescaling (INTER_LINEAR -> [-1.0, 1.0])",
            "total_parameters": int(model.count_params()),
            "classes_supported": num_classes,
            "architecture": "MobileNetV2 + GlobalAveragePooling2D + Dropout(0.2) + Dense(28, Softmax)"
        },
        "dataset_info": {
            "name": "PlantDoc (Cleaned & Stratified)",
            "total_images": split_counts["total"],
            "training_images": split_counts["train"],
            "validation_images": split_counts["val"],
            "test_images": split_counts["test"],
            "healthy_classes": len(class_data.get("healthy_classes", [])),
            "diseased_classes": len(class_data.get("diseased_classes", []))
        },
        "overall_metrics": {
            "accuracy": round(overall_accuracy, 4),
            "accuracy_percent": round(overall_accuracy * 100.0, 2),
            "loss": round(test_loss, 4),
            "total_test_samples": n_samples,
            "correct_predictions": total_tp,
            "incorrect_predictions": n_samples - total_tp,
            "macro_precision": round(macro_precision, 4),
            "macro_precision_percent": round(macro_precision * 100.0, 2),
            "macro_recall": round(macro_recall, 4),
            "macro_recall_percent": round(macro_recall * 100.0, 2),
            "macro_f1": round(macro_f1, 4),
            "macro_f1_percent": round(macro_f1 * 100.0, 2),
            "weighted_precision": round(weighted_precision, 4),
            "weighted_precision_percent": round(weighted_precision * 100.0, 2),
            "weighted_recall": round(weighted_recall, 4),
            "weighted_recall_percent": round(weighted_recall * 100.0, 2),
            "weighted_f1": round(weighted_f1, 4),
            "weighted_f1_percent": round(weighted_f1 * 100.0, 2)
        },
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "class_labels": [idx_to_class[i] for i in range(num_classes)]
    }

    json_path = reports_dir / "model_evaluation_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(eval_payload, f, indent=4)
    print(f"[*] Exported structured JSON: {json_path}")
    print("\n" + "=" * 80)
    print("MODEL EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    run_evaluation()
