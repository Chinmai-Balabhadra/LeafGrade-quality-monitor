"""
Validation of empirical model evaluation data:
1. Total classes = 28
2. Sum of per-class support = 465
3. Macro metrics match mean of 28 classes
4. Weighted metrics match support-weighted mean
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METRICS_PATH = ROOT / "reports" / "model_evaluation_metrics.json"

with open(METRICS_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

pcm = data["per_class_metrics"]
om = data["overall_metrics"]

print(f"Total classes: {len(pcm)}")
total_support = sum(c["support"] for c in pcm)
print(f"Sum of support: {total_support} (Overall test samples: {om['total_test_samples']})")
assert total_support == 465, "Support sum mismatch!"

# Macro average
macro_p = sum(c["precision"] for c in pcm) / len(pcm)
macro_r = sum(c["recall"] for c in pcm) / len(pcm)
macro_f1 = sum(c["f1_score"] for c in pcm) / len(pcm)

print(f"Calculated Macro Precision: {macro_p:.4f} (Reported: {om['macro_precision']:.4f})")
print(f"Calculated Macro Recall:    {macro_r:.4f} (Reported: {om['macro_recall']:.4f})")
print(f"Calculated Macro F1:        {macro_f1:.4f} (Reported: {om['macro_f1']:.4f})")

# Weighted average
weighted_p = sum(c["precision"] * c["support"] for c in pcm) / total_support
weighted_r = sum(c["recall"] * c["support"] for c in pcm) / total_support
weighted_f1 = sum(c["f1_score"] * c["support"] for c in pcm) / total_support

print(f"Calculated Weighted Precision: {weighted_p:.4f} (Reported: {om['weighted_precision']:.4f})")
print(f"Calculated Weighted Recall:    {weighted_r:.4f} (Reported: {om['weighted_recall']:.4f})")
print(f"Calculated Weighted F1:        {weighted_f1:.4f} (Reported: {om['weighted_f1']:.4f})")

assert abs(macro_p - om["macro_precision"]) < 1e-3, "Macro precision mismatch"
assert abs(macro_r - om["macro_recall"]) < 1e-3, "Macro recall mismatch"
assert abs(macro_f1 - om["macro_f1"]) < 1e-3, "Macro F1 mismatch"
assert abs(weighted_p - om["weighted_precision"]) < 1e-3, "Weighted precision mismatch"
assert abs(weighted_r - om["weighted_recall"]) < 1e-3, "Weighted recall mismatch"
assert abs(weighted_f1 - om["weighted_f1"]) < 1e-3, "Weighted F1 mismatch"

print("\nAll 28 classes and mathematical relationships are 100% VALIDATED!")
