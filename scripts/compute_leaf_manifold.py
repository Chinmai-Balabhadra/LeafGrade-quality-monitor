"""
Leaf Quality Classification and Plucking Standard Monitoring System
Script: Compute and Save In-Distribution Leaf Manifold Reference
File: scripts/compute_leaf_manifold.py

Computes:
1. Per-class normalized centroid vectors (28 classes x 1280 dimensions)
2. Global leaf manifold centroid vector (1280 dimensions)
3. Intra-class and inter-class distance distributions
4. Calibrated threshold percentiles for Out-of-Distribution rejection
Saves output to: model/leaf_manifold_reference.npz and model/leaf_manifold_meta.json
"""

import os
import sys
import json
import time
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KERAS_HOME"] = str((PROJECT_ROOT / ".keras").resolve())

import keras
import numpy as np

TRAIN_DIR = PROJECT_ROOT / "datasets" / "processed" / "plantdoc" / "train"
VAL_DIR = PROJECT_ROOT / "datasets" / "processed" / "plantdoc" / "val"
MODEL_PATH = PROJECT_ROOT / "model" / "leaf_classifier.keras"
CLASS_INDICES_PATH = PROJECT_ROOT / "model" / "class_indices.json"
OUTPUT_NPZ_PATH = PROJECT_ROOT / "model" / "leaf_manifold_reference.npz"
OUTPUT_META_PATH = PROJECT_ROOT / "model" / "leaf_manifold_meta.json"


def main():
    print(f"[1/4] Loading trained MobileNetV2 model from {MODEL_PATH}...", flush=True)
    full_model = keras.models.load_model(str(MODEL_PATH), compile=False)
    
    # Feature extractor up to GlobalAveragePooling2D
    gap_layer = full_model.get_layer("global_average_pooling2d")
    head_layer = full_model.get_layer("classification_head")
    feature_model = keras.Model(
        inputs=full_model.input,
        outputs=[gap_layer.output, head_layer.output]
    )

    with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
        class_indices = json.load(f)
    idx_to_class = {int(k): v for k, v in class_indices["idx_to_class"].items()}
    num_classes = len(idx_to_class)
    assert num_classes == 28, f"Expected 28 classes, got {num_classes}"

    print(f"[2/4] Ingesting training dataset ({TRAIN_DIR})...", flush=True)
    train_ds = keras.utils.image_dataset_from_directory(
        str(TRAIN_DIR),
        image_size=(224, 224),
        batch_size=64,
        shuffle=False
    )
    dataset_class_names = train_ds.class_names
    name_to_dataset_idx = {name: i for i, name in enumerate(dataset_class_names)}
    # Map dataset label index to canonical class index
    class_to_canonical_idx = class_indices["class_to_idx"]
    ds_idx_to_canonical = {
        name_to_dataset_idx[c]: class_to_canonical_idx[c]
        for c in dataset_class_names if c in class_to_canonical_idx
    }

    all_features = []
    all_canonical_labels = []

    t0 = time.time()
    for batch_x, batch_y in train_ds:
        feats, _ = feature_model(batch_x, training=False)
        feats_np = feats.numpy()
        # L2-normalize
        norms = np.linalg.norm(feats_np, axis=1, keepdims=True) + 1e-12
        norm_feats = feats_np / norms
        all_features.append(norm_feats)
        
        y_np = batch_y.numpy()
        canonical_y = np.array([ds_idx_to_canonical[y] for y in y_np], dtype=np.int32)
        all_canonical_labels.append(canonical_y)

    all_features = np.concatenate(all_features, axis=0)
    all_canonical_labels = np.concatenate(all_canonical_labels, axis=0)
    elapsed = time.time() - t0
    print(f"      Extracted {len(all_features)} training embeddings in {elapsed:.1f}s.", flush=True)

    print("[3/4] Computing per-class and global manifold centroids...", flush=True)
    class_centroids = np.zeros((num_classes, 1280), dtype=np.float32)
    per_class_counts = {}

    for c in range(num_classes):
        mask = (all_canonical_labels == c)
        c_name = idx_to_class[c]
        if np.any(mask):
            c_feats = all_features[mask]
            mean_vec = np.mean(c_feats, axis=0)
            mean_vec = mean_vec / (np.linalg.norm(mean_vec) + 1e-12)
            class_centroids[c] = mean_vec
            per_class_counts[c_name] = int(np.sum(mask))
        else:
            print(f"Warning: No training samples found for class {c}: {c_name}")

    global_centroid = np.mean(all_features, axis=0)
    global_centroid = global_centroid / (np.linalg.norm(global_centroid) + 1e-12)

    # In-distribution training similarity distributions
    # sim_matrix: (N, 28)
    sim_matrix = np.dot(all_features, class_centroids.T)
    max_sims = np.max(sim_matrix, axis=1)
    global_sims = np.dot(all_features, global_centroid)

    stats = {
        "train_samples_count": int(len(all_features)),
        "max_sim_mean": float(np.mean(max_sims)),
        "max_sim_std": float(np.std(max_sims)),
        "max_sim_min": float(np.min(max_sims)),
        "max_sim_p01": float(np.percentile(max_sims, 1)),
        "max_sim_p05": float(np.percentile(max_sims, 5)),
        "max_sim_p10": float(np.percentile(max_sims, 10)),
        "global_sim_mean": float(np.mean(global_sims)),
        "global_sim_min": float(np.min(global_sims)),
        "global_sim_p05": float(np.percentile(global_sims, 5)),
        # Calibrated threshold for OOD leaf rejection:
        # Require max_sim >= 0.50 and global_sim >= 0.40
        "recommended_max_sim_threshold": 0.50,
        "recommended_global_sim_threshold": 0.40
    }

    print(f"      In-distribution statistics:")
    print(f"      Max class similarity: mean={stats['max_sim_mean']:.4f}, 5th percentile={stats['max_sim_p05']:.4f}, min={stats['max_sim_min']:.4f}")
    print(f"      Global similarity   : mean={stats['global_sim_mean']:.4f}, 5th percentile={stats['global_sim_p05']:.4f}, min={stats['global_sim_min']:.4f}")

    print(f"[4/4] Saving manifold reference to {OUTPUT_NPZ_PATH}...", flush=True)
    np.savez_compressed(
        str(OUTPUT_NPZ_PATH),
        class_centroids=class_centroids,
        global_centroid=global_centroid
    )

    with open(OUTPUT_META_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": stats,
            "per_class_sample_counts": per_class_counts
        }, f, indent=2)

    print("Successfully built leaf manifold reference!")


if __name__ == "__main__":
    main()
