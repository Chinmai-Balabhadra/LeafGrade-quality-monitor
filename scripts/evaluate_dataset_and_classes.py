"""
Comprehensive diagnostic evaluation of PlantDoc dataset distribution,
Apple Scab predictions, Corn predictions, and class-level confusions.
"""
import os
import sys
import json
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import pandas as pd
import keras

def run_evaluation():
    processed_dir = PROJECT_ROOT / "datasets" / "processed" / "plantdoc"
    cleaned_dir = PROJECT_ROOT / "datasets" / "cleaned" / "plantdoc" / "original_classes"
    model_path = PROJECT_ROOT / "model" / "leaf_classifier.keras"
    indices_path = PROJECT_ROOT / "model" / "class_indices.json"
    
    with open(indices_path, "r", encoding="utf-8") as f:
        class_data = json.load(f)
    
    class_to_idx = class_data["class_to_idx"]
    idx_to_class = {int(k): v for k, v in class_data["idx_to_class"].items()}
    
    # 1. Dataset splits count
    print("=" * 80)
    print("PHASE 5: DATASET DISTRIBUTION & CLASS IMBALANCE")
    print("=" * 80)
    
    stats = []
    for cls_name in sorted(class_to_idx.keys()):
        train_count = len(list((processed_dir / "train" / cls_name).glob("*.jpg")))
        val_count = len(list((processed_dir / "val" / cls_name).glob("*.jpg")))
        test_count = len(list((processed_dir / "test" / cls_name).glob("*.jpg")))
        total_count = train_count + val_count + test_count
        stats.append({
            "Class Name": cls_name,
            "Class Index": class_to_idx[cls_name],
            "Train": train_count,
            "Val": val_count,
            "Test": test_count,
            "Total": total_count
        })
    
    df_dist = pd.DataFrame(stats)
    print(df_dist.to_string(index=False))
    print(f"\nTotal Cleaned Images: {df_dist['Total'].sum()}")
    print(f"Train: {df_dist['Train'].sum()}, Val: {df_dist['Val'].sum()}, Test: {df_dist['Test'].sum()}")
    
    # Apple Scab specific
    apple_scab_row = df_dist[df_dist["Class Name"] == "Apple_Scab_Leaf"].iloc[0]
    print(f"\nApple_Scab_Leaf count: Train={apple_scab_row['Train']}, Val={apple_scab_row['Val']}, Test={apple_scab_row['Test']}, Total={apple_scab_row['Total']}")
    
    # 2. Model Evaluation on all Apple Scab Images
    print("\n" + "=" * 80)
    print("EVALUATION OF ALL APPLE SCAB IMAGES ACROSS SPLITS")
    print("=" * 80)
    
    model = keras.models.load_model(str(model_path), compile=False)
    
    for split in ["test", "val", "train"]:
        split_dir = processed_dir / split / "Apple_Scab_Leaf"
        imgs = list(split_dir.glob("*.jpg"))
        if not imgs:
            continue
            
        print(f"\n--- Split: {split.upper()} (Total: {len(imgs)} images) ---")
        correct = 0
        predictions_counter = defaultdict(int)
        
        sample_details = []
        for img_path in sorted(imgs):
            # Load and preprocess exactly as LeafPredictor does
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32)
            inp = np.expand_dims(resized, axis=0)
            
            probs = model.predict(inp, verbose=0)[0]
            top1_idx = int(np.argmax(probs))
            top1_class = idx_to_class[top1_idx]
            top1_conf = float(probs[top1_idx])
            
            apple_scab_conf = float(probs[class_to_idx["Apple_Scab_Leaf"]])
            
            is_correct = (top1_class == "Apple_Scab_Leaf")
            if is_correct:
                correct += 1
            predictions_counter[top1_class] += 1
            
            top3_indices = np.argsort(probs)[::-1][:3]
            top3_str = ", ".join([f"{idx_to_class[i]}: {probs[i]*100:.1f}%" for i in top3_indices])
            
            sample_details.append({
                "Filename": img_path.name,
                "Predicted": top1_class,
                "Conf": f"{top1_conf*100:.1f}%",
                "AppleScabConf": f"{apple_scab_conf*100:.1f}%",
                "Correct": is_correct,
                "Top3": top3_str
            })
            
        print(f"Accuracy in {split}: {correct}/{len(imgs)} ({correct/len(imgs)*100:.1f}%)")
        print("Predicted distribution for Apple_Scab_Leaf images:")
        for pred_cls, count in sorted(predictions_counter.items(), key=lambda x: -x[1]):
            print(f"  {pred_cls}: {count} ({count/len(imgs)*100:.1f}%)")
            
        if split == "test":
            print("\nDetailed Test Samples for Apple_Scab_Leaf:")
            df_samples = pd.DataFrame(sample_details)
            print(df_samples[["Filename", "Predicted", "Conf", "AppleScabConf", "Correct"]].to_string(index=False))

    # 3. Test corn leaf images to see how they perform
    print("\n" + "=" * 80)
    print("EVALUATION OF CORN CLASSES IN TEST SPLIT")
    print("=" * 80)
    for corn_cls in ["Corn_leaf_blight", "Corn_Gray_leaf_spot", "Corn_rust_leaf"]:
        test_corn_dir = processed_dir / "test" / corn_cls
        imgs = list(test_corn_dir.glob("*.jpg"))
        correct = 0
        for p in imgs:
            bgr = cv2.imread(str(p))
            if bgr is None: continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32)
            inp = np.expand_dims(resized, axis=0)
            probs = model.predict(inp, verbose=0)[0]
            if idx_to_class[int(np.argmax(probs))] == corn_cls:
                correct += 1
        print(f"{corn_cls}: {correct}/{len(imgs)} correct ({correct/len(imgs)*100:.1f}%)")

if __name__ == "__main__":
    run_evaluation()
