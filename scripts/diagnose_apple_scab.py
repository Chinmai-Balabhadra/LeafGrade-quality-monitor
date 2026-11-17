"""
Diagnostic script to investigate test_Apple Scab Leaf_4.jpg and class mapping/preprocessing.
"""
import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import keras
from utils.predictor import LeafPredictor

def run_diagnostics():
    print("=" * 70)
    print("PHASE 1 & 2: VERIFY MODEL & CLASS INDICES")
    print("=" * 70)
    
    model_path = PROJECT_ROOT / "model" / "leaf_classifier.keras"
    indices_path = PROJECT_ROOT / "model" / "class_indices.json"
    
    with open(indices_path, "r", encoding="utf-8") as f:
        indices_data = json.load(f)
    
    class_to_idx = indices_data["class_to_idx"]
    idx_to_class = {int(k): v for k, v in indices_data["idx_to_class"].items()}
    
    print(f"Total classes in indices: {len(class_to_idx)}")
    for i in range(len(idx_to_class)):
        print(f"  Index {i:2d} -> {idx_to_class[i]}")
        
    model = keras.models.load_model(str(model_path), compile=False)
    print(f"\nModel output shape: {model.output_shape}")
    print(f"Model input shape: {model.input_shape}")
    
    # Inspect model layers
    print("\nModel Layers:")
    for idx, layer in enumerate(model.layers):
        print(f"  [{idx}] {layer.name} ({layer.__class__.__name__})")
        
    print("\n" + "=" * 70)
    print("PHASE 3 & 4: TEST ON test_Apple Scab Leaf_4.jpg")
    print("=" * 70)
    
    # Search for image
    img_candidates = list((PROJECT_ROOT / "datasets").rglob("test_Apple Scab Leaf_4.jpg"))
    print(f"Found {len(img_candidates)} matching files for test_Apple Scab Leaf_4.jpg:")
    for p in img_candidates:
        print(f"  - {p}")
        
    if not img_candidates:
        print("ERROR: test_Apple Scab Leaf_4.jpg not found!")
        return
        
    test_img_path = img_candidates[0]
    print(f"\nUsing test image: {test_img_path}")
    
    # Test via LeafPredictor
    predictor = LeafPredictor.get_instance()
    result = predictor.predict(str(test_img_path))
    print("\n--- LeafPredictor Result ---")
    print(f"Predicted class: {result['predicted_class']}")
    print(f"Species        : {result.get('species')}")
    print(f"Condition      : {result.get('condition')}")
    print(f"Confidence     : {result['confidence'] * 100:.2f}%")
    print("Top 3 Candidates:")
    for c in result.get("top_three", []):
        print(f"  {c['rank']}. {c['class_name']} ({c['species']} - {c['condition']}) -> {c['confidence'] * 100:.2f}% (index {c['class_index']})")
        
    print("\n" + "=" * 70)
    print("PHASE 5: DIRECT MODEL INFERENCE WITH DIFFERENT PREPROCESSINGS")
    print("=" * 70)
    
    img_bgr = cv2.imread(str(test_img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Preprocessing 1: Standard resized RGB [0, 255] float32 (current LeafPredictor)
    p1 = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32)
    inp1 = np.expand_dims(p1, axis=0)
    preds1 = model.predict(inp1, verbose=0)[0]
    
    # Preprocessing 2: Resized BGR [0, 255] float32 (what if BGR was passed?)
    p2 = cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32)
    inp2 = np.expand_dims(p2, axis=0)
    preds2 = model.predict(inp2, verbose=0)[0]
    
    # Preprocessing 3: Resized RGB pre-scaled to [-1, 1] (what if double scaling happens?)
    p3 = ((cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 127.5) - 1.0)
    inp3 = np.expand_dims(p3, axis=0)
    preds3 = model.predict(inp3, verbose=0)[0]
    
    # Preprocessing 4: Resized RGB scaled to [0, 1]
    p4 = (cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0)
    inp4 = np.expand_dims(p4, axis=0)
    preds4 = model.predict(inp4, verbose=0)[0]

    for name, probs in [
        ("Current (RGB [0, 255])", preds1),
        ("BGR [0, 255]", preds2),
        ("Pre-normalized [-1, 1]", preds3),
        ("Pre-normalized [0, 1]", preds4)
    ]:
        top3_idx = np.argsort(probs)[::-1][:3]
        print(f"\n--- {name} ---")
        for r, i in enumerate(top3_idx, 1):
            print(f"  {r}. [{i:2d}] {idx_to_class[i]}: {probs[i]*100:.2f}%")
        apple_scab_prob = probs[class_to_idx["Apple_Scab_Leaf"]]
        print(f"  -> Apple_Scab_Leaf (index {class_to_idx['Apple_Scab_Leaf']}) prob: {apple_scab_prob*100:.2f}%")

if __name__ == "__main__":
    run_diagnostics()
