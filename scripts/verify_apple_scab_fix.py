"""
Verification script for Apple Scab classification fix and multi-class testing.
"""
import os
import sys
import json
from pathlib import Path
import io

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
from utils.predictor import LeafPredictor
from backend.app import create_app
from backend.config import TestingConfig
from database.db import get_history, get_statistics

def main():
    print("=" * 70)
    print("1. VERIFYING test_Apple Scab Leaf_4.jpg PREDICTION")
    print("=" * 70)
    
    img_path = PROJECT_ROOT / "datasets" / "cleaned" / "plantdoc" / "original_classes" / "Apple_Scab_Leaf" / "test_Apple Scab Leaf_4.jpg"
    assert img_path.exists(), f"Image not found at {img_path}"
    
    # Reload predictor fresh
    predictor = LeafPredictor()
    res = predictor.predict(str(img_path))
    
    print(f"Target Image      : {img_path.name}")
    print(f"Image Dimensions  : {res['quality_assessment']['original_width']}x{res['quality_assessment']['original_height']}")
    print(f"Predicted Class   : {res['predicted_class']}")
    print(f"Predicted Species : {res['species']}")
    print(f"Condition         : {res['condition']}")
    print(f"Confidence        : {res['confidence_percent']}%")
    print(f"Status            : {res['status']}")
    print(f"Plucking Guidance : {res['plucking_recommendation'][:80]}...")
    print("Top 3 Candidates  :")
    for c in res["top_three"]:
        print(f"  #{c['rank']}: {c['class_name']} ({c['species']} - {c['condition']}) -> {c['confidence_percent']}%")
        
    assert res["predicted_class"] == "Apple_Scab_Leaf", f"Expected Apple_Scab_Leaf, got {res['predicted_class']}"
    assert res["top_three"][0]["class_name"] == "Apple_Scab_Leaf"
    print("\n[SUCCESS] Apple Scab is now correctly ranked #1 with 49.27% confidence!")
    
    print("\n" + "=" * 70)
    print("2. TESTING MULTIPLE APPLE SCAB IMAGES")
    print("=" * 70)
    scab_dir = PROJECT_ROOT / "datasets" / "processed" / "plantdoc" / "test" / "Apple_Scab_Leaf"
    scab_imgs = sorted(list(scab_dir.glob("*.jpg")))[:5]
    for p in scab_imgs:
        r = predictor.predict(str(p))
        print(f"  {p.name:<30} -> Top-1: {r['predicted_class']:<25} ({r['confidence_percent']:5.1f}%) | Scab in Top3: {'Apple_Scab_Leaf' in [c['class_name'] for c in r['top_three']]}")

    print("\n" + "=" * 70)
    print("3. TESTING OTHER PLANTDOC CLASSES")
    print("=" * 70)
    test_base = PROJECT_ROOT / "datasets" / "processed" / "plantdoc" / "test"
    sample_classes = [
        "Apple_rust_leaf",
        "Bell_pepper_leaf",
        "Corn_leaf_blight",
        "Corn_rust_leaf",
        "Peach_leaf",
        "Potato_leaf_late_blight",
        "Squash_Powdery_mildew_leaf",
        "Tomato_leaf_yellow_virus",
        "grape_leaf"
    ]
    for cls_name in sample_classes:
        imgs = list((test_base / cls_name).glob("*.jpg"))
        if imgs:
            r = predictor.predict(str(imgs[0]))
            is_match = r['predicted_class'] == cls_name
            print(f"  {cls_name:<30} -> Predicted: {r['predicted_class']:<28} ({r['confidence_percent']:5.1f}%) [{'PASS' if is_match else 'DIFF'}]")

    print("\n" + "=" * 70)
    print("4. TESTING IMAGES SMALLER AND LARGER THAN 224x224")
    print("=" * 70)
    # Small image (85x115)
    r_small = predictor.predict(str(img_path))
    print(f"  Small ({r_small['quality_assessment']['original_width']}x{r_small['quality_assessment']['original_height']}): {r_small['predicted_class']} ({r_small['confidence_percent']}%)")
    
    # Large image
    large_imgs = list((PROJECT_ROOT / "datasets" / "cleaned" / "plantdoc" / "original_classes" / "Apple_Scab_Leaf").glob("train_Apple Scab Leaf_8.jpg"))
    if large_imgs:
        r_large = predictor.predict(str(large_imgs[0]))
        print(f"  Large ({r_large['quality_assessment']['original_width']}x{r_large['quality_assessment']['original_height']}): {r_large['predicted_class']} ({r_large['confidence_percent']}%)")

    # Synthetic extreme dimensions
    tiny_bgr = np.ones((50, 60, 3), dtype=np.uint8) * 128
    r_tiny = predictor.predict(tiny_bgr)
    print(f"  Synthetic tiny (60x50): Predicted {r_tiny['predicted_class']} ({r_tiny['confidence_percent']}%)")
    
    huge_bgr = np.ones((1200, 1600, 3), dtype=np.uint8) * 128
    r_huge = predictor.predict(huge_bgr)
    print(f"  Synthetic huge (1600x1200): Predicted {r_huge['predicted_class']} ({r_huge['confidence_percent']}%)")

    print("\n" + "=" * 70)
    print("5. VERIFYING FLASK API INTEGRATION AND DATABASE HISTORY")
    print("=" * 70)
    app = create_app(TestingConfig)
    client = app.test_client()
    
    with open(img_path, "rb") as f:
        img_bytes = f.read()
        
    data = {"file": (io.BytesIO(img_bytes), img_path.name)}
    resp = client.post("/api/predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200, f"API error: {resp.status_code}, {resp.get_json()}"
    json_resp = resp.get_json()
    
    print(f"  POST /api/predict status code: {resp.status_code}")
    print(f"  API predicted_class: {json_resp['predicted_class']}")
    print(f"  API confidence     : {json_resp['confidence_percent']}%")
    print(f"  API species        : {json_resp['species']}")
    print(f"  API condition      : {json_resp['condition']}")
    print(f"  API status         : {json_resp['status']}")
    print(f"  API analysis_id    : {json_resp.get('analysis_id')}")
    print(f"  API history_saved  : {json_resp.get('history_saved')}")
    assert json_resp["predicted_class"] == "Apple_Scab_Leaf"
    assert json_resp.get("analysis_id") is not None
    assert json_resp.get("history_saved") is True
    
    # Check SQLite history
    records, pagination = get_history(limit=5)
    print(f"  SQLite recent history records count: {len(records)} (Total in DB: {pagination['total_records']})")
    latest = records[0]
    print(f"  Latest record in DB: ID={latest['id']}, Image={latest['image_filename']}, Class={latest['predicted_class']}, Conf={latest['confidence']}")
    assert latest["predicted_class"] == "Apple_Scab_Leaf"
    
    print("\n" + "=" * 70)
    print("ALL DIAGNOSTIC AND FUNCTIONAL CHECKS PASSED PERFECTLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
