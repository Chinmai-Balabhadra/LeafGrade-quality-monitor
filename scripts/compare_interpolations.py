import cv2
import keras
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import json

test_dir = Path("datasets/processed/plantdoc/test")
model = keras.models.load_model("model/leaf_classifier.keras", compile=False)

with open("model/class_indices.json", "r", encoding="utf-8") as f:
    class_to_idx = json.load(f)["class_to_idx"]

for interp_name, interp in [("INTER_AREA", cv2.INTER_AREA), ("INTER_LINEAR (bilinear)", cv2.INTER_LINEAR), ("INTER_CUBIC", cv2.INTER_CUBIC)]:
    correct = 0
    total = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    for cls_name, cls_idx in class_to_idx.items():
        cls_folder = test_dir / cls_name
        for p in cls_folder.glob("*.jpg"):
            bgr = cv2.imread(str(p))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            res = cv2.resize(rgb, (224, 224), interpolation=interp).astype(np.float32)
            inp = np.expand_dims(res, axis=0)
            preds = model.predict(inp, verbose=0)[0]
            pred_idx = int(np.argmax(preds))
            if pred_idx == cls_idx:
                correct += 1
                per_class_correct[cls_name] += 1
            per_class_total[cls_name] += 1
            total += 1
    acc = correct / total
    scab_corr = per_class_correct["Apple_Scab_Leaf"]
    scab_tot = per_class_total["Apple_Scab_Leaf"]
    print(f"{interp_name}: {correct}/{total} ({acc*100:.2f}%) | Apple_Scab_Leaf: {scab_corr}/{scab_tot} ({scab_corr/scab_tot*100:.1f}%)")
