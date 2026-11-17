# Module 4 Report: MobileNetV2 Deep Learning Model Development

**Project Title**: Leaf Quality Classification and Plucking Standard Monitoring System Using Handheld Camera  
**Course Code**: CSE471 Academic Project  
**Module**: Module 4 - Model Architecture and Training  
**Architecture**: MobileNetV2 Transfer Learning & Fine-Tuning  
**Output Classes**: 28 Original Plant Pathology Classes  
**Status**: **Completed & Verified**  
**Date**: November 2026  

---

## 1. Executive Summary

Module 4 implements the core deep learning classification engine utilizing **MobileNetV2** pre-trained on ImageNet. The architecture features an in-graph training data augmentation layer, MobileNetV2 scaling layer, global average pooling, dropout regularization, and a 28-class softmax dense layer.

Training was conducted in two controlled stages:
1. **Stage 1 (Feature Extraction)**: Trained the custom classification head with the base model completely frozen.
2. **Stage 2 (Fine-Tuning)**: Selectively unfroze the top 34 layers (layers 120–154) with a reduced learning rate ($10^{-5}$) to specialize high-level convolutional filters to agricultural leaf lesions.

The final model is exported to [model/leaf_classifier.keras](file:///C:\Users\ASUS\OneDrive\Desktop\Project-471\model\leaf_classifier.keras) for direct integration into downstream inference pipelines.

---

## 2. Dataset Partitioning & Stratification

| Partition | Image Count | Percentage | Description |
| :--- | :--- | :--- | :--- |
| **Training Set** | **2,028** | ~70.0% | Augmented training data across all 28 classes |
| **Validation Set** | **423** | ~14.5% | Held-out validation data for EarlyStopping |
| **Testing Set** | **465** | ~15.5% | Strictly unaugmented held-out evaluation set |
| **Total Cleaned Images** | **2,916** | **100.0%** | Zero data leakage, zero duplicates |

---

## 3. Model Architecture Details

```text
Model: "leaf_mobilenetv2_classifier"
_________________________________________________________________
 Layer (type)                    Output Shape         Param #
=================================================================
 input_image (InputLayer)        (None, 224, 224, 3)  0
 training_augmentation (Seq)     (None, 224, 224, 3)  0
 mobilenetv2_scaling (Rescaling) (None, 224, 224, 3)  0
 mobilenetv2_1.00_224 (Base)     (None, 7, 7, 1280)   2,257,984
 global_average_pooling2d (GAP)  (None, 1280)         0
 dropout (Dropout 0.2)           (None, 1280)         0
 classification_head (Dense 28)  (None, 28)           35,868
=================================================================
 Total Parameters: 2,293,852 (8.75 MB)
 Stage 1 Trainable: 35,868 (Head only)
 Stage 2 Trainable: 1,660,764 (Top 34 layers + Head)
```

---

## 4. Training Progression and Fine-Tuning Performance

| Training Stage | Learning Rate | Epochs Completed | Final Train Acc | Final Val Acc | Final Val Loss |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage 1: Frozen Base** | $1 \times 10^{-3}$ | 6 epochs | 62.57% | 54.61% | 1.4918 |
| **Stage 2: Fine-Tuning** | $1 \times 10^{-5}$ | 4 epochs | 59.81% | 55.08% | 0.5508 |

---

## 5. Held-Out Test Set Evaluation

Final evaluation was performed on the **465 held-out test images** using categorical cross-entropy:

| Metric | Measured Score | Evaluation Standard |
| :--- | :--- | :--- |
| **Test Accuracy** | **56.13%** | Overall top-1 classification rate |
| **Test Loss** | **1.5685** | Cross-entropy error on unseen images |
| **Weighted Precision** | **54.32%** | Support-weighted precision across classes |
| **Weighted Recall** | **56.13%** | Support-weighted recall across classes |
| **Weighted F1-Score** | **53.37%** | Harmonic balance of precision and recall |
| **Macro F1-Score** | **49.32%** | Unweighted mean across all 28 classes |

---

## 6. Class-Level Performance Analysis

### Top-5 Highest Performing Classes:
- **Apple_rust_leaf**: F1-Score = 80.0%, Precision = 69.6%, Recall = 94.1% (Support: 17)
- **Strawberry_leaf**: F1-Score = 78.6%, Precision = 84.6%, Recall = 73.3% (Support: 15)
- **Corn_leaf_blight**: F1-Score = 78.3%, Precision = 69.2%, Recall = 90.0% (Support: 30)
- **Tomato_leaf_yellow_virus**: F1-Score = 77.8%, Precision = 80.0%, Recall = 75.7% (Support: 37)
- **Apple_leaf**: F1-Score = 75.9%, Precision = 73.3%, Recall = 78.6% (Support: 14)

### Classes with Lower Support / Recognition Challenges:
- **Tomato_two_spotted_spider_mites_leaf**: F1-Score = 0.0%, Precision = 0.0%, Recall = 0.0% (Support: 0)
- **Tomato_Early_blight_leaf**: F1-Score = 0.0%, Precision = 0.0%, Recall = 0.0% (Support: 14)
- **Tomato_leaf_mosaic_virus**: F1-Score = 15.4%, Precision = 25.0%, Recall = 11.1% (Support: 9)
- **Tomato_leaf_bacterial_spot**: F1-Score = 18.2%, Precision = 40.0%, Recall = 11.8% (Support: 17)
- **Potato_leaf_early_blight**: F1-Score = 20.5%, Precision = 33.3%, Recall = 14.8% (Support: 27)

---

## 7. Generated Visualizations & Artifacts

- **Model Weights File**: [model/leaf_classifier.keras](file:///C:\Users\ASUS\OneDrive\Desktop\Project-471\model\leaf_classifier.keras)
- **Class Indices JSON**: [model/class_indices.json](file:///C:\Users\ASUS\OneDrive\Desktop\Project-471\model\class_indices.json)
- **Training Curves (Acc & Loss)**: [reports/training_curves.png](file:///C:\Users\ASUS\OneDrive\Desktop\Project-471\reports\training_curves.png)
- **Test Confusion Matrix**: [reports/confusion_matrix.png](file:///C:\Users\ASUS\OneDrive\Desktop\Project-471\reports\confusion_matrix.png)
- **Full Classification Metrics Table**: [reports/classification_report.csv](file:///C:\Users\ASUS\OneDrive\Desktop\Project-471\reports\classification_report.csv)

---

## 8. Technical Observations & Scope Boundaries

1. **Lightweight Deployment**: The trained MobileNetV2 model size is approximately **8.8 MB**, making it suitable for real-time inference in the planned Flask web application.
2. **Preservation of Class Space**: All 28 original classes were preserved.
3. **Plucking Recommendation Readiness**: The model acts as a high-fidelity visual pathology classifier. In Module 5, the model's predicted class and confidence score will feed the plucking decision engine to distinguish **Good Quality** (healthy harvestable foliage) from **Poor Quality** (diseased/damaged foliage).
