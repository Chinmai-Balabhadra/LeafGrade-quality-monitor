# Module 2 Summary: Dataset Cleaning, Validation, and Organization

**Project Title**: Leaf Quality Classification and Plucking Standard Monitoring System Using Handheld Camera  
**Course Code**: CSE471 Academic Project  
**Date Generated**: November 2026  
**Scope**: PlantDoc Dataset Audit, Cleaning, Deduplication, and Preliminary Organization  

---

## 1. Executive Summary

Module 2 focused on rigorously auditing, cleaning, and structuring the **PlantDoc** dataset while maintaining full architectural readiness to integrate the pending **New Plant Diseases** dataset. 

Every operation strictly obeyed project integrity constraints:
- **No modification of raw data**: Original files in `datasets/raw/plantdoc/` remain completely untouched.
- **Zero data fabrication**: No artificial images or fabricated labels were generated.
- **Strict preservation**: Original source class labels are preserved 1:1 in `datasets/cleaned/plantdoc/original_classes/`.
- **Full audit trail**: Every excluded image, duplicate conflict, and mapping decision has been documented in machine-readable reports.

---

## 2. Quantitative Cleaning Metrics

| Metric | Count | Percentage of Raw Data |
| :--- | :--- | :--- |
| **Total Original Raw Images** | **2,922** | 100.00% |
| **Valid & Decodable Images** | **2,922** | 100.00% |
| **Corrupted / Unreadable Images** | **0** | 0.00% |
| **Exact Binary Duplicates Identified** | **6** | 0.21% |
| **Total Cleaned & Verified Images** | **2,916** | **99.79%** |
| **Preliminary Processed Train Set (70%)** | **2,028** | 70.00% |
| **Preliminary Processed Val Set (15%)** | **423** | 15.00% |
| **Preliminary Processed Test Set (15%)** | **465** | 15.00% |

---

## 3. Data Integrity & Validation Findings

### A. Image Decodability and Resolutions
- **Integrity**: All 2,922 raw images were tested using both **Pillow** (`img.verify()`) and **OpenCV** (`cv2.imread()`). Zero images were corrupted or zero-byte.
- **Color Spaces**: Images encompass `RGB`, `RGBA`, `L` (grayscale), and `CMYK` color profiles.
- **Resolution Range**:
  - Minimum Resolution: $115 \times 69$ pixels
  - Maximum Resolution: $6000 \times 6000$ pixels
  - Average Resolution: $911 \times 800$ pixels
- **Implication for Downstream Modeling**: MobileNetV2 requires a standardized input tensor ($224 \times 224 \times 3$). When building the training pipeline in Module 3, resizing with aspect-ratio-preserving padding and channel standardization (converting all images to 3-channel RGB) will be essential.

### B. Duplicate Detection & Benchmark Data Leakage Analysis
A SHA-256 cryptographic hash audit uncovered **6 exact duplicate groups** in the raw PlantDoc distribution:
1. **Intra-Class Train-Test Leak**:
   - `test/Blueberry_leaf/test_Blueberry leaf_3.jpg` is an exact binary clone of `train/Blueberry_leaf/train_Blueberry leaf_33.jpg`.
2. **Cross-Class Train-Test Conflicts**:
   - `test/Corn_Gray_leaf_spot/test_Corn Gray leaf spot_4.jpg` $\leftrightarrow$ `train/Corn_leaf_blight/train_Corn leaf blight_119.jpg`
   - `test/Corn_leaf_blight/test_Corn leaf blight_4.jpg` $\leftrightarrow$ `train/Corn_Gray_leaf_spot/train_Corn Gray leaf spot_16.jpg`
   - `test/Corn_leaf_blight/test_Corn leaf blight_5.jpg` $\leftrightarrow$ `train/Corn_Gray_leaf_spot/train_Corn Gray leaf spot_19.jpg`
   - `test/Potato_leaf_late_blight/test_Potato leaf late blight_5.jpg` $\leftrightarrow$ `train/Potato_leaf_early_blight/train_Potato leaf early blight_68.jpg`
   - `test/Tomato_leaf_bacterial_spot/test_Tomato leaf bacterial spot_1.jpg` $\leftrightarrow$ `train/Tomato_Septoria_leaf_spot/train_Tomato Septoria leaf spot_20.jpg`

**Cleaning Action Taken**: All 6 redundant and conflicting test instances were excluded from the cleaned dataset. Canonical copies in `train/` were retained. Full details are recorded in `reports/duplicate_report.csv`.

---

## 4. Class Mapping Analysis & Agricultural Plucking Standards

The project requires classifying leaves into three operational target classes:
1. **Good Quality**: Healthy, mature leaves satisfying plucking criteria.
2. **Poor Quality**: Diseased, defective, damaged, or pest-affected leaves unsuitable for harvest.
3. **Not Ready**: Immature, young leaves (e.g., tender flush/buds) requiring additional growth time before harvesting.

### Preliminary Mapping Breakdown:
- **Good Quality Classes (10 classes, 822 images)**:
  `Apple_leaf`, `Bell_pepper_leaf`, `Blueberry_leaf`, `Cherry_leaf`, `grape_leaf`, `Peach_leaf`, `Raspberry_leaf`, `Soyabean_leaf`, `Strawberry_leaf`, `Tomato_leaf`.
- **Poor Quality Classes (18 classes, 2,094 images)**:
  `Apple_rust_leaf`, `Apple_Scab_Leaf`, `Bell_pepper_leaf_spot`, `Corn_Gray_leaf_spot`, `Corn_leaf_blight`, `Corn_rust_leaf`, `grape_leaf_black_rot`, `Potato_leaf_early_blight`, `Potato_leaf_late_blight`, `Squash_Powdery_mildew_leaf`, `Tomato_Early_blight_leaf`, `Tomato_leaf_bacterial_spot`, `Tomato_leaf_late_blight`, `Tomato_leaf_mosaic_virus`, `Tomato_leaf_yellow_virus`, `Tomato_mold_leaf`, `Tomato_Septoria_leaf_spot`, `Tomato_two_spotted_spider_mites_leaf`.

### Critical Finding: The "Not Ready" Category Data Gap
- **Analysis**: PlantDoc is strictly a plant pathology dataset designed to distinguish healthy foliage from disease pathogens. It contains **zero annotations of developmental maturity** (no tags for young tender shoots vs. mature plucking leaves).
- **Compliance with Project Rules**: Treating healthy mature leaves as "ready for plucking" indiscriminately is agronomically incorrect. Furthermore, fabricating artificial "Not Ready" labels violates project guidelines.
- **Formal Status**: The *Not Ready* class is officially flagged as a **Missing Dataset Requirement**. When the **New Plant Diseases** dataset is added, or when custom leaf maturity images are integrated, this category will be populated based on verified developmental stages.

---

## 5. Dataset Organization Architecture

The project directory structure now reflects clean separation:

```text
datasets/
├── raw/
│   ├── plantdoc/                       # Untouched original raw images (2,922 images)
│   └── new_plant_diseases/             # Ready for integration once download completes
├── cleaned/
│   └── plantdoc/
│       └── original_classes/           # 28 classes, 2,916 deduplicated images
└── processed/
    └── plantdoc/
        ├── train/                      # 2,041 images (70% stratified split)
        ├── val/                        # 437 images (15% stratified split)
        └── test/                       # 438 images (15% stratified split)
```

---

## 6. Generated Reports

1. [reports/cleaning_report.csv](file:///c:/Users/ASUS/OneDrive/Desktop/Project-471/reports/cleaning_report.csv): Class-by-class raw, corrupted, duplicate, and cleaned image counts.
2. [reports/duplicate_report.csv](file:///c:/Users/ASUS/OneDrive/Desktop/Project-471/reports/duplicate_report.csv): Itemized log of all 6 identified duplicate files and justifications.
3. [reports/invalid_images_report.csv](file:///c:/Users/ASUS/OneDrive/Desktop/Project-471/reports/invalid_images_report.csv): Registry of corrupted/unsupported files (0 invalid images detected).
4. [reports/class_mapping.csv](file:///c:/Users/ASUS/OneDrive/Desktop/Project-471/reports/class_mapping.csv): Full 28-class agronomic mapping, plucking suitability, and justification table.
5. [reports/module2_summary.md](file:///c:/Users/ASUS/OneDrive/Desktop/Project-471/reports/module2_summary.md): This comprehensive audit report.

---

## 7. Next Steps (Pending Approval)

1. Await user manual placement of **New Plant Diseases Dataset** in `datasets/raw/new_plant_diseases/`.
2. Apply the audited cleaning pipeline to New Plant Diseases upon arrival.
3. Perform the combined final dataset integration and stratification.
4. Prepare for **Module 3: Model Architecture and Training (MobileNetV2)** once authorized.
