# Dataset Inspection and Inventory Report
**Project Title**: Leaf Quality Classification and Plucking Standard Monitoring System Using Handheld Camera  
**Course Code**: CSE471 Academic Project  
**Module**: Module 1 - Project Setup and Dataset Collection  
**Report Generated**: Module 1 Initialization & Verification Phase  
**Artifact Reference**: `reports/dataset_inventory.csv`  

---

## 1. Executive Summary

This report documents the dataset collection, audit protocols, and inspection tools initialized during **Module 1** of the CSE471 project. The planned system monitors tea/crop leaf quality using handheld camera captures and categorizes leaves into three operational classes:
1. **Good Quality**: Healthy, mature leaves satisfying plucking criteria.
2. **Poor Quality**: Diseased, defective, damaged, or pest-affected leaves unsuitable for harvest.
3. **Not Ready**: Immature, young leaves (e.g., tender flush/buds) requiring additional growth time before harvesting.

Two primary external raw datasets have been integrated into the project architecture for manual placement:
- **PlantDoc Dataset**: A complex, real-world benchmark featuring field-captured images under uncontrolled illumination, background clutter, and natural leaf poses.
- **New Plant Diseases Dataset (PlantVillage derivation)**: A controlled-environment benchmark with laboratory-standardized lighting and uniform leaf orientations.

---

## 2. Directory Layout & Data Ingestion Setup

The project enforces strict directory separation between raw external sources, intermediate transformations, and modeling pipelines:

```text
datasets/
├── raw/
│   ├── plantdoc/               <-- Target folder for manual Kaggle PlantDoc download
│   │   └── .gitkeep
│   └── new_plant_diseases/     <-- Target folder for manual Kaggle New Plant Diseases download
│       └── .gitkeep
├── cleaned/                    <-- Reserved for Module 2 deduplication & noise filtering
│   └── .gitkeep
└── processed/                  <-- Reserved for Module 2 standardized resized & augmented splits
    └── .gitkeep
```

All raw dataset files are ignored by version control (`.gitignore`) to avoid bloat and ensure reproducible environment setup.

---

## 3. Dataset Characteristics & Audit Protocol

### A. Integrity & Corruption Detection
The exploration tools (`scripts/01_dataset_exploration.py` and `notebooks/01_dataset_exploration.ipynb`) utilize a two-tier verification check:
1. **PIL Integrity Verification (`Image.verify()`)**: Detects truncated JPEG blocks, broken PNG chunks, and malformed header signatures.
2. **OpenCV Pixel Decoding (`cv2.imread()`)**: Confirms that pixel arrays can be decoded into multidimensional NumPy arrays without raising segmentation faults or decompression exceptions.

### B. Image Dimension & Resolution Metrics
- The script calculates minimum, maximum, and average resolutions across each class folder.
- Standard input for the downstream MobileNetV2 architecture will be standardized to $224 \times 224 \times 3$ during Module 2.

### C. Current Inventory Status

| Dataset Name | Expected Source | Current Local Status | Local Target Path |
| :--- | :--- | :--- | :--- |
| **PlantDoc** | Kaggle: `daeyun/plantdoc-dataset` | Awaiting Manual Placement | `datasets/raw/plantdoc/` |
| **New Plant Diseases** | Kaggle: `vipoooool/new-plant-diseases-dataset` | Awaiting Manual Placement | `datasets/raw/new_plant_diseases/` |

---

## 4. Critical Academic & Engineering Insights

### Preserving Original Class Names
Original class labels (such as `Apple___Cedar_apple_rust`, `Tomato___healthy`, `Corn___Cercospora_leaf_spot`) are strictly preserved. Premature grouping into binary healthy/diseased buckets is prohibited until rigorous exploratory data analysis is completed.

### The "Not Ready" Category Gap (Missing Dataset Requirement)
- **Problem Statement**: Standard Kaggle leaf disease datasets (PlantDoc, New Plant Diseases) are curated specifically to discriminate plant pathology (healthy vs. specific pathogen diseases). They **do not** categorize developmental stages or leaf maturity.
- **Analysis**: Immature tea or crop leaves have distinct physiological traits (smaller surface area, lighter lime-green pigmentation, delicate leaf venation, tender cuticle thickness). Marking healthy mature leaves as "Good" and diseased leaves as "Poor" leaves the model with zero training signal for the "Not Ready" plucking state.
- **Action Required**: The "Not Ready" class cannot be synthesized or fabricated. In accordance with project integrity rules, this gap is formally reported as a **Missing Dataset Requirement**. For Module 2, a dedicated subset of young/immature leaf images must be collected or identified before training begins.

---

## 5. Verification Tools Summary

1. **CLI Script**: `scripts/01_dataset_exploration.py`
   - Executes non-destructively.
   - Generates summary statistics, detects corruptions, and writes `reports/dataset_inventory.csv`.
   - Generates `reports/class_distribution.png` once images are present.
2. **Interactive EDA Notebook**: `notebooks/01_dataset_exploration.ipynb`
   - Provides step-by-step visual exploration, category preview grids, and dimension histograms.
