# Leaf Quality Classification and Plucking Standard Monitoring System Using Handheld Camera

**Academic Course**: CSE471 Project  
**Author**: CSE471 Project Team  
**Level**: Third-Year B.Tech Computer Science and Engineering  
**Current Milestone**: Module 9 - Final System Testing, Performance Validation and Deployment (Completed)  

---

## 1. Project Overview

In agriculture and commercial plantation management (such as tea, herbs, and leafy vegetables), determining the optimal time to harvest leaves directly dictates product quality, grade, and economic yield. Manual inspection by human plucking labor is labor-intensive, subjective, and prone to error.

This project develops an AI-assisted computer vision system that analyzes leaf images captured using standard handheld cameras (e.g., smartphones or portable inspection devices). The system evaluates leaf condition, flags diseases or damage, checks maturity, and provides plucking recommendations through an intuitive web application interface.

---

## 2. Project Objectives

1. **Automated Leaf Ingestion**: Accept leaf photographs uploaded by users or captured via handheld camera.
2. **Computer Vision Processing**: Clean, standardize, and preprocess leaf imagery under varied outdoor field lighting and background conditions.
3. **Three-Tier Quality Categorization**:
   - **Good Quality**: Healthy, mature leaves that meet standard plucking criteria.
   - **Poor Quality**: Diseased, pest-damaged, scorched, or necrotic leaves that should not be harvested.
   - **Not Ready**: Immature or tender young leaves that require additional time on the plant to reach maturity.
4. **Plucking Recommendation**: Generate clear, actionable harvest recommendations based on quality assessment.
5. **Interactive Web Interface**: Provide a user-friendly Flask-based web application with image upload, instantaneous result display, and visual classification feedback.
6. **Historical Audit Log**: Maintain an analysis log (SQLite database) to track plucking trends over time.

---

## 3. Technology Stack

### Current Active Stack:
- **Programming Language**: Python 3.12.10 (64-bit)
- **Environment Management**: Python standard `venv` (`.venv`)
- **Deep Learning & Neural Networks**: TensorFlow 2.21.0, Keras 3.15.1 (MobileNetV2 standard input)
- **Image Processing & Validation**: OpenCV (`opencv-python` 5.0.0), Pillow (`PIL` 13.0.0)
- **Numerical & Data Processing**: NumPy 2.5.4, Pandas 3.1.1, Scikit-Learn 1.8.0
- **Visualization**: Matplotlib 3.11.2
- **Backend Web Server**: Flask 3.1.3, Flask-CORS 6.0.2
- **Database & Persistence**: SQLite3 (Python built-in standard library), WAL mode
- **Frontend Dashboard**: Vanilla HTML5, CSS3, ES6 JavaScript, MediaDevices Camera API, Font Awesome 6.5
- **Unit Testing**: Python standard `unittest` (71/71 tests passing)
- **Interactive Development**: Jupyter Notebook (`jupyter` 1.1.1, `ipykernel` 7.4.0)
- **CLI Utilities**: tqdm 4.70.1

---

## 4. Project Folder Structure

The project follows a modular, reproducible directory layout designed for collaborative software and machine learning engineering:

```text
Project-471/ (Leaf_Quality_Monitoring)
│
├── datasets/                       # Data storage (git-ignored for raw/large assets)
│   ├── raw/                        # Original, unaltered Kaggle downloads
│   │   ├── plantdoc/               # PlantDoc dataset images
│   │   └── new_plant_diseases/     # New Plant Diseases dataset images
│   ├── cleaned/                    # Cleaned, filtered, deduplicated images (Module 2)
│   │   └── plantdoc/original_classes/
│   └── processed/                  # Train/val/test split indices (Module 2)
│       └── plantdoc/
│
├── notebooks/                      # Jupyter notebooks for interactive analysis & EDA
│   ├── 01_dataset_exploration.ipynb# Module 1 exploration
│   └── 02_image_preprocessing.ipynb# Module 3 preprocessing pipeline
│
├── scripts/                        # Standalone executable automation scripts
│   ├── 01_dataset_exploration.py   # Module 1 audit script
│   └── 03_image_preprocessing.py   # Module 3 preprocessing pipeline
│
├── utils/                          # Reusable library modules
│   ├── preprocessing.py            # MobileNetV2 preprocessing utilities
│   └── predictor.py                # Standalone prediction pipeline & quality assessor
│
├── model/                          # Saved model weights, architecture configs (.h5, .keras)
│   ├── leaf_classifier.keras       # Trained MobileNetV2 classifier (22.03 MB)
│   └── class_indices.json          # 28-class JSON mapping dictionary
│
├── database/                       # SQLite persistent storage subsystem (Module 8)
│   ├── __init__.py                 # Database package exports
│   ├── schema.sql                  # DDL table definition & 4 performance indexes
│   ├── db.py                       # Data access layer, WAL mode, CRUD & analytics
│   └── leaf_analysis.db            # Embedded SQLite active database file
│
├── backend/                        # Flask backend application logic and REST APIs
│   ├── __init__.py                 # Backend application factory export
│   ├── app.py                      # Flask factory, blueprint registration, CORS, entry point
│   ├── config.py                   # Multi-environment configuration & limits
│   ├── errors.py                   # Standardized JSON error response handlers
│   ├── routes/                     # Blueprint routes (health, predict, history)
│   └── services/                   # Service layer (singleton prediction caching)
│
├── frontend/                       # Responsive web dashboard interface
│   ├── index.html                  # Single-page dashboard (Hero, Analysis, History, About)
│   ├── css/                        # Modular stylesheet system (style, dashboard, responsive)
│   └── js/                         # Client controllers (api, upload, prediction, history, main)
│
├── reports/                        # Audit logs, generated charts, and summaries
│   ├── dataset_inventory.csv       # Module 1 inventory
│   ├── dataset_inspection_report.md# Module 1 report
│   ├── cleaning_report.csv         # Module 2 cleaning audit
│   ├── class_mapping.csv           # Module 2 preliminary mapping
│   ├── duplicate_report.csv        # Module 2 duplicate images report
│   ├── invalid_images_report.csv   # Module 2 corrupted images report
│   ├── module2_summary.md          # Module 2 summary report
│   ├── preprocessing_report.md     # Module 3 preprocessing report
│   ├── module4_model_report.md     # Module 4 training & architecture report
│   ├── module5_prediction_report.md# Module 5 inference pipeline report
│   ├── module6_backend_report.md   # Module 6 Flask backend report
│   └── module8_database_report.md  # Module 8 SQLite persistence report
│
├── tests/                          # Automated unit and integration test suites (88 tests)
│   ├── test_preprocessing.py       # Module 3 preprocessing unit tests (9 tests)
│   ├── test_model.py               # Module 4 model evaluation tests (7 tests)
│   ├── test_predictor.py           # Module 5 prediction pipeline tests (12 tests)
│   ├── test_database.py            # Module 8 SQLite database tests (12 tests)
│   ├── test_api.py                 # Module 6 & 8 Flask REST API tests (23 tests)
│   ├── test_frontend_integration.py# Module 7 & 8 frontend integration tests (8 tests)
│   ├── test_edge_cases.py          # Module 9 edge cases & failure recovery tests (16 tests)
│   └── test_end_to_end.py          # Module 9 complete 15-step workflow verification (1 test)
│
├── .gitignore                      # Git exclusion rules for venv, data, and caches
├── requirements.txt                # Locked Python dependency manifest
├── README.md                       # Main project documentation
└── setup_info.md                   # Environment details and download instructions
```

---

## 5. Python Environment Setup Instructions

### Prerequisites:
- Python 3.11 or Python 3.12 installed on your machine.
- Git (optional, for version tracking).

### Step 1: Navigate to Project Directory
Open PowerShell or Command Prompt:
```powershell
cd "c:\Users\ASUS\OneDrive\Desktop\Project-471"
```

### Step 2: Virtual Environment Configuration
A dedicated virtual environment named `.venv` has already been created in this directory. If you ever need to recreate it from scratch:
```powershell
python -m venv .venv
```

### Step 3: Activate the Virtual Environment
- **Windows PowerShell**:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
  *(Note: If you encounter an execution policy error, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*
- **Windows Command Prompt (cmd.exe)**:
  ```cmd
  .\.venv\Scripts\activate.bat
  ```

---

## 6. Dependency Installation Commands

All initial dependencies are locked in `requirements.txt`. To install or reinstall them in your activated `.venv`:
```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install -r requirements.txt
```

---

## 7. Kaggle Dataset Placement Instructions

The system requires two benchmark datasets. Please download them manually:

### 1. PlantDoc Dataset
- **Download Link**: [https://www.kaggle.com/datasets/daeyun/plantdoc-dataset](https://www.kaggle.com/datasets/daeyun/plantdoc-dataset)
- **Extraction Path**: Extract the downloaded archive directly into:
  ```text
  datasets/raw/plantdoc/
  ```
- **Context**: Real-world field images taken under variable daylight with complex background vegetation.

### 2. New Plant Diseases Dataset
- **Download Link**: [https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- **Extraction Path**: Extract the downloaded archive directly into:
  ```text
  datasets/raw/new_plant_diseases/
  ```
- **Context**: Standardized laboratory captures of healthy and diseased leaves across multiple agricultural species.

*Note: Do not rename or alter original files or folders when extracting.*

---

## 8. Dataset Exploration Execution Instructions

### Option A: Command-Line Exploration Script
Run the automated dataset auditing script:
```powershell
.\.venv\Scripts\python.exe scripts/01_dataset_exploration.py
```
**What the script does**:
1. Scans `datasets/raw/plantdoc/` and `datasets/raw/new_plant_diseases/`.
2. Validates image integrity and identifies corrupted files.
3. Computes image resolutions (average, min, max) and catalogs file extensions.
4. Generates an inventory summary in `reports/dataset_inventory.csv`.
5. Exports a class distribution chart to `reports/class_distribution.png` once images are present.
6. Gracefully alerts you if datasets have not yet been placed.

### Option B: Interactive Jupyter Notebook
Launch Jupyter Notebook:
```powershell
.\.venv\Scripts\jupyter.exe notebook notebooks/01_dataset_exploration.ipynb
```
Select the `.venv` kernel to run the interactive cells, inspect sample images, and review statistical distributions.

---

## 9. Current Development Status

- [x] **Module 1: Project Setup and Dataset Collection** *(Completed)*
  - [x] Project directory structure established.
  - [x] Python virtual environment configured (`.venv`).
  - [x] Initial dependencies installed and verified (`numpy`, `pandas`, `matplotlib`, `opencv-python`, `pillow`, `jupyter`, `tqdm`).
  - [x] Exploration script created (`scripts/01_dataset_exploration.py`).
  - [x] Exploration notebook created (`notebooks/01_dataset_exploration.ipynb`).
  - [x] Dataset inventory schema initialized (`reports/dataset_inventory.csv`).
  - [x] Dataset audit and category mapping report generated (`reports/dataset_inspection_report.md`).
  - [x] Git ignore rules defined (`.gitignore`).
  - [x] Setup and project documentation completed (`README.md`, `setup_info.md`).

- [x] **Module 2: Dataset Cleaning and Organization** *(Completed)*
  - [x] PlantDoc dataset audit completed (2,922 images across 28 original classes).
  - [x] Image integrity validation (0 corrupt images, all valid JPEG/PNG).
  - [x] Exact duplicate detection via MD5 hashing (6 duplicates identified and excluded).
  - [x] Cleaned dataset organized in `datasets/cleaned/plantdoc/original_classes/` (2,916 valid images).
  - [x] Class mapping analysis (17 diseased categories, 11 healthy categories, missing immature leaf category noted).
  - [x] Stratified train/val/test splits created in `datasets/processed/plantdoc/` (70/15/15 ratio).
  - [x] Reports generated (`cleaning_report.csv`, `duplicate_report.csv`, `invalid_images_report.csv`, `class_mapping.csv`, `module2_summary.md`).

- [x] **Module 3: Image Preprocessing** *(Completed)*
  - [x] MobileNetV2-compatible preprocessing utilities created (`utils/preprocessing.py`).
  - [x] Robust image loading with OpenCV and PIL fallback.
  - [x] BGR to RGB color conversion for natural color alignment.
  - [x] Adaptive image resizing to $224 \times 224$ pixels (`cv2.INTER_AREA` downsampling).
  - [x] MobileNetV2 pixel normalization scaling to $[-1.0, 1.0]$: $x_{\text{norm}} = (x / 127.5) - 1.0$.
  - [x] Safe denormalization for visual display and plotting.
  - [x] Controlled training-only data augmentation (rotation $\pm 15^\circ$, zoom $0.92-1.08$, horizontal flip, shift $\pm 6\%$).
  - [x] Automated batch testing and audit over all 2,916 cleaned images with 100% success rate (`scripts/03_image_preprocessing.py`).
  - [x] Interactive Jupyter Notebook implemented (`notebooks/02_image_preprocessing.ipynb`).
  - [x] Unit test suite implemented and passing (`tests/test_preprocessing.py`).
  - [x] Comprehensive Markdown report and sample comparison figures (`reports/preprocessing_report.md`, `reports/preprocessing_samples/`).

- [x] **Module 4: Deep Learning Model Development Using MobileNetV2** *(Completed)*
  - [x] Environment and dependency verification (Python 3.12.10, TensorFlow 2.21.0, Keras 3.15.1, offline ImageNet weights).
  - [x] Pretrained MobileNetV2 architecture with in-graph data augmentation, rescaling, GAP, Dropout (0.2), and 28-class Softmax head.
  - [x] Two-stage training strategy: Frozen base feature extraction (Stage 1, $lr = 10^{-3}$) followed by fine-tuning top 34 layers (Stage 2, $lr = 10^{-5}$).
  - [x] Model checkpointing (`save_best_only=True`) and EarlyStopping with best weight restoration based on validation loss.
  - [x] Batch Normalization carefully kept in inference mode (`training=False`) during transfer learning to preserve ImageNet statistics.
  - [x] Zero data leakage: Stratified train (2,028), validation (423), and held-out test (465) splits verified with zero hash/path overlap.
  - [x] Held-out test set evaluation: **56.13% test accuracy**, **53.37% weighted F1-score**, and **49.32% macro F1-score** across 28 classes.
  - [x] Exported model and artifacts: `model/leaf_classifier.keras` (22.03 MB), `model/class_indices.json`, `reports/training_curves.png`, `reports/confusion_matrix.png`, `reports/classification_report.csv`, and `reports/module4_model_report.md`.
  - [x] Interactive walkthrough notebook implemented (`notebooks/03_model_development.ipynb`).
  - [x] Full unit testing and verification suite passing (16/16 unit tests, 11/11 audit criteria).

- [x] **Module 5: Model Evaluation and Prediction Pipeline** *(Completed)*
  - [x] Reusable single-image prediction module implemented (`utils/predictor.py`).
  - [x] Cached in-memory singleton model and class mapping loader.
  - [x] Multi-input format decoding (file paths, raw bytes, OpenCV NumPy arrays).
  - [x] Laplacian variance blur and sharpness detection ($L_{\text{var}} < 50.0$).
  - [x] MobileNetV2 preprocessing integration and full 28-class Softmax inference.
  - [x] Ranked candidate extraction: Top-3 classes with confidence, condition, and visual health type.
  - [x] Configurable confidence threshold (default 40.0%) with `UNCERTAIN_LOW_CONFIDENCE` handling.
  - [x] Agronomic plucking standard distinction (visual health separated from physical maturity).
  - [x] 100% JSON-serializable output schema ready for direct Flask API integration.
  - [x] Comprehensive demonstration script and test suite (`scripts/05_prediction_pipeline.py`).
  - [x] Diagnostic sample figures generated (`reports/prediction_samples/`).
  - [x] Unit test suite implemented and passing (`tests/test_predictor.py`, 12/12 passing; 28/28 across workspace).
  - [x] Comprehensive Markdown report generated (`reports/module5_prediction_report.md`).

- [x] **Module 6: Flask Backend Development** *(Completed)*
  - [x] Flask application factory and entry point implemented (`backend/app.py`).
  - [x] Service layer connection (`backend/services/prediction_service.py`) with singleton model caching.
  - [x] Diagnostic health check endpoint: `GET /api/health` with model readiness and zero path leaks.
  - [x] Single-image prediction endpoint: `POST /api/predict` (multipart upload, threshold overrides, Top-3, blur detection).
  - [x] Multi-environment configuration (`backend/config.py`, 16 MB upload limits, CORS).
  - [x] Standardized JSON error handlers (`backend/errors.py`, 400, 404, 405, 413, 415, 422, 500).
  - [x] Unit test suite implemented and passing (`tests/test_api.py`, 14/14 passing; 42/42 total across workspace).
  - [x] Comprehensive documentation report generated (`reports/module6_backend_report.md`).

- [x] **Module 7: Web Application Frontend Interface** *(Completed)*
  - [x] Modern, clean, responsive agricultural AI dashboard interface (`frontend/index.html`).
  - [x] Modular CSS design system (`frontend/css/style.css`, `frontend/css/dashboard.css`, `frontend/css/responsive.css`).
  - [x] Client JavaScript architecture (`frontend/js/api.js`, `frontend/js/upload.js`, `frontend/js/prediction.js`, `frontend/js/main.js`).
  - [x] Drag-and-drop file upload with format and file size validation (JPG, PNG, BMP, WEBP up to 16 MB).
  - [x] Real-time browser camera integration using HTML5 MediaDevices APIs with snapshot preview and retake.
  - [x] Direct multipart/form-data integration with Flask `POST /api/predict` and `GET /api/health`.
  - [x] Rich diagnostic results presentation: Top-3 candidate probability bars, confidence gauge, and blur alerts.
  - [x] Agronomic harvesting advisory distinguishing visual pathology from tactile leaf maturity.
  - [x] Responsive layout verified across mobile, tablet, and desktop viewports.
  - [x] Comprehensive integration test suite implemented and passing (`tests/test_frontend_integration.py`, 6/6; 48/48 across workspace).

- [x] **Module 8: SQLite Database Integration and Historical Analysis** *(Completed)*
  - [x] Embedded SQLite persistence initialized (`database/schema.sql`, `database/db.py`, `database/leaf_analysis.db`).
  - [x] Table schema created with 13 diagnostic fields and 4 performance B-tree indexes (`timestamp`, `species`, `visual_health`, `status`).
  - [x] WAL (Write-Ahead Logging) concurrency mode and connection timeout handling for robust multi-request read/write operations.
  - [x] Automated inference persistence integrated into `POST /api/predict` with isolated error containment (`history_saved` status).
  - [x] Path sanitization: Strictly stores image file basenames, completely eliminating OS and filesystem path leakage.
  - [x] Historical analysis REST API routes implemented (`backend/routes/history.py`):
    - `GET /api/history` (paginated, multi-field filtering by species/health/status, text substring search).
    - `GET /api/history/stats` (aggregated farm diagnostics: totals, health ratios, average confidence, species counts).
    - `GET /api/history/<id>` (individual diagnostic record retrieval with Top-3 probabilities and quality assessment).
    - `DELETE /api/history/<id>` (single record deletion).
    - `DELETE /api/history` (bulk clear requiring mandatory `?confirm=true` safeguard).
  - [x] Frontend history dashboard interface (`frontend/index.html`, `frontend/js/history.js`, `frontend/css/dashboard.css`):
    - 4 real-time analytics summary cards (Total Analyses, Healthy Foliage, Diseased Leaves, Average Confidence).
    - Live search bar with 300ms debounce and dynamic dropdown filters for species and health.
    - Responsive historical records table with color-coded health badges and plucking suitability indicators.
    - Diagnostic details modal with Top-3 candidate probability progress bars and quality metrics.
    - Live synchronization: new predictions automatically trigger a table and analytics refresh.
  - [x] Dedicated unit test suite implemented and passing (`tests/test_database.py`, 12/12 passing).
  - [x] REST API and frontend integration test suites updated and passing (71/71 tests passing across entire project).
  - [x] Comprehensive documentation report generated (`reports/module8_database_report.md`).

- [x] **Module 9: Final System Testing, Performance Validation, and Deployment** *(Completed)*
  - [x] Project-wide repository audit completed (model, class mappings, prediction pipeline, database, routes, frontend, configs).
  - [x] Complete 15-step End-to-End workflow verified (`tests/test_end_to_end.py`).
  - [x] Edge case & failure recovery test suite implemented (`tests/test_edge_cases.py`, 16/16 tests passing).
  - [x] Empirical system performance benchmarking completed (`scripts/benchmark_system_performance.py`, `reports/performance_benchmarks.json`):
    - Application Startup: 537.73 ms
    - Model Initialization (Cold Load): 8,032.77 ms
    - Single-Image Inference Latency (Warm): 480.29 ms mean (425.84 ms min)
    - Full End-to-End API Response Time: 615.51 ms mean (P95: 1,103.12 ms)
    - Database Operations: Insert 11.75 ms, Query 8.71 ms, Stats 9.24 ms
    - Process Memory: 30.61 MB start, 317.03 MB model footprint, 435.68 MB peak RSS
  - [x] Security verification completed (16 MB upload limit, OpenCV/PIL binary validation, parameterized SQL queries, path sanitization, CORS).
  - [x] Local production WSGI server configured with Waitress (`wsgi.py`).
  - [x] Production environment template created (`.env.example`).
  - [x] Frozen requirements updated with Waitress (`requirements.txt`).
  - [x] Final user guide & troubleshooting documentation completed.
  - [x] Comprehensive documentation report generated (`reports/final_system_testing_report.md`).
  - [x] Complete automated test suite passing: **88 / 88 tests passing (100%)** across 8 test suites in 25.60 seconds.
