# Setup and Environment Information
**Project**: Leaf Quality Classification and Plucking Standard Monitoring System Using Handheld Camera  
**Course Code**: CSE471 Academic Project  
**Module**: Module 1 - Project Setup and Dataset Collection  
**Date**: November 2026  

---

## 1. Environment Specifications

| Component | Specification |
| :--- | :--- |
| **Operating System** | Windows 11 (64-bit) |
| **Workspace Root** | `c:\Users\ASUS\OneDrive\Desktop\Project-471` |
| **Python Version** | Python 3.12.10 (64-bit) |
| **Virtual Environment** | `.venv` located at project root |
| **Package Installer** | pip 25.0.1 |
| **Version Control** | Git (`.gitignore` configured, no remote initialized) |

---

## 2. Installed Dependencies (Module 1)

The following core libraries were installed specifically for Module 1 dataset exploration and audit:

| Package | Version | Primary Purpose |
| :--- | :--- | :--- |
| `numpy` | 2.5.4 | Numerical matrix and array operations |
| `pandas` | 3.1.1 | Dataset inventory tables, tabular structuring, CSV exports |
| `matplotlib` | 3.11.2 | Class distribution histograms and EDA plotting |
| `opencv-python` | 5.0.0.93 | Image decoding, channel inspection, resolution extraction |
| `pillow` (PIL) | 13.0.0 | Image file format validation and corruption verification |
| `jupyter` | 1.1.1 | Interactive notebook environment for exploratory analysis |
| `tqdm` | 4.70.1 | Terminal progress bar indicators for batch scans |

*Note: Heavy deep-learning frameworks (e.g., TensorFlow/Keras) are deliberately deferred to Module 2 to maintain a lean environment for data audit.*

---

## 3. How to Activate the Virtual Environment

### Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

### Windows Command Prompt (cmd.exe):
```cmd
.\.venv\Scripts\activate.bat
```

---

## 4. Kaggle Dataset Download & Placement Guide

Two external datasets are required for this project. They must be downloaded manually:

### Dataset 1: PlantDoc Dataset
1. **Source**: [Kaggle - PlantDoc Dataset](https://www.kaggle.com/datasets/daeyun/plantdoc-dataset)
2. **Download**: Download the dataset archive (`.zip`).
3. **Extraction Target**:
   ```text
   datasets/raw/plantdoc/
   ```
4. **Characteristics**: Field-captured leaf images featuring real-world conditions, diverse angles, natural sunlight, and background foliage.

### Dataset 2: New Plant Diseases Dataset
1. **Source**: [Kaggle - New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
2. **Download**: Download the dataset archive (`.zip`).
3. **Extraction Target**:
   ```text
   datasets/raw/new_plant_diseases/
   ```
4. **Characteristics**: High-resolution, controlled-background leaf pathology images (healthy vs. diseased across multiple crop species).

---

## 5. Running the Exploration Tools

### 1. Run the Dataset Exploration Script:
```powershell
.\.venv\Scripts\python.exe scripts/01_dataset_exploration.py
```
This script will:
- Traverse all subdirectories inside `datasets/raw/plantdoc/` and `datasets/raw/new_plant_diseases/`.
- Validate file headers and decode pixels to detect corrupted images.
- Compute average, minimum, and maximum image dimensions per class.
- Generate or update `reports/dataset_inventory.csv`.
- Save `reports/class_distribution.png` once images are present.

### 2. Launch the Interactive Exploration Notebook:
```powershell
.\.venv\Scripts\jupyter.exe notebook notebooks/01_dataset_exploration.ipynb
```
Or open the notebook directly inside VS Code with the `.venv` kernel selected.
