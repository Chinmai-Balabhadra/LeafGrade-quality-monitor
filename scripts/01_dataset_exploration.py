"""
Leaf Quality Classification and Plucking Standard Monitoring System
Module 1: Dataset Exploration and Verification Script
File: scripts/01_dataset_exploration.py

Description:
    This script inspects, audits, and validates raw image datasets for the project:
    1. PlantDoc Dataset (expected at datasets/raw/plantdoc/)
    2. New Plant Diseases Dataset (expected at datasets/raw/new_plant_diseases/)

Capabilities:
    - Scans dataset directories and detects class subdirectories.
    - Counts images per class and overall.
    - Identifies valid image formats (.jpg, .jpeg, .png, .bmp, .webp, .tiff).
    - Checks for unreadable, truncated, or corrupted images.
    - Extracts image dimension statistics (height, width, aspect ratio, sample resolution).
    - Identifies empty folders.
    - Outputs a console distribution summary.
    - Exports a structured dataset inventory to reports/dataset_inventory.csv.
    - Generates class distribution visualization charts in reports/.
    - Gracefully handles missing directories or datasets not yet placed without crashing.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Root project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_RAW_DIR = PROJECT_ROOT / "datasets" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Set matplotlib config directory inside project reports to avoid permission warnings on Windows
os.environ.setdefault("MPLCONFIGDIR", str(REPORTS_DIR / ".mplconfig"))

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Supported image extensions
VALID_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}

DATASET_CONFIGS = [
    {
        "name": "PlantDoc",
        "path": DATASETS_RAW_DIR / "plantdoc",
        "kaggle_url": "https://www.kaggle.com/datasets/daeyun/plantdoc-dataset",
        "expected_role": "In-field, handheld camera images with complex backgrounds"
    },
    {
        "name": "New Plant Diseases",
        "path": DATASETS_RAW_DIR / "new_plant_diseases",
        "kaggle_url": "https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset",
        "expected_role": "Controlled-environment leaf disease and healthy benchmarks"
    }
]


def setup_environment() -> None:
    """Ensure required output directories exist."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / ".mplconfig").mkdir(parents=True, exist_ok=True)


def is_image_corrupted(file_path: Path) -> Tuple[bool, Tuple[int, int, int]]:
    """
    Validates if an image file is readable and uncorrupted using both PIL and OpenCV.
    
    Returns:
        Tuple of (is_corrupt: bool, dimensions: (height, width, channels))
    """
    # Check 1: PIL integrity verification
    try:
        with Image.open(file_path) as img:
            img.verify()
    except Exception:
        return True, (0, 0, 0)

    # Check 2: OpenCV decoding check (reads actual pixel data)
    try:
        img_cv = cv2.imread(str(file_path))
        if img_cv is None:
            return True, (0, 0, 0)
        h, w = img_cv.shape[:2]
        c = img_cv.shape[2] if len(img_cv.shape) > 2 else 1
        return False, (h, w, c)
    except Exception:
        return True, (0, 0, 0)


def scan_dataset(dataset_name: str, dataset_path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Scans a dataset directory structure, handling both single-tier and multi-tier
    (e.g., train/valid/test or class-only) organizations.
    
    Returns:
        Tuple of (inventory_records, empty_folders)
    """
    records: List[Dict[str, Any]] = []
    empty_folders: List[str] = []

    if not dataset_path.exists():
        return records, empty_folders

    # Walk directory to find all subdirectories
    subdirs = [p for p in dataset_path.glob("**/*") if p.is_dir()]
    all_dirs_to_check = [dataset_path] + subdirs

    for folder in all_dirs_to_check:
        # Ignore internal system/gitkeep files
        files = [f for f in folder.iterdir() if f.is_file() and f.name != ".gitkeep"]
        image_files = [f for f in files if f.suffix.lower() in VALID_IMAGE_EXTENSIONS]

        # Determine relative class name
        rel_path = folder.relative_to(dataset_path)
        rel_str = str(rel_path).replace("\\", "/")
        class_name = "root" if rel_str == "." else rel_str

        # If it has subdirectories and no images, it's an intermediate directory
        has_subdirs = any(p.is_dir() for p in folder.iterdir())
        if has_subdirs and len(image_files) == 0:
            continue

        if len(files) == 0:
            if class_name != "root":
                empty_folders.append(class_name)
            continue

        if len(image_files) == 0:
            # Non-image files present, but no images
            continue

        # Inspect images in this class
        formats_found = set()
        corrupted_count = 0
        heights = []
        widths = []
        sample_dim_str = "N/A"

        for img_path in image_files:
            ext = img_path.suffix.lower()
            formats_found.add(ext)

            is_corrupt, (h, w, c) = is_image_corrupted(img_path)
            if is_corrupt:
                corrupted_count += 1
            else:
                heights.append(h)
                widths.append(w)
                if sample_dim_str == "N/A":
                    sample_dim_str = f"{w}x{h}x{c}"

        dim_stats = "N/A"
        if heights and widths:
            min_dim = f"{min(widths)}x{min(heights)}"
            max_dim = f"{max(widths)}x{max(heights)}"
            avg_dim = f"{int(np.mean(widths))}x{int(np.mean(heights))}"
            dim_stats = f"Avg: {avg_dim}, Min: {min_dim}, Max: {max_dim}"

        records.append({
            "Dataset Name": dataset_name,
            "Original Class Name": class_name,
            "Number of Images": len(image_files),
            "Image Formats": ", ".join(sorted(formats_found)) if formats_found else "None",
            "Sample Image Dimensions": sample_dim_str,
            "Dimension Stats (Avg, Min, Max)": dim_stats,
            "Number of Corrupted Images": corrupted_count
        })

    return records, empty_folders


def generate_class_distribution_chart(df: pd.DataFrame, output_path: Path) -> None:
    """Generates a bar plot showing the image distribution per class."""
    if df.empty or "Number of Images" not in df.columns:
        print("[*] Skipping chart generation: no dataset records found.")
        return

    # Take top 30 classes if there are too many to fit on a single readable plot
    df_sorted = df.sort_values(by="Number of Images", ascending=False)
    plot_df = df_sorted.head(30)

    plt.figure(figsize=(14, 8))
    bars = plt.barh(plot_df["Original Class Name"], plot_df["Number of Images"], color="#2e7d32")
    plt.xlabel("Number of Images", fontsize=12)
    plt.ylabel("Class Name", fontsize=12)
    
    title = "Top 30 Class Distribution - Leaf Datasets"
    if len(df) > 30:
        title += f" (Showing top 30 of {len(df)} total classes)"
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.gca().invert_yaxis()
    plt.tight_layout()

    # Add count labels on bars
    max_val = max(df["Number of Images"]) if not df.empty else 1
    for bar in bars:
        width = bar.get_width()
        plt.text(width + max_val * 0.005, 
                 bar.get_y() + bar.get_height() / 2, 
                 f"{int(width)}", 
                 va='center', fontsize=9)

    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Distribution chart saved to: {output_path}")


def run_exploration() -> pd.DataFrame:
    """Main execution function for dataset exploration."""
    setup_environment()
    print("=" * 80)
    print(" CSE471: Leaf Quality Monitoring System - Dataset Exploration Audit ")
    print("=" * 80)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Raw Datasets Location: {DATASETS_RAW_DIR}\n")

    all_records: List[Dict[str, Any]] = []
    missing_datasets: List[Dict[str, Any]] = []
    empty_folders_detected: List[str] = []

    for cfg in DATASET_CONFIGS:
        name = cfg["name"]
        path = cfg["path"]
        print(f"[*] Scanning {name} Dataset...")
        print(f"    Target Path: {path}")

        records, empty_folders = scan_dataset(name, path)
        if records:
            all_records.extend(records)
            total_imgs_in_ds = sum(r["Number of Images"] for r in records)
            print(f"    [STATUS] Found {len(records)} class folder(s) with {total_imgs_in_ds:,} image(s).")
        else:
            print(f"    [STATUS] Missing or no image files placed yet.")
            missing_datasets.append(cfg)

        if empty_folders:
            empty_folders_detected.extend([f"{name} -> {ef}" for ef in empty_folders])

    print("\n" + "-" * 80)
    print(" EXPLORATION AUDIT SUMMARY")
    print("-" * 80)

    inventory_csv_path = REPORTS_DIR / "dataset_inventory.csv"
    chart_png_path = REPORTS_DIR / "class_distribution.png"

    if all_records:
        df = pd.DataFrame(all_records)
        total_images = df["Number of Images"].sum()
        total_corrupt = df["Number of Corrupted Images"].sum()

        print(f"Total Categories/Folders Identified : {len(df)}")
        print(f"Total Images Found                  : {total_images:,}")
        print(f"Total Corrupted/Unreadable Images   : {total_corrupt}")
        
        # Save CSV
        df.to_csv(inventory_csv_path, index=False)
        print(f"[+] Dataset inventory saved to       : {inventory_csv_path}")

        # Generate chart
        generate_class_distribution_chart(df, chart_png_path)

        # Print detailed preview
        print("\nDataset Class Distribution Preview (First 10 records):")
        print(df[["Dataset Name", "Original Class Name", "Number of Images", "Image Formats", "Sample Image Dimensions"]].head(10).to_string(index=False))

    else:
        print("[!] No image datasets detected yet in datasets/raw/.")
        # Create an informative template CSV with correct schema
        empty_df = pd.DataFrame(columns=[
            "Dataset Name",
            "Original Class Name",
            "Number of Images",
            "Image Formats",
            "Sample Image Dimensions",
            "Dimension Stats (Avg, Min, Max)",
            "Number of Corrupted Images"
        ])
        empty_df.to_csv(inventory_csv_path, index=False)
        print(f"[+] Schema template created at       : {inventory_csv_path}")

    if empty_folders_detected:
        print("\n[!] Empty Folders Detected:")
        for ef in empty_folders_detected:
            print(f"    - {ef}")

    if missing_datasets:
        print("\n" + "=" * 80)
        print(" ACTION REQUIRED: KAGGLE DATASET PLACEMENT INSTRUCTIONS")
        print("=" * 80)
        for ds in missing_datasets:
            print(f"\n[>] {ds['name']} Dataset:")
            print(f"    1. Download from Kaggle: {ds['kaggle_url']}")
            print(f"    2. Extract the downloaded ZIP file.")
            print(f"    3. Place the extracted image folders directly into:")
            print(f"       {ds['path']}")
            print(f"    Role: {ds['expected_role']}")
        print("\nAfter placing your datasets, re-run this script to generate full inventory & charts:")
        print("    .\\.venv\\Scripts\\python.exe scripts/01_dataset_exploration.py")
        print("=" * 80)

    # Note regarding category mapping and 'Not Ready' requirement
    print("\n[CRITICAL NOTE ON CSE471 QUALITY CATEGORY MAPPINGS]:")
    print(" - Original class names must be preserved without premature relabeling.")
    print(" - Neither PlantDoc nor New Plant Diseases automatically maps 1:1 to 'Good Quality',")
    print("   'Poor Quality', or 'Not Ready'.")
    print(" - The 'Not Ready' category requires reliable immature/tender leaf labels.")
    print(" - If no immature leaf images are present in the downloaded datasets, 'Not Ready'")
    print("   must be flagged as a missing dataset requirement before training in Module 2.")
    print("=" * 80)

    return pd.DataFrame(all_records) if all_records else empty_df


if __name__ == "__main__":
    run_exploration()
