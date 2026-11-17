# Module 3 Report: Image Preprocessing Pipeline

**Project Title**: Leaf Quality Classification and Plucking Standard Monitoring System Using Handheld Camera  
**Course Code**: CSE471 Academic Project  
**Module**: Module 3 - Image Preprocessing  
**Status**: **Fully Verified and Completed**  
**Date**: November 2026  

---

## 1. Executive Summary

Module 3 establishes a standardized, reproducible, and non-destructive image preprocessing pipeline tailored specifically for **MobileNetV2** deep learning architecture. All operations strictly preserve original dataset files and prepare tensors directly for downstream model training and real-time handheld camera inference.

---

## 2. Quantitative Preprocessing Statistics

| Metric | Result | Description |
| :--- | :--- | :--- |
| **Total Cleaned Images Inspected** | **2,916** | All images in `datasets/cleaned/plantdoc/original_classes/` |
| **Successfully Processed Images** | **2,916** | Decoded, resized, color-standardized, and normalized |
| **Failed Image Loads** | **0** | Zero broken, truncated, or unreadable images |
| **Processing Success Rate** | **100.00%** | Full verification across all 28 classes |
| **Original Dimensions Range** | Min: 115x69 &#124; Max: 6000x6000 &#124; Avg: 908x802 | Wide distribution of handheld and field resolutions |
| **Target Model Input Dimensions** | **224 × 224 × 3** | Height: 224, Width: 224, Channels: 3 (RGB) |
| **Normalization Value Range** | **[-1.00, 1.00]** | Exactly matches MobileNetV2 standard $[-1.0, 1.0]$ |
| **Global Pixel Mean** | **-0.079** | Centered near zero across dataset |

---

## 3. Pipeline Architecture & Implementation Details

### A. Image Ingestion and Loading
- **Method**: Implemented in `utils/preprocessing.py` via `load_image()`.
- **Primary Reader**: OpenCV (`cv2.imread()`) with automated PIL fallback (`Image.open()`) to seamlessly handle diverse color spaces (`RGB`, `RGBA`, `L`, `CMYK`) without data loss.

### B. Color Space Standardization
- **Method**: OpenCV natively loads in BGR channel order.
- **Conversion**: Converted to RGB via `bgr_to_rgb()` (`cv2.COLOR_BGR2RGB`), ensuring compatibility with Matplotlib plotting and MobileNetV2 channel weights.

### C. Adaptive Image Resizing
- **Target Dimensions**: $224 \times 224$ pixels.
- **Interpolation Strategy**:
  - Downsampling: `cv2.INTER_AREA` (preserves fine leaf vein structures, eliminates aliasing artifacts).
  - Upsampling: `cv2.INTER_LINEAR` (smooth bilinear interpolation).

### D. MobileNetV2 Normalization
- **Standard Formula**:
  $$x_{\text{norm}} = \frac{x}{127.5} - 1.0$$
- **Range**: Scales 8-bit unsigned integers $[0, 255]$ into floating-point numbers in $[-1.0, 1.0]$.
- **Framework Support**: Implemented via `mobilenet_v2_preprocess()` with seamless support for `tensorflow.keras.applications.mobilenet_v2.preprocess_input` and an identical NumPy vector implementation.
- **Consistency Guarantee**: Zero double-normalization or scaling conflicts. The exact same transformation is utilized for training and single-image handheld inference.

### E. Controlled Training-Only Data Augmentation
- **Rule**: Applied **strictly** when `is_training=True`. Validation and test splits remain pristine and unaugmented.
- **Transformations**:
  1. *Horizontal Flip*: 50% probability (`cv2.flip(img, 1)`).
  2. *Small Rotation*: Random angle in $[-15^\circ, +15^\circ]$ (simulates hand tilt on phone camera).
  3. *Small Zoom*: Scaling factor in $[0.92, 1.08]$ (simulates phone distance to leaf).
  4. *Minor Translation*: Shift in $[-6\%, +6\%]$ (simulates off-center framing).
  5. *Border Handling*: `cv2.BORDER_REFLECT_101` eliminates artificial black borders.

---

## 4. Before-and-After Visualizations

Visual verification figures were generated and saved in `reports/preprocessing_samples/`. Each figure showcases:
1. Original Raw Image
2. Resized Image ($224 \times 224$)
3. RGB Conversion
4. MobileNetV2 Model Input (Denormalized display of $[-1, 1]$ tensor)
5. Training Augmentation Sample 1
6. Training Augmentation Sample 2

Generated visualization artifacts:
- [preprocessing_Apple_leaf.png](file:///c:/Users/ASUS/OneDrive/Desktop/Project-471/reports/preprocessing_samples/preprocessing_Apple_leaf.png)
- [preprocessing_Tomato_Early_blight_leaf.png](file:///c:/Users/ASUS/OneDrive/Desktop/Project-471/reports/preprocessing_samples/preprocessing_Tomato_Early_blight_leaf.png)
- [preprocessing_Corn_rust_leaf.png](file:///c:/Users/ASUS/OneDrive/Desktop/Project-471/reports/preprocessing_samples/preprocessing_Corn_rust_leaf.png)
- [preprocessing_grape_leaf.png](file:///c:/Users/ASUS/OneDrive/Desktop/Project-471/reports/preprocessing_samples/preprocessing_grape_leaf.png)
- [preprocessing_Bell_pepper_leaf_spot.png](file:///c:/Users/ASUS/OneDrive/Desktop/Project-471/reports/preprocessing_samples/preprocessing_Bell_pepper_leaf_spot.png)
- [preprocessing_Potato_leaf_late_blight.png](file:///c:/Users/ASUS/OneDrive/Desktop/Project-471/reports/preprocessing_samples/preprocessing_Potato_leaf_late_blight.png)

---

## 5. Technology and Dependency Compatibility

- **Python Version**: 3.12.10 (64-bit)
- **TensorFlow Status**: Available (TensorFlow installed in environment)
- **OpenCV Version**: 5.0.0.93
- **NumPy Version**: 2.5.4
- **Preservation Check**: No original files were modified or deleted; no duplicate full-dataset copies were created.

---

## 6. Limitations & Considerations for Module 4

1. **Resolution Downsampling**: High-resolution captures ($6000 \times 6000$) downscaled to $224 \times 224$ compress microscopic bacterial lesions into sub-pixel regions. High-quality `cv2.INTER_AREA` interpolation mitigates this.
2. **Pending Dataset**: The **New Plant Diseases Dataset** will be preprocessed using this identical `utils/preprocessing.py` pipeline once placed.
3. **Maturity Gap**: The missing *Not Ready* immature leaf class remains unpopulated until dedicated immature leaf images are integrated.
