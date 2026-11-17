"""
Leaf Quality Classification and Plucking Standard Monitoring System
Module 3: Image Preprocessing Utilities
File: utils/preprocessing.py

Description:
    Provides reusable, robust computer vision preprocessing functions tailored for
    the MobileNetV2 architecture (standard input: 224x224x3, values in [-1, 1]):
    - Safe image loading with OpenCV and PIL fallback.
    - Color space conversion (BGR -> RGB).
    - Image resizing with adaptive interpolation (cv2.INTER_AREA for downscaling).
    - MobileNetV2 normalization (scaling to [-1, 1]).
    - Safe denormalization for visual display and plotting.
    - Controlled training-only data augmentation (rotation, zoom, horizontal flip, shift).
    - End-to-end single-image and batch processing pipelines.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

import cv2
import numpy as np
from PIL import Image

# Global constants
TARGET_IMAGE_SIZE = (224, 224)  # Width, Height expected by MobileNetV2
VALID_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}

# TensorFlow availability flag (cached check)
_HAS_TF_CACHE: Optional[bool] = None

def check_tensorflow_available() -> bool:
    """Checks whether TensorFlow / Keras is available without importing unless queried."""
    global _HAS_TF_CACHE
    if _HAS_TF_CACHE is None:
        try:
            import importlib.util
            _HAS_TF_CACHE = importlib.util.find_spec("tensorflow") is not None
        except Exception:
            _HAS_TF_CACHE = False
    return _HAS_TF_CACHE

HAS_TENSORFLOW = check_tensorflow_available()


def load_image(image_path: Union[str, Path]) -> Optional[np.ndarray]:
    """
    Safely loads an image from disk using OpenCV with PIL fallback.
    
    Args:
        image_path: Path to the image file.
        
    Returns:
        np.ndarray in BGR uint8 format, or None if the file cannot be decoded.
    """
    path_obj = Path(image_path)
    if not path_obj.is_file():
        return None

    # Check extension
    if path_obj.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
        return None

    # Primary check: OpenCV decoding
    try:
        # Pass path as string (handles Windows path formatting)
        img_bgr = cv2.imread(str(path_obj), cv2.IMREAD_COLOR)
        if img_bgr is not None and img_bgr.size > 0:
            return img_bgr
    except Exception:
        pass

    # Secondary check: PIL fallback (e.g. for unusual EXIF or CMYK profiles)
    try:
        with Image.open(path_obj) as pil_img:
            # Convert to standard RGB first
            pil_rgb = pil_img.convert("RGB")
            np_rgb = np.array(pil_rgb)
            img_bgr = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)
            return img_bgr
    except Exception:
        return None


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Converts a BGR image array (OpenCV standard) to RGB (Matplotlib/Keras standard).
    
    Args:
        image: Input image array (H, W, 3) in BGR format.
        
    Returns:
        RGB image array (H, W, 3).
    """
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = TARGET_IMAGE_SIZE,
    interpolation: Optional[int] = None
) -> np.ndarray:
    """
    Resizes an image to the target dimensions (width, height) using bilinear interpolation.
    
    Defaults to cv2.INTER_LINEAR (bilinear interpolation) to maintain bit-for-bit
    consistency with Keras image_dataset_from_directory training pipeline.
    
    Args:
        image: Input image array.
        target_size: Target (width, height) tuple. Default: (224, 224).
        interpolation: Optional explicit cv2 interpolation flag (defaults to cv2.INTER_LINEAR).
        
    Returns:
        Resized image array with shape (height, width, channels).
    """
    target_w, target_h = target_size

    if interpolation is None:
        interpolation = cv2.INTER_LINEAR

    return cv2.resize(image, (target_w, target_h), interpolation=interpolation)


def mobilenet_v2_preprocess(image: Union[np.ndarray, Any]) -> Union[np.ndarray, Any]:
    """
    Applies the official MobileNetV2 preprocessing standard.
    Scales pixel values from [0, 255] to [-1.0, 1.0]:
        x = (x / 127.5) - 1.0
        
    Compatible with both tf.Tensor inputs and NumPy array inputs.
    Bit-for-bit identical to tensorflow.keras.applications.mobilenet_v2.preprocess_input(mode='tf').
    
    Args:
        image: RGB image array or Tensor with values in [0, 255].
        
    Returns:
        Float32 array or Tensor with values scaled to [-1.0, 1.0].
    """
    if HAS_TENSORFLOW and hasattr(image, "dtype") and not isinstance(image, np.ndarray):
        # Native TensorFlow / Keras Tensor (e.g. within tf.data pipeline or Keras layer)
        try:
            from keras.applications.mobilenet_v2 import preprocess_input as tf_mobilenet_preprocess
        except (ImportError, ModuleNotFoundError):
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as tf_mobilenet_preprocess  # type: ignore[import-untyped, import-not-found]
        return tf_mobilenet_preprocess(image)

    # High-performance NumPy implementation (mode='tf' formula)
    img_float = np.asarray(image, dtype=np.float32)
    return (img_float / 127.5) - 1.0


def denormalize_for_display(image: np.ndarray) -> np.ndarray:
    """
    Converts a MobileNetV2-normalized image (values in [-1.0, 1.0]) back to
    standard [0, 255] uint8 RGB for visualization and plotting.
    
    Args:
        image: Array with values in [-1.0, 1.0].
        
    Returns:
        np.ndarray of uint8 RGB values in [0, 255].
    """
    # Inverse formula: (x + 1.0) * 127.5
    img_display = (image + 1.0) * 127.5
    return np.clip(img_display, 0, 255).astype(np.uint8)


def apply_training_augmentation(
    image: np.ndarray,
    rng: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """
    Applies controlled, realistic data augmentation for leaf imagery during training:
    - Random horizontal flip (50% probability)
    - Small random rotation (between -15 and +15 degrees)
    - Small random zoom (scale factor between 0.90 and 1.10)
    - Minor translation (shift by up to +/- 8% of image dimensions)
    
    Preserves leaf pathology, lesion texture, and physiological appearance without
    distorting diagnostic visual features.
    
    Args:
        image: RGB or BGR image array.
        rng: Optional seeded np.random.RandomState instance for deterministic testing.
        
    Returns:
        Augmented image array of same dimensions and dtype.
    """
    if rng is None:
        rng = np.random.RandomState()

    augmented = image.copy()
    h, w = augmented.shape[:2]

    # 1. Random Horizontal Flip (p=0.5)
    if rng.rand() > 0.5:
        augmented = cv2.flip(augmented, 1)

    # 2. Random Rotation (-15 to +15 deg) and Zoom (0.92 to 1.08)
    angle = rng.uniform(-15.0, 15.0)
    scale = rng.uniform(0.92, 1.08)
    center = (w / 2.0, h / 2.0)

    rot_matrix = cv2.getRotationMatrix2D(center, angle, scale)

    # 3. Minor Translation (+/- 6% shift)
    tx = rng.uniform(-0.06, 0.06) * w
    ty = rng.uniform(-0.06, 0.06) * h
    rot_matrix[0, 2] += tx
    rot_matrix[1, 2] += ty

    # Warp affine with reflection border to avoid synthetic black margins
    augmented = cv2.warpAffine(
        augmented,
        rot_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    return augmented


def preprocess_single_image(
    image_path: Union[str, Path],
    target_size: Tuple[int, int] = TARGET_IMAGE_SIZE,
    is_training: bool = False,
    rng: Optional[np.random.RandomState] = None
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    End-to-end preprocessing pipeline for a single image file:
    1. Loads image safely via OpenCV.
    2. Converts BGR to RGB.
    3. Resizes to target_size (224x224).
    4. Applies augmentation ONLY if is_training=True.
    5. Normalizes using MobileNetV2 preprocess_input (values in [-1.0, 1.0]).
    
    Args:
        image_path: Path to the raw or cleaned image.
        target_size: Output resolution tuple (width, height).
        is_training: If True, applies training data augmentation.
        rng: Optional RandomState for deterministic augmentations.
        
    Returns:
        Tuple of (preprocessed_tensor, metadata_dict).
        If loading fails, returns (None, error_metadata_dict).
    """
    path_obj = Path(image_path)
    metadata = {
        "file_name": path_obj.name,
        "file_path": str(path_obj),
        "status": "failed",
        "original_shape": None,
        "preprocessed_shape": None,
        "is_training": is_training,
        "error": None
    }

    # Step 1: Load image
    bgr_img = load_image(path_obj)
    if bgr_img is None:
        metadata["error"] = "Failed to decode image"
        return None, metadata

    metadata["original_shape"] = bgr_img.shape

    # Step 2: Convert BGR to RGB
    rgb_img = bgr_to_rgb(bgr_img)

    # Step 3: Resize to target size (224, 224)
    resized_rgb = resize_image(rgb_img, target_size=target_size)

    # Step 4: Apply Augmentation (Training ONLY)
    if is_training:
        processed_rgb = apply_training_augmentation(resized_rgb, rng=rng)
    else:
        processed_rgb = resized_rgb

    # Step 5: MobileNetV2 Normalization (values in [-1, 1])
    normalized_tensor = mobilenet_v2_preprocess(processed_rgb)

    metadata["status"] = "success"
    metadata["preprocessed_shape"] = normalized_tensor.shape
    metadata["min_val"] = float(np.min(normalized_tensor))
    metadata["max_val"] = float(np.max(normalized_tensor))
    metadata["mean_val"] = float(np.mean(normalized_tensor))

    return normalized_tensor, metadata


def preprocess_batch(
    image_paths: List[Union[str, Path]],
    target_size: Tuple[int, int] = TARGET_IMAGE_SIZE,
    is_training: bool = False
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Preprocesses a batch of image file paths into a 4D NumPy tensor (N, 224, 224, 3)
    and an accompanying list of metadata records.
    
    Args:
        image_paths: List of image file paths.
        target_size: Output resolution tuple (width, height).
        is_training: If True, applies training data augmentation.
        
    Returns:
        Tuple of (batch_tensor, metadata_list).
    """
    tensors = []
    metadata_list = []
    
    for p in image_paths:
        tensor, meta = preprocess_single_image(p, target_size=target_size, is_training=is_training)
        if tensor is not None and meta["status"] == "success":
            tensors.append(tensor)
            metadata_list.append(meta)
            
    if tensors:
        batch_array = np.stack(tensors, axis=0).astype(np.float32)
    else:
        batch_array = np.empty((0, target_size[1], target_size[0], 3), dtype=np.float32)
        
    return batch_array, metadata_list
