"""
Leaf Quality Classification and Plucking Standard Monitoring System
Module 5: Single-Image Prediction Pipeline & Inference Utilities
File: utils/predictor.py

Description:
    Provides a production-grade, API-ready inference pipeline for single leaf images:
    - Cached singleton model and class mapping loader.
    - Multi-input format support (file path, raw bytes, OpenCV BGR / RGB NumPy array).
    - Image quality assessment (Laplacian variance sharpness, blur detection).
    - MobileNetV2 preprocessing integration.
    - Softmax probability distribution extraction and Top-3 ranked candidate parsing.
    - Configurable confidence thresholding with explicit uncertainty flagging.
    - Agronomic quality categorization and separate plucking readiness recommendation.
    - 100% JSON-serializable outputs for direct Flask API integration.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

import cv2
import numpy as np

from utils.preprocessing import bgr_to_rgb, resize_image, TARGET_IMAGE_SIZE

# Ensure TensorFlow/Keras environment configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("KERAS_HOME", str(PROJECT_ROOT / ".keras"))

logger = logging.getLogger(__name__)

# Default file paths
DEFAULT_MODEL_PATH = PROJECT_ROOT / "model" / "leaf_classifier.keras"
DEFAULT_CLASS_INDICES_PATH = PROJECT_ROOT / "model" / "class_indices.json"

# Supported image file extensions
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

# Production Thresholds
DEFAULT_CONFIDENCE_THRESHOLD = 0.50  # 50% confidence threshold for certain classification
BLUR_LAPLACIAN_THRESHOLD = 50.0      # Variance below 50 indicates high blur / out-of-focus camera


class LeafPredictor:
    """
    Production-ready leaf quality predictor encapsulating model inference,
    quality analysis, and agricultural recommendation logic.
    """

    _instance: Optional["LeafPredictor"] = None

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        class_indices_path: Optional[Union[str, Path]] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        blur_threshold: float = BLUR_LAPLACIAN_THRESHOLD
    ):
        """
        Initializes the LeafPredictor.

        Args:
            model_path: Path to the trained Keras model (.keras).
            class_indices_path: Path to class_indices.json.
            confidence_threshold: Minimum confidence required for certain classification.
            blur_threshold: Laplacian variance threshold below which images are flagged as blurry.
        """
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.class_indices_path = Path(class_indices_path) if class_indices_path else DEFAULT_CLASS_INDICES_PATH
        self.confidence_threshold = float(confidence_threshold)
        self.blur_threshold = float(blur_threshold)

        self.model: Optional[Any] = None
        self.class_indices_data: Dict[str, Any] = {}
        self.idx_to_class: Dict[int, str] = {}
        self.class_to_idx: Dict[str, int] = {}
        self.metadata: Dict[str, Any] = {}

        self._load_class_indices()
        self._load_model()

        # Initialize Out-of-Distribution (OOD) Leaf Validator
        from utils.leaf_validator import RobustLeafValidator
        self.validator = RobustLeafValidator.get_instance()
        if self.model is not None:
            self.validator.bind_feature_extractor(self.model)

    def _load_class_indices(self) -> None:
        """Loads and parses the class mapping and plucking metadata."""
        if not self.class_indices_path.exists():
            raise FileNotFoundError(f"Class indices file not found at: {self.class_indices_path}")

        with open(self.class_indices_path, "r", encoding="utf-8") as f:
            self.class_indices_data = json.load(f)

        # Parse idx_to_class with integer keys
        raw_idx = self.class_indices_data.get("idx_to_class", {})
        self.idx_to_class = {int(k): v for k, v in raw_idx.items()}
        self.class_to_idx = self.class_indices_data.get("class_to_idx", {})
        self.metadata = self.class_indices_data.get("metadata", {})

        if len(self.idx_to_class) != 28:
            raise ValueError(f"Expected 28 classes in mapping, found {len(self.idx_to_class)}")

    def _load_model(self) -> None:
        """Loads the trained Keras MobileNetV2 model into memory."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")

        try:
            import keras
            self.model = keras.models.load_model(str(self.model_path), compile=False)
            logger.info(f"Loaded leaf classifier model from: {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load Keras model from {self.model_path}: {e}") from e

    @classmethod
    def get_instance(cls) -> "LeafPredictor":
        """Singleton accessor to reuse the loaded model across requests."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def assess_image_quality(image_bgr: np.ndarray, blur_threshold: float = BLUR_LAPLACIAN_THRESHOLD) -> Dict[str, Any]:
        """
        Calculates image quality metrics including sharpness (Laplacian variance),
        brightness, and contrast to detect low-quality field captures.

        Args:
            image_bgr: Loaded BGR image as NumPy uint8 array.
            blur_threshold: Threshold below which image is considered blurry.

        Returns:
            Dict containing quality metrics and blur flag.
        """
        if image_bgr is None or image_bgr.size == 0:
            return {"is_blurry": True, "laplacian_variance": 0.0, "status": "INVALID_IMAGE"}

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_brightness = float(np.mean(gray))
        contrast_std = float(np.std(gray))

        is_blurry = bool(laplacian_var < blur_threshold)
        quality_rating = "Sharp" if not is_blurry else ("Very Blurry" if laplacian_var < (blur_threshold / 2) else "Slightly Blurry")

        return {
            "is_blurry": is_blurry,
            "quality_rating": quality_rating,
            "laplacian_variance": round(laplacian_var, 2),
            "mean_brightness": round(mean_brightness, 2),
            "contrast_std": round(contrast_std, 2),
            "original_height": int(image_bgr.shape[0]),
            "original_width": int(image_bgr.shape[1]),
            "channels": int(image_bgr.shape[2]) if len(image_bgr.shape) == 3 else 1
        }

    def decode_image_input(
        self,
        image_input: Union[str, Path, bytes, bytearray, np.ndarray]
    ) -> Tuple[Optional[np.ndarray], Optional[str], Optional[str]]:
        """
        Decodes diverse input types into a standardized BGR uint8 NumPy array.

        Returns:
            Tuple of (image_bgr, source_name, error_message)
        """
        if isinstance(image_input, (str, Path)):
            path_obj = Path(image_input)
            source_name = path_obj.name

            if not path_obj.exists():
                return None, source_name, f"File does not exist: {path_obj}"

            if not path_obj.is_file():
                return None, source_name, f"Path is not a regular file: {path_obj}"

            ext = path_obj.suffix.lower()
            if ext not in SUPPORTED_IMAGE_EXTENSIONS:
                return None, source_name, f"Unsupported file extension '{ext}'. Supported formats: {sorted(list(SUPPORTED_IMAGE_EXTENSIONS))}"

            # Check for 0-byte or corrupted file
            if path_obj.stat().st_size == 0:
                return None, source_name, "File is empty (0 bytes)."

            # Load via OpenCV
            img_bgr = cv2.imread(str(path_obj))
            if img_bgr is None:
                # Fallback to PIL
                try:
                    from PIL import Image
                    with Image.open(path_obj) as pil_img:
                        pil_rgb = pil_img.convert("RGB")
                        img_bgr = cv2.cvtColor(np.array(pil_rgb), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    return None, source_name, f"Corrupted or unreadable image file: {e}"

            return img_bgr, source_name, None

        elif isinstance(image_input, (bytes, bytearray)):
            source_name = "in_memory_bytes"
            if len(image_input) == 0:
                return None, source_name, "Received empty byte array (0 bytes)."

            nparr = np.frombuffer(image_input, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                return None, source_name, "Failed to decode in-memory bytes into an image."
            return img_bgr, source_name, None

        elif isinstance(image_input, np.ndarray):
            source_name = "numpy_array"
            if image_input.size == 0:
                return None, source_name, "Received empty NumPy array."

            # Check dimensionality
            if len(image_input.shape) == 2:
                img_bgr = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            elif len(image_input.shape) == 3 and image_input.shape[2] == 3:
                # Assume uint8 BGR
                if image_input.dtype != np.uint8:
                    img_bgr = np.clip(image_input, 0, 255).astype(np.uint8)
                else:
                    img_bgr = image_input.copy()
            else:
                return None, source_name, f"Invalid array dimensions: {image_input.shape}. Expected (H, W, 3) or (H, W)."

            return img_bgr, source_name, None

        else:
            return None, "unknown", f"Unsupported input type: {type(image_input)}. Expected file path, bytes, or NumPy array."

    def predict(
        self,
        image_input: Union[str, Path, bytes, bytearray, np.ndarray],
        confidence_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end prediction on a single leaf image.

        Args:
            image_input: File path, byte stream, or NumPy array.
            confidence_threshold: Optional custom confidence threshold override.

        Returns:
            JSON-serializable dictionary with predictions, top-3 candidates,
            quality assessment, plucking recommendations, and diagnostic status.
        """
        eff_threshold = float(confidence_threshold) if confidence_threshold is not None else self.confidence_threshold

        # 1. Decode Image Input
        img_bgr, source_name, decode_error = self.decode_image_input(image_input)
        if decode_error is not None or img_bgr is None:
            return {
                "status": "ERROR",
                "message": decode_error,
                "source": source_name,
                "predicted_class": None,
                "confidence": 0.0,
                "top_three": [],
                "quality_category": "Unknown / Unsupported",
                "plucking_recommendation": "Unable to evaluate: image input could not be loaded or processed.",
                "quality_assessment": {"status": "DECODE_FAILED"}
            }

        # 2. Image Quality & Sharpness Assessment
        quality_meta = self.assess_image_quality(img_bgr, self.blur_threshold)

        # 3. Out-of-Distribution (OOD) Input Validation BEFORE MobileNetV2 28-Class Classifier
        is_valid_leaf, val_message, val_code, val_details = self.validator.validate(img_bgr)
        if not is_valid_leaf:
            logger.info(f"OOD input rejected ({val_code}) for source '{source_name}': {val_message}")
            return {
                "status": "REJECTED_NOT_LEAF",
                "message": val_message,
                "error": "REJECTED_NOT_LEAF",
                "is_leaf": False,
                "source": source_name,
                "predicted_class": None,
                "predicted_index": None,
                "confidence": 0.0,
                "confidence_percent": 0.0,
                "confidence_threshold_applied": round(eff_threshold, 4),
                "is_above_threshold": False,
                "confidence_warning": val_message,
                "blur_warning": quality_meta.get("is_blurry") and f"Laplacian variance ({quality_meta['laplacian_variance']}) indicates blur.",
                "visual_health_category": "Invalid Input",
                "is_visually_healthy": False,
                "plucking_recommendation": val_message,
                "species": "Non-Leaf / Out-of-Distribution",
                "condition": "Rejected",
                "plucking_suitability": "Not Applicable",
                "top_three": [],
                "quality_assessment": quality_meta,
                "validation_details": {
                    "stage": val_code,
                    "rejection_reason": val_message,
                    **val_details
                },
                "disclaimer": "The input image was rejected by Out-of-Distribution validation because no valid plant leaf was detected."
            }

        # 4. Preprocess for MobileNetV2 (consistent with Keras training pipeline)
        # Convert BGR to RGB using shared preprocessing utility
        img_rgb = bgr_to_rgb(img_bgr)
        # Resize to standard 224x224 using cv2.INTER_LINEAR (bilinear interpolation matching model training)
        img_resized = resize_image(img_rgb, target_size=TARGET_IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
        # Convert to float32 batch [1, 224, 224, 3] in [0, 255] range
        # Note: In-graph Rescaling layer inside leaf_classifier.keras performs the [-1, 1] scaling
        input_tensor = np.expand_dims(img_resized.astype(np.float32), axis=0)

        # 4. Model Inference
        try:
            probabilities = self.model.predict(input_tensor, verbose=0)[0]
        except Exception as e:
            return {
                "status": "ERROR",
                "message": f"Inference execution failed: {e}",
                "source": source_name,
                "predicted_class": None,
                "confidence": 0.0,
                "top_three": [],
                "quality_category": "Unknown / Unsupported",
                "plucking_recommendation": "Inference failure.",
                "quality_assessment": quality_meta
            }

        # Normalize probabilities defensively
        prob_sum = float(np.sum(probabilities))
        if prob_sum > 0:
            probabilities = probabilities / prob_sum

        # 5. Extract Ranked Candidates (Top-3)
        top_indices = np.argsort(probabilities)[::-1][:3]
        top_candidates: List[Dict[str, Any]] = []

        for rank, idx in enumerate(top_indices, start=1):
            class_name = self.idx_to_class.get(int(idx), "Unknown")
            conf = float(probabilities[idx])
            class_meta = self.metadata.get(class_name, {})

            top_candidates.append({
                "rank": rank,
                "class_name": class_name,
                "class_index": int(idx),
                "confidence": round(conf, 4),
                "confidence_percent": round(conf * 100.0, 2),
                "species": class_meta.get("species", "Unknown"),
                "condition": class_meta.get("condition", "Unknown"),
                "visual_health_type": class_meta.get("type", "Unknown"),
                "quality_category": class_meta.get("quality_category", "Unknown"),
                "default_recommendation": class_meta.get("default_recommendation", "No specific agronomic guidance.")
            })

        # Top-1 decision
        top1 = top_candidates[0]
        pred_class = top1["class_name"]
        pred_index = top1["class_index"]
        pred_confidence = top1["confidence"]
        top_meta = self.metadata.get(pred_class, {})

        is_healthy = top_meta.get("type") == "Healthy"
        quality_category = top_meta.get("quality_category", "Unknown")

        # 6. Confidence Threshold Handling & Plucking Assessment
        if pred_confidence < eff_threshold:
            status = "UNCERTAIN_LOW_CONFIDENCE"
            confidence_warning = (
                f"Model prediction confidence ({pred_confidence * 100:.2f}%) is below the required certainty threshold "
                f"({eff_threshold * 100:.1f}%). Moderate confidence / uncertain prediction. The leaf image may have poor lighting, "
                f"heavy background foliage, atypical lesions, or camera motion blur. Classification should be treated as an unconfirmed candidate suggestion."
            )
            plucking_recommendation = (
                "Uncertain visual classification: Do not make harvesting decisions based on this prediction. "
                "Re-capture a sharp, centered photograph of the single leaf under clear daylight."
            )
        else:
            status = "SUCCESS"
            confidence_warning = None
            if is_healthy:
                plucking_recommendation = (
                    "Visually Healthy Foliage: Leaf exhibits uniform color and zero visible fungal or bacterial lesions. "
                    "However, visual health alone does NOT guarantee plucking readiness. Field inspection required to confirm "
                    "physical maturity, leaf size, tenderness, and harvest standard."
                )
            else:
                plucking_recommendation = (
                    f"Visually Damaged/Diseased ({top_meta.get('condition', 'Lesions detected')}): "
                    "Do NOT pluck for commercial processing. Isolate or prune affected leaf to prevent pathogen dissemination."
                )

        # Include blur alert if detected
        if quality_meta["is_blurry"]:
            blur_warning = (
                f"Image sharpness warning: Laplacian variance ({quality_meta['laplacian_variance']}) is below threshold "
                f"({self.blur_threshold}). Motion blur or camera defocus may degrade classification reliability."
            )
        else:
            blur_warning = None

        # 7. Construct Final API-Ready Output Dictionary
        result: Dict[str, Any] = {
            "status": status,
            "source": source_name,
            "predicted_class": pred_class,
            "predicted_index": pred_index,
            "confidence": round(pred_confidence, 4),
            "confidence_percent": round(pred_confidence * 100.0, 2),
            "confidence_threshold_applied": round(eff_threshold, 4),
            "is_above_threshold": bool(pred_confidence >= eff_threshold),
            "confidence_warning": confidence_warning,
            "blur_warning": blur_warning,
            "visual_health_category": quality_category,
            "is_visually_healthy": bool(is_healthy),
            "plucking_recommendation": plucking_recommendation,
            "species": top_meta.get("species", "Unknown"),
            "condition": top_meta.get("condition", "Unknown"),
            "plucking_suitability": top_meta.get("plucking_suitability", "Not Harvestable"),
            "top_three": top_candidates,
            "quality_assessment": quality_meta,
            "disclaimer": (
                "DISCLAIMER: Predicted class and confidence scores represent statistical patterns from MobileNetV2 transfer learning "
                "on PlantDoc dataset images. Predictions do not provide an absolute biological guarantee. Handheld plucking standard "
                "decisions must combine visual classification with tactile maturity inspection."
            )
        }

        return result


# Convenience module-level predictor function
_GLOBAL_PREDICTOR: Optional[LeafPredictor] = None

def get_default_predictor() -> LeafPredictor:
    """Returns or creates the cached singleton LeafPredictor instance."""
    global _GLOBAL_PREDICTOR
    if _GLOBAL_PREDICTOR is None:
        _GLOBAL_PREDICTOR = LeafPredictor()
    return _GLOBAL_PREDICTOR


def predict_leaf(
    image_input: Union[str, Path, bytes, bytearray, np.ndarray],
    confidence_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Convenience function for single-image prediction using the singleton LeafPredictor.

    Args:
        image_input: File path, bytes, or NumPy array.
        confidence_threshold: Optional confidence threshold override.

    Returns:
        JSON-serializable prediction result dictionary.
    """
    predictor = get_default_predictor()
    return predictor.predict(image_input, confidence_threshold=confidence_threshold)
