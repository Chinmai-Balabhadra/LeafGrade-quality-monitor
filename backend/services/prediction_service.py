"""
Leaf Quality Classification and Plucking Standard Monitoring System
Module 6: Prediction Service Layer
File: backend/services/prediction_service.py
"""

import io
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

from werkzeug.datastructures import FileStorage
from werkzeug.exceptions import BadRequest, UnprocessableEntity

from backend.config import BaseConfig
from utils.predictor import LeafPredictor, get_default_predictor

logger = logging.getLogger(__name__)


class PredictionService:
    """
    Service layer providing clean abstraction between HTTP endpoints
    and the underlying MobileNetV2 predictor.
    """

    def __init__(self, config: Optional[BaseConfig] = None):
        self.config = config or BaseConfig()
        # Initialize or retrieve singleton predictor
        self.predictor: LeafPredictor = get_default_predictor()

    def validate_file_upload(self, file: FileStorage) -> Tuple[str, bytes]:
        """
        Validates the incoming HTTP file upload.

        Args:
            file: Werkzeug FileStorage object from request.files.

        Returns:
            Tuple of (filename, file_bytes)

        Raises:
            BadRequest: If missing, empty filename, unsupported extension, or 0 bytes.
        """
        if not file or not file.filename:
            raise BadRequest("No file was submitted or filename is empty.")

        filename = file.filename.strip()
        ext = Path(filename).suffix.lower()

        if ext not in self.config.ALLOWED_EXTENSIONS:
            allowed_str = ", ".join(sorted(list(self.config.ALLOWED_EXTENSIONS)))
            raise BadRequest(
                f"File format '{ext}' is not supported. Allowed image extensions: {allowed_str}"
            )

        file_bytes = file.read()
        if not file_bytes or len(file_bytes) == 0:
            raise BadRequest("Uploaded file is empty (0 bytes).")

        return filename, file_bytes

    def predict_image(
        self,
        file: Union[FileStorage, bytes],
        confidence_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Executes leaf quality prediction on an uploaded file or byte stream.

        Args:
            file: FileStorage object or raw bytes.
            confidence_threshold: Optional threshold override.

        Returns:
            Structured JSON-ready prediction result dictionary.

        Raises:
            BadRequest: For client validation failures.
            UnprocessableEntity: If file cannot be decoded as an image.
        """
        if isinstance(file, FileStorage):
            filename, file_bytes = self.validate_file_upload(file)
        elif isinstance(file, (bytes, bytearray)):
            filename = "in_memory_bytes"
            file_bytes = file
            if len(file_bytes) == 0:
                raise BadRequest("Byte stream is empty (0 bytes).")
        else:
            raise BadRequest(f"Unsupported file input type: {type(file)}")

        # Check threshold
        eff_threshold = (
            float(confidence_threshold)
            if confidence_threshold is not None
            else self.config.DEFAULT_CONFIDENCE_THRESHOLD
        )
        if eff_threshold < 0.0 or eff_threshold > 1.0:
            raise BadRequest("confidence_threshold must be a float between 0.0 and 1.0")

        # Execute prediction via utils.predictor
        result = self.predictor.predict(file_bytes, confidence_threshold=eff_threshold)

        # Map decode failures to HTTP 422 Unprocessable Entity
        if result.get("status") == "ERROR":
            error_msg = result.get("message", "Unable to process or decode image.")
            logger.warning(f"Prediction failed on {filename}: {error_msg}")
            raise UnprocessableEntity(error_msg)

        # Attach original filename in source field
        result["source"] = filename
        return result

    def get_service_status(self) -> Dict[str, Any]:
        """
        Returns diagnostic health metadata for the prediction service
        without exposing sensitive internal file paths.
        """
        is_model_loaded = bool(self.predictor.model is not None)
        classes_count = len(self.predictor.idx_to_class)
        model_ready = is_model_loaded and classes_count == 28

        return {
            "model_ready": model_ready,
            "architecture": "MobileNetV2 Transfer Learning",
            "supported_classes_count": classes_count,
            "default_confidence_threshold": self.config.DEFAULT_CONFIDENCE_THRESHOLD,
            "blur_detection_enabled": True
        }


# Module-level singleton service instance
_GLOBAL_PREDICTION_SERVICE: Optional[PredictionService] = None

def get_prediction_service() -> PredictionService:
    """Returns or creates the cached PredictionService instance."""
    global _GLOBAL_PREDICTION_SERVICE
    if _GLOBAL_PREDICTION_SERVICE is None:
        _GLOBAL_PREDICTION_SERVICE = PredictionService()
    return _GLOBAL_PREDICTION_SERVICE
