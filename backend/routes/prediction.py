"""
Leaf Quality Classification and Plucking Standard Monitoring System
Module 6: Single-Image Prediction API Route
File: backend/routes/prediction.py
"""

import logging
from typing import Optional
from flask import Blueprint, request, jsonify, Response
from werkzeug.exceptions import BadRequest

from backend.services.prediction_service import get_prediction_service
from database.db import insert_analysis

logger = logging.getLogger(__name__)
prediction_bp = Blueprint("prediction", __name__, url_prefix="/api")


@prediction_bp.route("/predict", methods=["POST"])
def predict_endpoint() -> Response:
    """
    Accepts an uploaded leaf image, executes MobileNetV2 prediction,
    persists the analysis record to SQLite, and returns structured
    classification and plucking guidance.

    Expected Request:
        Multipart Form-Data with file field named 'file' or 'image'.
        Optional query/form parameter: 'confidence_threshold' (float 0.0 - 1.0).

    Returns:
        JSON response with HTTP 200 (Success / Uncertain), 400 (Bad Request),
        413 (File Too Large), 422 (Unprocessable Image), or 500 (Internal Error).
    """
    # 1. Check for file in request
    if not request.files:
        raise BadRequest("No multipart file data was submitted in the request.")

    file_obj = request.files.get("file") or request.files.get("image")
    if file_obj is None or file_obj.filename == "":
        raise BadRequest("Missing file upload. Please provide an image under field name 'file' or 'image'.")

    # 2. Extract optional confidence threshold
    threshold_str: Optional[str] = (
        request.form.get("confidence_threshold") or
        request.args.get("confidence_threshold") or
        request.args.get("threshold")
    )
    confidence_threshold: Optional[float] = None
    if threshold_str is not None:
        try:
            confidence_threshold = float(threshold_str)
            if not (0.0 <= confidence_threshold <= 1.0):
                raise ValueError()
        except ValueError:
            raise BadRequest("Invalid 'confidence_threshold'. Must be a float between 0.0 and 1.0.")

    # 3. Delegate to Prediction Service
    service = get_prediction_service()
    result = service.predict_image(file_obj, confidence_threshold=confidence_threshold)

    # 4. Handle Rejected Non-Leaf Images (OOD Guardrail - do not save to SQLite)
    if result.get("status") == "REJECTED_NOT_LEAF" or result.get("is_leaf") is False:
        logger.info(f"OOD validation rejected upload '{result.get('source')}': {result.get('message')}")
        result["analysis_id"] = None
        result["history_saved"] = False
        return jsonify(result), 422

    # 5. Persist valid leaf analysis record to SQLite database
    try:
        analysis_id = insert_analysis(result)
        result["analysis_id"] = analysis_id
        result["history_saved"] = True
        result["history_warning"] = None
    except Exception as db_err:
        logger.error(f"Failed to persist analysis record to database: {db_err}")
        result["analysis_id"] = None
        result["history_saved"] = False
        result["history_warning"] = (
            "Database storage warning: Prediction completed successfully, but the record "
            "could not be saved to historical database."
        )

    return jsonify(result), 200
