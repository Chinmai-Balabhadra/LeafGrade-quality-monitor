"""
Leaf Quality Classification and Plucking Standard Monitoring System
Model Evaluation API Blueprint
File: backend/routes/model_evaluation.py
"""

import json
import logging
from pathlib import Path
from flask import Blueprint, jsonify, send_file, Response, current_app

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
FRONTEND_ASSETS_DIR = PROJECT_ROOT / "frontend" / "assets"
METRICS_JSON_PATH = REPORTS_DIR / "model_evaluation_metrics.json"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"
ASSET_CONFUSION_MATRIX_PATH = FRONTEND_ASSETS_DIR / "confusion_matrix.png"

logger = logging.getLogger(__name__)

model_bp = Blueprint("model", __name__, url_prefix="/api/model")


@model_bp.route("/evaluation", methods=["GET"])
@model_bp.route("/evaluation/", methods=["GET"])
def get_model_evaluation() -> Response:
    """
    Returns empirical performance metrics of the trained MobileNetV2 model
    evaluated on the held-out PlantDoc test set (465 samples).
    """
    if not METRICS_JSON_PATH.exists():
        return jsonify({
            "status": "ERROR",
            "message": "Model evaluation metrics report not found on server."
        }), 404

    try:
        with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        response = jsonify(data)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Cache-Control"] = "public, max-age=3600"
        return response, 200
    except Exception as e:
        logger.error(f"Failed to read model evaluation metrics: {e}")
        return jsonify({
            "status": "ERROR",
            "message": f"Failed to load evaluation metrics: {str(e)}"
        }), 500


@model_bp.route("/confusion-matrix", methods=["GET"])
@model_bp.route("/confusion-matrix/", methods=["GET"])
@model_bp.route("/confusion_matrix", methods=["GET"])
@model_bp.route("/confusion_matrix/", methods=["GET"])
@model_bp.route("/confusion-matrix.png", methods=["GET"])
@model_bp.route("/confusion_matrix.png", methods=["GET"])
def get_confusion_matrix_image() -> Response:
    """
    Serves the high-resolution 28x28 confusion matrix visualization image.
    Supports PNG binary stream, caching headers, and JSON descriptor queries.
    """
    from flask import request

    # Locate image file across designated locations
    target_path = None
    if CONFUSION_MATRIX_PATH.exists():
        target_path = CONFUSION_MATRIX_PATH
    elif ASSET_CONFUSION_MATRIX_PATH.exists():
        target_path = ASSET_CONFUSION_MATRIX_PATH

    if target_path is None or not target_path.exists():
        logger.error(f"Confusion matrix image not found at {CONFUSION_MATRIX_PATH} or {ASSET_CONFUSION_MATRIX_PATH}")
        return jsonify({
            "status": "ERROR",
            "message": "Confusion matrix image not found on server."
        }), 404

    # Optional JSON descriptor format if explicitly requested
    if request.args.get("format", "").lower() == "json":
        return jsonify({
            "status": "SUCCESS",
            "endpoint": "/api/model/confusion-matrix",
            "filename": target_path.name,
            "file_size_bytes": target_path.stat().st_size,
            "dimensions": "28x28",
            "content_type": "image/png"
        }), 200

    response = send_file(
        str(target_path.resolve()),
        mimetype="image/png",
        as_attachment=False,
        download_name="confusion_matrix.png"
    )
    # Enable browser caching and cross-origin resource sharing for dashboard image tags
    response.headers["Cache-Control"] = "public, max-age=3600"
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response
