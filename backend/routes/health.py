"""
Leaf Quality Classification and Plucking Standard Monitoring System
Module 6: Health Check API Route
File: backend/routes/health.py
"""

from datetime import datetime, timezone
from flask import Blueprint, jsonify, Response
from backend.services.prediction_service import get_prediction_service

health_bp = Blueprint("health", __name__, url_prefix="/api")


@health_bp.route("/health", methods=["GET"])
def health_check() -> Response:
    """
    Returns system and service health status.
    Guarantees no sensitive filesystem paths are exposed.
    """
    try:
        service = get_prediction_service()
        service_status = service.get_service_status()
        backend_status = "healthy" if service_status.get("model_ready") else "degraded"
    except Exception as e:
        service_status = {"error": str(e), "model_ready": False}
        backend_status = "unhealthy"

    payload = {
        "status": backend_status,
        "service": "Leaf Quality Classification Backend API",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_status": service_status
    }
    return jsonify(payload), (200 if backend_status != "unhealthy" else 503)
