"""
Leaf Quality Classification and Plucking Standard Monitoring System
Module 8: Historical Analysis API Routes
File: backend/routes/history.py
"""

import logging
from typing import Optional
from flask import Blueprint, request, jsonify, Response
from werkzeug.exceptions import BadRequest, NotFound

from database.db import (
    get_history,
    get_analysis_by_id,
    get_statistics,
    delete_analysis,
    clear_all_history
)

logger = logging.getLogger(__name__)

history_bp = Blueprint("history", __name__, url_prefix="/api")


@history_bp.route("/history", methods=["GET"])
def list_history() -> Response:
    """
    Retrieves paginated historical leaf analysis records with filtering and search.

    Query Parameters:
        page (int): Page number (default: 1).
        limit (int): Records per page (default: 10, max: 100).
        species (str): Filter by crop species (e.g. 'Apple').
        visual_health (str): Filter by 'Healthy' or 'Diseased'.
        status (str): Filter by 'Certain' or 'Uncertain'.
        search (str): Substring search across class, species, condition, filename.
        start_date (str): ISO timestamp lower bound.
        end_date (str): ISO timestamp upper bound.
    """
    try:
        page = int(request.args.get("page", 1))
        limit = int(request.args.get("limit", 10))
        if page < 1 or limit < 1:
            raise BadRequest("Invalid 'page' or 'limit'. Must be positive integers.")
    except ValueError:
        raise BadRequest("Invalid 'page' or 'limit'. Must be positive integers.")

    species = request.args.get("species")
    visual_health = request.args.get("visual_health")
    status = request.args.get("status")
    search = request.args.get("search")
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")

    records, pagination = get_history(
        page=page,
        limit=limit,
        species=species,
        visual_health=visual_health,
        status=status,
        search=search,
        start_date=start_date,
        end_date=end_date
    )

    return jsonify({
        "status": "SUCCESS",
        "records": records,
        "pagination": pagination
    }), 200


@history_bp.route("/history/stats", methods=["GET"])
def history_statistics() -> Response:
    """Returns aggregate summary analytics over all past leaf analyses."""
    stats = get_statistics()
    return jsonify({
        "status": "SUCCESS",
        "statistics": stats
    }), 200


@history_bp.route("/history/<int:analysis_id>", methods=["GET"])
def get_record(analysis_id: int) -> Response:
    """Retrieves full diagnostic details for a specific analysis record."""
    record = get_analysis_by_id(analysis_id)
    if record is None:
        raise NotFound(f"Historical analysis record #{analysis_id} was not found.")

    return jsonify({
        "status": "SUCCESS",
        "record": record
    }), 200


@history_bp.route("/history/<int:analysis_id>", methods=["DELETE"])
def delete_record(analysis_id: int) -> Response:
    """Deletes a single analysis record by ID."""
    success = delete_analysis(analysis_id)
    if not success:
        raise NotFound(f"Historical record #{analysis_id} not found or already deleted.")

    return jsonify({
        "status": "SUCCESS",
        "message": f"Historical record #{analysis_id} deleted successfully.",
        "deleted_id": analysis_id
    }), 200


@history_bp.route("/history", methods=["DELETE"])
def clear_history() -> Response:
    """
    Clears all analysis history records with mandatory confirmation safeguard.

    Requires:
        Query param '?confirm=true' or JSON body {"confirm": true}.
    """
    confirm_param = request.args.get("confirm", "").lower() in ("true", "1", "yes")
    confirm_json = False
    if request.is_json:
        confirm_json = bool(request.get_json(silent=True, default={}).get("confirm"))

    if not (confirm_param or confirm_json):
        raise BadRequest(
            "Confirmation required to delete all records. Pass '?confirm=true' as a query parameter or '{\"confirm\": true}' in JSON body."
        )

    deleted_count = clear_all_history()
    return jsonify({
        "status": "SUCCESS",
        "message": f"All {deleted_count} historical records have been successfully cleared.",
        "deleted_count": deleted_count
    }), 200
