"""
Leaf Quality Classification and Plucking Standard Monitoring System
Module 6: Standardized JSON Error Handlers
File: backend/errors.py
"""

from typing import Dict, Any, Tuple
from flask import jsonify, Response
from werkzeug.exceptions import HTTPException


def format_error_response(error_type: str, message: str, status_code: int) -> Tuple[Response, int]:
    """Generates a standardized JSON error response."""
    payload: Dict[str, Any] = {
        "status": "ERROR",
        "error": error_type,
        "message": message,
        "status_code": status_code
    }
    return jsonify(payload), status_code


def register_error_handlers(app) -> None:
    """Registers global JSON error handlers on the Flask app."""

    @app.errorhandler(400)
    def handle_bad_request(e):
        msg = e.description if isinstance(e, HTTPException) else "Bad request."
        return format_error_response("BAD_REQUEST", msg, 400)

    @app.errorhandler(404)
    def handle_not_found(e):
        return format_error_response("NOT_FOUND", "The requested resource was not found on this server.", 404)

    @app.errorhandler(405)
    def handle_method_not_allowed(e):
        return format_error_response("METHOD_NOT_ALLOWED", "HTTP method not allowed for this endpoint.", 405)

    @app.errorhandler(413)
    def handle_payload_too_large(e):
        return format_error_response("PAYLOAD_TOO_LARGE", "Uploaded file exceeds maximum allowed size (16 MB limit).", 413)

    @app.errorhandler(415)
    def handle_unsupported_media_type(e):
        msg = e.description if isinstance(e, HTTPException) else "Unsupported media type."
        return format_error_response("UNSUPPORTED_MEDIA_TYPE", msg, 415)

    @app.errorhandler(422)
    def handle_unprocessable_entity(e):
        msg = e.description if isinstance(e, HTTPException) else "Unprocessable image entity."
        return format_error_response("UNPROCESSABLE_ENTITY", msg, 422)

    @app.errorhandler(500)
    def handle_internal_server_error(e):
        return format_error_response("INTERNAL_SERVER_ERROR", "An unexpected server error occurred during processing.", 500)
