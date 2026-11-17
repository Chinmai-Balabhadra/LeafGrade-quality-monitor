"""
Leaf Quality Classification and Plucking Standard Monitoring System
Module 6: Flask Application Entry Point & Factory
File: backend/app.py
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, request, send_from_directory, abort
from flask_cors import CORS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Avoid Matplotlib font cache warnings and ensure Keras cache
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "reports" / ".mplconfig"))
os.environ.setdefault("KERAS_HOME", str(PROJECT_ROOT / ".keras"))

from backend.config import BaseConfig, get_config
from backend.errors import register_error_handlers
from backend.routes.health import health_bp
from backend.routes.prediction import prediction_bp
from backend.routes.history import history_bp
from backend.routes.model_evaluation import model_bp

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("backend.app")


def create_app(config_class: Optional[BaseConfig] = None) -> Flask:
    """
    Application factory initializing Flask, registering CORS, blueprints,
    configuration, and global error handlers.
    """
    app = Flask(__name__)

    # Apply configuration
    if config_class is None:
        config_class = get_config()
    app.config.from_object(config_class)

    # Configure CORS for local development and web frontend
    CORS(
        app,
        resources={r"/api/*": {"origins": app.config.get("CORS_ORIGINS", "*")}},
        methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"]
    )

    # Register blueprints
    app.register_blueprint(health_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(history_bp)
    app.register_blueprint(model_bp)

    # Register standardized error handlers
    register_error_handlers(app)

    # Frontend static asset & page routes (explicit non-conflicting paths)
    frontend_dir = PROJECT_ROOT / "frontend"

    @app.route("/css/<path:filename>", methods=["GET"])
    def serve_css(filename):
        return send_from_directory(str(frontend_dir / "css"), filename)

    @app.route("/js/<path:filename>", methods=["GET"])
    def serve_js(filename):
        return send_from_directory(str(frontend_dir / "js"), filename)

    @app.route("/assets/<path:filename>", methods=["GET"])
    def serve_assets(filename):
        return send_from_directory(str(frontend_dir / "assets"), filename)

    @app.route("/reports/<path:filename>", methods=["GET"])
    def serve_reports(filename):
        reports_dir = PROJECT_ROOT / "reports"
        return send_from_directory(str(reports_dir), filename)

    @app.route("/app", methods=["GET"])
    @app.route("/index.html", methods=["GET"])
    def serve_frontend_ui():
        return send_from_directory(str(frontend_dir), "index.html")

    # Evaluation route aliases ensuring 404 immunity
    @app.route("/api/model_evaluation", methods=["GET"])
    @app.route("/api/evaluation", methods=["GET"])
    def route_evaluation_alias():
        from backend.routes.model_evaluation import get_model_evaluation
        return get_model_evaluation()

    @app.route("/api/confusion-matrix", methods=["GET"])
    @app.route("/api/confusion_matrix", methods=["GET"])
    def route_confusion_matrix_alias():
        from backend.routes.model_evaluation import get_confusion_matrix_image
        return get_confusion_matrix_image()

    # Root welcome route (serves frontend HTML to web browsers, JSON to API clients)
    @app.route("/", methods=["GET"])
    def root():
        accept_header = request.headers.get("Accept", "")
        # If browser navigates directly, serve frontend UI
        if "text/html" in accept_header and "application/json" not in accept_header:
            return send_from_directory(str(frontend_dir), "index.html")

        return jsonify({
            "service": "Leaf Quality Classification & Plucking Standard Monitoring System API",
            "version": "1.0.0",
            "course": "CSE471 Academic Project",
            "endpoints": {
                "health": "/api/health",
                "predict": "/api/predict",
                "history": "/api/history",
                "history_stats": "/api/history/stats",
                "model_evaluation": "/api/model/evaluation",
                "confusion_matrix": "/api/model/confusion-matrix",
                "web_app": "/index.html"
            }
        }), 200

    logger.info(f"Flask application initialized with config: {config_class.__name__}")
    return app


# Default app instance for WSGI runners
app = create_app()


if __name__ == "__main__":
    host = app.config.get("HOST", "127.0.0.1")
    port = app.config.get("PORT", 5000)
    debug = app.config.get("DEBUG", True)

    print(f"\n================================================================================")
    print(f" Starting Leaf Quality Classification Backend Server on http://{host}:{port}")
    print(f" Endpoints:")
    print(f"   - GET  http://{host}:{port}/api/health")
    print(f"   - POST http://{host}:{port}/api/predict")
    print(f"================================================================================\n")

    app.run(host=host, port=port, debug=debug)
