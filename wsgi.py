"""
Leaf Quality Classification and Plucking Standard Monitoring System
Module 9: Production WSGI Application Entry Point
File: wsgi.py

Runs the Flask application using Waitress WSGI production server on Windows.
Usage:
    python wsgi.py
"""

import logging
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure Matplotlib and Keras offline cache directories
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "reports" / ".mplconfig"))
os.environ.setdefault("KERAS_HOME", str(PROJECT_ROOT / ".keras"))
os.environ.setdefault("FLASK_ENV", "production")

from backend.app import create_app
from backend.config import ProductionConfig, get_config

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("wsgi")

# Instantiate application with ProductionConfig
app = create_app(ProductionConfig)
application = app  # Standard WSGI callable alias


if __name__ == "__main__":
    from waitress import serve

    host = app.config.get("HOST", "127.0.0.1")
    port = app.config.get("PORT", 5000)
    threads = int(os.environ.get("WAITRESS_THREADS", 4))

    print("\n" + "=" * 80)
    print(" AgriLeaf AI - Leaf Quality Classification & Plucking Monitoring System")
    print(" Production WSGI Server (Waitress)")
    print("=" * 80)
    print(f" Binding:          http://{host}:{port}")
    print(f" Web Dashboard:    http://{host}:{port}/")
    print(f" Health Check API: http://{host}:{port}/api/health")
    print(f" Predict API:      http://{host}:{port}/api/predict")
    print(f" History API:      http://{host}:{port}/api/history")
    print(f" Worker Threads:   {threads}")
    print("=" * 80 + "\n")

    logger.info(f"Starting Waitress production server on {host}:{port} with {threads} threads...")
    serve(app, host=host, port=port, threads=threads)
