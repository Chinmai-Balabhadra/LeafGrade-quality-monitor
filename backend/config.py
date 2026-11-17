"""
Leaf Quality Classification and Plucking Standard Monitoring System
Module 6: Flask Backend Configuration
File: backend/config.py
"""

import os
from pathlib import Path
from typing import Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class BaseConfig:
    """Base application configuration."""
    SECRET_KEY: str = os.environ.get("SECRET_KEY", "cse471-leaf-quality-monitoring-secret-key")
    
    # Upload limits (16 MB maximum payload)
    MAX_CONTENT_LENGTH: int = int(os.environ.get("MAX_CONTENT_LENGTH", 16 * 1024 * 1024))
    
    # Allowed image file extensions
    ALLOWED_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    # Model and inference parameters
    DEFAULT_CONFIDENCE_THRESHOLD: float = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.40))
    DEFAULT_BLUR_THRESHOLD: float = float(os.environ.get("BLUR_THRESHOLD", 50.0))
    
    # Paths to model assets
    MODEL_PATH: Path = PROJECT_ROOT / "model" / "leaf_classifier.keras"
    CLASS_INDICES_PATH: Path = PROJECT_ROOT / "model" / "class_indices.json"
    
    # CORS settings
    CORS_ORIGINS: str = os.environ.get("CORS_ORIGINS", "*")
    
    # Server configuration
    HOST: str = os.environ.get("HOST", "127.0.0.1")
    PORT: int = int(os.environ.get("PORT", 5000))
    DEBUG: bool = False
    TESTING: bool = False


class DevelopmentConfig(BaseConfig):
    """Development environment configuration."""
    DEBUG: bool = True


class TestingConfig(BaseConfig):
    """Testing environment configuration."""
    TESTING: bool = True
    DEBUG: bool = False


class ProductionConfig(BaseConfig):
    """Production environment configuration."""
    DEBUG: bool = False
    TESTING: bool = False


CONFIG_MAP = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig
}


def get_config(config_name: str = None) -> BaseConfig:
    """Retrieves configuration class based on environment or passed name."""
    if not config_name:
        config_name = os.environ.get("FLASK_ENV", "development").lower()
    return CONFIG_MAP.get(config_name, DevelopmentConfig)
