"""
Leaf Quality Classification and Plucking Standard Monitoring System
Module 9: Automated System Performance Benchmarking & Resource Profiling
File: scripts/benchmark_system_performance.py
"""

import io
import json
import logging
import os
import platform
import sys
import time
from pathlib import Path
import numpy as np
import psutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure Matplotlib and Keras offline cache
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "reports" / ".mplconfig"))
os.environ.setdefault("KERAS_HOME", str(PROJECT_ROOT / ".keras"))
os.environ.setdefault("FLASK_ENV", "testing")

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger("benchmark")


def get_system_specifications():
    """Extracts host machine hardware and software specifications."""
    mem = psutil.virtual_memory()
    return {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "total_ram_gb": round(mem.total / (1024 ** 3), 2),
        "available_ram_gb": round(mem.available / (1024 ** 3), 2),
        "python_version": sys.version.split()[0],
        "python_build": sys.version
    }


def run_benchmarks():
    """Executes empirical benchmarks across all system components."""
    logger.info("Initializing system performance benchmarks...")
    process = psutil.Process(os.getpid())
    rss_start_mb = process.memory_info().rss / (1024 * 1024)

    specs = get_system_specifications()
    logger.info(f"Host System: {specs['processor']} | {specs['physical_cores']}C/{specs['logical_cores']}T | {specs['total_ram_gb']} GB RAM")

    # --------------------------------------------------------------------------
    # 1. Application Factory Startup Latency
    # --------------------------------------------------------------------------
    t0 = time.perf_counter()
    from backend.app import create_app
    from backend.config import TestingConfig
    app = create_app(TestingConfig)
    app_startup_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    logger.info(f"Flask App Startup: {app_startup_time_ms:.2f} ms")

    # --------------------------------------------------------------------------
    # 2. Model Initialization Time (Cold Load)
    # --------------------------------------------------------------------------
    t0 = time.perf_counter()
    from utils.predictor import LeafPredictor
    predictor = LeafPredictor()
    model_init_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    rss_after_model_mb = process.memory_info().rss / (1024 * 1024)
    model_memory_footprint_mb = round(rss_after_model_mb - rss_start_mb, 2)
    logger.info(f"Model Initialization: {model_init_time_ms:.2f} ms | Model RAM Footprint: {model_memory_footprint_mb:.2f} MB")

    # Locate test image
    test_img_dir = PROJECT_ROOT / "datasets" / "processed" / "plantdoc" / "test" / "Apple_leaf"
    test_imgs = list(test_img_dir.glob("*.jpg"))
    sample_img_path = str(test_imgs[0]) if test_imgs else None
    if not sample_img_path:
        raise FileNotFoundError("Sample test image not found in test split.")

    with open(sample_img_path, "rb") as f:
        sample_img_bytes = f.read()

    # --------------------------------------------------------------------------
    # 3. Single-Image Inference Latency (Cold vs Warm)
    # --------------------------------------------------------------------------
    t0 = time.perf_counter()
    _ = predictor.predict(sample_img_bytes)
    inference_cold_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    logger.info(f"Inference Latency (Cold): {inference_cold_ms:.2f} ms")

    warm_inference_times = []
    for _ in range(15):
        t0 = time.perf_counter()
        _ = predictor.predict(sample_img_bytes)
        warm_inference_times.append((time.perf_counter() - t0) * 1000.0)

    inference_warm_mean = round(float(np.mean(warm_inference_times)), 2)
    inference_warm_min = round(float(np.min(warm_inference_times)), 2)
    inference_warm_max = round(float(np.max(warm_inference_times)), 2)
    inference_warm_std = round(float(np.std(warm_inference_times)), 2)
    inference_warm_p95 = round(float(np.percentile(warm_inference_times, 95)), 2)
    logger.info(f"Inference Latency (Warm, 15 runs): Mean={inference_warm_mean:.2f} ms | Min={inference_warm_min:.2f} ms | Max={inference_warm_max:.2f} ms | P95={inference_warm_p95:.2f} ms")

    # --------------------------------------------------------------------------
    # 4. End-to-End API Response Time (POST /api/predict)
    # --------------------------------------------------------------------------
    client = app.test_client()
    api_latencies = []
    for _ in range(15):
        t0 = time.perf_counter()
        resp = client.post(
            "/api/predict",
            data={"file": (io.BytesIO(sample_img_bytes), "bench_api.jpg")},
            content_type="multipart/form-data"
        )
        assert resp.status_code == 200
        api_latencies.append((time.perf_counter() - t0) * 1000.0)

    api_mean = round(float(np.mean(api_latencies)), 2)
    api_min = round(float(np.min(api_latencies)), 2)
    api_max = round(float(np.max(api_latencies)), 2)
    api_std = round(float(np.std(api_latencies)), 2)
    api_p95 = round(float(np.percentile(api_latencies, 95)), 2)
    logger.info(f"API Latency (POST /api/predict, 15 runs): Mean={api_mean:.2f} ms | Min={api_min:.2f} ms | Max={api_max:.2f} ms | P95={api_p95:.2f} ms")

    # --------------------------------------------------------------------------
    # 5. Database Insertion & Query Latency
    # --------------------------------------------------------------------------
    from database.db import insert_analysis, get_history, get_statistics
    sample_record = {
        "source": "bench_leaf.jpg",
        "species": "Apple",
        "predicted_class": "Apple_scab_healthy",
        "confidence": 0.945,
        "visual_health": "Healthy",
        "is_visually_healthy": True,
        "condition": "Healthy Foliage",
        "plucking_suitability": "CONDITIONAL_VERIFICATION",
        "status": "SUCCESS",
        "plucking_recommendation": "Field maturity inspection required.",
        "top_three": [{"rank": 1, "class_name": "Apple_scab_healthy", "confidence": 0.945}],
        "quality_assessment": {"laplacian_variance": 380.0, "quality_rating": "Sharp"}
    }

    db_insert_times = []
    inserted_ids = []
    for _ in range(25):
        t0 = time.perf_counter()
        new_id = insert_analysis(sample_record)
        db_insert_times.append((time.perf_counter() - t0) * 1000.0)
        inserted_ids.append(new_id)

    db_insert_mean = round(float(np.mean(db_insert_times)), 2)
    db_insert_min = round(float(np.min(db_insert_times)), 2)
    db_insert_max = round(float(np.max(db_insert_times)), 2)
    logger.info(f"DB Insertion Latency (25 runs): Mean={db_insert_mean:.2f} ms | Min={db_insert_min:.2f} ms | Max={db_insert_max:.2f} ms")

    db_query_times = []
    for _ in range(25):
        t0 = time.perf_counter()
        _ = get_history(page=1, limit=10, species="Apple")
        db_query_times.append((time.perf_counter() - t0) * 1000.0)

    db_query_mean = round(float(np.mean(db_query_times)), 2)
    db_query_min = round(float(np.min(db_query_times)), 2)
    db_query_max = round(float(np.max(db_query_times)), 2)
    logger.info(f"DB Filtered Query Latency (25 runs): Mean={db_query_mean:.2f} ms | Min={db_query_min:.2f} ms | Max={db_query_max:.2f} ms")

    db_stats_times = []
    for _ in range(25):
        t0 = time.perf_counter()
        _ = get_statistics()
        db_stats_times.append((time.perf_counter() - t0) * 1000.0)

    db_stats_mean = round(float(np.mean(db_stats_times)), 2)
    logger.info(f"DB Aggregation Stats Latency (25 runs): Mean={db_stats_mean:.2f} ms")

    # --------------------------------------------------------------------------
    # 6. Overall Memory Profile
    # --------------------------------------------------------------------------
    rss_peak_mb = round(process.memory_info().rss / (1024 * 1024), 2)
    logger.info(f"Peak Process RSS: {rss_peak_mb:.2f} MB")

    # Package Results
    benchmark_results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "system_specifications": specs,
        "benchmarks": {
            "application_startup_time_ms": app_startup_time_ms,
            "model_initialization_time_ms": model_init_time_ms,
            "model_memory_footprint_mb": model_memory_footprint_mb,
            "process_rss_start_mb": round(rss_start_mb, 2),
            "process_rss_peak_mb": rss_peak_mb,
            "single_image_inference_cold_ms": inference_cold_ms,
            "single_image_inference_warm": {
                "runs": 15,
                "mean_ms": inference_warm_mean,
                "min_ms": inference_warm_min,
                "max_ms": inference_warm_max,
                "std_ms": inference_warm_std,
                "p95_ms": inference_warm_p95
            },
            "api_request_e2e": {
                "endpoint": "POST /api/predict",
                "runs": 15,
                "mean_ms": api_mean,
                "min_ms": api_min,
                "max_ms": api_max,
                "std_ms": api_std,
                "p95_ms": api_p95
            },
            "database_operations": {
                "insertion": {
                    "runs": 25,
                    "mean_ms": db_insert_mean,
                    "min_ms": db_insert_min,
                    "max_ms": db_insert_max
                },
                "filtered_query": {
                    "runs": 25,
                    "mean_ms": db_query_mean,
                    "min_ms": db_query_min,
                    "max_ms": db_query_max
                },
                "aggregation_stats": {
                    "runs": 25,
                    "mean_ms": db_stats_mean
                }
            }
        }
    }

    report_path = PROJECT_ROOT / "reports" / "performance_benchmarks.json"
    report_path.write_text(json.dumps(benchmark_results, indent=2), encoding="utf-8")
    logger.info(f"Benchmarking results saved to: {report_path}")
    return benchmark_results


if __name__ == "__main__":
    run_benchmarks()
