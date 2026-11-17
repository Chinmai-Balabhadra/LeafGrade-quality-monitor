# Module 6 Report: Flask Backend Development

**Project Title**: Leaf Quality Classification and Plucking Standard Monitoring System Using Handheld Camera  
**Course Code**: CSE471 Academic Project  
**Module**: Module 6 – Flask Backend Development  
**Component**: [backend/app.py](file:///C:/Users/ASUS/OneDrive/Desktop/Project-471/backend/app.py)  
**Status**: **Completed & Fully Verified (42/42 Tests Passing)**  
**Date**: November 2026  

---

## 1. Executive Summary

Module 6 creates a simple, robust, production-structured **Flask backend service** that connects the MobileNetV2 deep learning prediction pipeline ([utils/predictor.py](file:///C:/Users/ASUS/OneDrive/Desktop/Project-471/utils/predictor.py)) to web, mobile, or handheld camera frontends.

The backend is developed using clean modular architectural patterns (Application Factory, Blueprints, Service Layer, and Global Error Handlers). Model inference logic remains strictly encapsulated within the prediction service, guaranteeing zero duplication of deep learning code.

---

## 2. Backend Architecture and File Organization

```text
backend/
├── __init__.py                     # Package export (app, create_app)
├── app.py                          # Flask application factory, CORS, blueprints, entry point
├── config.py                       # Configuration classes (Dev, Test, Prod), upload limits
├── errors.py                       # Standardized JSON error response handlers
├── routes/
│   ├── __init__.py                 # Routes blueprint registration
│   ├── health.py                   # GET /api/health (Diagnostic status, model availability)
│   └── prediction.py               # POST /api/predict (Multipart image upload and inference)
└── services/
    ├── __init__.py                 # Services package
    └── prediction_service.py       # Service layer interfacing with utils/predictor.py
```

### Key Architectural Safeguards:
1. **Singleton Model Caching**: Uses `PredictionService` and `LeafPredictor.get_instance()` to load the 22.03 MB MobileNetV2 model into memory once at application launch, eliminating per-request I/O latency.
2. **CORS Enabled**: Configured via `flask-cors` for cross-origin requests from local frontend development servers (`http://127.0.0.1:5500`, `http://localhost:3000`, etc.).
3. **Payload & Format Restrictions**: Rejects files larger than 16 MB (`MAX_CONTENT_LENGTH`) and restricts extensions strictly to allowed image formats (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`).
4. **Information Leakage Prevention**: The health check endpoint reports system readiness without exposing internal directory structures or absolute operating system file paths.

---

## 3. API Endpoints Specification

### 3.1 Service Welcome Index
- **URL**: `GET /`
- **Description**: Returns API service identification and endpoint index.
- **Status Code**: `200 OK`
- **Example Response**:
```json
{
  "service": "Leaf Quality Classification & Plucking Standard Monitoring System API",
  "version": "1.0.0",
  "course": "CSE471 Academic Project",
  "endpoints": {
    "health": "/api/health",
    "predict": "/api/predict"
  }
}
```

---

### 3.2 System & Model Health Check
- **URL**: `GET /api/health`
- **Description**: Diagnostic endpoint to verify server responsiveness, model readiness, and class registry.
- **Status Code**: `200 OK` (or `503 Service Unavailable` if model loading fails)
- **Example Response**:
```json
{
  "status": "healthy",
  "service": "Leaf Quality Classification Backend API",
  "version": "1.0.0",
  "timestamp": "2026-11-16T04:13:00.123456+00:00",
  "model_status": {
    "model_ready": true,
    "architecture": "MobileNetV2 Transfer Learning",
    "supported_classes_count": 28,
    "default_confidence_threshold": 0.4,
    "blur_detection_enabled": true
  }
}
```

---

### 3.3 Single-Image Prediction & Plucking Advisory
- **URL**: `POST /api/predict`
- **Description**: Ingests an uploaded leaf photograph, executes MobileNetV2 classification, checks image quality, and delivers an agronomic harvesting recommendation.
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file` or `image` (*Required*, File): Raw image binary file.
  - `confidence_threshold` (*Optional*, Float `0.0`–`1.0`): Custom certainty threshold (default `0.40`).

#### A. Successful Prediction Response (`200 OK`)
```json
{
  "status": "SUCCESS",
  "source": "test_Apple leaf_1.jpg",
  "predicted_class": "Apple_leaf",
  "predicted_index": 1,
  "confidence": 0.8109,
  "confidence_percent": 81.09,
  "confidence_threshold_applied": 0.4,
  "is_above_threshold": true,
  "confidence_warning": null,
  "blur_warning": null,
  "visual_health_category": "Good Quality (Visually Healthy)",
  "is_visually_healthy": true,
  "species": "Apple",
  "condition": "Healthy foliage",
  "plucking_suitability": "Proxy Benchmark (Tree foliage, not commercial plucking crop)",
  "plucking_recommendation": "Visually Healthy Foliage: Leaf exhibits uniform color and zero visible fungal or bacterial lesions. However, visual health alone does NOT guarantee plucking readiness. Field inspection required to confirm physical maturity, leaf size, tenderness, and harvest standard.",
  "top_three": [
    {
      "rank": 1,
      "class_name": "Apple_leaf",
      "class_index": 1,
      "confidence": 0.8109,
      "confidence_percent": 81.09,
      "species": "Apple",
      "condition": "Healthy foliage",
      "visual_health_type": "Healthy",
      "quality_category": "Good Quality (Visually Healthy)",
      "default_recommendation": "Inspect leaf maturity and physical dimensions before plucking (foliage is visually healthy)."
    },
    {
      "rank": 2,
      "class_name": "Cherry_leaf",
      "class_index": 6,
      "confidence": 0.0991,
      "confidence_percent": 9.91,
      "species": "Cherry",
      "condition": "Healthy foliage",
      "visual_health_type": "Healthy",
      "quality_category": "Good Quality (Visually Healthy)",
      "default_recommendation": "Inspect leaf maturity and physical dimensions before plucking (foliage is visually healthy)."
    },
    {
      "rank": 3,
      "class_name": "Apple_Scab_Leaf",
      "class_index": 0,
      "confidence": 0.0305,
      "confidence_percent": 3.05,
      "species": "Apple",
      "condition": "Apple scab (Venturia inaequalis)",
      "visual_health_type": "Diseased",
      "quality_category": "Poor Quality (Visually Diseased/Damaged)",
      "default_recommendation": "Do not pluck (leaf exhibits disease lesions or pathological damage). Discard or quarantine."
    }
  ],
  "quality_assessment": {
    "is_blurry": false,
    "quality_rating": "Sharp",
    "laplacian_variance": 542.18,
    "mean_brightness": 128.45,
    "contrast_std": 56.12,
    "original_height": 261,
    "original_width": 193,
    "channels": 3
  },
  "disclaimer": "DISCLAIMER: Predicted class and confidence scores represent statistical patterns from MobileNetV2 transfer learning on PlantDoc dataset images. Predictions do not provide an absolute biological guarantee. Handheld plucking standard decisions must combine visual classification with tactile maturity inspection."
}
```

#### B. Sub-Threshold / Ambiguous Prediction Response (`200 OK`)
When top confidence falls below the configured threshold ($T = 0.40$ or custom override):
```json
{
  "status": "UNCERTAIN_LOW_CONFIDENCE",
  "predicted_class": "Corn_leaf_blight",
  "confidence": 0.3214,
  "confidence_percent": 32.14,
  "confidence_threshold_applied": 0.4,
  "is_above_threshold": false,
  "confidence_warning": "Model prediction confidence (32.14%) is below the required certainty threshold (40.0%). The leaf image may have poor lighting, heavy background foliage, atypical lesions, or camera motion blur. Classification should be treated as an unconfirmed suggestion.",
  "plucking_recommendation": "Uncertain visual classification: Do not make harvesting decisions based on this prediction. Re-capture a sharp, centered photograph of the single leaf under clear daylight.",
  "visual_health_category": "Borderline / Ambiguous"
}
```

---

## 4. Standardized Error Handling

All client and server errors return a consistent, parseable JSON schema:

```json
{
  "status": "ERROR",
  "error": "ERROR_CODE",
  "message": "Human-readable explanation of error.",
  "status_code": 400
}
```

| HTTP Status | Error Code | Trigger Condition |
| :--- | :--- | :--- |
| `400 Bad Request` | `BAD_REQUEST` | Missing `file` field, empty filename, unsupported file extension (`.txt`, `.pdf`), invalid threshold value (`>1.0` or `<0.0`), or 0-byte file. |
| `404 Not Found` | `NOT_FOUND` | Accessing an undefined API route. |
| `405 Method Not Allowed`| `METHOD_NOT_ALLOWED` | Submitting a `GET` request to `POST /api/predict`. |
| `413 Payload Too Large` | `PAYLOAD_TOO_LARGE` | Uploaded image exceeds 16 MB. |
| `422 Unprocessable` | `UNPROCESSABLE_ENTITY`| Corrupted binary payload that cannot be decoded into a valid 3-channel image by OpenCV/PIL. |
| `500 Server Error` | `INTERNAL_SERVER_ERROR`| Unhandled exception during prediction execution. |

---

## 5. Automated Unit Testing Results

A dedicated test suite was implemented in [tests/test_api.py](file:///C:/Users/ASUS/OneDrive/Desktop/Project-471/tests/test_api.py) using Flask's test client:

```text
test_01_root_endpoint                                        ... ok
test_02_health_endpoint                                      ... ok
test_03_predict_valid_image                                  ... ok
test_04_predict_valid_image_image_field_name                 ... ok
test_05_predict_missing_file_payload                         ... ok
test_06_predict_empty_filename                               ... ok
test_07_predict_unsupported_file_extension                   ... ok
test_08_predict_empty_zero_byte_file                         ... ok
test_09_predict_corrupted_image_bytes                        ... ok
test_10_predict_custom_confidence_threshold_uncertain        ... ok
test_11_predict_invalid_confidence_threshold                 ... ok
test_12_not_found_endpoint                                   ... ok
test_13_method_not_allowed                                   ... ok
test_14_cors_headers_present                                 ... ok

----------------------------------------------------------------------
Ran 14 tests in 8.491s
OK (All 14 API tests passed cleanly)
```

### Cumulative Project Test Suite Status
Running the complete test discovery across all modules:
- [tests/test_api.py](file:///C:/Users/ASUS/OneDrive/Desktop/Project-471/tests/test_api.py): 14 Tests (Endpoints, validation, error handlers)
- [tests/test_predictor.py](file:///C:/Users/ASUS/OneDrive/Desktop/Project-471/tests/test_predictor.py): 12 Tests (Inference, Top-3, blur detection, JSON)
- [tests/test_model.py](file:///C:/Users/ASUS/OneDrive/Desktop/Project-471/tests/test_model.py): 7 Tests (Architecture, weights, Softmax parity)
- [tests/test_preprocessing.py](file:///C:/Users/ASUS/OneDrive/Desktop/Project-471/tests/test_preprocessing.py): 9 Tests (Resizing, normalization range $[-1, 1]$, BGR/RGB)
- **Total Test Suite**: **42 / 42 Tests Passed (100% Success Rate in 14.46s)**.

---

## 6. How to Run the Flask Server

### Option A: Direct Python Execution
```powershell
.\.venv\Scripts\python.exe backend/app.py
```

### Option B: Flask CLI
```powershell
$env:FLASK_APP = "backend.app:app"
$env:FLASK_ENV = "development"
.\.venv\Scripts\flask.exe run --host=127.0.0.1 --port=5000
```

### Option C: Testing via cURL
```powershell
# 1. Health check
curl http://127.0.0.1:5000/api/health

# 2. Prediction request with leaf image
curl -X POST -F "file=@datasets/processed/plantdoc/test/Apple_leaf/sample.jpg" http://127.0.0.1:5000/api/predict
```

---

## 7. Remaining Limitations & Agricultural Considerations

1. **Standalone Deployment**: The backend is ready for local development. For production deployment on edge devices or cloud nodes, an asynchronous WSGI server (such as Waitress or Gunicorn) will be recommended.
2. **Stateless Operation**: The backend does not yet record historical user queries; persistent SQLite database logging will be integrated in subsequent modules.
3. **Frontend Decoupling**: HTML/CSS/JS frontend templates and static file routing will connect to this backend in the upcoming module.

---

## 8. Conclusion and Module 7 Readiness

Module 6 is completed and verified against all functional requirements. The Flask backend is fully operational, thoroughly tested, and ready to connect to user-facing frontend interfaces upon user approval.
