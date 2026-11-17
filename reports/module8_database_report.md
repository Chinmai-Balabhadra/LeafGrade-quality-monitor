# Module 8 Report: SQLite Database Integration and Historical Analysis

**Project Title**: Leaf Quality Classification and Plucking Standard Monitoring System Using Handheld Camera  
**Course Code**: CSE471 Academic Project  
**Module**: Module 8 – SQLite Database Integration and Historical Analysis  
**Database Component**: [database/db.py](file:///C:/Users/ASUS/OneDrive/Desktop/Project-471/database/db.py)  
**Schema Definition**: [database/schema.sql](file:///C:/Users/ASUS/OneDrive/Desktop/Project-471/database/schema.sql)  
**Active Database File**: [database/leaf_analysis.db](file:///C:/Users/ASUS/OneDrive/Desktop/Project-471/database/leaf_analysis.db)  
**Historical API Routes**: [backend/routes/history.py](file:///C:/Users/ASUS/OneDrive/Desktop/Project-471/backend/routes/history.py)  
**Frontend View Controller**: [frontend/js/history.js](file:///C:/Users/ASUS/OneDrive/Desktop/Project-471/frontend/js/history.js)  
**Status**: **Completed & Fully Verified (71/71 Tests Passing)**  
**Date**: November 2026  

---

## 1. Executive Summary

Module 8 introduces persistent local storage and historical quality tracking for the Leaf Quality Classification and Plucking Standard Monitoring System. Prior to Module 8, leaf inference predictions were ephemeral and lost upon page refresh. 

With this module, all leaf analyses executed through `POST /api/predict` are systematically committed to a lightweight, embedded **SQLite** database (`database/leaf_analysis.db`). The system provides a complete historical analysis subsystem enabling farmers, field inspectors, and agricultural researchers to:
1. Review chronological leaf inspection records across field operations.
2. Filter inspections dynamically by crop species, visual health category (`Healthy` vs `Diseased`), or prediction confidence status (`Certain` vs `Uncertain`).
3. Search inspection logs by partial crop names, specific pathology terms, or image filenames.
4. Monitor aggregated farm health trends (overall healthy-to-diseased ratio, crop distributions, average model confidence).
5. Inspect granular image diagnostics, blur ratings, Laplacian sharpness values, and Top-3 candidate probabilities in a dedicated detail modal.
6. Safely manage data lifecycle with single-record deletion and clear-all capabilities protected by double confirmation safeguards.

The implementation strictly utilizes Python's built-in `sqlite3` library, adhering to zero external database service overhead (no MySQL, PostgreSQL, or MongoDB required).

---

## 2. Database Architecture and Configuration

```text
Project-471/
├── database/
│   ├── __init__.py             # Package exports (db interface functions)
│   ├── schema.sql              # DDL schema definition & performance indexes
│   ├── db.py                   # Data access layer, connection pooling, queries & analytics
│   └── leaf_analysis.db        # Active SQLite database file
├── backend/
│   ├── app.py                  # Registers history_bp blueprint & updated CORS rules
│   └── routes/
│       ├── prediction.py       # Auto-saves analysis records upon inference completion
│       └── history.py          # REST API endpoints for history retrieval, filtering, stats, deletion
├── frontend/
│   ├── index.html              # Historical analysis dashboard section, stats cards, table, modal
│   ├── css/
│   │   ├── dashboard.css       # History table, badges, toolbar, stats cards, and modal dialog styles
│   │   └── responsive.css      # Mobile & tablet viewports for historical table & modal
│   └── js/
│       ├── api.js              # Extended ApiService with history, stats, and deletion methods
│       ├── history.js          # HistoryController (data fetching, debounce search, pagination, modal)
│       ├── prediction.js       # Auto-triggers HistoryController.refresh() on completed analysis
│       └── main.js             # Bootstraps HistoryController on DOM ready
└── tests/
    ├── test_database.py        # Dedicated unit tests for SQLite database operations
    ├── test_api.py             # Extended with 9 history endpoint tests (23 tests total)
    └── test_frontend_integration.py # Extended with history DOM and API contract assertions
```

### 2.1 SQLite Pragmas and Concurrency Modes
In [database/db.py](file:///C:/Users/ASUS/OneDrive/Desktop/Project-471/database/db.py), database connections are established with the following performance and integrity settings:
- **`PRAGMA journal_mode=WAL;` (Write-Ahead Logging)**: Enables concurrent readers and writers without lock contention, crucial for web application request concurrency.
- **`PRAGMA foreign_keys=ON;`**: Enforces relational constraints.
- **`conn.row_factory = sqlite3.Row`**: Enables dictionary-like attribute access on query results.
- **Connection Timeout**: Configured to 15.0 seconds to prevent `database is locked` exceptions under concurrent write bursts.

---

## 3. Database Schema and Data Dictionary

The table definition in [database/schema.sql](file:///C:/Users/ASUS/OneDrive/Desktop/Project-471/database/schema.sql) is optimized for agricultural diagnostics:

```sql
CREATE TABLE IF NOT EXISTS leaf_analysis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    image_filename TEXT NOT NULL,
    species TEXT NOT NULL,
    predicted_class TEXT NOT NULL,
    confidence REAL NOT NULL,
    visual_health TEXT NOT NULL,
    condition TEXT NOT NULL,
    plucking_suitability TEXT NOT NULL,
    prediction_status TEXT NOT NULL,
    top_three_json TEXT,
    quality_assessment_json TEXT,
    plucking_recommendation TEXT
);
```

### Data Dictionary

| Column Name | SQLite Type | Description / Constraints | Example Value |
| :--- | :--- | :--- | :--- |
| `id` | `INTEGER` | Primary Key, Auto-incremented identifier | `1` |
| `timestamp` | `TEXT` | UTC ISO 8601 formatted timestamp | `2026-11-16T04:45:01.372Z` |
| `image_filename` | `TEXT` | Sanitized basename of the uploaded image file | `leaf_sample_01.jpg` |
| `species` | `TEXT` | Plant / crop common name | `Apple`, `Tomato`, `Grape` |
| `predicted_class` | `TEXT` | Model predicted class from the 28 PlantDoc classes | `Apple_scab_healthy` |
| `confidence` | `REAL` | Raw softmax probability score ($0.0 \le p \le 1.0$) | `0.9423` |
| `visual_health` | `TEXT` | Visual condition classification (`Healthy` or `Diseased`) | `Healthy` |
| `condition` | `TEXT` | Descriptive pathological condition or symptom | `Healthy Foliage` |
| `plucking_suitability`| `TEXT` | Agronomic harvesting guidance indicator | `CONDITIONAL_VERIFICATION` |
| `prediction_status` | `TEXT` | Inference reliability classification (`Certain` or `Uncertain`)| `Certain` |
| `top_three_json` | `TEXT` | Serialized JSON array of top-3 candidate classes | `[{"rank":1,"class_name":"..."}]` |
| `quality_assessment_json`| `TEXT` | Serialized JSON object of Laplacian variance & blur | `{"laplacian_variance":425.6}` |
| `plucking_recommendation`| `TEXT` | Full agronomic guidance advisory message | `Visually healthy. Verify bud maturity.` |

---

## 4. Query Optimization & Indexing Strategy

To maintain sub-5 millisecond query execution times even as the log grows to tens of thousands of records, four dedicated B-tree indexes were created:

```sql
CREATE INDEX IF NOT EXISTS idx_leaf_analysis_timestamp ON leaf_analysis(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_leaf_analysis_species ON leaf_analysis(species);
CREATE INDEX IF NOT EXISTS idx_leaf_analysis_visual_health ON leaf_analysis(visual_health);
CREATE INDEX IF NOT EXISTS idx_leaf_analysis_status ON leaf_analysis(prediction_status);
```

- **`idx_leaf_analysis_timestamp`**: Accelerates reverse-chronological pagination (`ORDER BY timestamp DESC LIMIT :limit OFFSET :offset`).
- **`idx_leaf_analysis_species`**: Optimizes crop filtering in dashboards.
- **`idx_leaf_analysis_visual_health`**: Accelerates healthy vs. diseased statistical aggregations.
- **`idx_leaf_analysis_status`**: Speeds up filtering for low-confidence warnings.

---

## 5. Privacy, Security, and Path Isolation

1. **Basename Path Sanitization**:
   In [database/db.py](file:///C:/Users/ASUS/OneDrive/Desktop/Project-471/database/db.py), raw file paths (e.g. `C:\Users\Admin\Desktop\Project-471\datasets\leaf.jpg`) are stripped using `Path(raw_source).name` before database insertion. Only `leaf.jpg` is persisted. This strictly prevents accidental leakage of user filenames, directory structures, or server operating system details.
2. **Zero Raw Image Storage**:
   Raw image bytes are never stored directly in the database, preventing unbounded database file bloat and preserving database performance.
3. **Graceful Persistence Isolation**:
   In [backend/routes/prediction.py](file:///C:/Users/ASUS/OneDrive/Desktop/Project-471/backend/routes/prediction.py), the call to `insert_analysis(result)` is wrapped in an isolated `try...except` block. If a database disk write fails (e.g. disk quota full or file permissions issue), the endpoint logs a warning, sets `history_saved: False`, and still delivers the model classification result to the user without raising an HTTP 500 error.

---

## 6. Historical Analysis REST API Specification

All historical analysis endpoints are registered under the `/api` prefix via `history_bp` in [backend/routes/history.py](file:///C:/Users/ASUS/OneDrive/Desktop/Project-471/backend/routes/history.py):

| Method | Endpoint | Query Parameters | Description | Status Code |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/api/history` | `page`, `limit`, `species`, `visual_health`, `status`, `search` | Paginated list of historical inspection records | `200 OK` |
| `GET` | `/api/history/stats` | *None* | Aggregated analytics & distributions | `200 OK` |
| `GET` | `/api/history/<id>` | *None* | Full diagnostic details for record ID | `200 OK` / `404 Not Found` |
| `DELETE` | `/api/history/<id>` | *None* | Deletes a single record by ID | `200 OK` / `404 Not Found` |
| `DELETE` | `/api/history` | `confirm=true` (required) | Clears all records & resets ID sequence | `200 OK` / `400 Bad Request` |

### Sample Response: `GET /api/history/stats`
```json
{
  "status": "SUCCESS",
  "statistics": {
    "total_analyses": 48,
    "total_records": 48,
    "healthy_count": 32,
    "diseased_count": 16,
    "uncertain_count": 3,
    "healthy_percentage": 66.7,
    "diseased_percentage": 33.3,
    "average_confidence": 0.8924,
    "average_confidence_percent": 89.24,
    "by_species": {
      "Apple": 18,
      "Tomato": 14,
      "Grape": 10,
      "Corn": 6
    },
    "species_distribution": [
      {"species": "Apple", "count": 18},
      {"species": "Tomato", "count": 14},
      {"species": "Grape", "count": 10},
      {"species": "Corn", "count": 6}
    ]
  }
}
```

---

## 7. Frontend Historical Analysis Dashboard Interface

The frontend dashboard in [frontend/index.html](file:///C:/Users/ASUS/OneDrive/Desktop/Project-471/frontend/index.html) and [frontend/js/history.js](file:///C:/Users/ASUS/OneDrive/Desktop/Project-471/frontend/js/history.js) provides an intuitive, responsive user experience:

1. **Analytics Summary Cards**:
   - **Total Analyses**: Real-time counter of total inspections performed.
   - **Healthy Leaves**: Count of foliage free of visual pathology.
   - **Diseased / Damaged**: Count of infected or pest-damaged leaves.
   - **Average Confidence**: Mean neural network classification confidence across all evaluations.
2. **Interactive Search & Filter Toolbar**:
   - **Search Input**: Debounced (300ms) live search querying crop species, class name, disease condition, or filename.
   - **Crop Species Dropdown**: Dynamically populated with unique crop varieties present in the database.
   - **Visual Health Dropdown**: Filters by `Healthy` or `Diseased`.
   - **Clear Log Button**: Triggers confirmation dialog to safely wipe history.
3. **Inspection History Data Table**:
   - Formats UTC timestamps to user's localized date and time.
   - Visual health badges (`badge-healthy` green vs. `badge-diseased` red).
   - Plucking standard indicators (`Conditional`, `Do Not Pluck`, `Deferred`).
   - Certainty icons denoting confident vs. uncertain classifications.
   - Quick action buttons for viewing full diagnostics or deleting single records.
4. **Diagnostic Details Modal**:
   - Accessible by clicking the eye icon on any historical row.
   - Displays full image quality metrics (Laplacian sharpness variance, mean brightness).
   - Renders animated Top-3 probability breakdown bars.
   - Outlines complete agronomic advisory text with plucking maturity instructions.
5. **Seamless Real-Time Synchronization**:
   - When a user uploads or captures a new leaf in `#analysis`, `PredictionController` automatically calls `HistoryController.refresh()`. The table and statistics update instantly without requiring a full page refresh.

---

## 8. Database Backup, Maintenance, and Recovery Procedures

### 8.1 Online Hot Backup
Because SQLite is configured in WAL mode, backups can be performed live while the web service is running without blocking read or write queries:

```bash
# SQLite CLI online backup
sqlite3 database/leaf_analysis.db ".backup 'database/backup_leaf_analysis.db'"
```

### 8.2 Integrity Check
```bash
sqlite3 database/leaf_analysis.db "PRAGMA integrity_check;"
# Expected output: ok
```

### 8.3 Compaction & Defragmentation (Vacuuming)
After large deletion batches (e.g. clearing old logs):
```bash
sqlite3 database/leaf_analysis.db "VACUUM;"
```

---

## 9. Comprehensive Test Suite Verification

The project includes an exhaustive automated test suite verifying all modules:

```text
tests/
├── test_preprocessing.py         [9 Tests  - All Passing]
├── test_model.py                 [7 Tests  - All Passing]
├── test_predictor.py             [12 Tests - All Passing]
├── test_database.py              [12 Tests - All Passing]
├── test_api.py                   [23 Tests - All Passing]
└── test_frontend_integration.py  [8 Tests  - All Passing]
```

### Test Discovery Execution
```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

### Output Summary
```text
----------------------------------------------------------------------
Ran 71 tests in 25.137s

OK (100% Passing, 0 Errors, 0 Failures)
```

---

## 10. Summary of Deliverables

| Deliverable | Path | Description |
| :--- | :--- | :--- |
| **SQL Schema** | `database/schema.sql` | Table definition, columns, and 4 performance indexes |
| **Database Module** | `database/db.py` | Connection pooling, WAL mode, CRUD, pagination, and analytics |
| **Database File** | `database/leaf_analysis.db` | Embedded SQLite persistent database file |
| **History Routes** | `backend/routes/history.py` | REST API routes (`GET`, `DELETE` with confirm guard) |
| **Prediction Route Update**| `backend/routes/prediction.py` | Integrated `insert_analysis(result)` with fail-safe error handling |
| **App Registration** | `backend/app.py` | Registered `history_bp` and updated CORS for DELETE |
| **History Controller** | `frontend/js/history.js` | Frontend controller for table, search, filters, pagination, modal |
| **API Client Update** | `frontend/js/api.js` | Added history API client methods |
| **Prediction UI Sync** | `frontend/js/prediction.js` | Added auto-refresh hook to `HistoryController.refresh()` |
| **Dashboard UI** | `frontend/index.html` | History section, stats cards, table, pagination, modal dialog |
| **Styling** | `frontend/css/dashboard.css` | Complete styles for history components |
| **Responsive Rules** | `frontend/css/responsive.css` | Media query adaptations for mobile/tablet screens |
| **Database Tests** | `tests/test_database.py` | 12 comprehensive unit tests for database layer |
| **API Tests** | `tests/test_api.py` | Expanded to 23 tests covering history endpoints |
| **Frontend Integration Tests**| `tests/test_frontend_integration.py`| Expanded to 8 tests covering history DOM & API contracts |
| **Module Report** | `reports/module8_database_report.md` | Comprehensive architectural and empirical report |
