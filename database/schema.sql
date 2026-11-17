-- ==============================================================================
-- Leaf Quality Classification and Plucking Standard Monitoring System
-- Module 8: SQLite Database Schema
-- File: database/schema.sql
-- ==============================================================================

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

-- Performance indexes for frequent query patterns (filtering, sorting, search)
CREATE INDEX IF NOT EXISTS idx_leaf_analysis_timestamp ON leaf_analysis(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_leaf_analysis_species ON leaf_analysis(species);
CREATE INDEX IF NOT EXISTS idx_leaf_analysis_visual_health ON leaf_analysis(visual_health);
CREATE INDEX IF NOT EXISTS idx_leaf_analysis_status ON leaf_analysis(prediction_status);
