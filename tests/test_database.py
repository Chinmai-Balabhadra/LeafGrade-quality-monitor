"""
Leaf Quality Classification and Plucking Standard Monitoring System
Module 8: Automated Test Suite for SQLite Database Layer
File: tests/test_database.py
"""

import os
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from database.db import (
    init_db,
    get_connection,
    insert_analysis,
    get_analysis_by_id,
    get_history,
    get_statistics,
    delete_analysis,
    clear_all_history
)


class TestDatabaseLayer(unittest.TestCase):
    """Unit tests for SQLite database operations, queries, and analytics."""

    def setUp(self):
        """Create a temporary directory and isolated test database file."""
        self.test_dir = tempfile.mkdtemp(prefix="test_leaf_db_")
        self.db_path = Path(self.test_dir) / "test_leaf_analysis.db"
        init_db(self.db_path)

    def tearDown(self):
        """Clean up temporary test directory."""
        if Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def _sample_healthy_record(self, filename="apple_healthy_001.jpg"):
        return {
            "source": filename,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "species": "Apple",
            "predicted_class": "Apple_scab_healthy",
            "confidence": 0.942,
            "visual_health": "Healthy",
            "is_visually_healthy": True,
            "condition": "Healthy Foliage",
            "plucking_suitability": "CONDITIONAL_VERIFICATION",
            "status": "SUCCESS",
            "plucking_recommendation": "Visually healthy. Inspect bud maturity in field before harvesting.",
            "top_three": [
                {"rank": 1, "class_name": "Apple_scab_healthy", "confidence": 0.942, "confidence_percent": 94.2, "visual_health_type": "Healthy"},
                {"rank": 2, "class_name": "Apple_Black_rot", "confidence": 0.038, "confidence_percent": 3.8, "visual_health_type": "Diseased"},
                {"rank": 3, "class_name": "Apple_Cedar_rust", "confidence": 0.020, "confidence_percent": 2.0, "visual_health_type": "Diseased"}
            ],
            "quality_assessment": {
                "laplacian_variance": 425.6,
                "mean_brightness": 128.4,
                "is_blurry": False,
                "quality_rating": "Sharp"
            }
        }

    def _sample_diseased_record(self, filename="tomato_blight_002.jpg"):
        return {
            "source": filename,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "species": "Tomato",
            "predicted_class": "Tomato_Early_blight",
            "confidence": 0.887,
            "visual_health": "Diseased",
            "is_visually_healthy": False,
            "condition": "Early Blight Infection",
            "plucking_suitability": "REJECT",
            "status": "SUCCESS",
            "plucking_recommendation": "Pathology detected. Do not harvest diseased foliage.",
            "top_three": [
                {"rank": 1, "class_name": "Tomato_Early_blight", "confidence": 0.887, "confidence_percent": 88.7, "visual_health_type": "Diseased"},
                {"rank": 2, "class_name": "Tomato_Late_blight", "confidence": 0.071, "confidence_percent": 7.1, "visual_health_type": "Diseased"},
                {"rank": 3, "class_name": "Tomato_healthy", "confidence": 0.042, "confidence_percent": 4.2, "visual_health_type": "Healthy"}
            ],
            "quality_assessment": {
                "laplacian_variance": 195.2,
                "mean_brightness": 115.0,
                "is_blurry": False,
                "quality_rating": "Acceptable"
            }
        }

    def test_01_init_db(self):
        """Verify init_db creates the leaf_analysis table and performance indexes."""
        conn = get_connection(self.db_path)
        try:
            # Check table existence
            table_row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='leaf_analysis';"
            ).fetchone()
            self.assertIsNotNone(table_row, "leaf_analysis table should exist")

            # Check indexes
            indexes = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='leaf_analysis';"
            ).fetchall()
            index_names = {row["name"] for row in indexes}
            self.assertIn("idx_leaf_analysis_timestamp", index_names)
            self.assertIn("idx_leaf_analysis_species", index_names)
            self.assertIn("idx_leaf_analysis_visual_health", index_names)
            self.assertIn("idx_leaf_analysis_status", index_names)
        finally:
            conn.close()

    def test_02_insert_healthy_record(self):
        """Verify inserting a healthy leaf analysis record returns positive integer ID."""
        record = self._sample_healthy_record()
        analysis_id = insert_analysis(record, db_path=self.db_path)
        self.assertIsInstance(analysis_id, int)
        self.assertGreater(analysis_id, 0)

        # Retrieve and verify fields
        saved = get_analysis_by_id(analysis_id, db_path=self.db_path)
        self.assertIsNotNone(saved)
        self.assertEqual(saved["id"], analysis_id)
        self.assertEqual(saved["species"], "Apple")
        self.assertEqual(saved["visual_health"], "Healthy")
        self.assertEqual(saved["plucking_suitability"], "CONDITIONAL_VERIFICATION")
        self.assertEqual(saved["prediction_status"], "Certain")
        self.assertAlmostEqual(saved["confidence"], 0.942, places=3)
        self.assertEqual(len(saved["top_three"]), 3)
        self.assertEqual(saved["quality_assessment"]["quality_rating"], "Sharp")

    def test_03_insert_diseased_record(self):
        """Verify inserting a diseased leaf analysis record."""
        record = self._sample_diseased_record()
        analysis_id = insert_analysis(record, db_path=self.db_path)
        self.assertGreater(analysis_id, 0)

        saved = get_analysis_by_id(analysis_id, db_path=self.db_path)
        self.assertEqual(saved["species"], "Tomato")
        self.assertEqual(saved["visual_health"], "Diseased")
        self.assertEqual(saved["plucking_suitability"], "REJECT")

    def test_04_get_analysis_by_id_not_found(self):
        """Verify get_analysis_by_id returns None for non-existent IDs."""
        saved = get_analysis_by_id(99999, db_path=self.db_path)
        self.assertIsNone(saved)

    def test_05_filename_sanitization(self):
        """Verify filesystem directories and absolute paths are stripped to basename only."""
        record = self._sample_healthy_record(
            filename="C:\\Users\\Farmer\\Desktop\\Project-471\\datasets\\cleaned\\apple_leaf.jpg"
        )
        analysis_id = insert_analysis(record, db_path=self.db_path)
        saved = get_analysis_by_id(analysis_id, db_path=self.db_path)
        self.assertEqual(saved["image_filename"], "apple_leaf.jpg")
        self.assertNotIn("Users", saved["image_filename"])
        self.assertNotIn("C:", saved["image_filename"])

    def test_06_pagination(self):
        """Verify pagination metadata and record counts across multiple pages."""
        # Insert 15 records
        for i in range(15):
            rec = self._sample_healthy_record(filename=f"leaf_{i:02d}.jpg")
            insert_analysis(rec, db_path=self.db_path)

        # Page 1, limit 6
        records_p1, meta_p1 = get_history(page=1, limit=6, db_path=self.db_path)
        self.assertEqual(len(records_p1), 6)
        self.assertEqual(meta_p1["page"], 1)
        self.assertEqual(meta_p1["total_records"], 15)
        self.assertEqual(meta_p1["total_pages"], 3)
        self.assertTrue(meta_p1["has_next"])
        self.assertFalse(meta_p1["has_prev"])

        # Page 2, limit 6
        records_p2, meta_p2 = get_history(page=2, limit=6, db_path=self.db_path)
        self.assertEqual(len(records_p2), 6)
        self.assertEqual(meta_p2["page"], 2)
        self.assertTrue(meta_p2["has_next"])
        self.assertTrue(meta_p2["has_prev"])

        # Page 3, limit 6 (remaining 3 records)
        records_p3, meta_p3 = get_history(page=3, limit=6, db_path=self.db_path)
        self.assertEqual(len(records_p3), 3)
        self.assertFalse(meta_p3["has_next"])
        self.assertTrue(meta_p3["has_prev"])

    def test_07_filter_by_species(self):
        """Verify species query filtering."""
        insert_analysis(self._sample_healthy_record(filename="apple1.jpg"), db_path=self.db_path)
        insert_analysis(self._sample_healthy_record(filename="apple2.jpg"), db_path=self.db_path)
        insert_analysis(self._sample_diseased_record(filename="tomato1.jpg"), db_path=self.db_path)

        records, meta = get_history(species="Apple", db_path=self.db_path)
        self.assertEqual(meta["total_records"], 2)
        for r in records:
            self.assertEqual(r["species"], "Apple")

        records_tomato, meta_tomato = get_history(species="Tomato", db_path=self.db_path)
        self.assertEqual(meta_tomato["total_records"], 1)
        self.assertEqual(records_tomato[0]["species"], "Tomato")

    def test_08_filter_by_visual_health(self):
        """Verify visual health filtering (Healthy vs Diseased)."""
        insert_analysis(self._sample_healthy_record(filename="apple1.jpg"), db_path=self.db_path)
        insert_analysis(self._sample_healthy_record(filename="apple2.jpg"), db_path=self.db_path)
        insert_analysis(self._sample_diseased_record(filename="tomato1.jpg"), db_path=self.db_path)

        healthy_records, healthy_meta = get_history(visual_health="Healthy", db_path=self.db_path)
        self.assertEqual(healthy_meta["total_records"], 2)

        diseased_records, diseased_meta = get_history(visual_health="Diseased", db_path=self.db_path)
        self.assertEqual(diseased_meta["total_records"], 1)

    def test_09_search_functionality(self):
        """Verify text search across predicted_class, species, condition, and filename."""
        insert_analysis(self._sample_healthy_record(filename="apple_foliage.jpg"), db_path=self.db_path)
        insert_analysis(self._sample_diseased_record(filename="tomato_fungus.jpg"), db_path=self.db_path)

        # Search by species
        res, meta = get_history(search="apple", db_path=self.db_path)
        self.assertEqual(meta["total_records"], 1)

        # Search by disease term
        res, meta = get_history(search="Early_blight", db_path=self.db_path)
        self.assertEqual(meta["total_records"], 1)

        # Search by filename
        res, meta = get_history(search="fungus", db_path=self.db_path)
        self.assertEqual(meta["total_records"], 1)

    def test_10_get_statistics(self):
        """Verify analytics aggregation calculations."""
        # When empty
        stats_empty = get_statistics(db_path=self.db_path)
        self.assertEqual(stats_empty["total_records"], 0)
        self.assertEqual(stats_empty["average_confidence"], 0.0)

        # Add 2 healthy (conf 0.94) + 1 diseased (conf 0.88)
        insert_analysis(self._sample_healthy_record(), db_path=self.db_path)
        insert_analysis(self._sample_healthy_record(), db_path=self.db_path)
        insert_analysis(self._sample_diseased_record(), db_path=self.db_path)

        stats = get_statistics(db_path=self.db_path)
        self.assertEqual(stats["total_records"], 3)
        self.assertEqual(stats["healthy_count"], 2)
        self.assertEqual(stats["diseased_count"], 1)
        self.assertAlmostEqual(stats["average_confidence"], (0.942 + 0.942 + 0.887) / 3.0, places=2)
        self.assertEqual(stats["by_species"]["Apple"], 2)
        self.assertEqual(stats["by_species"]["Tomato"], 1)

    def test_11_delete_single_record(self):
        """Verify deleting an existing record by ID."""
        id1 = insert_analysis(self._sample_healthy_record(), db_path=self.db_path)
        id2 = insert_analysis(self._sample_diseased_record(), db_path=self.db_path)

        # Delete id1
        success = delete_analysis(id1, db_path=self.db_path)
        self.assertTrue(success)

        # Confirm id1 is gone but id2 persists
        self.assertIsNone(get_analysis_by_id(id1, db_path=self.db_path))
        self.assertIsNotNone(get_analysis_by_id(id2, db_path=self.db_path))

        # Deleting already deleted id returns False
        self.assertFalse(delete_analysis(id1, db_path=self.db_path))

    def test_12_clear_all_history(self):
        """Verify clearing all history records and resetting autoincrement sequence."""
        insert_analysis(self._sample_healthy_record(), db_path=self.db_path)
        insert_analysis(self._sample_diseased_record(), db_path=self.db_path)

        deleted_count = clear_all_history(db_path=self.db_path)
        self.assertEqual(deleted_count, 2)

        # Verify table is now empty
        stats = get_statistics(db_path=self.db_path)
        self.assertEqual(stats["total_records"], 0)

        # Next inserted record should restart ID at 1
        new_id = insert_analysis(self._sample_healthy_record(), db_path=self.db_path)
        self.assertEqual(new_id, 1)


if __name__ == "__main__":
    unittest.main()
