#!/usr/bin/env python3
"""
Unit tests for the Random Forest land cover classification model.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from train_random_forest import LandCoverClassifier, MODISFeatureExtractor


class TestMODISFeatureExtractor(unittest.TestCase):
    """Unit tests for MODISFeatureExtractor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_data_dir = Path("data/processed")
        self.extractor = MODISFeatureExtractor(self.test_data_dir)

    def test_extract_spectral_features(self):
        """Test spectral feature extraction."""
        # Create test data
        test_data = np.random.randint(1, 17, size=(10, 10))

        features = self.extractor.extract_spectral_features(test_data)

        # Should return 8 spectral features
        self.assertEqual(len(features), 8)
        self.assertTrue(np.all(np.isfinite(features)))

    def test_extract_spatial_features(self):
        """Test spatial feature extraction."""
        test_data = np.random.randint(1, 17, size=(10, 10))

        features = self.extractor.extract_spatial_features(test_data)

        # Should return 4 spatial features
        self.assertEqual(len(features), 4)
        self.assertTrue(np.all(np.isfinite(features)))

    def test_extract_texture_features(self):
        """Test texture feature extraction."""
        test_data = np.random.randint(1, 17, size=(10, 10))

        features = self.extractor.extract_texture_features(test_data)

        # Should return 4 texture features
        self.assertEqual(len(features), 4)
        self.assertTrue(np.all(np.isfinite(features)))

    def test_create_feature_names(self):
        """Test feature names creation."""
        feature_names = self.extractor.create_feature_names()

        # Should have 16 base features (without coordinate features in base method)
        expected_count = 8 + 4 + 4  # spectral + spatial + texture
        self.assertEqual(len(feature_names), expected_count)

        # Check for expected feature types
        spectral_features = [name for name in feature_names if name.startswith("spectral_")]
        self.assertEqual(len(spectral_features), 8)


class TestLandCoverClassifier(unittest.TestCase):
    """Unit tests for LandCoverClassifier class."""

    def setUp(self):
        """Set up test fixtures."""
        self.classifier = LandCoverClassifier("random_forest")

    def test_init(self):
        """Test classifier initialization."""
        self.assertEqual(self.classifier.algorithm, "random_forest")
        self.assertIsNone(self.classifier.model)
        self.assertEqual(len(self.classifier.class_names), 17)

    def test_create_model_random_forest(self):
        """Test Random Forest model creation."""
        model = self.classifier.create_model(n_estimators=10, max_depth=5)

        self.assertIsNotNone(model)
        self.assertEqual(model.n_estimators, 10)
        self.assertEqual(model.max_depth, 5)

    def test_create_model_gradient_boosting(self):
        """Test Gradient Boosting model creation."""
        classifier = LandCoverClassifier("gradient_boosting")
        model = classifier.create_model(n_estimators=10, max_depth=3)

        self.assertIsNotNone(model)
        self.assertEqual(model.n_estimators, 10)
        self.assertEqual(model.max_depth, 3)

    def test_train_with_synthetic_data(self):
        """Test model training with synthetic data."""
        # Create synthetic data
        n_samples = 1000
        n_features = 18
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(1, 8, n_samples)  # 7 land cover classes

        self.classifier.feature_names = [f"feature_{i}" for i in range(n_features)]

        # Train model
        metrics, report = self.classifier.train(X, y, n_estimators=10, max_depth=5)

        # Check that model is trained
        self.assertIsNotNone(self.classifier.model)

        # Check metrics
        self.assertIn("accuracy", metrics)
        self.assertIn("f1_macro", metrics)
        self.assertIn("training_time", metrics)

        # Check that accuracy is reasonable (>0 for random data)
        self.assertGreater(metrics["accuracy"], 0)
        self.assertLess(metrics["training_time"], 60)  # Should train quickly

    def test_feature_importance(self):
        """Test feature importance extraction."""
        # Train with synthetic data first
        n_samples = 500
        n_features = 10
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(1, 5, n_samples)

        self.classifier.feature_names = [f"feature_{i}" for i in range(n_features)]
        self.classifier.train(X, y, n_estimators=10, max_depth=3)

        # Get feature importance
        importance_df = self.classifier.get_feature_importance()

        self.assertIsInstance(importance_df, pd.DataFrame)
        self.assertEqual(len(importance_df), n_features)
        self.assertIn("feature", importance_df.columns)
        self.assertIn("importance", importance_df.columns)

    def test_save_load_model(self):
        """Test model saving and loading."""
        # Train a simple model
        n_samples = 100
        n_features = 5
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(1, 3, n_samples)

        self.classifier.feature_names = [f"feature_{i}" for i in range(n_features)]
        self.classifier.train(X, y, n_estimators=5, max_depth=3)

        # Save model to temporary file
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            self.classifier.save_model(tmp_path)

            # Create new classifier and load model
            new_classifier = LandCoverClassifier()
            new_classifier.load_model(tmp_path)

            # Check that model was loaded correctly
            self.assertEqual(new_classifier.algorithm, "random_forest")
            self.assertIsNotNone(new_classifier.model)
            self.assertEqual(len(new_classifier.feature_names), n_features)

        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""

    def test_full_pipeline_synthetic(self):
        """Test the complete pipeline with synthetic data."""
        # Create synthetic TIFF-like data
        extractor = MODISFeatureExtractor(Path("data/processed"))

        # Get the actual feature count from the extractor
        base_feature_names = extractor.create_feature_names()
        # Add coordinate features as done in the actual pipeline
        full_feature_names = base_feature_names + ["coord_row", "coord_col"]
        n_features = len(full_feature_names)
        n_samples = 200

        # Create synthetic features and labels with correct dimensions
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(1, 8, n_samples)

        # Create classifier
        classifier = LandCoverClassifier("random_forest")
        classifier.feature_names = full_feature_names

        # Train model
        metrics, report = classifier.train(X, y, n_estimators=10, max_depth=5)

        # Verify results
        self.assertIsNotNone(classifier.model)
        self.assertGreater(metrics["accuracy"], 0)
        self.assertIsInstance(report, dict)

        # Test feature importance
        importance_df = classifier.get_feature_importance()
        self.assertFalse(importance_df.empty)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
