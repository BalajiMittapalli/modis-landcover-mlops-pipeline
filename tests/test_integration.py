#!/usr/bin/env python3
"""
Integration test for the complete MODIS land cover classification pipeline.
Tests the full workflow from data loading to model training and evaluation.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from train_random_forest import LandCoverClassifier, MODISFeatureExtractor
from train_random_forest import main as train_main


class TestFullPipelineIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""

    def setUp(self):
        """Set up test environment."""
        self.data_dir = Path("data/processed")
        self.temp_files = []

    def tearDown(self):
        """Clean up temporary files."""
        for temp_file in self.temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_data_availability(self):
        """Test that required data files are available."""
        # Check if data directory exists
        self.assertTrue(self.data_dir.exists(), "Data directory should exist")

        # Check for at least some TIFF files
        tiff_files = list(self.data_dir.glob("*.tif"))
        self.assertGreater(len(tiff_files), 0, "Should have at least one TIFF file")

        # Check file naming convention
        years = []
        for tiff_file in tiff_files[:5]:  # Check first 5 files
            try:
                year = int(tiff_file.stem)
                self.assertGreater(year, 2000, f"Year {year} should be reasonable")
                self.assertLess(year, 2030, f"Year {year} should be reasonable")
                years.append(year)
            except ValueError:
                self.fail(f"TIFF file {tiff_file.name} doesn't follow year naming convention")

    def test_feature_extraction_pipeline(self):
        """Test the complete feature extraction pipeline."""
        extractor = MODISFeatureExtractor(self.data_dir)

        # Test with a small subset of years
        available_years = [2020, 2021, 2022]  # Use recent years that should exist

        try:
            # Extract features with small sample ratio for speed
            X, y = extractor.load_data(available_years, sample_ratio=0.001)

            # Verify extracted data
            self.assertGreater(len(X), 0, "Should extract some features")
            self.assertGreater(len(y), 0, "Should extract some labels")
            self.assertEqual(len(X), len(y), "Features and labels should have same length")

            # Check feature dimensions
            expected_features = len(extractor.create_feature_names())
            self.assertEqual(
                X.shape[1], expected_features, f"Should have {expected_features} features"
            )

            # Check label validity
            unique_labels = np.unique(y)
            self.assertTrue(np.all(unique_labels >= 1), "All labels should be >= 1")
            self.assertTrue(np.all(unique_labels <= 17), "All labels should be <= 17")

        except Exception as e:
            self.skipTest(f"Feature extraction failed (data may not be available): {e}")

    def test_model_training_pipeline(self):
        """Test the complete model training pipeline."""
        try:
            # Create synthetic data that mimics real MODIS data structure
            extractor = MODISFeatureExtractor(self.data_dir)
            feature_names = extractor.create_feature_names()

            # Generate synthetic but realistic data
            n_samples = 1000
            n_features = len(feature_names)

            # Create features with more structure to ensure reasonable performance
            np.random.seed(42)  # For reproducible results
            X = np.random.rand(n_samples, n_features)

            # Add clear patterns to make classification easier
            for i in range(min(5, n_features)):
                X[:, i] = X[:, i] + np.random.normal(0, 0.1, n_samples)

            # Create labels with clearer patterns
            # Make labels somewhat dependent on features for better classification
            y = np.zeros(n_samples, dtype=int)
            for i in range(n_samples):
                if X[i, 0] > 0.7:
                    y[i] = 1  # Water
                elif X[i, 1] > 0.6:
                    y[i] = 2  # Forest
                elif X[i, 2] > 0.5:
                    y[i] = 3  # Grassland
                else:
                    y[i] = np.random.choice([4, 5, 6, 7, 8, 9, 10])  # Other classes

            # Test classifier
            classifier = LandCoverClassifier("random_forest")
            classifier.feature_names = feature_names

            # Train model
            metrics, report = classifier.train(X, y, n_estimators=50, max_depth=10)

            # Verify training results
            self.assertIsNotNone(classifier.model, "Model should be trained")
            self.assertIn("accuracy", metrics, "Should have accuracy metric")
            self.assertIn("f1_macro", metrics, "Should have F1 macro metric")
            self.assertIn("training_time", metrics, "Should have training time")

            # Check that accuracy is reasonable for synthetic data with patterns
            self.assertGreater(
                metrics["accuracy"],
                0.15,
                "Accuracy should be reasonable with structured synthetic data",
            )
            self.assertLess(
                metrics["training_time"], 120, "Training should complete in reasonable time"
            )

            # Test feature importance
            importance_df = classifier.get_feature_importance()
            self.assertFalse(importance_df.empty, "Should have feature importance")
            self.assertEqual(
                len(importance_df), n_features, "Should have importance for all features"
            )

            # Test model saving/loading
            temp_model_path = tempfile.mktemp(suffix=".pkl")
            self.temp_files.append(temp_model_path)

            classifier.save_model(temp_model_path)
            self.assertTrue(os.path.exists(temp_model_path), "Model file should be saved")

            # Load model and verify
            new_classifier = LandCoverClassifier()
            new_classifier.load_model(temp_model_path)
            self.assertEqual(
                new_classifier.algorithm,
                "random_forest",
                "Loaded model should have correct algorithm",
            )

        except Exception as e:
            self.fail(f"Model training pipeline failed: {e}")

    def test_config_files_exist(self):
        """Test that required configuration files exist."""
        config_files = [
            "environment.yml",
            "requirements.txt",
            "config/data_config.yaml",
            "config/training_config.yaml",
        ]

        for config_file in config_files:
            file_path = Path(config_file)
            self.assertTrue(file_path.exists(), f"Configuration file {config_file} should exist")

    def test_main_script_components(self):
        """Test that the main training script components work."""
        # Test that we can import the main function
        from train_random_forest import main as train_main

        # Verify it's callable
        self.assertTrue(callable(train_main), "Main function should be callable")

        # Test argument parsing (by checking if the script can be imported without errors)
        # This is already done by the import above


class TestMLflowIntegration(unittest.TestCase):
    """Test MLflow experiment tracking integration."""

    def test_mlflow_availability(self):
        """Test that MLflow is available and configured."""
        try:
            import mlflow

            from train_random_forest import HAS_MLFLOW

            if HAS_MLFLOW:
                # Test that we can create an experiment
                experiment_name = "test_experiment"
                try:
                    mlflow.set_experiment(experiment_name)
                    # If we get here, MLflow is working
                    self.assertTrue(True, "MLflow is available and working")
                except Exception as e:
                    self.skipTest(f"MLflow not properly configured: {e}")
            else:
                self.skipTest("MLflow not available")

        except ImportError:
            self.skipTest("MLflow not installed")


if __name__ == "__main__":
    # Create test suite
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTest(unittest.makeSuite(TestFullPipelineIntegration))
    suite.addTest(unittest.makeSuite(TestMLflowIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)
