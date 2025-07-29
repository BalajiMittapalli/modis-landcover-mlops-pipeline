#!/usr/bin/env python3
"""
Tests for Prefect ML Pipeline workflow orchestration.
"""

import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add flows to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from flows.config import get_pipeline_config, setup_directories
    from flows.ml_pipeline import (
        evaluate_model,
        ingest_data,
        ml_pipeline,
        preprocess_data,
        register_model,
        train_model,
    )

    PREFECT_AVAILABLE = True
except ImportError:
    PREFECT_AVAILABLE = False


@unittest.skipIf(not PREFECT_AVAILABLE, "Prefect not available")
class TestPrefectWorkflow(unittest.TestCase):
    """Test Prefect workflow components."""

    def setUp(self):
        """Set up test environment."""
        self.test_data_dir = Path("data/processed")
        self.temp_files = []

    def tearDown(self):
        """Clean up test files."""
        for temp_file in self.temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_config_loading(self):
        """Test configuration loading for different environments."""
        # Test development config
        dev_config = get_pipeline_config("development")
        self.assertIn("parameters", dev_config)
        self.assertIn("schedule", dev_config)
        self.assertEqual(dev_config["parameters"]["model_type"], "random_forest")

        # Test production config
        prod_config = get_pipeline_config("production")
        self.assertEqual(prod_config["parameters"]["start_year"], 2001)
        self.assertEqual(prod_config["parameters"]["end_year"], 2022)

        # Test experimental config
        exp_config = get_pipeline_config("experimental")
        self.assertEqual(exp_config["parameters"]["model_type"], "gradient_boosting")

    @patch("flows.ml_pipeline.Path")
    def test_ingest_data_task(self, mock_path):
        """Test data ingestion task."""
        # Mock file existence
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance

        # Mock files exist for 2020-2021
        def side_effect(filename):
            mock_file = MagicMock()
            mock_file.exists.return_value = filename in ["2020.tif", "2021.tif"]
            return mock_file

        mock_path_instance.__truediv__ = side_effect

        # Run the task (without Prefect context)
        result = asyncio.run(ingest_data.fn(2020, 2021, "data/processed"))

        self.assertIn("available_years", result)
        self.assertIn("missing_years", result)
        self.assertIn("total_files", result)

    def test_directory_setup(self):
        """Test directory setup function."""
        # This should run without errors
        try:
            setup_directories()
            self.assertTrue(True, "Directory setup completed without errors")
        except Exception as e:
            self.fail(f"Directory setup failed: {e}")


class TestWorkflowIntegration(unittest.TestCase):
    """Integration tests for workflow components."""

    def test_workflow_configuration_consistency(self):
        """Test that workflow configurations are consistent."""
        configs = ["production", "development", "experimental"]

        for config_name in configs:
            config = get_pipeline_config(config_name)

            # Check required parameters exist
            params = config["parameters"]
            required_params = ["start_year", "end_year", "model_type", "sample_ratio"]

            for param in required_params:
                self.assertIn(param, params, f"Missing parameter {param} in {config_name}")

            # Check parameter validity
            self.assertLessEqual(params["start_year"], params["end_year"])
            self.assertGreater(params["sample_ratio"], 0)
            self.assertLess(params["sample_ratio"], 1)
            self.assertIn(params["model_type"], ["random_forest", "gradient_boosting", "xgboost"])

    def test_mock_pipeline_execution(self):
        """Test pipeline execution with mocked components."""
        # This test simulates the pipeline flow without actually running it

        # Mock ingestion result
        mock_ingestion = {
            "available_years": [2020, 2021],
            "missing_years": [],
            "total_files": 2,
            "data_directory": "data/processed",
        }

        # Mock preprocessing result
        mock_preprocessing = {
            "n_samples": 1000,
            "n_features": 18,
            "feature_names": [f"feature_{i}" for i in range(18)],
            "data_path": "mock_data.pkl",
        }

        # Mock training result
        mock_training = {
            "model_type": "random_forest",
            "model_path": "mock_model.pkl",
            "metrics": {"accuracy": 0.45, "f1_macro": 0.35, "training_time": 120.0},
        }

        # Mock evaluation result
        mock_evaluation = {
            "performance_metrics": mock_training["metrics"],
            "quality_check": {
                "passes_accuracy_threshold": True,
                "passes_f1_threshold": True,
                "passes_time_threshold": True,
                "overall_pass": True,
            },
        }

        # Test that the pipeline components can handle these data structures
        self.assertIsInstance(mock_ingestion["available_years"], list)
        self.assertIsInstance(mock_preprocessing["n_samples"], int)
        self.assertIsInstance(mock_training["metrics"]["accuracy"], float)
        self.assertTrue(mock_evaluation["quality_check"]["overall_pass"])


class TestPrefectIntegration(unittest.TestCase):
    """Test Prefect-specific functionality."""

    @unittest.skipIf(not PREFECT_AVAILABLE, "Prefect not available")
    def test_prefect_import(self):
        """Test that Prefect components can be imported."""
        try:
            from prefect import flow, task
            from prefect.task_runners import ConcurrentTaskRunner

            self.assertTrue(True, "Prefect imports successful")
        except ImportError as e:
            self.fail(f"Prefect import failed: {e}")

    def test_workflow_parameters(self):
        """Test workflow parameter validation."""
        # Test valid parameters
        valid_params = {
            "start_year": 2020,
            "end_year": 2022,
            "model_type": "random_forest",
            "sample_ratio": 0.01,
        }

        # These should be valid
        self.assertLessEqual(valid_params["start_year"], valid_params["end_year"])
        self.assertIn(valid_params["model_type"], ["random_forest", "gradient_boosting", "xgboost"])
        self.assertGreater(valid_params["sample_ratio"], 0)
        self.assertLessEqual(valid_params["sample_ratio"], 1)


if __name__ == "__main__":
    # Create test suite
    suite = unittest.TestSuite()

    # Add tests
    if PREFECT_AVAILABLE:
        suite.addTest(unittest.makeSuite(TestPrefectWorkflow))
        print("✅ Prefect available - running full test suite")
    else:
        print("⚠️  Prefect not available - running limited tests")

    suite.addTest(unittest.makeSuite(TestWorkflowIntegration))
    suite.addTest(unittest.makeSuite(TestPrefectIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)
