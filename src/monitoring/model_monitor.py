#!/usr/bin/env python3
"""
Basic model monitoring script for MODIS land cover classification.
Monitors model performance and data drift.
"""

import argparse
import json
import logging
import os
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from scipy import stats
from sklearn.metrics import accuracy_score, classification_report, f1_score

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelMonitor:
    """Monitor model performance and data drift."""

    def __init__(self, model_path: str, reference_data_path: Optional[str] = None):
        self.model_path = model_path
        self.reference_data_path = reference_data_path
        self.model_data = None
        self.reference_stats = None
        self.alerts = []

        self.load_model()
        if reference_data_path and os.path.exists(reference_data_path):
            self.load_reference_data()

    def load_model(self):
        """Load the trained model."""
        try:
            with open(self.model_path, "rb") as f:
                self.model_data = pickle.load(f)
            logger.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def load_reference_data(self):
        """Load reference data for drift detection."""
        try:
            with open(self.reference_data_path, "rb") as f:
                reference_data = pickle.load(f)

            # Calculate statistics for drift detection
            self.reference_stats = {}
            if "X" in reference_data:
                X_ref = reference_data["X"]
                self.reference_stats = {
                    "means": np.mean(X_ref, axis=0),
                    "stds": np.std(X_ref, axis=0),
                    "mins": np.min(X_ref, axis=0),
                    "maxs": np.max(X_ref, axis=0),
                }

            logger.info(f"Reference data loaded from {self.reference_data_path}")
        except Exception as e:
            logger.warning(f"Could not load reference data: {e}")

    def calculate_performance_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate model performance metrics."""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
            "timestamp": datetime.now().isoformat(),
        }

        # Per-class metrics
        report = classification_report(y_true, y_pred, output_dict=True)
        metrics["per_class"] = report

        return metrics

    def detect_data_drift(self, X_new: np.ndarray, threshold: float = 0.05) -> Dict:
        """Detect data drift using statistical tests."""
        if self.reference_stats is None:
            logger.warning("No reference data available for drift detection")
            return {"drift_detected": False, "reason": "No reference data"}

        drift_results = {
            "drift_detected": False,
            "features_with_drift": [],
            "drift_scores": {},
            "timestamp": datetime.now().isoformat(),
        }

        feature_names = self.model_data.get(
            "feature_names", [f"feature_{i}" for i in range(X_new.shape[1])]
        )

        for i, feature_name in enumerate(feature_names):
            if i >= X_new.shape[1]:
                break

            # Statistical tests for drift
            new_feature = X_new[:, i]
            ref_mean = self.reference_stats["means"][i]
            ref_std = self.reference_stats["stds"][i]

            # Z-test for mean shift
            new_mean = np.mean(new_feature)
            new_std = np.std(new_feature)

            # Calculate z-score for mean difference
            if ref_std > 0:
                z_score = abs(new_mean - ref_mean) / (ref_std / np.sqrt(len(new_feature)))
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

                drift_results["drift_scores"][feature_name] = {
                    "z_score": float(z_score),
                    "p_value": float(p_value),
                    "new_mean": float(new_mean),
                    "ref_mean": float(ref_mean),
                    "drift_detected": p_value < threshold,
                }

                if p_value < threshold:
                    drift_results["features_with_drift"].append(feature_name)
                    drift_results["drift_detected"] = True

        return drift_results

    def monitor_model_server(self, server_url: str) -> Dict:
        """Monitor the model server health and performance."""
        try:
            # Health check
            health_response = requests.get(f"{server_url}/health", timeout=10)
            health_status = health_response.json()

            # Model info
            info_response = requests.get(f"{server_url}/model_info", timeout=10)
            model_info = info_response.json()

            monitoring_result = {
                "server_healthy": health_status.get("status") == "healthy",
                "model_loaded": health_status.get("model_loaded", False),
                "model_info": model_info,
                "response_time": health_response.elapsed.total_seconds(),
                "timestamp": datetime.now().isoformat(),
            }

            # Test prediction endpoint
            test_features = np.random.rand(1, len(model_info["feature_names"])).tolist()
            pred_response = requests.post(
                f"{server_url}/predict", json={"features": test_features}, timeout=30
            )

            if pred_response.status_code == 200:
                monitoring_result["prediction_endpoint_working"] = True
                monitoring_result[
                    "prediction_response_time"
                ] = pred_response.elapsed.total_seconds()
            else:
                monitoring_result["prediction_endpoint_working"] = False
                monitoring_result["prediction_error"] = pred_response.text

            return monitoring_result

        except Exception as e:
            logger.error(f"Error monitoring server: {e}")
            return {
                "server_healthy": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def generate_alert(self, alert_type: str, message: str, severity: str = "warning"):
        """Generate monitoring alert."""
        alert = {
            "type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": datetime.now().isoformat(),
        }

        self.alerts.append(alert)
        logger.warning(f"ALERT [{severity.upper()}] {alert_type}: {message}")

        return alert

    def save_monitoring_report(self, report: Dict, output_dir: str = "monitoring_reports"):
        """Save monitoring report to file."""
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(output_dir, f"monitoring_report_{timestamp}.json")

        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Monitoring report saved to {report_file}")
        return report_file


def main():
    """Main monitoring function."""
    parser = argparse.ArgumentParser(description="Monitor MODIS land cover classification model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="random_forest_full_dataset_model.pkl",
        help="Path to the trained model",
    )
    parser.add_argument(
        "--reference_data", type=str, help="Path to reference data for drift detection"
    )
    parser.add_argument(
        "--server_url",
        type=str,
        default="http://localhost:5001",
        help="URL of the model server to monitor",
    )
    parser.add_argument(
        "--monitor_interval", type=int, default=300, help="Monitoring interval in seconds"
    )
    parser.add_argument(
        "--run_once", action="store_true", help="Run monitoring once instead of continuously"
    )

    args = parser.parse_args()

    # Initialize monitor
    monitor = ModelMonitor(args.model_path, args.reference_data)

    def run_monitoring_cycle():
        """Run one cycle of monitoring."""
        logger.info("Starting monitoring cycle...")

        monitoring_report = {
            "timestamp": datetime.now().isoformat(),
            "model_path": args.model_path,
            "alerts": [],
        }

        # Monitor server if URL provided
        if args.server_url:
            logger.info(f"Monitoring server at {args.server_url}")
            server_status = monitor.monitor_model_server(args.server_url)
            monitoring_report["server_status"] = server_status

            # Generate alerts based on server status
            if not server_status.get("server_healthy", False):
                alert = monitor.generate_alert(
                    "server_health", "Model server is not healthy", "critical"
                )
                monitoring_report["alerts"].append(alert)

            if server_status.get("response_time", 0) > 5.0:
                alert = monitor.generate_alert(
                    "performance",
                    f"Slow server response time: {server_status['response_time']:.2f}s",
                    "warning",
                )
                monitoring_report["alerts"].append(alert)

        # Add general model info
        monitoring_report["model_info"] = {
            "algorithm": monitor.model_data.get("algorithm", "unknown"),
            "n_features": len(monitor.model_data.get("feature_names", [])),
            "feature_names": monitor.model_data.get("feature_names", []),
        }

        # Save report
        report_file = monitor.save_monitoring_report(monitoring_report)

        logger.info(f"Monitoring cycle completed. Report: {report_file}")
        return monitoring_report

    if args.run_once:
        # Run once
        run_monitoring_cycle()
    else:
        # Run continuously
        logger.info(f"Starting continuous monitoring (interval: {args.monitor_interval}s)")

        while True:
            try:
                run_monitoring_cycle()
                time.sleep(args.monitor_interval)
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring cycle: {e}")
                time.sleep(60)  # Wait 1 minute before retrying


if __name__ == "__main__":
    main()
