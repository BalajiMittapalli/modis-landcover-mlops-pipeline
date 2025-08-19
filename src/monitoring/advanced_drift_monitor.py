#!/usr/bin/env python3
"""
Advanced Model Drift Monitoring System for MODIS Land Cover Classification.
Implements comprehensive drift detection including data drift, concept drift, and performance monitoring.
"""

import json
import logging
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import seaborn as sns
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AdvancedDriftMonitor:
    """
    Advanced drift monitoring system with multiple detection methods.

    Features:
    - Statistical drift detection (KS test, Chi-square, Jensen-Shannon divergence)
    - Concept drift detection using performance degradation
    - Population stability index (PSI)
    - Data quality monitoring
    - Automated alerting and visualization
    """

    def __init__(
        self,
        model_path: str,
        reference_data_path: Optional[str] = None,
        alert_thresholds: Optional[Dict] = None,
        monitoring_config: Optional[Dict] = None,
    ):
        self.model_path = model_path
        self.reference_data_path = reference_data_path
        self.model_data = None
        self.reference_data = None
        self.reference_stats = None
        self.monitoring_history = []
        self.alerts = []

        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            "psi_threshold": 0.2,  # Population Stability Index
            "ks_threshold": 0.05,  # Kolmogorov-Smirnov test
            "performance_degradation": 0.1,  # 10% performance drop
            "data_quality_threshold": 0.95,  # 95% data quality
            "drift_feature_ratio": 0.3,  # 30% of features showing drift
        }

        # Monitoring configuration
        self.config = monitoring_config or {
            "monitoring_window_hours": 24,
            "reference_window_days": 30,
            "min_samples_for_detection": 100,
            "enable_visualizations": True,
            "save_reports": True,
        }

        self.load_model()
        if reference_data_path and os.path.exists(reference_data_path):
            self.load_reference_data()

    def load_model(self):
        """Load the trained model and metadata."""
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
                self.reference_data = pickle.load(f)

            # Calculate comprehensive reference statistics
            if "X" in self.reference_data:
                X_ref = self.reference_data["X"]
                self.reference_stats = self._calculate_comprehensive_stats(X_ref)

            logger.info(f"Reference data loaded from {self.reference_data_path}")
        except Exception as e:
            logger.warning(f"Could not load reference data: {e}")

    def _calculate_comprehensive_stats(self, X: np.ndarray) -> Dict:
        """Calculate comprehensive statistics for reference data."""
        stats_dict = {
            "means": np.mean(X, axis=0),
            "stds": np.std(X, axis=0),
            "mins": np.min(X, axis=0),
            "maxs": np.max(X, axis=0),
            "medians": np.median(X, axis=0),
            "q25": np.percentile(X, 25, axis=0),
            "q75": np.percentile(X, 75, axis=0),
            "skewness": stats.skew(X, axis=0),
            "kurtosis": stats.kurtosis(X, axis=0),
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "timestamp": datetime.now().isoformat(),
        }

        # Calculate histograms for PSI calculation
        stats_dict["histograms"] = {}
        for i in range(X.shape[1]):
            hist, bin_edges = np.histogram(X[:, i], bins=20, density=True)
            stats_dict["histograms"][i] = {"hist": hist, "bin_edges": bin_edges}

        return stats_dict

    def detect_data_drift_comprehensive(
        self, X_new: np.ndarray, feature_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Comprehensive data drift detection using multiple statistical tests.
        """
        if self.reference_stats is None:
            logger.warning("No reference data available for drift detection")
            return {"drift_detected": False, "reason": "No reference data"}

        if X_new.shape[0] < self.config["min_samples_for_detection"]:
            logger.warning(f"Insufficient samples for drift detection: {X_new.shape[0]}")
            return {"drift_detected": False, "reason": "Insufficient samples"}

        feature_names = feature_names or [f"feature_{i}" for i in range(X_new.shape[1])]

        drift_results = {
            "drift_detected": False,
            "features_with_drift": [],
            "drift_methods": {
                "kolmogorov_smirnov": {},
                "population_stability_index": {},
                "jensen_shannon_divergence": {},
                "statistical_tests": {},
            },
            "overall_drift_score": 0.0,
            "data_quality_score": 0.0,
            "timestamp": datetime.now().isoformat(),
        }

        # Calculate new data statistics
        new_stats = self._calculate_comprehensive_stats(X_new)

        # 1. Kolmogorov-Smirnov Test
        ks_results = self._ks_drift_detection(X_new, feature_names)
        drift_results["drift_methods"]["kolmogorov_smirnov"] = ks_results

        # 2. Population Stability Index
        psi_results = self._psi_drift_detection(X_new, feature_names)
        drift_results["drift_methods"]["population_stability_index"] = psi_results

        # 3. Jensen-Shannon Divergence
        js_results = self._js_drift_detection(X_new, feature_names)
        drift_results["drift_methods"]["jensen_shannon_divergence"] = js_results

        # 4. Statistical Tests (means, variances)
        stat_results = self._statistical_drift_detection(X_new, feature_names)
        drift_results["drift_methods"]["statistical_tests"] = stat_results

        # 5. Data Quality Assessment
        drift_results["data_quality_score"] = self._assess_data_quality(X_new)

        # Aggregate results
        drift_results = self._aggregate_drift_results(drift_results)

        # Generate alerts if necessary
        self._generate_drift_alerts(drift_results)

        return drift_results

    def _ks_drift_detection(self, X_new: np.ndarray, feature_names: List[str]) -> Dict:
        """Kolmogorov-Smirnov test for drift detection."""
        ks_results = {}
        X_ref = self.reference_data["X"]

        for i, feature_name in enumerate(feature_names):
            if i >= X_new.shape[1]:
                break

            ks_stat, p_value = stats.ks_2samp(X_ref[:, i], X_new[:, i])

            ks_results[feature_name] = {
                "ks_statistic": float(ks_stat),
                "p_value": float(p_value),
                "drift_detected": p_value < self.alert_thresholds["ks_threshold"],
                "severity": "high" if p_value < 0.01 else "medium" if p_value < 0.05 else "low",
            }

        return ks_results

    def _psi_drift_detection(self, X_new: np.ndarray, feature_names: List[str]) -> Dict:
        """Population Stability Index for drift detection."""
        psi_results = {}

        for i, feature_name in enumerate(feature_names):
            if i >= X_new.shape[1]:
                break

            # Get reference histogram
            ref_hist_data = self.reference_stats["histograms"][i]
            ref_hist = ref_hist_data["hist"]
            bin_edges = ref_hist_data["bin_edges"]

            # Calculate histogram for new data using same bins
            new_hist, _ = np.histogram(X_new[:, i], bins=bin_edges, density=True)

            # Normalize to avoid division by zero
            ref_hist = ref_hist + 1e-10
            new_hist = new_hist + 1e-10

            # Calculate PSI
            psi = np.sum((new_hist - ref_hist) * np.log(new_hist / ref_hist))

            psi_results[feature_name] = {
                "psi_score": float(psi),
                "drift_detected": psi > self.alert_thresholds["psi_threshold"],
                "severity": ("high" if psi > 0.25 else "medium" if psi > 0.1 else "low"),
            }

        return psi_results

    def _js_drift_detection(self, X_new: np.ndarray, feature_names: List[str]) -> Dict:
        """Jensen-Shannon divergence for drift detection."""
        js_results = {}
        X_ref = self.reference_data["X"]

        for i, feature_name in enumerate(feature_names):
            if i >= X_new.shape[1]:
                break

            # Calculate histograms
            ref_hist, bin_edges = np.histogram(X_ref[:, i], bins=20, density=True)
            new_hist, _ = np.histogram(X_new[:, i], bins=bin_edges, density=True)

            # Normalize
            ref_hist = ref_hist / np.sum(ref_hist) + 1e-10
            new_hist = new_hist / np.sum(new_hist) + 1e-10

            # Calculate JS divergence
            m = 0.5 * (ref_hist + new_hist)
            js_div = 0.5 * stats.entropy(ref_hist, m) + 0.5 * stats.entropy(new_hist, m)

            js_results[feature_name] = {
                "js_divergence": float(js_div),
                "drift_detected": js_div > 0.1,  # Threshold for JS divergence
                "severity": "high" if js_div > 0.3 else "medium" if js_div > 0.1 else "low",
            }

        return js_results

    def _statistical_drift_detection(self, X_new: np.ndarray, feature_names: List[str]) -> Dict:
        """Statistical tests for mean and variance shifts."""
        stat_results = {}

        for i, feature_name in enumerate(feature_names):
            if i >= X_new.shape[1]:
                break

            # T-test for mean difference
            ref_feature = self.reference_data["X"][:, i]
            new_feature = X_new[:, i]

            t_stat, t_p_value = stats.ttest_ind(ref_feature, new_feature)

            # F-test for variance difference
            f_stat = np.var(new_feature) / np.var(ref_feature)
            f_p_value = 2 * min(
                stats.f.cdf(f_stat, len(new_feature) - 1, len(ref_feature) - 1),
                1 - stats.f.cdf(f_stat, len(new_feature) - 1, len(ref_feature) - 1),
            )

            stat_results[feature_name] = {
                "t_statistic": float(t_stat),
                "t_p_value": float(t_p_value),
                "f_statistic": float(f_stat),
                "f_p_value": float(f_p_value),
                "mean_drift": float(t_p_value) < 0.05,
                "variance_drift": float(f_p_value) < 0.05,
                "drift_detected": float(t_p_value) < 0.05 or float(f_p_value) < 0.05,
            }

        return stat_results

    def _assess_data_quality(self, X_new: np.ndarray) -> float:
        """Assess data quality of new data."""
        quality_issues = 0
        total_checks = 0

        # Check for missing values
        total_checks += 1
        if np.isnan(X_new).any():
            quality_issues += 1

        # Check for infinite values
        total_checks += 1
        if np.isinf(X_new).any():
            quality_issues += 1

        # Check for extreme outliers (beyond 5 standard deviations)
        total_checks += 1
        z_scores = np.abs(stats.zscore(X_new, axis=0, nan_policy="omit"))
        if np.any(z_scores > 5):
            quality_issues += 1

        # Check for constant features
        total_checks += 1
        feature_stds = np.std(X_new, axis=0)
        if np.any(feature_stds == 0):
            quality_issues += 1

        return (total_checks - quality_issues) / total_checks

    def _aggregate_drift_results(self, drift_results: Dict) -> Dict:
        """Aggregate drift detection results from multiple methods."""
        drift_features = set()
        drift_scores = []

        # Collect all features with drift
        for method_results in drift_results["drift_methods"].values():
            for feature, result in method_results.items():
                if isinstance(result, dict) and result.get("drift_detected", False):
                    drift_features.add(feature)
                    # Collect numeric drift scores
                    for key, value in result.items():
                        if key in ["psi_score", "js_divergence", "ks_statistic"]:
                            drift_scores.append(value)

        # Calculate overall drift metrics
        drift_results["features_with_drift"] = list(drift_features)
        drift_results["drift_detected"] = len(drift_features) > 0
        drift_results["drift_feature_ratio"] = len(drift_features) / len(
            drift_results["drift_methods"]["kolmogorov_smirnov"]
        )
        drift_results["overall_drift_score"] = np.mean(drift_scores) if drift_scores else 0.0

        # Determine severity
        if drift_results["drift_feature_ratio"] > self.alert_thresholds["drift_feature_ratio"]:
            drift_results["severity"] = "high"
        elif drift_results["drift_feature_ratio"] > 0.1:
            drift_results["severity"] = "medium"
        else:
            drift_results["severity"] = "low"

        return drift_results

    def detect_concept_drift(
        self, X_new: np.ndarray, y_true: np.ndarray, model_url: Optional[str] = None
    ) -> Dict:
        """
        Detect concept drift by monitoring model performance degradation.
        """
        concept_drift_results = {
            "concept_drift_detected": False,
            "performance_metrics": {},
            "performance_degradation": 0.0,
            "timestamp": datetime.now().isoformat(),
        }

        try:
            # Get predictions from model
            if model_url:
                # Get predictions from deployed model
                predictions = self._get_model_predictions(X_new, model_url)
            else:
                # Use local model if available
                model = self.model_data.get("model")
                if model:
                    predictions = model.predict(X_new)
                else:
                    logger.warning("No model available for concept drift detection")
                    return concept_drift_results

            # Calculate current performance
            current_accuracy = accuracy_score(y_true, predictions)
            current_f1 = f1_score(y_true, predictions, average="macro")

            concept_drift_results["performance_metrics"] = {
                "accuracy": float(current_accuracy),
                "f1_macro": float(current_f1),
                "classification_report": classification_report(
                    y_true, predictions, output_dict=True
                ),
            }

            # Compare with reference performance if available
            if "reference_performance" in self.model_data:
                ref_accuracy = self.model_data["reference_performance"]["accuracy"]
                performance_degradation = (ref_accuracy - current_accuracy) / ref_accuracy

                concept_drift_results["performance_degradation"] = float(performance_degradation)
                concept_drift_results["concept_drift_detected"] = (
                    performance_degradation > self.alert_thresholds["performance_degradation"]
                )

                # Generate alert if significant degradation
                if concept_drift_results["concept_drift_detected"]:
                    self._generate_alert(
                        "concept_drift",
                        f"Performance degradation detected: {performance_degradation:.2%}",
                        "high",
                    )

        except Exception as e:
            logger.error(f"Error in concept drift detection: {e}")
            concept_drift_results["error"] = str(e)

        return concept_drift_results

    def _get_model_predictions(self, X: np.ndarray, model_url: str) -> np.ndarray:
        """Get predictions from deployed model."""
        try:
            response = requests.post(
                f"{model_url}/predict",
                json={"features": X.tolist()},
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()
            return np.array(result.get("predictions", []))
        except Exception as e:
            logger.error(f"Error getting model predictions: {e}")
            raise

    def _generate_drift_alerts(self, drift_results: Dict):
        """Generate alerts based on drift detection results."""
        if drift_results["drift_detected"]:
            severity = drift_results.get("severity", "medium")
            drift_ratio = drift_results.get("drift_feature_ratio", 0)

            message = (
                f"Data drift detected in {len(drift_results['features_with_drift'])} features "
                f"({drift_ratio:.1%} of total features)"
            )

            self._generate_alert("data_drift", message, severity)

        if drift_results["data_quality_score"] < self.alert_thresholds["data_quality_threshold"]:
            self._generate_alert(
                "data_quality",
                f"Data quality degraded: {drift_results['data_quality_score']:.2%}",
                "medium",
            )

    def _generate_alert(self, alert_type: str, message: str, severity: str = "warning"):
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

    def create_drift_visualization(
        self, drift_results: Dict, output_dir: str = "monitoring_reports"
    ):
        """Create comprehensive drift visualization dashboard."""
        if not self.config.get("enable_visualizations", True):
            return None

        os.makedirs(output_dir, exist_ok=True)

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Drift Detection Summary",
                "Feature Drift Scores",
                "Data Quality Metrics",
                "Drift Timeline",
            ),
            specs=[
                [{"type": "indicator"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "scatter"}],
            ],
        )

        # 1. Overall drift indicator
        drift_score = drift_results.get("overall_drift_score", 0)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=drift_score,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Overall Drift Score"},
                gauge={
                    "axis": {"range": [None, 1]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 0.1], "color": "lightgray"},
                        {"range": [0.1, 0.3], "color": "yellow"},
                        {"range": [0.3, 1], "color": "red"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 0.3,
                    },
                },
            ),
            row=1,
            col=1,
        )

        # 2. Feature drift scores
        if "kolmogorov_smirnov" in drift_results["drift_methods"]:
            ks_results = drift_results["drift_methods"]["kolmogorov_smirnov"]
            features = list(ks_results.keys())
            ks_scores = [ks_results[f]["ks_statistic"] for f in features]

            fig.add_trace(
                go.Bar(x=features[:10], y=ks_scores[:10], name="KS Statistic"), row=1, col=2
            )

        # 3. Data quality score
        quality_score = drift_results.get("data_quality_score", 1.0)
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[quality_score, quality_score],
                mode="lines+markers",
                name="Data Quality",
                line=dict(color="green" if quality_score > 0.95 else "red"),
            ),
            row=2,
            col=1,
        )

        # 4. Historical drift timeline (if history exists)
        if self.monitoring_history:
            timestamps = [h["timestamp"] for h in self.monitoring_history[-20:]]
            drift_scores_hist = [
                h.get("overall_drift_score", 0) for h in self.monitoring_history[-20:]
            ]

            fig.add_trace(
                go.Scatter(
                    x=timestamps, y=drift_scores_hist, mode="lines+markers", name="Drift History"
                ),
                row=2,
                col=2,
            )

        # Update layout
        fig.update_layout(
            title=f"Model Drift Monitoring Dashboard - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            showlegend=True,
            height=800,
        )

        # Save visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_file = os.path.join(output_dir, f"drift_dashboard_{timestamp}.html")
        fig.write_html(viz_file)

        logger.info(f"Drift visualization saved to {viz_file}")
        return viz_file

    def save_monitoring_report(
        self, drift_results: Dict, output_dir: str = "monitoring_reports"
    ) -> str:
        """Save comprehensive monitoring report."""
        if not self.config.get("save_reports", True):
            return None

        os.makedirs(output_dir, exist_ok=True)

        # Create comprehensive report
        report = {
            "monitoring_metadata": {
                "timestamp": datetime.now().isoformat(),
                "model_path": self.model_path,
                "monitoring_config": self.config,
                "alert_thresholds": self.alert_thresholds,
            },
            "drift_detection_results": drift_results,
            "alerts": self.alerts,
            "recommendations": self._generate_recommendations(drift_results),
        }

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(output_dir, f"comprehensive_monitoring_report_{timestamp}.json")

        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Add to monitoring history
        self.monitoring_history.append(drift_results)

        logger.info(f"Comprehensive monitoring report saved to {report_file}")
        return report_file

    def _generate_recommendations(self, drift_results: Dict) -> List[str]:
        """Generate actionable recommendations based on monitoring results."""
        recommendations = []

        if drift_results.get("drift_detected", False):
            recommendations.append(
                "Data drift detected. Consider retraining the model with recent data."
            )

            if drift_results.get("drift_feature_ratio", 0) > 0.5:
                recommendations.append(
                    "Significant feature drift detected. Review data pipeline and feature engineering."
                )

            drift_features = drift_results.get("features_with_drift", [])
            if drift_features:
                recommendations.append(
                    f"Focus on features with drift: {', '.join(drift_features[:5])}"
                )

        if drift_results.get("data_quality_score", 1.0) < 0.9:
            recommendations.append(
                "Data quality issues detected. Review data preprocessing and validation."
            )

        if drift_results.get("concept_drift_detected", False):
            recommendations.append(
                "Concept drift detected. Model performance has degraded significantly."
            )

        if not recommendations:
            recommendations.append("No significant drift detected. Continue monitoring.")

        return recommendations


def main():
    """Main monitoring function with enhanced drift detection."""
    import argparse

    parser = argparse.ArgumentParser(description="Advanced Model Drift Monitoring")
    parser.add_argument("--model_path", required=True, help="Path to the trained model")
    parser.add_argument("--reference_data", help="Path to reference data for drift detection")
    parser.add_argument("--new_data", help="Path to new data for monitoring")
    parser.add_argument("--server_url", help="URL of the model server")
    parser.add_argument(
        "--output_dir", default="monitoring_reports", help="Output directory for reports"
    )
    parser.add_argument("--config", help="Path to monitoring configuration file")

    args = parser.parse_args()

    # Load configuration if provided
    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = json.load(f)

    # Initialize advanced monitor
    monitor = AdvancedDriftMonitor(
        model_path=args.model_path,
        reference_data_path=args.reference_data,
        monitoring_config=config,
    )

    # Load new data for monitoring
    if args.new_data and os.path.exists(args.new_data):
        with open(args.new_data, "rb") as f:
            new_data = pickle.load(f)

        X_new = new_data.get("X")
        y_new = new_data.get("y")

        if X_new is not None:
            logger.info("Starting comprehensive drift detection...")

            # Data drift detection
            drift_results = monitor.detect_data_drift_comprehensive(X_new)

            # Concept drift detection if labels available
            if y_new is not None:
                concept_results = monitor.detect_concept_drift(X_new, y_new, args.server_url)
                drift_results["concept_drift"] = concept_results

            # Create visualizations
            monitor.create_drift_visualization(drift_results, args.output_dir)

            # Save comprehensive report
            monitor.save_monitoring_report(drift_results, args.output_dir)

            # Print summary
            print("\n" + "=" * 50)
            print("DRIFT MONITORING SUMMARY")
            print("=" * 50)
            print(f"Data Drift Detected: {drift_results.get('drift_detected', False)}")
            print(f"Features with Drift: {len(drift_results.get('features_with_drift', []))}")
            print(f"Overall Drift Score: {drift_results.get('overall_drift_score', 0):.3f}")
            print(f"Data Quality Score: {drift_results.get('data_quality_score', 1.0):.3f}")

            if drift_results.get("concept_drift"):
                concept_drift = drift_results["concept_drift"]
                print(
                    f"Concept Drift Detected: {concept_drift.get('concept_drift_detected', False)}"
                )
                print(
                    f"Performance Degradation: {concept_drift.get('performance_degradation', 0):.2%}"
                )

            print(f"Active Alerts: {len(monitor.alerts)}")
            print("=" * 50)
        else:
            logger.error("No data found in the provided file")
    else:
        logger.info("No new data provided. Running server monitoring only...")

        if args.server_url:
            # Basic server monitoring
            from model_monitor import ModelMonitor

            basic_monitor = ModelMonitor(args.model_path, args.reference_data)
            server_result = basic_monitor.monitor_model_server(args.server_url)
            logger.info(f"Server monitoring result: {server_result}")


if __name__ == "__main__":
    main()
