#!/usr/bin/env python3
"""
ML Pipeline for MODIS Land Cover Classification using Prefect 3.x

This orchestration pipeline manages the complete ML workflow:
- Data ingestion from MODIS sources
- Data preprocessing and feature extraction
- Model training with different algorithms
- Model evaluation and validation
- Model registration and deployment

Author: NASA Land Cover Classification Team
Date: January 2025
"""

import json
import logging
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Prefect 3.x imports
try:
    from prefect import flow, get_run_logger, task
    from prefect.artifacts import create_markdown_artifact
    from prefect.task_runners import ConcurrentTaskRunner

    PREFECT_AVAILABLE = True
except ImportError:
    # Fallback for environments without Prefect
    PREFECT_AVAILABLE = False

    def flow(**kwargs):
        def decorator(func):
            return func

        return decorator

    def task(**kwargs):
        def decorator(func):
            return func

        return decorator

    def get_run_logger():
        return logging.getLogger(__name__)


# Import our ML components
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root / "src" / "models"))

from train_random_forest import LandCoverClassifier, MODISFeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@task(retries=3, retry_delay_seconds=60, timeout_seconds=1800)  # 30 minutes timeout
def ingest_data(start_year: int, end_year: int, data_dir: str = "data/processed") -> Dict:
    """
    Ingest MODIS data for specified year range.

    Args:
        start_year: Starting year for data ingestion
        end_year: Ending year for data ingestion
        data_dir: Directory containing MODIS data files

    Returns:
        Dictionary with ingestion results
    """
    task_logger = get_run_logger()
    task_logger.info(f"Starting data ingestion for years {start_year}-{end_year}")

    data_path = Path(data_dir)
    available_files = []
    missing_files = []

    for year in range(start_year, end_year + 1):
        tiff_file = data_path / f"{year}.tif"
        if tiff_file.exists():
            available_files.append(year)
            task_logger.info(f"✅ Found data for year {year}")
        else:
            missing_files.append(year)
            task_logger.warning(f"❌ Missing data for year {year}")

    ingestion_result = {
        "available_years": available_files,
        "missing_years": missing_files,
        "total_files": len(available_files),
        "data_directory": str(data_path),
        "ingestion_timestamp": datetime.now().isoformat(),
    }

    if not available_files:
        raise ValueError(f"No data files found for years {start_year}-{end_year}")

    task_logger.info(f"Data ingestion completed: {len(available_files)} files available")
    return ingestion_result


@task(retries=3, retry_delay_seconds=60, timeout_seconds=1800)
def preprocess_data(
    ingestion_result: Dict,
    sample_ratio: float = 0.01,
    feature_types: List[str] = ["spectral", "spatial", "texture"],
) -> Dict:
    """
    Preprocess MODIS data and extract features.

    Args:
        ingestion_result: Results from data ingestion task
        sample_ratio: Ratio of pixels to sample from each file
        feature_types: Types of features to extract

    Returns:
        Dictionary with preprocessing results and extracted features
    """
    task_logger = get_run_logger()
    task_logger.info("Starting data preprocessing and feature extraction")

    # Initialize feature extractor
    data_dir = Path(ingestion_result["data_directory"])
    extractor = MODISFeatureExtractor(data_dir)

    # Extract features from available years
    available_years = ingestion_result["available_years"]
    task_logger.info(f"Extracting features from {len(available_years)} years of data")

    X, y = extractor.load_data(available_years, sample_ratio)

    # Calculate data statistics
    preprocessing_result = {
        "n_samples": len(X),
        "n_features": X.shape[1],
        "feature_names": extractor.create_feature_names() + ["coord_row", "coord_col"],
        "class_distribution": {
            int(cls): int(count) for cls, count in zip(*np.unique(y, return_counts=True))
        },
        "sample_ratio": sample_ratio,
        "years_processed": available_years,
        "preprocessing_timestamp": datetime.now().isoformat(),
    }

    # Save preprocessed data temporarily
    temp_data_path = "temp_preprocessed_data.pkl"
    with open(temp_data_path, "wb") as f:
        pickle.dump({"X": X, "y": y, "metadata": preprocessing_result}, f)

    preprocessing_result["data_path"] = temp_data_path

    task_logger.info(f"Preprocessing completed: {len(X)} samples, {X.shape[1]} features")
    return preprocessing_result


@task(retries=3, retry_delay_seconds=60, timeout_seconds=1800)
def train_model(
    preprocessing_result: Dict,
    model_type: str = "random_forest",
    model_params: Optional[Dict] = None,
) -> Dict:
    """
    Train machine learning model.

    Args:
        preprocessing_result: Results from preprocessing task
        model_type: Type of model to train
        model_params: Model hyperparameters

    Returns:
        Dictionary with training results
    """
    task_logger = get_run_logger()
    task_logger.info(f"Starting model training with {model_type}")

    # Load preprocessed data
    with open(preprocessing_result["data_path"], "rb") as f:
        data = pickle.load(f)

    X, y = data["X"], data["y"]

    # Set default parameters
    if model_params is None:
        model_params = {
            "n_estimators": 200,
            "max_depth": 20,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
        }

    # Initialize classifier
    classifier = LandCoverClassifier(model_type)
    classifier.feature_names = preprocessing_result["feature_names"]

    # Train model
    start_time = datetime.now()
    metrics, report = classifier.train(X, y, **model_params)
    training_duration = (datetime.now() - start_time).total_seconds()

    # Save trained model
    model_filename = f"{model_type}_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    classifier.save_model(model_filename)

    training_result = {
        "model_type": model_type,
        "model_path": model_filename,
        "metrics": metrics,
        "model_params": model_params,
        "training_duration": training_duration,
        "n_samples_trained": len(X),
        "training_timestamp": datetime.now().isoformat(),
        "classification_report": report,
    }

    task_logger.info(f"Model training completed: {metrics['accuracy']:.4f} accuracy")
    return training_result


@task(retries=2, retry_delay_seconds=30, timeout_seconds=900)
def evaluate_model(training_result: Dict, preprocessing_result: Dict) -> Dict:
    """
    Evaluate trained model performance.

    Args:
        training_result: Results from training task
        preprocessing_result: Results from preprocessing task

    Returns:
        Dictionary with evaluation results
    """
    task_logger = get_run_logger()
    task_logger.info("Starting model evaluation")

    # Load model
    classifier = LandCoverClassifier()
    classifier.load_model(training_result["model_path"])

    # Get feature importance
    feature_importance = classifier.get_feature_importance()

    # Calculate additional metrics
    metrics = training_result["metrics"]

    evaluation_result = {
        "model_path": training_result["model_path"],
        "performance_metrics": {
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "f1_weighted": metrics["f1_weighted"],
            "cv_score": f"{metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}",
            "training_time": metrics["training_time"],
        },
        "feature_importance": feature_importance.head(10).to_dict("records"),
        "model_size_mb": os.path.getsize(training_result["model_path"]) / (1024 * 1024),
        "evaluation_timestamp": datetime.now().isoformat(),
    }

    # Determine if model meets quality thresholds
    quality_thresholds = {
        "min_accuracy": 0.35,
        "min_f1_macro": 0.25,
        "max_training_time": 3600,  # 1 hour
    }

    evaluation_result["quality_check"] = {
        "passes_accuracy_threshold": metrics["accuracy"] >= quality_thresholds["min_accuracy"],
        "passes_f1_threshold": metrics["f1_macro"] >= quality_thresholds["min_f1_macro"],
        "passes_time_threshold": metrics["training_time"]
        <= quality_thresholds["max_training_time"],
        "thresholds": quality_thresholds,
    }

    overall_quality = all(
        evaluation_result["quality_check"][key]
        for key in ["passes_accuracy_threshold", "passes_f1_threshold", "passes_time_threshold"]
    )
    evaluation_result["quality_check"]["overall_pass"] = overall_quality

    task_logger.info(
        f"Model evaluation completed. Quality check: {'✅ PASS' if overall_quality else '❌ FAIL'}"
    )
    return evaluation_result


@task(retries=2, retry_delay_seconds=30, timeout_seconds=600)
def register_model(training_result: Dict, evaluation_result: Dict) -> Dict:
    """
    Register model if it meets quality requirements.

    Args:
        training_result: Results from training task
        evaluation_result: Results from evaluation task

    Returns:
        Dictionary with registration results
    """
    task_logger = get_run_logger()
    task_logger.info("Starting model registration")

    # Check if model passes quality gates
    if not evaluation_result["quality_check"]["overall_pass"]:
        task_logger.warning("Model does not meet quality requirements. Skipping registration.")
        return {
            "registered": False,
            "reason": "Quality check failed",
            "quality_issues": [
                key
                for key, value in evaluation_result["quality_check"].items()
                if key.startswith("passes_") and not value
            ],
            "registration_timestamp": datetime.now().isoformat(),
        }

    # Create model registry entry
    model_registry = {
        "model_id": f"modis-landcover-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "model_path": training_result["model_path"],
        "model_type": training_result["model_type"],
        "version": "1.0.0",
        "accuracy": evaluation_result["performance_metrics"]["accuracy"],
        "f1_score": evaluation_result["performance_metrics"]["f1_macro"],
        "training_timestamp": training_result["training_timestamp"],
        "registration_timestamp": datetime.now().isoformat(),
        "status": "active",
        "metadata": {
            "training_samples": training_result["n_samples_trained"],
            "model_size_mb": evaluation_result["model_size_mb"],
            "feature_count": len(training_result.get("feature_names", [])),
            "model_params": training_result["model_params"],
        },
    }

    # Save to model registry (simple JSON file for now)
    registry_path = "model_registry.json"
    if os.path.exists(registry_path):
        with open(registry_path, "r") as f:
            registry = json.load(f)
    else:
        registry = {"models": []}

    registry["models"].append(model_registry)

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    # Copy model to production directory
    production_model_path = f"production_models/{model_registry['model_id']}.pkl"
    os.makedirs("production_models", exist_ok=True)

    import shutil

    shutil.copy2(training_result["model_path"], production_model_path)

    registration_result = {
        "registered": True,
        "model_id": model_registry["model_id"],
        "production_path": production_model_path,
        "registry_path": registry_path,
        "registration_timestamp": datetime.now().isoformat(),
    }

    task_logger.info(f"Model registered successfully: {model_registry['model_id']}")
    return registration_result


@task(retries=1)
def send_notification(
    flow_result: Dict, success: bool = True, slack_webhook_url: Optional[str] = None
) -> Dict:
    """
    Send notification about pipeline completion.

    Args:
        flow_result: Complete flow results
        success: Whether the pipeline succeeded
        slack_webhook_url: Slack webhook URL for notifications

    Returns:
        Dictionary with notification results
    """
    task_logger = get_run_logger()

    # Prepare notification message
    if success:
        message = f"""
🎉 *MODIS Land Cover ML Pipeline - SUCCESS*

📊 *Results Summary:*
• Model Type: {flow_result.get('training_result', {}).get('model_type', 'Unknown')}
• Accuracy: {flow_result.get('evaluation_result', {}).get('performance_metrics', {}).get('accuracy', 'N/A'):.4f}
• F1 Score: {flow_result.get('evaluation_result', {}).get('performance_metrics', {}).get('f1_macro', 'N/A'):.4f}
• Training Time: {flow_result.get('training_result', {}).get('training_duration', 'N/A'):.2f}s
• Samples: {flow_result.get('training_result', {}).get('n_samples_trained', 'N/A'):,}

✅ *Model Registration:* {'Registered' if flow_result.get('registration_result', {}).get('registered') else 'Skipped'}
🔗 *Pipeline completed successfully*
"""
    else:
        message = f"""
❌ *MODIS Land Cover ML Pipeline - FAILED*

💥 *Pipeline encountered errors during execution*

 *Please check logs for detailed error information*
"""

    notification_result = {
        "message_sent": False,
        "notification_timestamp": datetime.now().isoformat(),
        "message": message,
    }

    # Send to Slack if webhook is configured
    if slack_webhook_url:
        try:
            import requests

            payload = {
                "text": message,
                "username": "NASA MODIS Pipeline",
                "icon_emoji": ":satellite:",
                "channel": "#ml-alerts",  # You can customize this
            }

            response = requests.post(slack_webhook_url, json=payload, timeout=10)

            if response.status_code == 200:
                notification_result["message_sent"] = True
                notification_result["channel"] = "slack"
                task_logger.info("✅ Slack notification sent successfully")
            else:
                task_logger.error(
                    f"❌ Slack notification failed: {response.status_code} - {response.text}"
                )
                notification_result["error"] = f"HTTP {response.status_code}: {response.text}"

        except Exception as e:
            task_logger.error(f"❌ Failed to send Slack notification: {e}")
            notification_result["error"] = str(e)
    else:
        task_logger.info("No Slack webhook configured, logging notification only")

    # Always log the message
    task_logger.info(f"Pipeline notification: {message}")

    return notification_result


@flow(
    name="modis-landcover-ml-pipeline",
    description="Complete ML pipeline for MODIS land cover classification",
    timeout_seconds=7200,  # 2 hours total timeout
    retries=1,
    retry_delay_seconds=300,
)
def ml_pipeline(
    start_year: int = 2020,
    end_year: int = 2022,
    model_type: str = "random_forest",
    sample_ratio: float = 0.01,
    model_params: Optional[Dict] = None,
    slack_webhook_url: Optional[str] = None,
) -> Dict:
    """
    Complete ML pipeline for MODIS land cover classification.

    Args:
        start_year: Starting year for training data
        end_year: Ending year for training data
        model_type: Type of ML model to train
        sample_ratio: Ratio of pixels to sample from each file
        model_params: Model hyperparameters
        slack_webhook_url: Slack webhook URL for notifications

    Returns:
        Dictionary with complete pipeline results
    """
    flow_logger = get_run_logger()
    flow_logger.info(f"🚀 Starting MODIS Land Cover ML Pipeline")
    flow_logger.info(
        f"Parameters: years={start_year}-{end_year}, model={model_type}, sample_ratio={sample_ratio}"
    )

    pipeline_start_time = datetime.now()

    try:
        # Task 1: Data Ingestion
        flow_logger.info("📥 Step 1: Data Ingestion")
        ingestion_result = ingest_data(start_year, end_year)

        # Task 2: Data Preprocessing
        flow_logger.info("🔄 Step 2: Data Preprocessing")
        preprocessing_result = preprocess_data(ingestion_result, sample_ratio)

        # Task 3: Model Training
        flow_logger.info("🤖 Step 3: Model Training")
        training_result = train_model(preprocessing_result, model_type, model_params)

        # Task 4: Model Evaluation
        flow_logger.info("📊 Step 4: Model Evaluation")
        evaluation_result = evaluate_model(training_result, preprocessing_result)

        # Task 5: Model Registration
        flow_logger.info("📝 Step 5: Model Registration")
        registration_result = register_model(training_result, evaluation_result)

        # Compile complete results
        pipeline_duration = (datetime.now() - pipeline_start_time).total_seconds()

        complete_result = {
            "pipeline_success": True,
            "pipeline_duration": pipeline_duration,
            "ingestion_result": ingestion_result,
            "preprocessing_result": preprocessing_result,
            "training_result": training_result,
            "evaluation_result": evaluation_result,
            "registration_result": registration_result,
            "pipeline_timestamp": datetime.now().isoformat(),
        }

        # Create pipeline artifact
        markdown_report = f"""
# MODIS Land Cover ML Pipeline Report

## Pipeline Summary
- **Status**: ✅ SUCCESS
- **Duration**: {pipeline_duration:.2f} seconds
- **Model Type**: {model_type}
- **Years Processed**: {start_year}-{end_year}

## Results
- **Accuracy**: {evaluation_result['performance_metrics']['accuracy']:.4f}
- **F1 Score**: {evaluation_result['performance_metrics']['f1_macro']:.4f}
- **Training Samples**: {training_result['n_samples_trained']:,}
- **Model Registered**: {'Yes' if registration_result['registered'] else 'No'}

## Data Summary
- **Files Processed**: {len(ingestion_result['available_years'])}
- **Total Samples**: {preprocessing_result['n_samples']:,}
- **Features**: {preprocessing_result['n_features']}
"""

        # Log pipeline report
        flow_logger.info(f"Pipeline Report:\n{markdown_report}")

        # Send success notification
        flow_logger.info("📨 Step 6: Sending Success Notification")
        notification_result = send_notification(complete_result, True, slack_webhook_url)
        complete_result["notification_result"] = notification_result

        flow_logger.info(f"🎉 Pipeline completed successfully in {pipeline_duration:.2f} seconds")
        return complete_result

    except Exception as e:
        flow_logger.error(f"❌ Pipeline failed: {str(e)}")

        # Send failure notification
        failure_result = {"error": str(e), "pipeline_success": False}
        notification_result = send_notification(failure_result, False, slack_webhook_url)

        raise e


# Deployment configuration (Prefect 3.x compatible)
def create_deployment_config():
    """Create deployment configuration for Prefect 3.x."""

    config = {
        "name": "modis-landcover-ml-pipeline-prod",
        "description": "Production deployment of MODIS land cover classification ML pipeline",
        "tags": ["ml", "modis", "landcover", "production"],
        "parameters": {
            "start_year": 2020,
            "end_year": 2022,
            "model_type": "random_forest",
            "sample_ratio": 0.01,
            "slack_webhook_url": os.getenv("SLACK_WEBHOOK_URL"),
        },
        "schedule": {"cron": "0 2 * * 1", "timezone": "UTC"},  # Weekly on Monday at 2 AM UTC
    }

    print("📋 Deployment configuration for Prefect 3.x:")
    print("   To deploy, create prefect.yaml and run 'prefect deploy'")

    return config


if __name__ == "__main__":
    # For direct execution (testing without Prefect server)
    print("🚀 Running MODIS ML Pipeline directly...")

    try:
        result = ml_pipeline(
            start_year=2020,
            end_year=2022,
            model_type="random_forest",
            sample_ratio=0.005,
            slack_webhook_url=os.getenv("SLACK_WEBHOOK_URL"),
        )
        print(f"✅ Pipeline completed successfully!")
        print(f"📊 Results: {result}")
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback

        traceback.print_exc()
