#!/usr/bin/env python3
"""
Quick runner for MODIS Land Cover Classification Pipeline.
Executes the complete ML pipeline without requiring Prefect UI setup.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Try to import Prefect components
try:
    from flows.ml_pipeline import ml_pipeline

    PREFECT_AVAILABLE = True
    print("✅ Prefect available - running orchestrated pipeline")
except ImportError:
    PREFECT_AVAILABLE = False
    print("⚠️  Prefect not available - falling back to direct training")

# Fallback import
from train_random_forest import main as train_main


def run_orchestrated_pipeline():
    """Run the complete Prefect-orchestrated pipeline."""
    print("🚀 Starting MODIS Land Cover Classification Pipeline with Prefect...")
    print("-" * 70)

    try:
        # Run with development parameters (faster for testing)
        result = ml_pipeline(
            start_year=2020,
            end_year=2022,
            model_type="random_forest",
            sample_ratio=0.005,  # Small sample for quick testing
            model_params={
                "n_estimators": 50,  # Fewer trees for speed
                "max_depth": 10,
                "min_samples_split": 5,
            },
            slack_webhook_url=os.getenv("SLACK_WEBHOOK_URL"),
        )

        print("\n" + "=" * 70)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)

        if result.get("pipeline_success"):
            training_result = result.get("training_result", {})
            evaluation_result = result.get("evaluation_result", {})
            registration_result = result.get("registration_result", {})

            print(f"📊 MODEL PERFORMANCE:")
            print(
                f"   • Accuracy: {evaluation_result.get('performance_metrics', {}).get('accuracy', 'N/A'):.4f}"
            )
            print(
                f"   • F1 Score: {evaluation_result.get('performance_metrics', {}).get('f1_macro', 'N/A'):.4f}"
            )
            print(f"   • Training Time: {training_result.get('training_duration', 'N/A'):.2f}s")
            print(f"   • Samples: {training_result.get('n_samples_trained', 'N/A'):,}")

            print(f"\n📁 FILES GENERATED:")
            print(f"   • Model: {training_result.get('model_path', 'N/A')}")
            print(
                f"   • Registry: {'model_registry.json' if registration_result.get('registered') else 'Not registered'}"
            )

            print(f"\n🕒 PIPELINE DURATION: {result.get('pipeline_duration', 'N/A'):.2f} seconds")

            if registration_result.get("registered"):
                print(f"✅ Model registered with ID: {registration_result.get('model_id')}")
            else:
                print(f"⚠️  Model not registered: {registration_result.get('reason', 'Unknown')}")

        print("-" * 70)
        return result

    except Exception as e:
        print(f"\n❌ PIPELINE FAILED: {str(e)}")
        print(f"💡 Try running with: python train_random_forest.py --use_all_years")
        raise


def run_fallback_training():
    """Run fallback training without orchestration."""
    print("🔄 Running direct model training (no orchestration)...")
    print("-" * 70)

    try:
        # Simulate command line arguments for training
        original_argv = sys.argv
        sys.argv = [
            "train_random_forest.py",
            "--use_all_years",
            "--sample_ratio",
            "0.005",
            "--n_estimators",
            "50",
            "--max_depth",
            "10",
        ]

        # Run training
        train_main()

        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("📁 Check for generated model file: random_forest_full_dataset_model.pkl")
        print("💡 For orchestration features, install Prefect: pip install prefect>=2.10.0")
        print("-" * 70)

        # Restore original arguments
        sys.argv = original_argv

    except Exception as e:
        print(f"\n❌ TRAINING FAILED: {str(e)}")
        raise


def main():
    """Main execution function."""
    print("🌍 NASA MODIS Land Cover Classification Pipeline")
    print("🛰️  Processing satellite data for land cover classification")
    print("=" * 70)

    # Check if data directory exists
    data_dir = Path("data/processed")
    if not data_dir.exists():
        print("❌ Error: Data directory 'data/processed' not found!")
        print("💡 Please ensure MODIS data files are available in data/processed/")
        return

    # Count available data files
    tiff_files = list(data_dir.glob("*.tif"))
    print(f"📊 Found {len(tiff_files)} MODIS data files")

    if len(tiff_files) == 0:
        print("❌ Error: No MODIS data files found!")
        print("💡 Please add .tif files to data/processed/ directory")
        return

    print(f"📅 Data available for years: {sorted([int(f.stem) for f in tiff_files])}")
    print()

    # Run appropriate pipeline
    if PREFECT_AVAILABLE:
        run_orchestrated_pipeline()
    else:
        run_fallback_training()


def show_help():
    """Show help information."""
    print(
        """
🚀 MODIS Land Cover Classification Pipeline Runner

USAGE:
    python run_pipeline.py              # Run with default parameters
    python run_pipeline.py --help       # Show this help

REQUIREMENTS:
    • Python 3.10+
    • Conda environment: nasa-landcover-mlops
    • MODIS data files in data/processed/
    • At least 4GB RAM

ORCHESTRATION FEATURES:
    With Prefect installed:
    • Task retries and error handling
    • Progress monitoring and logging
    • Model registry management
    • Slack notifications (if configured)
    • Pipeline artifacts and reports

    Without Prefect:
    • Direct model training
    • Basic logging
    • Model file generation

NEXT STEPS:
    1. Check generated model files
    2. View results in MLflow (if available)
    3. Deploy model using: python src/deployment/model_server.py
    4. Monitor with: python src/monitoring/model_monitor.py

For full orchestration setup, see: ORCHESTRATION_GUIDE.md
"""
    )


if __name__ == "__main__":
    # Check for help flag
    if len(sys.argv) > 1 and sys.argv[1] in ["--help", "-h", "help"]:
        show_help()
        sys.exit(0)

    # Run the pipeline
    main()
