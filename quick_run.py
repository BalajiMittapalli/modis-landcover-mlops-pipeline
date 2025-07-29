#!/usr/bin/env python3
"""
Quick Run Script for NASA MODIS Land Cover Classification Pipeline
This script provides the fastest way to execute the complete ML pipeline.
"""

import asyncio
import os
import subprocess
import sys
from pathlib import Path


def check_environment():
    """Check if required packages are installed."""
    try:
        import mlflow
        import numpy as np
        import pandas as pd
        import prefect
        import sklearn

        return True, "All packages available"
    except ImportError as e:
        return False, f"Missing package: {e}"


def run_prefect_pipeline():
    """Run the pipeline using Prefect orchestration."""
    print("🚀 Starting NASA MODIS Pipeline with Prefect Orchestration...")

    try:
        # Import and run the flow
        from flows.ml_pipeline import ml_pipeline

        # Run the flow
        result = asyncio.run(ml_pipeline())

        if result:
            print("✅ Pipeline completed successfully!")
            print("📊 Check MLflow UI at http://localhost:5000 for results")
            return True
        else:
            print("❌ Pipeline failed - check logs for details")
            return False

    except Exception as e:
        print(f"❌ Error running Prefect pipeline: {e}")
        return False


def run_direct_training():
    """Run training directly without Prefect (fallback)."""
    print("🔄 Running direct training (no orchestration)...")

    try:
        # Import and run training
        import sys

        sys.path.append(".")

        from train_random_forest import main

        # Run training
        print("🤖 Training model...")
        main()

        print("✅ Training completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Error in direct training: {e}")
        return False


def main():
    """Main execution function."""
    print("=" * 60)
    print("  NASA MODIS Land Cover Classification - Quick Run")
    print("=" * 60)

    # Check environment
    env_ok, env_msg = check_environment()
    if not env_ok:
        print(f"❌ Environment check failed: {env_msg}")
        print("📋 Please run: conda env create -f environment.yml")
        print("📋 Then activate: conda activate nasa-landcover-mlops")
        return 1

    print("✅ Environment check passed")

    # Try Prefect first, then fallback
    print("\n🎯 Attempting Prefect orchestration...")

    if run_prefect_pipeline():
        print("\n🎉 SUCCESS: Pipeline completed with orchestration!")
    else:
        print("\n🔄 Prefect failed, trying direct execution...")
        if run_direct_training():
            print("\n🎉 SUCCESS: Training completed directly!")
        else:
            print("\n❌ Both orchestration and direct training failed")
            print("📋 Check the logs above for specific errors")
            print("📋 Consult ORCHESTRATION_GUIDE.md for detailed troubleshooting")
            return 1

    print("\n📊 Next Steps:")
    print("   • View results in MLflow: http://localhost:5000")
    print("   • Check model files in models/ directory")
    print("   • Review logs for detailed execution info")
    print("   • See ORCHESTRATION_GUIDE.md for advanced options")

    return 0


if __name__ == "__main__":
    sys.exit(main())
