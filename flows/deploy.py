#!/usr/bin/env python3
"""
Deploy Prefect flows for MODIS Land Cover Classification Pipeline
"""

import asyncio
import os
from pathlib import Path

from prefect import flow
from prefect.client.orchestration import get_client
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule, IntervalSchedule

from flows.ml_pipeline import ml_pipeline


async def deploy_ml_pipeline():
    """Deploy the main ML pipeline."""

    # Production deployment
    prod_deployment = Deployment.build_from_flow(
        flow=ml_pipeline,
        name="modis-landcover-production",
        description="Production ML pipeline for MODIS land cover classification",
        tags=["ml", "modis", "landcover", "production"],
        parameters={
            "start_year": 2001,
            "end_year": 2022,
            "model_type": "random_forest",
            "sample_ratio": 0.01,
            "model_params": {
                "n_estimators": 200,
                "max_depth": 20,
                "min_samples_split": 10,
                "min_samples_leaf": 5,
            },
            "slack_webhook_url": os.getenv("SLACK_WEBHOOK_URL"),
        },
        schedule=CronSchedule(cron="0 2 * * 1", timezone="UTC"),  # Weekly on Monday 2 AM UTC
        work_pool_name="default-agent-pool",
    )

    # Development deployment
    dev_deployment = Deployment.build_from_flow(
        flow=ml_pipeline,
        name="modis-landcover-development",
        description="Development ML pipeline for testing and experimentation",
        tags=["ml", "modis", "landcover", "development"],
        parameters={
            "start_year": 2020,
            "end_year": 2022,
            "model_type": "random_forest",
            "sample_ratio": 0.005,  # Smaller sample for faster development
            "model_params": {"n_estimators": 50, "max_depth": 10},
        },
        # No schedule for dev - run manually
        work_pool_name="default-agent-pool",
    )

    # Experimental deployment for testing different algorithms
    experimental_deployment = Deployment.build_from_flow(
        flow=ml_pipeline,
        name="modis-landcover-experimental",
        description="Experimental ML pipeline for algorithm testing",
        tags=["ml", "modis", "landcover", "experimental"],
        parameters={
            "start_year": 2020,
            "end_year": 2021,
            "model_type": "gradient_boosting",
            "sample_ratio": 0.002,
            "model_params": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1},
        },
        work_pool_name="default-agent-pool",
    )

    # Apply deployments
    deployments = [prod_deployment, dev_deployment, experimental_deployment]

    for deployment in deployments:
        deployment_id = await deployment.apply()
        print(f"✅ Deployed: {deployment.name} (ID: {deployment_id})")

    print("\n🚀 All deployments created successfully!")
    print("\nTo run deployments:")
    print("prefect deployment run 'modis-landcover-ml-pipeline/modis-landcover-production'")
    print("prefect deployment run 'modis-landcover-ml-pipeline/modis-landcover-development'")
    print("prefect deployment run 'modis-landcover-ml-pipeline/modis-landcover-experimental'")


async def main():
    """Main deployment function."""
    print("🔧 Setting up Prefect deployments for MODIS Land Cover Classification...")

    # Check if Prefect server is running
    try:
        async with get_client() as client:
            server_info = await client.api_healthcheck()
            print(f"✅ Connected to Prefect server: {server_info}")
    except Exception as e:
        print(f"❌ Could not connect to Prefect server: {e}")
        print("Please start Prefect server with: prefect server start")
        return

    # Deploy flows
    await deploy_ml_pipeline()

    print("\n📋 Next steps:")
    print("1. Start a Prefect agent: prefect agent start --pool default-agent-pool")
    print("2. View flows in UI: prefect server start (if not running)")
    print("3. Monitor runs: http://localhost:4200")


if __name__ == "__main__":
    asyncio.run(main())
