#!/usr/bin/env python3
"""
Configuration and utilities for Prefect workflow orchestration.
"""

import os
from pathlib import Path
from typing import Any, Dict

# Prefect configuration
PREFECT_CONFIG = {
    "server": {"host": "127.0.0.1", "port": 4200, "api_url": "http://127.0.0.1:4200/api"},
    "agent": {"work_pool": "default-agent-pool", "query_interval": 10},
    "storage": {"artifacts_dir": "artifacts", "logs_dir": "logs/prefect"},
}

# Default pipeline parameters
DEFAULT_PIPELINE_PARAMS = {
    "production": {
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
    },
    "development": {
        "start_year": 2020,
        "end_year": 2022,
        "model_type": "random_forest",
        "sample_ratio": 0.005,
        "model_params": {"n_estimators": 50, "max_depth": 10},
    },
    "experimental": {
        "start_year": 2020,
        "end_year": 2021,
        "model_type": "gradient_boosting",
        "sample_ratio": 0.002,
        "model_params": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1},
    },
}

# Schedule configurations
SCHEDULE_CONFIG = {
    "production": {
        "cron": "0 2 * * 1",  # Weekly on Monday at 2 AM UTC
        "timezone": "UTC",
        "description": "Weekly production model training",
    },
    "development": None,  # Manual runs only
    "experimental": None,  # Manual runs only
}

# Notification settings
NOTIFICATION_CONFIG = {
    "slack": {
        "webhook_url": os.getenv("SLACK_WEBHOOK_URL"),
        "channel": "#ml-alerts",
        "enabled": bool(os.getenv("SLACK_WEBHOOK_URL")),
    },
    "email": {"enabled": False, "smtp_server": None, "recipients": []},  # Configure as needed
}


def get_pipeline_config(environment: str = "development") -> Dict[str, Any]:
    """Get configuration for specific environment."""
    return {
        "parameters": DEFAULT_PIPELINE_PARAMS.get(
            environment, DEFAULT_PIPELINE_PARAMS["development"]
        ),
        "schedule": SCHEDULE_CONFIG.get(environment),
        "notifications": NOTIFICATION_CONFIG,
        "prefect": PREFECT_CONFIG,
    }


def setup_directories():
    """Create necessary directories for Prefect workflows."""
    directories = ["artifacts", "logs/prefect", "production_models", "temp"]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")


if __name__ == "__main__":
    print("🔧 Setting up Prefect workflow directories...")
    setup_directories()

    print("\n📋 Configuration Summary:")
    for env in ["production", "development", "experimental"]:
        config = get_pipeline_config(env)
        print(f"\n{env.upper()}:")
        print(f"  - Years: {config['parameters']['start_year']}-{config['parameters']['end_year']}")
        print(f"  - Model: {config['parameters']['model_type']}")
        print(f"  - Sample Ratio: {config['parameters']['sample_ratio']}")
        if config["schedule"]:
            print(f"  - Schedule: {config['schedule']['cron']}")
        else:
            print(f"  - Schedule: Manual")

    print(f"\n🔔 Notifications:")
    print(f"  - Slack: {'Enabled' if NOTIFICATION_CONFIG['slack']['enabled'] else 'Disabled'}")
    print(f"  - Email: {'Enabled' if NOTIFICATION_CONFIG['email']['enabled'] else 'Disabled'}")

    print("\n✅ Setup complete!")
