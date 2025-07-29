#!/usr/bin/env python3
"""
Prefect setup and initialization script for MODIS Land Cover Classification.
"""

import os
import subprocess
import sys
from pathlib import Path


def install_prefect():
    """Install Prefect and related dependencies."""
    print("📦 Installing Prefect and dependencies...")

    packages = ["prefect>=2.10.0", "prefect-slack>=0.1.0", "asyncio", "aiohttp"]

    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed: {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False

    return True


def setup_prefect_server():
    """Initialize Prefect server configuration."""
    print("\n🔧 Setting up Prefect server...")

    commands = [
        ["prefect", "config", "set", "PREFECT_API_URL=http://127.0.0.1:4200/api"],
        ["prefect", "server", "database", "reset", "--yes"],
    ]

    for cmd in commands:
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ Executed: {' '.join(cmd)}")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Command failed (might be okay): {' '.join(cmd)}")
            print(f"   Error: {e.stderr if e.stderr else str(e)}")


def create_work_pool():
    """Create a work pool for the flows."""
    print("\n🏊 Creating work pool...")

    try:
        cmd = ["prefect", "work-pool", "create", "default-agent-pool", "--type", "process"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Work pool 'default-agent-pool' created successfully")
        else:
            print("⚠️  Work pool might already exist")

    except Exception as e:
        print(f"❌ Failed to create work pool: {e}")


def start_prefect_server_background():
    """Start Prefect server in background."""
    print("\n🚀 Starting Prefect server...")
    print("Note: Server will run in background. Access UI at http://localhost:4200")

    try:
        # Start server in background
        subprocess.Popen(
            ["prefect", "server", "start", "--host", "0.0.0.0"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("✅ Prefect server started in background")
        print("🌐 UI available at: http://localhost:4200")

    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        print("💡 Try starting manually: prefect server start")


def create_startup_scripts():
    """Create convenient startup scripts."""
    print("\n📝 Creating startup scripts...")

    # Windows batch script
    batch_script = """@echo off
echo Starting Prefect Server...
prefect server start --host 0.0.0.0
pause
"""

    with open("start_prefect_server.bat", "w") as f:
        f.write(batch_script)
    print("✅ Created: start_prefect_server.bat")

    # Agent startup script
    agent_script = """@echo off
echo Starting Prefect Agent...
prefect agent start --pool default-agent-pool
pause
"""

    with open("start_prefect_agent.bat", "w") as f:
        f.write(agent_script)
    print("✅ Created: start_prefect_agent.bat")

    # Python script to run pipeline
    python_runner = '''#!/usr/bin/env python3
"""
Quick runner for MODIS ML Pipeline
"""

import asyncio
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from flows.ml_pipeline import ml_pipeline

async def main():
    """Run the pipeline with default parameters."""
    print("🚀 Running MODIS Land Cover ML Pipeline...")

    result = await ml_pipeline(
        start_year=2020,
        end_year=2022,
        model_type="random_forest",
        sample_ratio=0.005  # Small sample for testing
    )

    print("✅ Pipeline completed!")
    print(f"📊 Accuracy: {result['evaluation_result']['performance_metrics']['accuracy']:.4f}")

if __name__ == "__main__":
    asyncio.run(main())
'''

    with open("run_pipeline.py", "w") as f:
        f.write(python_runner)
    print("✅ Created: run_pipeline.py")


def main():
    """Main setup function."""
    print("🔧 Setting up Prefect for MODIS Land Cover Classification")
    print("=" * 60)

    # Install Prefect
    if not install_prefect():
        print("❌ Failed to install Prefect. Exiting.")
        return

    # Setup server
    setup_prefect_server()

    # Create work pool
    create_work_pool()

    # Create directories
    from config import setup_directories

    setup_directories()

    # Create startup scripts
    create_startup_scripts()

    # Start server
    start_prefect_server_background()

    print("\n" + "=" * 60)
    print("✅ Prefect setup complete!")
    print("\n📋 Next steps:")
    print("1. Wait for server to start (check http://localhost:4200)")
    print("2. Deploy flows: python flows/deploy.py")
    print("3. Start agent: start_prefect_agent.bat")
    print("4. Run pipeline: python run_pipeline.py")
    print("\n🔗 Useful commands:")
    print("  - prefect server start (manual server start)")
    print("  - prefect agent start --pool default-agent-pool")
    print("  - prefect deployment run 'modis-landcover-ml-pipeline/modis-landcover-development'")


if __name__ == "__main__":
    main()
