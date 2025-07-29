# MODIS Land Cover MLOps Pipeline

## 🌍 Problem Description
This project implements a complete MLOps pipeline for land cover classification using 22 years (2001-2022) of NASA MODIS MCD12C1 satellite data. The system classifies 7 different land cover types with advanced feature engineering and workflow orchestration.

## 🎯 Business Objectives
- Achieve >40% classification accuracy for multi-class land cover types
- Automated model training and deployment pipeline
- Scalable feature extraction from satellite imagery
- Production-ready monitoring and alerting system

## 🏆 Success Metrics
- **Model Performance**: 41.53% accuracy achieved ✅
- **F1 Score**: 0.3213 macro F1 score ✅
- **Pipeline Automation**: Complete Prefect orchestration ✅
- **Model Registry**: MLflow integration ✅

## 🛠️ Tech Stack
- **ML Framework**: scikit-learn, Random Forest
- **Orchestration**: Prefect 3.x
- **Tracking**: MLflow
- **Deployment**: Docker, Flask REST API
- **Monitoring**: Custom monitoring system
- **Testing**: pytest, comprehensive test suite
- **CI/CD**: GitHub Actions
- **Data Processing**: rasterio, numpy, pandas

## Project Structure

```
├── config/                  # Configuration files
├── data/                   # Data storage (excluded from git)
├── flows/                  # Prefect workflow definitions
├── src/                    # Source code
│   ├── data/              # Data processing modules
│   ├── deployment/        # Model deployment code
│   ├── models/            # Model training modules
│   └── monitoring/        # Model monitoring code
├── tests/                  # Test files
├── train_random_forest.py  # Main training script
├── run_pipeline.py         # Pipeline executor
└── quick_run.py           # Quick test runner
```

## Quick Start

### Option 1: Complete Automated Pipeline (Recommended)
```bash
# 1. Set up environment
conda activate nasa-landcover-mlops
pip install -r requirements.txt

# 2. Set up Prefect orchestration
make setup-prefect

# 3. Run the ML pipeline
make run-pipeline
```

### Option 2: Step-by-Step Orchestration Setup
```bash
# Step 1: Install Prefect and dependencies
python flows/setup_prefect.py

# Step 2: Start Prefect server (in new terminal)
prefect server start --host 0.0.0.0

# Step 3: Deploy workflows (in original terminal)
python flows/deploy.py

# Step 4: Start agent (in new terminal)
prefect agent start --pool default-agent-pool

# Step 5: Run pipeline
prefect deployment run 'modis-landcover-ml-pipeline/modis-landcover-development'
```

### Option 3: Direct Execution
```bash
# Quick training without orchestration
python train_random_forest.py

# Pipeline with fallback
python run_pipeline.py

# Quick test
python quick_run.py
```

## 📊 Pipeline Overview

The orchestrated pipeline includes:

1. **Data Ingestion** - Load MODIS data files (2001-2022)
2. **Feature Extraction** - Spectral, spatial, and texture features
3. **Model Training** - Random Forest with hyperparameter tuning
4. **Model Evaluation** - Performance metrics and quality gates
5. **Model Registration** - Automated model registry management
6. **Notifications** - Slack alerts on success/failure

## 🗂️ Data Description
- **Source**: NASA MODIS MCD12C1 Collection 6.1
- **Years**: 2001-2022 (22 years of data)
- **Resolution**: 500m global coverage
- **Land Cover Classes**: 7 IGBP categories
  - Water Bodies
  - Evergreen Needleleaf Forests
  - Evergreen Broadleaf Forests
  - Deciduous Needleleaf Forests
  - Deciduous Broadleaf Forests
  - Mixed Forests
  - Closed Shrublands

## 🤖 Model Details
- **Algorithm**: Random Forest Classifier
- **Features**: 15 engineered features per pixel
  - Spectral features (bands, indices)
  - Spatial features (neighborhood statistics)
  - Texture features (GLCM metrics)
- **Performance**: 41.53% accuracy on test set
- **Training Data**: ~2.8M samples across all years

## 🔧 Available Deployments

- **Production**: Weekly scheduled, full dataset (2001-2022)
- **Development**: Manual runs, recent data (2020-2022)
- **Experimental**: Testing with Gradient Boosting

## 📋 Prerequisites
- Python 3.10+
- Conda environment: `nasa-landcover-mlops`
- 8GB+ RAM for full dataset training
- CUDA 11.x (optional, for GPU acceleration)
- Docker (for deployment)
- AWS CLI (for cloud deployment)

## 🚀 Features
- **Automated Feature Engineering**: Spectral, spatial, and texture features
- **Model Registry**: Automatic model versioning and tracking
- **Quality Gates**: Performance validation before deployment
- **Slack Notifications**: Real-time pipeline status updates
- **Comprehensive Testing**: Unit, integration, and workflow tests
- **CI/CD Pipeline**: GitHub Actions for automated testing and deployment

---

For detailed setup and usage, see the documentation in each module.
