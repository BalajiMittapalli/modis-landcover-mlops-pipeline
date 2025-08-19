# MODIS Land Cover MLOps Pipeline

## 🌍 Problem Description
This project implements a **production-ready MLOps pipeline** for land cover classification using 22 years (2001-2022) of NASA MODIS MCD12C1 satellite data. The system classifies 7 different land cover types with advanced feature engineering, comprehensive monitoring, and automated deployment capabilities.

## 🎯 Business Objectives
- Achieve >40% classification accuracy for multi-class land cover types
- Automated model training and deployment pipeline with CI/CD
- Scalable feature extraction from satellite imagery
- Production-ready monitoring and alerting system
- Multi-cloud deployment capabilities (Azure/AWS)
- Real-time model drift detection and performance monitoring

## 🏆 Success Metrics
- **Model Performance**: 41.53% accuracy achieved ✅
- **F1 Score**: 0.3213 macro F1 score ✅
- **Pipeline Automation**: Complete Prefect orchestration ✅
- **Model Registry**: Enhanced MLflow integration with validation ✅
- **CI/CD**: Automated GitHub Actions pipeline ✅
- **Infrastructure**: Multi-cloud Terraform deployment ✅
- **Monitoring**: Advanced drift detection and real-time dashboard ✅

## 🛠️ Tech Stack

### **Core ML & Data Processing**
- **ML Framework**: scikit-learn, Random Forest
- **Data Processing**: rasterio, numpy, pandas
- **Feature Engineering**: Advanced spectral, spatial, and texture features

### **MLOps & Orchestration**
- **Orchestration**: Prefect 3.x
- **Tracking**: MLflow with enhanced model registry
- **Model Monitoring**: Advanced drift detection (KS, PSI, Jensen-Shannon)
- **Real-time Dashboard**: Plotly Dash for monitoring

### **Deployment & Infrastructure**
- **Containerization**: Docker with multi-stage builds
- **Deployment**: Flask REST API with health checks
- **Infrastructure as Code**: Terraform (Azure & AWS)
- **Cloud Platforms**: Azure Container Instances, AWS ECS Fargate

### **CI/CD & Testing**
- **CI/CD**: GitHub Actions with automated pipelines
- **Testing**: pytest with comprehensive test suite
- **Quality Gates**: Pre-commit hooks, performance validation
- **Integration Testing**: REST API endpoint testing

### **Monitoring & Observability**
- **Drift Detection**: Statistical tests for data/model drift
- **Performance Monitoring**: Real-time metrics and alerts
- **Logging**: Structured logging with Azure/AWS integration
- **Notifications**: Automated alerts and status reporting

## Project Structure

```
├── .github/workflows/       # CI/CD GitHub Actions pipelines
│   └── mlops-pipeline.yml  # Complete MLOps automation
├── config/                  # Configuration files
├── data/                   # Data storage (excluded from git)
├── flows/                  # Prefect workflow definitions
├── infrastructure/         # Infrastructure as Code
│   ├── terraform/         # Multi-cloud Terraform templates
│   └── deploy.sh         # Deployment automation
├── src/                    # Source code
│   ├── data/              # Data processing modules
│   ├── deployment/        # Model deployment code
│   ├── models/            # Model training & enhanced registry
│   └── monitoring/        # Advanced monitoring & drift detection
│       ├── advanced_drift_monitor.py  # Statistical drift detection
│       └── dashboard.py   # Real-time monitoring dashboard
├── tests/                  # Comprehensive test suite
│   ├── test_integration.py # REST API & integration tests
│   ├── test_model.py      # Model unit tests
│   └── test_prefect_workflow.py # Workflow tests
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
5. **Model Registration** - Enhanced MLflow registry with validation
6. **Drift Detection** - Advanced statistical monitoring
7. **Deployment** - Multi-cloud containerized deployment
8. **Monitoring** - Real-time performance dashboard
9. **Notifications** - Automated alerts and status reporting

## 🚀 Advanced MLOps Features

### **🔍 Advanced Model Monitoring**
- **Drift Detection**: Statistical tests including Kolmogorov-Smirnov, Population Stability Index, and Jensen-Shannon divergence
- **Real-time Dashboard**: Interactive monitoring dashboard with live metrics
- **Performance Tracking**: Continuous model performance monitoring with alerts
- **Data Quality Monitoring**: Automated data validation and quality checks

### **🔄 CI/CD Pipeline**
- **Automated Testing**: Unit, integration, and workflow tests on every commit
- **Quality Gates**: Performance validation before deployment
- **Multi-cloud Deployment**: Automated deployment to Azure and AWS
- **Model Registry Integration**: Automatic model registration and staging

### **☁️ Infrastructure as Code**
- **Terraform Templates**: Complete infrastructure definitions for Azure and AWS
- **Container Orchestration**: Azure Container Instances and AWS ECS Fargate
- **Load Balancing**: Application Load Balancer with health checks
- **Monitoring Integration**: Azure Application Insights and AWS CloudWatch

### **🛡️ Enhanced Model Registry**
- **Model Validation**: Automated performance validation before registration
- **Staging Pipeline**: Development → Staging → Production promotion
- **Version Management**: Comprehensive model versioning and metadata tracking
- **Rollback Capabilities**: Easy model rollback and comparison

### **📊 Real-time Monitoring Dashboard**
- **System Health**: Server status, response times, and resource utilization
- **Model Performance**: Real-time prediction accuracy and drift metrics
- **Data Quality**: Input data validation and anomaly detection
- **Alert Management**: Configurable alerts for various monitoring conditions

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

### **Production Deployments**
- **Azure Container Instances**: Scalable cloud deployment with monitoring
- **AWS ECS Fargate**: Serverless container deployment
- **Local Docker**: Containerized local deployment for testing

### **Deployment Environments**
- **Production**: Weekly scheduled, full dataset (2001-2022), multi-cloud
- **Staging**: Automated testing environment with quality gates
- **Development**: Manual runs, recent data (2020-2022)
- **Experimental**: Testing with alternative algorithms and features

### **Monitoring & Alerting**
- **Drift Detection**: Automated model and data drift monitoring
- **Performance Alerts**: Real-time notifications for performance degradation
- **System Health**: Infrastructure monitoring and alerting
- **Dashboard Access**: Web-based monitoring dashboard

## 📋 Prerequisites
- Python 3.10+
- Conda environment: `nasa-landcover-mlops`
- 8GB+ RAM for full dataset training
- Docker (for containerized deployment)
- Terraform (for infrastructure deployment)
- Cloud provider credentials (Azure/AWS for cloud deployment)
- CUDA 11.x (optional, for GPU acceleration)

## 🚀 Quick Start Guides

### **Option 1: Complete MLOps Pipeline (Recommended)**
```bash
# 1. Set up environment
conda activate nasa-landcover-mlops
pip install -r requirements.txt

# 2. Configure cloud credentials (optional)
export AZURE_CREDENTIALS="your-azure-credentials"
export AWS_ACCESS_KEY_ID="your-aws-key"

# 3. Run complete pipeline with monitoring
make setup-prefect
make run-pipeline

# 4. Access monitoring dashboard
python src/monitoring/dashboard.py
# Visit: http://localhost:8050
```

### **Option 2: CI/CD Deployment**
```bash
# Push to main branch triggers automatic:
# - Testing and validation
# - Model training and evaluation
# - Multi-cloud deployment
# - Monitoring setup

git push origin main
# Check GitHub Actions for pipeline status
```

### **Option 3: Local Development**
```bash
# Quick training and testing
python train_random_forest.py
python src/models/enhanced_model_registry.py --model_path production_models/latest.pkl

# Run drift monitoring
python src/monitoring/advanced_drift_monitor.py

# Start monitoring dashboard
python src/monitoring/dashboard.py
```

### **Option 4: Infrastructure Deployment**
```bash
# Deploy to Azure
cd infrastructure/terraform
terraform init
terraform plan -var="deploy_azure=true"
terraform apply

# Deploy to AWS
terraform plan -var="deploy_aws=true"
terraform apply
```

## ✨ Complete Feature Set

### **🤖 Machine Learning**
- **Advanced Feature Engineering**: Spectral, spatial, and texture features
- **Model Registry**: Enhanced MLflow integration with validation and staging
- **Performance Monitoring**: Real-time accuracy and drift detection
- **Quality Gates**: Automated performance validation before deployment

### **🔄 MLOps & Automation**
- **CI/CD Pipeline**: Complete GitHub Actions workflow automation
- **Infrastructure as Code**: Terraform templates for multi-cloud deployment
- **Container Orchestration**: Docker with Azure/AWS deployment
- **Workflow Orchestration**: Prefect 3.x with automated scheduling

### **📊 Monitoring & Observability**
- **Drift Detection**: Statistical tests (KS, PSI, Jensen-Shannon divergence)
- **Real-time Dashboard**: Interactive monitoring with live metrics
- **Performance Tracking**: Continuous model and system monitoring
- **Alert System**: Automated notifications and status reporting

### **🧪 Testing & Validation**
- **Comprehensive Testing**: Unit, integration, and workflow tests
- **REST API Testing**: Automated endpoint validation
- **Performance Testing**: Model quality and response time validation
- **Infrastructure Testing**: Deployment and scaling validation

### **☁️ Cloud & Deployment**
- **Multi-cloud Support**: Azure Container Instances and AWS ECS
- **Auto-scaling**: Dynamic resource allocation based on demand
- **Load Balancing**: High availability with health checks
- **Security**: Secure credential management and encrypted communications

### **📈 Analytics & Reporting**
- **Model Metrics**: Comprehensive performance analytics
- **Usage Analytics**: API usage and performance tracking
- **Cost Monitoring**: Cloud resource usage and cost optimization
- **Audit Trails**: Complete logging and traceability

---

## 🎯 Production-Ready MLOps Pipeline
This project demonstrates enterprise-grade MLOps practices with comprehensive monitoring, automated deployment, and multi-cloud infrastructure support. Perfect for scaling satellite imagery analysis to production environments.

For detailed setup and usage, see the documentation in each module.
