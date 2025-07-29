# Makefile for NASA Land Cover Classification Project

.PHONY: help setup train evaluate test lint format install-hooks deploy monitor clean docker-build docker-run setup-prefect run-pipeline

# Default target
help:
	@echo "Available targets:"
	@echo "  setup       - Set up the development environment"
	@echo "  setup-prefect - Set up Prefect workflow orchestration"
	@echo "  train       - Train the land cover classification model"
	@echo "  run-pipeline - Run the complete ML pipeline with Prefect"
	@echo "  evaluate    - Evaluate the trained model"
	@echo "  test        - Run unit and integration tests"
	@echo "  lint        - Run code linting"
	@echo "  format      - Format code with black and isort"
	@echo "  install-hooks - Install pre-commit hooks"
	@echo "  deploy      - Deploy model server locally"
	@echo "  monitor     - Run model monitoring"
	@echo "  clean       - Clean up generated files"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run  - Run Docker container"

# Set up development environment
setup:
	@echo "Setting up development environment..."
	conda env create -f environment.yml -n nasa-landcover-mlops || conda env update -f environment.yml -n nasa-landcover-mlops
	pip install -r requirements.txt
	pip install pytest pytest-cov flake8 black isort pre-commit
	@echo "Environment setup complete!"

# Set up Prefect workflow orchestration
setup-prefect:
	@echo "Setting up Prefect workflow orchestration..."
	python flows/setup_prefect.py
	@echo "Prefect setup complete!"

# Install pre-commit hooks
install-hooks:
	@echo "Installing pre-commit hooks..."
	pre-commit install
	@echo "Pre-commit hooks installed!"

# Train the model (traditional approach)
train:
	@echo "Training land cover classification model..."
	python train_random_forest.py --use_all_years --sample_ratio 0.01
	@echo "Training complete!"

# Run the complete ML pipeline with Prefect
run-pipeline:
	@echo "Running ML pipeline with Prefect orchestration..."
	python run_pipeline.py
	@echo "Pipeline complete!"

# Run tests
test:
	@echo "Running tests..."
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "Tests complete!"

# Run linting
lint:
	@echo "Running linting..."
	flake8 src/ train_random_forest.py flows/ --max-line-length=100
	@echo "Linting complete!"

# Format code
format:
	@echo "Formatting code..."
	black --line-length=100 src/ train_random_forest.py flows/
	isort --profile black --line-length=100 src/ train_random_forest.py flows/
	@echo "Code formatting complete!"

# Deploy model server
deploy:
	@echo "Starting model server..."
	python src/deployment/model_server.py

# Run model monitoring
monitor:
	@echo "Starting model monitoring..."
	python src/monitoring/model_monitor.py --run_once

# Evaluate model
evaluate:
	@echo "Evaluating model..."
	python -c "import pickle; import pandas as pd; \
		with open('random_forest_full_dataset_model.pkl', 'rb') as f: \
			model_data = pickle.load(f); \
		print('Model Algorithm:', model_data['algorithm']); \
		print('Features:', len(model_data['feature_names'])); \
		print('Classes:', len(model_data['label_encoder'].classes_))"

# Clean up generated files
clean:
	@echo "Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache htmlcov .coverage monitoring_reports temp_*.pkl
	@echo "Cleanup complete!"

# Docker targets
docker-build:
	@echo "Building Docker image..."
	docker build -t nasa-landcover-classifier .

docker-run:
	@echo "Running Docker container..."
	docker run -p 5001:5001 nasa-landcover-classifier
