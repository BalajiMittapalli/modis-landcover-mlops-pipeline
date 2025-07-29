#!/usr/bin/env powershell
# Git Deployment Script - Phase by Phase
# Automated deployment to GitHub repository

Write-Host "🚀 NASA MODIS MLOps Pipeline - GitHub Deployment" -ForegroundColor Green
Write-Host "============================================================"

$repo_url = "https://github.com/BalajiMittapalli/modis-landcover-mlops-pipeline.git"

# Check if git is initialized
if (-not (Test-Path ".git")) {
    Write-Host "📋 Initial Setup..." -ForegroundColor Yellow
    git init
    git remote add origin $repo_url
    git branch -M main
    Write-Host "✅ Git initialized" -ForegroundColor Green
} else {
    Write-Host "✅ Git already initialized" -ForegroundColor Green
}

# Phase 1: Data Pipeline
Write-Host "`n1️⃣  Phase 1: Data Pipeline" -ForegroundColor Cyan
git add src/data/ config/ src/utils.py environment.yml requirements.txt README.md
git commit -m "data pipeline"
git push -u origin main
Write-Host "✅ Phase 1 deployed" -ForegroundColor Green

# Phase 2: ML Training
Write-Host "`n2️⃣  Phase 2: ML Training" -ForegroundColor Cyan
git add src/models/ configs/ tests/test_phase2.py model_registry.json production_models/ train_random_forest.py
git commit -m "ml training"
git push origin main
Write-Host "✅ Phase 2 deployed" -ForegroundColor Green

# Phase 3: Orchestration
Write-Host "`n3️⃣  Phase 3: Orchestration" -ForegroundColor Cyan
git add flows/ run_pipeline.py ORCHESTRATION_GUIDE.md PHASE3_EXPLANATION.md
git commit -m "workflow orchestration"
git push origin main
Write-Host "✅ Phase 3 deployed" -ForegroundColor Green

# Phase 4: Production Setup
Write-Host "`n4️⃣  Phase 4: Production Setup" -ForegroundColor Cyan
git add docker-compose.yml Dockerfile .github/ src/deployment/ src/monitoring/ Makefile .pre-commit-config.yaml setup.cfg
git commit -m "production deployment"
git push origin main
Write-Host "✅ Phase 4 deployed" -ForegroundColor Green

Write-Host "`n🎉 ALL PHASES DEPLOYED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "📍 Repository: $repo_url" -ForegroundColor White
Write-Host "✅ Clean phase-wise commits completed" -ForegroundColor Green
