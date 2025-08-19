#!/bin/bash
# Infrastructure Deployment Script for MODIS Land Cover MLOps

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
TERRAFORM_DIR="infrastructure/terraform"
ENV=${1:-dev}
PLAN_FILE="terraform-${ENV}.plan"

echo -e "${BLUE}🚀 MODIS Land Cover MLOps Infrastructure Deployment${NC}"
echo -e "${BLUE}=================================================${NC}"
echo -e "Environment: ${GREEN}${ENV}${NC}"
echo ""

# Check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}🔍 Checking prerequisites...${NC}"

    # Check if Terraform is installed
    if ! command -v terraform &> /dev/null; then
        echo -e "${RED}❌ Terraform is not installed. Please install Terraform first.${NC}"
        exit 1
    fi

    # Check if Azure CLI is installed (if deploying to Azure)
    if ! command -v az &> /dev/null; then
        echo -e "${YELLOW}⚠️  Azure CLI is not installed. Azure deployment will be skipped.${NC}"
    fi

    # Check if AWS CLI is installed (if deploying to AWS)
    if ! command -v aws &> /dev/null; then
        echo -e "${YELLOW}⚠️  AWS CLI is not installed. AWS deployment will be skipped.${NC}"
    fi

    echo -e "${GREEN}✅ Prerequisites check completed${NC}"
}

# Authenticate with cloud providers
authenticate() {
    echo -e "${YELLOW}🔑 Authenticating with cloud providers...${NC}"

    # Azure authentication
    if command -v az &> /dev/null; then
        echo "Checking Azure authentication..."
        if ! az account show &> /dev/null; then
            echo -e "${YELLOW}Please login to Azure:${NC}"
            az login
        fi

        # Set subscription if provided
        if [[ -n "${AZURE_SUBSCRIPTION_ID}" ]]; then
            az account set --subscription "${AZURE_SUBSCRIPTION_ID}"
        fi

        echo -e "${GREEN}✅ Azure authentication successful${NC}"
    fi

    # AWS authentication
    if command -v aws &> /dev/null; then
        echo "Checking AWS authentication..."
        if ! aws sts get-caller-identity &> /dev/null; then
            echo -e "${RED}❌ AWS authentication failed. Please configure AWS credentials.${NC}"
            echo "Run: aws configure"
            exit 1
        fi
        echo -e "${GREEN}✅ AWS authentication successful${NC}"
    fi
}

# Initialize Terraform
init_terraform() {
    echo -e "${YELLOW}🏗️  Initializing Terraform...${NC}"

    cd "${TERRAFORM_DIR}"

    # Create terraform.tfvars if it doesn't exist
    if [[ ! -f "terraform.tfvars" ]]; then
        echo -e "${YELLOW}Creating terraform.tfvars from example...${NC}"
        cp terraform.tfvars.example terraform.tfvars
        echo -e "${YELLOW}⚠️  Please edit terraform.tfvars with your specific configuration${NC}"
    fi

    # Initialize Terraform
    terraform init

    echo -e "${GREEN}✅ Terraform initialization completed${NC}"
}

# Plan infrastructure changes
plan_infrastructure() {
    echo -e "${YELLOW}📋 Planning infrastructure changes...${NC}"

    cd "${TERRAFORM_DIR}"

    # Run terraform plan
    terraform plan \
        -var="environment=${ENV}" \
        -out="${PLAN_FILE}"

    echo -e "${GREEN}✅ Infrastructure planning completed${NC}"
    echo -e "${BLUE}Plan saved to: ${PLAN_FILE}${NC}"
}

# Apply infrastructure changes
apply_infrastructure() {
    echo -e "${YELLOW}🚀 Applying infrastructure changes...${NC}"

    cd "${TERRAFORM_DIR}"

    # Apply the plan
    terraform apply "${PLAN_FILE}"

    echo -e "${GREEN}✅ Infrastructure deployment completed${NC}"
}

# Output deployment information
output_deployment_info() {
    echo -e "${YELLOW}📊 Retrieving deployment information...${NC}"

    cd "${TERRAFORM_DIR}"

    # Get Terraform outputs
    echo -e "${BLUE}Deployment Outputs:${NC}"
    terraform output

    # Save outputs to file
    terraform output -json > "outputs-${ENV}.json"
    echo -e "${GREEN}✅ Outputs saved to outputs-${ENV}.json${NC}"
}

# Deploy model to infrastructure
deploy_model() {
    echo -e "${YELLOW}📦 Deploying model to infrastructure...${NC}"

    cd "${TERRAFORM_DIR}"

    # Get deployment targets
    AZURE_REGISTRY=$(terraform output -raw azure_container_registry_login_server 2>/dev/null || echo "")
    AWS_ECR_URL=$(terraform output -raw aws_ecr_repository_url 2>/dev/null || echo "")

    # Build and push Docker image to Azure
    if [[ -n "${AZURE_REGISTRY}" ]]; then
        echo -e "${BLUE}Deploying to Azure...${NC}"

        # Get ACR credentials
        AZURE_USERNAME=$(terraform output -raw azure_container_registry_admin_username)
        AZURE_PASSWORD=$(az acr credential show --name "${AZURE_REGISTRY%.*}" --query "passwords[0].value" -o tsv)

        # Login to ACR
        echo "${AZURE_PASSWORD}" | docker login "${AZURE_REGISTRY}" --username "${AZURE_USERNAME}" --password-stdin

        # Build and push image
        cd ../../  # Go back to project root
        docker build -t "${AZURE_REGISTRY}/modis-landcover-classifier:latest" .
        docker push "${AZURE_REGISTRY}/modis-landcover-classifier:latest"

        echo -e "${GREEN}✅ Model deployed to Azure${NC}"
    fi

    # Build and push Docker image to AWS
    if [[ -n "${AWS_ECR_URL}" ]]; then
        echo -e "${BLUE}Deploying to AWS...${NC}"

        # Get AWS region and account
        AWS_REGION=$(terraform output -raw aws_region 2>/dev/null || echo "us-east-1")

        # Login to ECR
        aws ecr get-login-password --region "${AWS_REGION}" | \
            docker login --username AWS --password-stdin "${AWS_ECR_URL}"

        # Build and push image
        cd ../../  # Go back to project root
        docker build -t "${AWS_ECR_URL}:latest" .
        docker push "${AWS_ECR_URL}:latest"

        # Update ECS service to use new image
        ECS_CLUSTER=$(terraform output -raw aws_ecs_cluster_name)
        ECS_SERVICE=$(terraform output -raw aws_ecs_service_name)

        aws ecs update-service \
            --cluster "${ECS_CLUSTER}" \
            --service "${ECS_SERVICE}" \
            --force-new-deployment \
            --region "${AWS_REGION}"

        echo -e "${GREEN}✅ Model deployed to AWS${NC}"
    fi
}

# Setup monitoring
setup_monitoring() {
    echo -e "${YELLOW}📊 Setting up monitoring...${NC}"

    cd "${TERRAFORM_DIR}"

    # Get monitoring endpoints
    AZURE_INSIGHTS_KEY=$(terraform output -raw azure_application_insights_instrumentation_key 2>/dev/null || echo "")
    AWS_LOG_GROUP=$(terraform output -raw aws_cloudwatch_log_group_name 2>/dev/null || echo "")

    if [[ -n "${AZURE_INSIGHTS_KEY}" ]]; then
        echo -e "${BLUE}Azure Application Insights configured${NC}"
        echo "Instrumentation Key: ${AZURE_INSIGHTS_KEY}"
    fi

    if [[ -n "${AWS_LOG_GROUP}" ]]; then
        echo -e "${BLUE}AWS CloudWatch Logs configured${NC}"
        echo "Log Group: ${AWS_LOG_GROUP}"
    fi

    echo -e "${GREEN}✅ Monitoring setup completed${NC}"
}

# Main deployment function
main() {
    echo -e "${BLUE}Starting deployment for environment: ${ENV}${NC}"

    check_prerequisites
    authenticate
    init_terraform
    plan_infrastructure

    # Ask for confirmation before applying
    echo -e "${YELLOW}Do you want to apply these changes? (y/N):${NC}"
    read -r response
    if [[ "${response}" =~ ^[Yy]$ ]]; then
        apply_infrastructure
        output_deployment_info
        deploy_model
        setup_monitoring

        echo ""
        echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
        echo -e "${BLUE}Next steps:${NC}"
        echo "1. Test the deployed endpoints"
        echo "2. Configure monitoring alerts"
        echo "3. Set up automated backups"
        echo "4. Review security settings"
    else
        echo -e "${YELLOW}Deployment cancelled.${NC}"
    fi
}

# Cleanup function
cleanup() {
    echo -e "${YELLOW}🧹 Cleaning up infrastructure...${NC}"

    cd "${TERRAFORM_DIR}"

    echo -e "${RED}⚠️  This will destroy all infrastructure. Are you sure? (y/N):${NC}"
    read -r response
    if [[ "${response}" =~ ^[Yy]$ ]]; then
        terraform destroy -var="environment=${ENV}" -auto-approve
        echo -e "${GREEN}✅ Infrastructure destroyed${NC}"
    else
        echo -e "${YELLOW}Cleanup cancelled.${NC}"
    fi
}

# Help function
show_help() {
    echo "MODIS Land Cover MLOps Infrastructure Deployment"
    echo ""
    echo "Usage: $0 [COMMAND] [ENVIRONMENT]"
    echo ""
    echo "Commands:"
    echo "  deploy    Deploy infrastructure (default)"
    echo "  plan      Plan infrastructure changes only"
    echo "  destroy   Destroy infrastructure"
    echo "  help      Show this help message"
    echo ""
    echo "Environments:"
    echo "  dev       Development environment (default)"
    echo "  staging   Staging environment"
    echo "  prod      Production environment"
    echo ""
    echo "Examples:"
    echo "  $0 deploy dev"
    echo "  $0 plan prod"
    echo "  $0 destroy staging"
}

# Parse command line arguments
case "${1}" in
    "plan")
        ENV=${2:-dev}
        check_prerequisites
        authenticate
        init_terraform
        plan_infrastructure
        ;;
    "destroy")
        ENV=${2:-dev}
        cleanup
        ;;
    "help")
        show_help
        ;;
    "deploy"|"")
        ENV=${1:-dev}
        main
        ;;
    *)
        echo -e "${RED}Unknown command: ${1}${NC}"
        show_help
        exit 1
        ;;
esac
