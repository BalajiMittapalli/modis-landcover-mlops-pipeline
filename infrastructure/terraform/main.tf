# Main Terraform configuration for MODIS Land Cover MLOps Infrastructure
# This creates cloud infrastructure for model deployment and monitoring

terraform {
  required_version = ">= 1.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.1"
    }
  }

  # Backend configuration for state management
  backend "azurerm" {
    # Configure these values in terraform.tfvars or environment variables
    # resource_group_name  = "terraform-state-rg"
    # storage_account_name = "terraformstatexxxxxx"
    # container_name      = "tfstate"
    # key                 = "modis-mlops.terraform.tfstate"
  }
}

# Configure providers
provider "azurerm" {
  features {}
}

provider "aws" {
  region = var.aws_region
}

# Random suffix for globally unique resource names
resource "random_string" "suffix" {
  length  = 8
  special = false
  upper   = false
}

# Local values for resource naming
locals {
  project_name = "modis-landcover-mlops"
  environment  = var.environment

  common_tags = {
    Project     = local.project_name
    Environment = local.environment
    CreatedBy   = "terraform"
    Owner       = var.resource_owner
  }

  resource_suffix = "${local.environment}-${random_string.suffix.result}"
}

# Azure Resource Group
resource "azurerm_resource_group" "main" {
  count    = var.deploy_azure ? 1 : 0
  name     = "${local.project_name}-rg-${local.resource_suffix}"
  location = var.azure_location
  tags     = local.common_tags
}

# Azure Container Registry
resource "azurerm_container_registry" "acr" {
  count               = var.deploy_azure ? 1 : 0
  name                = replace("${local.project_name}acr${random_string.suffix.result}", "-", "")
  resource_group_name = azurerm_resource_group.main[0].name
  location           = azurerm_resource_group.main[0].location
  sku                = "Standard"
  admin_enabled      = true
  tags               = local.common_tags
}

# Azure Container Instances for model serving
resource "azurerm_container_group" "model_server" {
  count               = var.deploy_azure ? 1 : 0
  name                = "${local.project_name}-aci-${local.resource_suffix}"
  location           = azurerm_resource_group.main[0].location
  resource_group_name = azurerm_resource_group.main[0].name
  ip_address_type    = "Public"
  dns_name_label     = "${local.project_name}-${local.resource_suffix}"
  os_type            = "Linux"
  tags               = local.common_tags

  container {
    name   = "model-server"
    image  = "${azurerm_container_registry.acr[0].login_server}/modis-landcover-classifier:latest"
    cpu    = var.container_cpu
    memory = var.container_memory

    ports {
      port     = 5001
      protocol = "TCP"
    }

    environment_variables = {
      FLASK_ENV = "production"
      MODEL_PATH = "/app/production_models"
    }

    # Health probe
    liveness_probe {
      http_get {
        path   = "/health"
        port   = 5001
        scheme = "Http"
      }
      initial_delay_seconds = 30
      period_seconds       = 30
      timeout_seconds      = 5
      failure_threshold    = 3
    }

    readiness_probe {
      http_get {
        path   = "/health"
        port   = 5001
        scheme = "Http"
      }
      initial_delay_seconds = 15
      period_seconds       = 10
      timeout_seconds      = 5
      failure_threshold    = 3
    }
  }

  depends_on = [azurerm_container_registry.acr]
}

# Azure Application Insights for monitoring
resource "azurerm_log_analytics_workspace" "main" {
  count               = var.deploy_azure ? 1 : 0
  name                = "${local.project_name}-law-${local.resource_suffix}"
  location           = azurerm_resource_group.main[0].location
  resource_group_name = azurerm_resource_group.main[0].name
  sku                = "PerGB2018"
  retention_in_days  = var.log_retention_days
  tags               = local.common_tags
}

resource "azurerm_application_insights" "main" {
  count               = var.deploy_azure ? 1 : 0
  name                = "${local.project_name}-ai-${local.resource_suffix}"
  location           = azurerm_resource_group.main[0].location
  resource_group_name = azurerm_resource_group.main[0].name
  workspace_id       = azurerm_log_analytics_workspace.main[0].id
  application_type   = "web"
  tags               = local.common_tags
}

# Azure Storage Account for MLflow
resource "azurerm_storage_account" "mlflow" {
  count                    = var.deploy_azure ? 1 : 0
  name                     = replace("${local.project_name}mlflow${random_string.suffix.result}", "-", "")
  resource_group_name      = azurerm_resource_group.main[0].name
  location                = azurerm_resource_group.main[0].location
  account_tier            = "Standard"
  account_replication_type = "LRS"
  tags                    = local.common_tags
}

resource "azurerm_storage_container" "mlflow_artifacts" {
  count                 = var.deploy_azure ? 1 : 0
  name                  = "mlflow-artifacts"
  storage_account_name  = azurerm_storage_account.mlflow[0].name
  container_access_type = "private"
}

# AWS Infrastructure
# VPC for AWS resources
resource "aws_vpc" "main" {
  count                = var.deploy_aws ? 1 : 0
  cidr_block          = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = merge(local.common_tags, {
    Name = "${local.project_name}-vpc-${local.resource_suffix}"
  })
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  count  = var.deploy_aws ? 1 : 0
  vpc_id = aws_vpc.main[0].id

  tags = merge(local.common_tags, {
    Name = "${local.project_name}-igw-${local.resource_suffix}"
  })
}

# Public Subnets
resource "aws_subnet" "public" {
  count  = var.deploy_aws ? 2 : 0
  vpc_id = aws_vpc.main[0].id

  cidr_block              = "10.0.${count.index + 1}.0/24"
  availability_zone       = data.aws_availability_zones.available[0].names[count.index]
  map_public_ip_on_launch = true

  tags = merge(local.common_tags, {
    Name = "${local.project_name}-public-subnet-${count.index + 1}-${local.resource_suffix}"
  })
}

# Route Table
resource "aws_route_table" "public" {
  count  = var.deploy_aws ? 1 : 0
  vpc_id = aws_vpc.main[0].id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main[0].id
  }

  tags = merge(local.common_tags, {
    Name = "${local.project_name}-public-rt-${local.resource_suffix}"
  })
}

resource "aws_route_table_association" "public" {
  count          = var.deploy_aws ? 2 : 0
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public[0].id
}

# ECR Repository
resource "aws_ecr_repository" "model_repo" {
  count                = var.deploy_aws ? 1 : 0
  name                 = "${local.project_name}/model-server"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = local.common_tags
}

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  count = var.deploy_aws ? 1 : 0
  name  = "${local.project_name}-cluster-${local.resource_suffix}"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = local.common_tags
}

# ECS Task Definition
resource "aws_ecs_task_definition" "model_server" {
  count                    = var.deploy_aws ? 1 : 0
  family                   = "${local.project_name}-model-server"
  requires_compatibilities = ["FARGATE"]
  network_mode            = "awsvpc"
  cpu                     = var.fargate_cpu
  memory                  = var.fargate_memory
  execution_role_arn      = aws_iam_role.ecs_execution[0].arn
  task_role_arn          = aws_iam_role.ecs_task[0].arn

  container_definitions = jsonencode([
    {
      name  = "model-server"
      image = "${aws_ecr_repository.model_repo[0].repository_url}:latest"

      portMappings = [
        {
          containerPort = 5001
          hostPort      = 5001
          protocol      = "tcp"
        }
      ]

      environment = [
        {
          name  = "FLASK_ENV"
          value = "production"
        }
      ]

      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:5001/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.ecs[0].name
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "ecs"
        }
      }

      essential = true
    }
  ])

  tags = local.common_tags
}

# ECS Service
resource "aws_ecs_service" "model_server" {
  count           = var.deploy_aws ? 1 : 0
  name            = "${local.project_name}-service"
  cluster         = aws_ecs_cluster.main[0].id
  task_definition = aws_ecs_task_definition.model_server[0].arn
  desired_count   = var.ecs_desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = aws_subnet.public[*].id
    security_groups  = [aws_security_group.ecs[0].id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.model_server[0].arn
    container_name   = "model-server"
    container_port   = 5001
  }

  depends_on = [aws_lb_listener.model_server]

  tags = local.common_tags
}

# Application Load Balancer
resource "aws_lb" "model_server" {
  count              = var.deploy_aws ? 1 : 0
  name               = "${local.project_name}-alb-${random_string.suffix.result}"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb[0].id]
  subnets           = aws_subnet.public[*].id

  tags = local.common_tags
}

resource "aws_lb_target_group" "model_server" {
  count       = var.deploy_aws ? 1 : 0
  name        = "${local.project_name}-tg-${random_string.suffix.result}"
  port        = 5001
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main[0].id
  target_type = "ip"

  health_check {
    enabled             = true
    healthy_threshold   = 2
    interval            = 30
    matcher             = "200"
    path                = "/health"
    port                = "traffic-port"
    protocol            = "HTTP"
    timeout             = 5
    unhealthy_threshold = 2
  }

  tags = local.common_tags
}

resource "aws_lb_listener" "model_server" {
  count             = var.deploy_aws ? 1 : 0
  load_balancer_arn = aws_lb.model_server[0].arn
  port              = "80"
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.model_server[0].arn
  }
}

# Security Groups
resource "aws_security_group" "alb" {
  count       = var.deploy_aws ? 1 : 0
  name        = "${local.project_name}-alb-sg-${local.resource_suffix}"
  description = "Security group for ALB"
  vpc_id      = aws_vpc.main[0].id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.common_tags
}

resource "aws_security_group" "ecs" {
  count       = var.deploy_aws ? 1 : 0
  name        = "${local.project_name}-ecs-sg-${local.resource_suffix}"
  description = "Security group for ECS tasks"
  vpc_id      = aws_vpc.main[0].id

  ingress {
    from_port       = 5001
    to_port         = 5001
    protocol        = "tcp"
    security_groups = [aws_security_group.alb[0].id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.common_tags
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "ecs" {
  count             = var.deploy_aws ? 1 : 0
  name              = "/ecs/${local.project_name}"
  retention_in_days = var.log_retention_days

  tags = local.common_tags
}

# S3 Bucket for MLflow artifacts
resource "aws_s3_bucket" "mlflow_artifacts" {
  count  = var.deploy_aws ? 1 : 0
  bucket = "${local.project_name}-mlflow-artifacts-${random_string.suffix.result}"

  tags = local.common_tags
}

resource "aws_s3_bucket_versioning" "mlflow_artifacts" {
  count  = var.deploy_aws ? 1 : 0
  bucket = aws_s3_bucket.mlflow_artifacts[0].id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "mlflow_artifacts" {
  count  = var.deploy_aws ? 1 : 0
  bucket = aws_s3_bucket.mlflow_artifacts[0].id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# IAM Roles for ECS
resource "aws_iam_role" "ecs_execution" {
  count = var.deploy_aws ? 1 : 0
  name  = "${local.project_name}-ecs-execution-role-${local.resource_suffix}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_execution" {
  count      = var.deploy_aws ? 1 : 0
  role       = aws_iam_role.ecs_execution[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "ecs_task" {
  count = var.deploy_aws ? 1 : 0
  name  = "${local.project_name}-ecs-task-role-${local.resource_suffix}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

# Data sources
data "aws_availability_zones" "available" {
  count = var.deploy_aws ? 1 : 0
  state = "available"
}
