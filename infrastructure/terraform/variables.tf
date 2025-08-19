# Input variables for MODIS Land Cover MLOps Infrastructure

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "resource_owner" {
  description = "Owner of the resources for tagging"
  type        = string
  default     = "mlops-team"
}

variable "deploy_azure" {
  description = "Whether to deploy Azure infrastructure"
  type        = bool
  default     = true
}

variable "deploy_aws" {
  description = "Whether to deploy AWS infrastructure"
  type        = bool
  default     = true
}

# Azure Configuration
variable "azure_location" {
  description = "Azure region for resource deployment"
  type        = string
  default     = "East US"
}

variable "container_cpu" {
  description = "CPU allocation for Azure Container Instance"
  type        = number
  default     = 2
}

variable "container_memory" {
  description = "Memory allocation for Azure Container Instance (GB)"
  type        = number
  default     = 4
}

# AWS Configuration
variable "aws_region" {
  description = "AWS region for resource deployment"
  type        = string
  default     = "us-east-1"
}

variable "fargate_cpu" {
  description = "CPU allocation for ECS Fargate tasks"
  type        = number
  default     = 1024  # 1 vCPU
}

variable "fargate_memory" {
  description = "Memory allocation for ECS Fargate tasks (MB)"
  type        = number
  default     = 2048  # 2 GB
}

variable "ecs_desired_count" {
  description = "Desired number of ECS tasks"
  type        = number
  default     = 2
}

# Common Configuration
variable "log_retention_days" {
  description = "Log retention period in days"
  type        = number
  default     = 30
}

# MLflow Configuration
variable "mlflow_backend_store_uri" {
  description = "MLflow backend store URI"
  type        = string
  default     = ""
}

variable "mlflow_default_artifact_root" {
  description = "MLflow default artifact root"
  type        = string
  default     = ""
}

# Monitoring Configuration
variable "enable_monitoring" {
  description = "Enable monitoring and alerting"
  type        = bool
  default     = true
}

variable "alert_email" {
  description = "Email address for alerts"
  type        = string
  default     = ""
}

# Security Configuration
variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to access the application"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # Restrict this in production
}

# Scaling Configuration
variable "auto_scaling_enabled" {
  description = "Enable auto-scaling for the application"
  type        = bool
  default     = true
}

variable "min_capacity" {
  description = "Minimum number of instances"
  type        = number
  default     = 1
}

variable "max_capacity" {
  description = "Maximum number of instances"
  type        = number
  default     = 10
}

variable "target_cpu_utilization" {
  description = "Target CPU utilization for auto-scaling"
  type        = number
  default     = 70
}

# Backup Configuration
variable "backup_retention_days" {
  description = "Backup retention period in days"
  type        = number
  default     = 7
}

variable "enable_point_in_time_recovery" {
  description = "Enable point-in-time recovery for databases"
  type        = bool
  default     = true
}
