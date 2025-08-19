# Output values for MODIS Land Cover MLOps Infrastructure

# Azure Outputs
output "azure_resource_group_name" {
  description = "Name of the Azure resource group"
  value       = var.deploy_azure ? azurerm_resource_group.main[0].name : null
}

output "azure_container_registry_login_server" {
  description = "Login server URL for Azure Container Registry"
  value       = var.deploy_azure ? azurerm_container_registry.acr[0].login_server : null
}

output "azure_container_registry_admin_username" {
  description = "Admin username for Azure Container Registry"
  value       = var.deploy_azure ? azurerm_container_registry.acr[0].admin_username : null
  sensitive   = true
}

output "azure_container_instance_fqdn" {
  description = "FQDN of the Azure Container Instance"
  value       = var.deploy_azure ? azurerm_container_group.model_server[0].fqdn : null
}

output "azure_container_instance_ip" {
  description = "Public IP address of the Azure Container Instance"
  value       = var.deploy_azure ? azurerm_container_group.model_server[0].ip_address : null
}

output "azure_application_insights_instrumentation_key" {
  description = "Instrumentation key for Azure Application Insights"
  value       = var.deploy_azure ? azurerm_application_insights.main[0].instrumentation_key : null
  sensitive   = true
}

output "azure_application_insights_connection_string" {
  description = "Connection string for Azure Application Insights"
  value       = var.deploy_azure ? azurerm_application_insights.main[0].connection_string : null
  sensitive   = true
}

output "azure_storage_account_name" {
  description = "Name of the Azure Storage Account for MLflow"
  value       = var.deploy_azure ? azurerm_storage_account.mlflow[0].name : null
}

output "azure_storage_account_primary_key" {
  description = "Primary access key for Azure Storage Account"
  value       = var.deploy_azure ? azurerm_storage_account.mlflow[0].primary_access_key : null
  sensitive   = true
}

# AWS Outputs
output "aws_vpc_id" {
  description = "ID of the AWS VPC"
  value       = var.deploy_aws ? aws_vpc.main[0].id : null
}

output "aws_public_subnet_ids" {
  description = "IDs of the AWS public subnets"
  value       = var.deploy_aws ? aws_subnet.public[*].id : null
}

output "aws_ecr_repository_url" {
  description = "URL of the AWS ECR repository"
  value       = var.deploy_aws ? aws_ecr_repository.model_repo[0].repository_url : null
}

output "aws_ecs_cluster_name" {
  description = "Name of the AWS ECS cluster"
  value       = var.deploy_aws ? aws_ecs_cluster.main[0].name : null
}

output "aws_ecs_service_name" {
  description = "Name of the AWS ECS service"
  value       = var.deploy_aws ? aws_ecs_service.model_server[0].name : null
}

output "aws_load_balancer_dns_name" {
  description = "DNS name of the AWS Application Load Balancer"
  value       = var.deploy_aws ? aws_lb.model_server[0].dns_name : null
}

output "aws_load_balancer_zone_id" {
  description = "Zone ID of the AWS Application Load Balancer"
  value       = var.deploy_aws ? aws_lb.model_server[0].zone_id : null
}

output "aws_s3_mlflow_bucket_name" {
  description = "Name of the S3 bucket for MLflow artifacts"
  value       = var.deploy_aws ? aws_s3_bucket.mlflow_artifacts[0].bucket : null
}

output "aws_cloudwatch_log_group_name" {
  description = "Name of the CloudWatch log group"
  value       = var.deploy_aws ? aws_cloudwatch_log_group.ecs[0].name : null
}

# Deployment URLs
output "model_endpoint_urls" {
  description = "URLs for accessing the deployed model endpoints"
  value = {
    azure = var.deploy_azure ? "http://${azurerm_container_group.model_server[0].fqdn}:5001" : null
    aws   = var.deploy_aws ? "http://${aws_lb.model_server[0].dns_name}" : null
  }
}

# Resource identifiers for CI/CD
output "deployment_targets" {
  description = "Deployment targets for CI/CD pipelines"
  value = {
    azure = var.deploy_azure ? {
      resource_group    = azurerm_resource_group.main[0].name
      container_group   = azurerm_container_group.model_server[0].name
      registry_server   = azurerm_container_registry.acr[0].login_server
      registry_username = azurerm_container_registry.acr[0].admin_username
    } : null

    aws = var.deploy_aws ? {
      cluster_name    = aws_ecs_cluster.main[0].name
      service_name    = aws_ecs_service.model_server[0].name
      repository_url  = aws_ecr_repository.model_repo[0].repository_url
      task_definition = aws_ecs_task_definition.model_server[0].family
    } : null
  }
}

# MLflow Configuration
output "mlflow_configuration" {
  description = "MLflow configuration for tracking and registry"
  value = {
    azure = var.deploy_azure ? {
      artifact_store = "azure://${azurerm_storage_account.mlflow[0].name}.blob.core.windows.net/mlflow-artifacts"
      tracking_uri   = "sqlite:///mlflow.db"  # Can be updated to use Azure SQL Database
    } : null

    aws = var.deploy_aws ? {
      artifact_store = "s3://${aws_s3_bucket.mlflow_artifacts[0].bucket}"
      tracking_uri   = "sqlite:///mlflow.db"  # Can be updated to use RDS
    } : null
  }
}

# Monitoring Endpoints
output "monitoring_endpoints" {
  description = "Monitoring and logging endpoints"
  value = {
    azure = var.deploy_azure ? {
      application_insights = azurerm_application_insights.main[0].app_id
      log_analytics       = azurerm_log_analytics_workspace.main[0].workspace_id
    } : null

    aws = var.deploy_aws ? {
      cloudwatch_logs = aws_cloudwatch_log_group.ecs[0].name
    } : null
  }
}

# Security Information
output "security_groups" {
  description = "Security group IDs for network access control"
  value = var.deploy_aws ? {
    alb_security_group = aws_security_group.alb[0].id
    ecs_security_group = aws_security_group.ecs[0].id
  } : null
}

# Auto-scaling Information
output "scaling_configuration" {
  description = "Auto-scaling configuration details"
  value = {
    aws_ecs_desired_count = var.deploy_aws ? aws_ecs_service.model_server[0].desired_count : null
    min_capacity         = var.min_capacity
    max_capacity         = var.max_capacity
    target_cpu_util      = var.target_cpu_utilization
  }
}

# Connection Strings and Keys (Sensitive)
output "connection_strings" {
  description = "Connection strings for various services"
  value = {
    azure_storage = var.deploy_azure ? azurerm_storage_account.mlflow[0].primary_connection_string : null
    azure_insights = var.deploy_azure ? azurerm_application_insights.main[0].connection_string : null
  }
  sensitive = true
}
