terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket = "rag-enterprise-tfstate"
    key    = "terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "rag-enterprise"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

# VPC
module "vpc" {
  source = "./modules/vpc"

  project_name = var.project_name
  environment  = var.environment
}

# ECR Repository
module "ecr" {
  source = "./modules/ecr"

  project_name = var.project_name
}

# ECS Cluster + Service
module "ecs" {
  source = "./modules/ecs"

  project_name    = var.project_name
  environment     = var.environment
  vpc_id          = module.vpc.vpc_id
  private_subnets = module.vpc.private_subnet_ids
  public_subnets  = module.vpc.public_subnet_ids
  ecr_repo_url    = module.ecr.repository_url
  container_port  = 8000
  cpu             = 512
  memory          = 1024
  desired_count   = 2

  environment_variables = {
    ENVIRONMENT      = var.environment
    LLM_PROVIDER     = "groq"
    VECTOR_STORE_TYPE = "chroma"
    LOG_LEVEL        = "INFO"
  }

  secrets = {
    JWT_SECRET_KEY = module.secrets.jwt_secret_arn
    GROQ_API_KEY   = module.secrets.groq_api_key_arn
  }
}

# Secrets Manager
module "secrets" {
  source = "./modules/secrets"

  project_name = var.project_name
  environment  = var.environment
}
