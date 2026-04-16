resource "aws_secretsmanager_secret" "jwt_secret" {
  name = "${var.project_name}/${var.environment}/jwt-secret-key"
}

resource "aws_secretsmanager_secret" "groq_api_key" {
  name = "${var.project_name}/${var.environment}/groq-api-key"
}

variable "project_name" { type = string }
variable "environment" { type = string }

output "jwt_secret_arn" { value = aws_secretsmanager_secret.jwt_secret.arn }
output "groq_api_key_arn" { value = aws_secretsmanager_secret.groq_api_key.arn }
