class RAGEnterpriseError(Exception):
    """Base exception for the application."""


class AuthenticationError(RAGEnterpriseError):
    """Raised when authentication fails."""


class AuthorizationError(RAGEnterpriseError):
    """Raised when user lacks required permissions."""


class DocumentIngestionError(RAGEnterpriseError):
    """Raised when document ingestion fails."""


class VectorStoreError(RAGEnterpriseError):
    """Raised when vector store operations fail."""


class LLMError(RAGEnterpriseError):
    """Raised when LLM operations fail."""


class GuardrailViolation(RAGEnterpriseError):
    """Raised when input or output fails guardrail checks."""

    def __init__(self, message: str, violation_type: str = "unknown"):
        super().__init__(message)
        self.violation_type = violation_type
