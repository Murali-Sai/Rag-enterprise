"""Compliance audit trail for financial services.

Every query, document access, and RBAC decision is logged to an
append-only audit trail. This is critical for regulatory compliance
(SEC Rule 17a-4, FINRA 4511, Dodd-Frank recordkeeping requirements).
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from src.common.logging import get_logger
from src.config import settings

logger = get_logger(__name__)

AUDIT_LOG_DIR = Path(settings.project_root) / "audit_logs"


def _ensure_audit_dir() -> None:
    AUDIT_LOG_DIR.mkdir(parents=True, exist_ok=True)


def log_query_audit(
    user_id: int,
    username: str,
    user_roles: list[str],
    query: str,
    retrieved_departments: list[str],
    documents_accessed: int,
    guardrail_flags: list[str],
    information_barriers_applied: list[str],
    response_length: int,
) -> None:
    """Log every RAG query for compliance audit trail."""
    _ensure_audit_dir()

    entry = {
        "event_type": "rag_query",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "username": username,
        "user_roles": user_roles,
        "query": query,
        "retrieved_departments": retrieved_departments,
        "documents_accessed": documents_accessed,
        "guardrail_flags": guardrail_flags,
        "information_barriers_applied": information_barriers_applied,
        "response_length": response_length,
    }

    # Append to daily audit log file
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_file = AUDIT_LOG_DIR / f"audit_{date_str}.jsonl"

    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")

    logger.info(
        "audit_log_written",
        event="rag_query",
        user=username,
        barriers=len(information_barriers_applied),
        flags=len(guardrail_flags),
    )


def log_document_access(
    user_id: int,
    username: str,
    user_roles: list[str],
    action: str,  # "ingest", "retrieve", "delete"
    department: str,
    document_name: str,
    access_granted: bool,
) -> None:
    """Log document access events for regulatory recordkeeping."""
    _ensure_audit_dir()

    entry = {
        "event_type": "document_access",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "username": username,
        "user_roles": user_roles,
        "action": action,
        "department": department,
        "document_name": document_name,
        "access_granted": access_granted,
    }

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_file = AUDIT_LOG_DIR / f"audit_{date_str}.jsonl"

    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


def log_auth_event(
    username: str,
    event: str,  # "login_success", "login_failure", "token_expired", "registration"
    ip_address: str | None = None,
) -> None:
    """Log authentication events."""
    _ensure_audit_dir()

    entry = {
        "event_type": "auth_event",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "username": username,
        "event": event,
        "ip_address": ip_address,
    }

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_file = AUDIT_LOG_DIR / f"audit_{date_str}.jsonl"

    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")
