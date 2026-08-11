"""The endpoint paths Project 6 §5.1 names, mounted over the existing handlers.

The spec asks for three routes by name — `POST /v1/ask`, `GET /v1/documents`
and `POST /v1/ingest` — and this API grew a different vocabulary: `/query`,
`/documents`, `/documents/ingest`, with no version prefix anywhere.

Aliases rather than renames, because the existing paths are load-bearing in
places a rename would break silently:

- the Streamlit dashboard posts to `/query` with a `retrieval_mode`
- the landing page's copyable examples, and the smoke tests in every handoff
- `README.md`'s curl walkthrough, which reviewers run
- the MCP server

A rename would fix a documentation gap by breaking the demo, which is the
wrong trade. Two names for one route costs an entry in the OpenAPI schema.

**Delegation, not reimplementation.** Each alias calls the same implementation
as the path it shadows, so there is one copy of the compliance layer and the
guardrails.

The rate limit needed more than delegation, and the first version of this file
got it wrong. **slowapi keys a limit on the request path**, so an alias that
merely calls the decorated handler is charged to its own bucket: measured, 21
requests alternating between `/query` and `/v1/ask` all returned 200 against a
20/minute limit, because each path had spent only half of it. On a public demo
whose limit exists to cap abuse, adding an alias would have quietly doubled
what one client can spend, with nothing erroring and nothing logged.

So both routes carry `@limiter.shared_limit(..., scope=QUERY_LIMIT_SCOPE)` and
delegate to the undecorated `answer_query`. Each route is charged exactly once,
against one bucket. A test pins the combined budget.
"""

from fastapi import APIRouter, Depends, Request, Response, UploadFile, status

from src.api.deps import get_current_user, get_rbac_retriever, require_role
from src.api.middleware import limiter
from src.api.routes.documents import ingest, list_documents
from src.api.routes.query import QUERY_LIMIT_SCOPE, answer_query
from src.auth.models import User
from src.common.schemas import (
    DocumentIngestResponse,
    IndexedDocumentsResponse,
    QueryRequest,
    QueryResponse,
)
from src.config import settings
from src.retrieval.retriever import Retriever

router = APIRouter(prefix="/v1", tags=["v1"])


@router.post("/ask", response_model=QueryResponse)
@limiter.shared_limit(settings.rate_limit, scope=QUERY_LIMIT_SCOPE)
async def ask(
    request: Request,
    response: Response,
    payload: QueryRequest,
    user: User = Depends(get_current_user),
    retriever: Retriever = Depends(get_rbac_retriever),
) -> QueryResponse:
    """Project 6 §5.1's name for `POST /query`. Same handler, same budget."""
    return await answer_query(request, response, payload, user, retriever)


@router.get("/documents", response_model=IndexedDocumentsResponse)
async def documents(user: User = Depends(get_current_user)) -> IndexedDocumentsResponse:
    """Project 6 §5.1's name for `GET /documents`."""
    return await list_documents(user=user)


@router.post(
    "/ingest",
    response_model=DocumentIngestResponse,
    status_code=status.HTTP_201_CREATED,
)
async def ingest_document(
    file: UploadFile,
    department: str,
    access_roles: str,
    user: User = Depends(require_role("admin")),
) -> DocumentIngestResponse:
    """Project 6 §5.1's name for `POST /documents/ingest`.

    Inherits the shipped deployment's refusal: `ALLOW_RUNTIME_INGEST=false` is
    baked into the image, so this returns 403 there. The demo's admin password
    is published on the landing page, which makes admin a public role and a
    write path to a fingerprinted corpus unacceptable.
    """
    return await ingest(
        file=file,
        department=department,
        access_roles=access_roles,
        user=user,
    )
