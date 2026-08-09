"""HyDE — Hypothetical Document Embeddings.

Queries and documents don't live in the same shape. A user asks "What was
Apple's total net revenue?" — eight words, interrogative. The chunk that
answers it is a paragraph of declarative filing prose, or a table of figures.
Embedding both with the same bi-encoder and comparing them asks the model to
bridge that asymmetry, which is exactly what it's weakest at.

HyDE (Gao et al. 2022) closes the gap from the other side: ask an LLM to
*write* the passage that would answer the question, then embed that instead.
The hypothetical passage is usually wrong on specifics — it may invent a
revenue figure — but it's wrong in the right shape, using the vocabulary and
register of the corpus. Retrieval matches document-to-document rather than
question-to-document.

The invented specifics never reach the user: the hypothetical text is used
only as a retrieval key. Generation still runs on the original question over
the real retrieved chunks.

Cost: one extra LLM call per query, on the critical path before retrieval.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.common.logging import get_logger
from src.config import settings

logger = get_logger(__name__)

HYDE_SYSTEM_PROMPT = """You write passages that imitate the style of SEC 10-K annual report filings.

Given a question, write a short passage (2-4 sentences) that reads as if it were \
excerpted from the relevant section of a 10-K filing and answers the question.

Rules:
- Match the register of a real filing: formal, declarative, specific.
- Use the vocabulary a filing would use ("net sales", "fiscal year", "the Company").
- Include plausible specifics (figures, dates, segment names) — this text is used \
only to find real passages, never shown to anyone, so invented numbers are fine \
and actually help the match.
- Do not hedge, do not say you are uncertain, do not mention that this is hypothetical.
- Output only the passage."""

hyde_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", HYDE_SYSTEM_PROMPT),
        ("human", "Question: {question}\n\nPassage:"),
    ]
)


def generate_hypothetical_document(query: str) -> str:
    """Write a synthetic filing passage that would answer the query.

    Returns the original query unchanged if generation fails — a degraded
    retrieval is better than a failed request, and the caller can't tell the
    difference apart from the logged warning.
    """
    from src.generation.llm_factory import get_llm

    try:
        chain = hyde_prompt | get_llm() | StrOutputParser()
        hypothetical = chain.invoke({"question": query}).strip()
    except Exception as e:
        logger.warning(
            "hyde_generation_failed",
            error=str(e),
            error_type=type(e).__name__,
            fallback="original_query",
        )
        return query

    if not hypothetical:
        logger.warning("hyde_generation_empty", fallback="original_query")
        return query

    logger.info(
        "hyde_document_generated",
        query_preview=query[:60],
        hypothetical_chars=len(hypothetical),
    )
    return hypothetical


def build_retrieval_query(query: str) -> str:
    """The text actually embedded for retrieval.

    Appends the original question to the hypothetical passage when
    HYDE_INCLUDE_QUERY is set. Pure HyDE discards the question entirely,
    which can lose literal terms that matter here — a ticker or an "Item 7A"
    the LLM happened not to echo. Keeping both hedges that at the cost of
    diluting the passage.
    """
    hypothetical = generate_hypothetical_document(query)
    if hypothetical == query:  # generation failed; nothing to combine
        return query
    if settings.hyde_include_query:
        return f"{query}\n\n{hypothetical}"
    return hypothetical
