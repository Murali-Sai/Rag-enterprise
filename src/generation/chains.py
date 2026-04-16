from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.common.logging import get_logger
from src.generation.llm_factory import get_llm
from src.generation.prompts import rag_prompt

logger = get_logger(__name__)


def format_documents(docs: list[Document]) -> str:
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file", "Unknown")
        department = doc.metadata.get("department", "Unknown")

        header_parts = [f"Source: {source}", f"Department: {department}"]

        # EDGAR-specific fields
        ticker = doc.metadata.get("ticker")
        filing_type = doc.metadata.get("filing_type")
        section = doc.metadata.get("section_name")
        filing_date = doc.metadata.get("filing_date")

        if ticker:
            header_parts.append(f"Company: {ticker}")
        if filing_type:
            header_parts.append(f"Filing: {filing_type}")
        if section:
            header_parts.append(f"Section: {section}")
        if filing_date:
            header_parts.append(f"Date: {filing_date}")

        header = ", ".join(header_parts)
        formatted.append(f"[Document {i}] ({header})\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def create_rag_chain():  # noqa: ANN201
    llm = get_llm()

    chain = (
        {
            "context": lambda x: format_documents(x["documents"]),
            "question": lambda x: x["question"],
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    return chain


def query_with_context(question: str, documents: list[Document]) -> str:
    chain = create_rag_chain()
    result = chain.invoke({"question": question, "documents": documents})
    logger.info(
        "rag_generation_complete",
        question_preview=question[:50],
        context_docs=len(documents),
        answer_length=len(result),
    )
    return result
