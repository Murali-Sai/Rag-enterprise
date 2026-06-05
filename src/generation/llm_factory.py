from langchain_core.language_models import BaseChatModel

from src.common.logging import get_logger
from src.config import LLMProvider, settings

logger = get_logger(__name__)


def _provider_has_key(provider: LLMProvider) -> bool:
    """Return True if the API key for the given provider is configured."""
    return {
        LLMProvider.GROQ: bool(settings.groq_api_key),
        LLMProvider.GEMINI: bool(settings.google_api_key),
        LLMProvider.OPENAI: bool(settings.openai_api_key),
        LLMProvider.HUGGINGFACE: bool(settings.huggingface_api_key),
    }.get(provider, False)


def _resolve_provider(provider: LLMProvider | None) -> LLMProvider:
    """Pick a usable provider.

    Prefer the explicitly configured provider, but if its API key is missing,
    fall back to whichever provider actually has a key. This makes deployment
    robust to LLM_PROVIDER / API-key mismatches (a common foot-gun).
    """
    provider = provider or settings.llm_provider
    if _provider_has_key(provider):
        return provider

    # Fallback priority: Gemini -> Groq -> OpenAI -> HuggingFace
    for candidate in (
        LLMProvider.GEMINI,
        LLMProvider.GROQ,
        LLMProvider.OPENAI,
        LLMProvider.HUGGINGFACE,
    ):
        if _provider_has_key(candidate):
            logger.warning(
                "llm_provider_fallback",
                requested=provider.value,
                using=candidate.value,
                reason="no API key for requested provider",
            )
            return candidate

    raise ValueError(
        "No LLM API key configured. Set one of GROQ_API_KEY, GOOGLE_API_KEY, "
        "OPENAI_API_KEY, or HUGGINGFACE_API_KEY (and optionally LLM_PROVIDER)."
    )


def create_llm(provider: LLMProvider | None = None) -> BaseChatModel:
    provider = _resolve_provider(provider)

    if provider == LLMProvider.GROQ:
        from langchain_groq import ChatGroq

        logger.info("initializing_llm", provider="groq", model="llama-3.3-70b-versatile")
        return ChatGroq(
            api_key=settings.groq_api_key,
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=1024,
        )

    if provider == LLMProvider.GEMINI:
        from langchain_google_genai import ChatGoogleGenerativeAI

        logger.info("initializing_llm", provider="gemini", model="gemini-2.5-flash")
        return ChatGoogleGenerativeAI(
            google_api_key=settings.google_api_key,
            model="gemini-2.5-flash",
            temperature=0.1,
            max_output_tokens=1024,
        )

    if provider == LLMProvider.OPENAI:
        from langchain_openai import ChatOpenAI

        logger.info("initializing_llm", provider="openai", model="gpt-4o-mini")
        return ChatOpenAI(
            api_key=settings.openai_api_key,
            model="gpt-4o-mini",
            temperature=0.1,
            max_tokens=1024,
        )

    if provider == LLMProvider.HUGGINGFACE:
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        logger.info("initializing_llm", provider="huggingface")
        endpoint = HuggingFaceEndpoint(
            huggingfacehub_api_token=settings.huggingface_api_key,
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            temperature=0.1,
            max_new_tokens=1024,
        )
        return ChatHuggingFace(llm=endpoint)

    raise ValueError(f"Unsupported LLM provider: {provider}")


_llm_instance: BaseChatModel | None = None


def get_llm() -> BaseChatModel:
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = create_llm()
    return _llm_instance
