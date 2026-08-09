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


def has_usable_provider() -> bool:
    """True if any provider has a key, i.e. create_llm() would succeed.

    The readiness probe uses this rather than naming providers itself — an
    OpenAI-only deployment reporting llm=not_configured because the check
    predated OpenAI becoming the default is exactly the kind of drift a probe
    is supposed to catch, not cause.
    """
    return any(_provider_has_key(p) for p in LLMProvider)


def _resolve_provider(provider: LLMProvider | None) -> LLMProvider:
    """Pick a usable provider.

    Prefer the explicitly configured provider, but if its API key is missing,
    fall back to whichever provider actually has a key. This makes deployment
    robust to LLM_PROVIDER / API-key mismatches (a common foot-gun).
    """
    provider = provider or settings.llm_provider
    if _provider_has_key(provider):
        return provider

    # Fallback priority: OpenAI -> Gemini -> Groq -> HuggingFace. OpenAI leads
    # because it is the configured default; a fallback that reorders the list
    # would answer from a different model than the one the run is tagged with.
    for candidate in (
        LLMProvider.OPENAI,
        LLMProvider.GEMINI,
        LLMProvider.GROQ,
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


DEFAULT_MAX_TOKENS = 1024


def create_llm(
    provider: LLMProvider | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
) -> BaseChatModel:
    """Build a chat model.

    `model` overrides the provider's default — the RAGAS judge uses it to stay
    on gpt-4o-mini while runtime generation moved to gpt-4o, so raising the
    answer model doesn't silently change what grades the eval.

    `max_tokens` overrides the 1024-token cap. Generation wants a cap: an
    answer that long has stopped being an answer. A judge does not — RAGAS
    faithfulness emits one structured verdict per extracted statement, so its
    output grows with the answer it is grading, and hitting the cap raises
    LLMDidNotFinishException and drops the row from the aggregate rather than
    truncating it. That failure is silent in the score and heavily
    non-uniform: it lands on exactly the long, multi-claim answers, so the
    mean is taken over the short ones.
    """
    provider = _resolve_provider(provider)
    max_tokens = max_tokens or DEFAULT_MAX_TOKENS

    if provider == LLMProvider.GROQ:
        from langchain_groq import ChatGroq

        name = model or "llama-3.3-70b-versatile"
        logger.info("initializing_llm", provider="groq", model=name)
        return ChatGroq(
            api_key=settings.groq_api_key,
            model=name,
            temperature=0.1,
            max_tokens=max_tokens,
        )

    if provider == LLMProvider.GEMINI:
        from langchain_google_genai import ChatGoogleGenerativeAI

        name = model or "gemini-2.5-flash"
        logger.info("initializing_llm", provider="gemini", model=name)
        return ChatGoogleGenerativeAI(
            google_api_key=settings.google_api_key,
            model=name,
            temperature=0.1,
            max_output_tokens=max_tokens,
        )

    if provider == LLMProvider.OPENAI:
        from langchain_openai import ChatOpenAI

        name = model or settings.openai_model
        logger.info("initializing_llm", provider="openai", model=name)
        return ChatOpenAI(
            api_key=settings.openai_api_key,
            model=name,
            temperature=0.1,
            max_tokens=max_tokens,
        )

    if provider == LLMProvider.HUGGINGFACE:
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        repo = model or "mistralai/Mistral-7B-Instruct-v0.3"
        logger.info("initializing_llm", provider="huggingface", model=repo)
        endpoint = HuggingFaceEndpoint(
            huggingfacehub_api_token=settings.huggingface_api_key,
            repo_id=repo,
            temperature=0.1,
            max_new_tokens=max_tokens,
        )
        return ChatHuggingFace(llm=endpoint)

    raise ValueError(f"Unsupported LLM provider: {provider}")


_llm_instance: BaseChatModel | None = None


def get_llm() -> BaseChatModel:
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = create_llm()
    return _llm_instance
