"""Tests for the embedding provider factory.

The provider decides the vector dimensionality of the whole index (1536 for
text-embedding-3-small, 384 for MiniLM), so picking the wrong one does not
raise — it produces a collection that cannot be queried by the other provider.
These tests pin the selection logic and the failure mode when the OpenAI
provider is configured without a key.
"""

from unittest.mock import patch

import pytest

from src.config import EmbeddingProvider
from src.ingestion import embeddings as embeddings_module
from src.ingestion.embeddings import (
    active_embedding_model_name,
    create_embedding_model,
    get_embedding_model,
    reset_embedding_model,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    reset_embedding_model()
    yield
    reset_embedding_model()


class TestProviderSelection:
    def test_openai_provider_builds_openai_embeddings(self):
        with (
            patch.object(
                embeddings_module.settings, "embedding_provider", EmbeddingProvider.OPENAI
            ),
            patch.object(embeddings_module.settings, "openai_api_key", "sk-test"),
            patch.object(
                embeddings_module.settings, "openai_embedding_model", "text-embedding-3-small"
            ),
        ):
            model = create_embedding_model()

        assert type(model).__name__ == "OpenAIEmbeddings"
        assert model.model == "text-embedding-3-small"

    def test_huggingface_provider_requires_no_api_key(self):
        """The local provider is the offline/CI path — it must not need a key."""
        with (
            patch.object(
                embeddings_module.settings, "embedding_provider", EmbeddingProvider.HUGGINGFACE
            ),
            patch.object(embeddings_module.settings, "openai_api_key", ""),
            patch("langchain_huggingface.HuggingFaceEmbeddings") as hf,
        ):
            create_embedding_model()

        assert hf.called
        assert hf.call_args.kwargs["encode_kwargs"] == {"normalize_embeddings": True}

    def test_openai_provider_without_key_fails_loudly(self):
        """Better to raise than to fall back and silently build a 384-dim index
        that the configured provider can never query."""
        with (
            patch.object(
                embeddings_module.settings, "embedding_provider", EmbeddingProvider.OPENAI
            ),
            patch.object(embeddings_module.settings, "openai_api_key", ""),
            pytest.raises(ValueError, match="OPENAI_API_KEY"),
        ):
            create_embedding_model()


class TestCaching:
    def test_model_is_built_once_and_reused(self):
        with patch.object(
            embeddings_module, "create_embedding_model", return_value=object()
        ) as factory:
            first = get_embedding_model()
            second = get_embedding_model()

        assert first is second
        assert factory.call_count == 1

    def test_reset_forces_a_rebuild(self):
        """Scripts switch provider mid-process by setting env then resetting;
        without this the first model built would leak into the new config."""
        with patch.object(
            embeddings_module, "create_embedding_model", side_effect=[object(), object()]
        ) as factory:
            first = get_embedding_model()
            reset_embedding_model()
            second = get_embedding_model()

        assert first is not second
        assert factory.call_count == 2


class TestActiveModelName:
    def test_name_is_provider_qualified(self):
        """Eval results record this. 'all-MiniLM-L6-v2' alone would not say
        which provider produced the vectors once both are in play."""
        with (
            patch.object(
                embeddings_module.settings, "embedding_provider", EmbeddingProvider.OPENAI
            ),
            patch.object(
                embeddings_module.settings, "openai_embedding_model", "text-embedding-3-small"
            ),
        ):
            assert active_embedding_model_name() == "openai/text-embedding-3-small"

        with (
            patch.object(
                embeddings_module.settings, "embedding_provider", EmbeddingProvider.HUGGINGFACE
            ),
            patch.object(
                embeddings_module.settings,
                "embedding_model",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
        ):
            assert (
                active_embedding_model_name()
                == "huggingface/sentence-transformers/all-MiniLM-L6-v2"
            )
