from enum import Enum
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    GROQ = "groq"
    GEMINI = "gemini"
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"


class EmbeddingProvider(str, Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"


class VectorStoreType(str, Enum):
    CHROMA = "chroma"
    OPENSEARCH = "opensearch"


class ChunkingStrategy(str, Enum):
    # Fixed-size character windows with overlap. The baseline: no knowledge of
    # sentences, paragraphs, or sections, so it will cut a revenue table away
    # from its header mid-row. Present to be beaten, and to make "structure
    # awareness helps" a measured claim rather than an assumed one.
    FIXED = "fixed"
    # Structure-aware: splits on paragraph, then line, then sentence
    # boundaries, falling back to characters only when a run of text has none.
    # For filings this stacks on the parser, which has already split the
    # document into Items — so a chunk stays inside one section.
    RECURSIVE = "recursive"
    # Splits where consecutive sentence embeddings diverge, i.e. where the
    # subject changes rather than where the character budget runs out.
    SEMANTIC = "semantic"


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # LLM
    llm_provider: LLMProvider = LLMProvider.OPENAI
    groq_api_key: str = ""
    google_api_key: str = ""
    huggingface_api_key: str = ""
    openai_api_key: str = ""

    # Runtime generation model for the OpenAI provider. Separate from
    # eval_judge_model below so raising the answer model doesn't silently
    # change (or multiply the cost of) what grades the eval.
    openai_model: str = "gpt-4o"

    # Embeddings
    #
    # OpenAI text-embedding-3-small is the default: 1536 dimensions against
    # MiniLM's 384, and an 8k-token input window against MiniLM's 256. That
    # window matters more than the dimensionality here — MiniLM silently
    # truncates anything past ~1,300 characters of filing prose at embedding
    # time, so a chunk's tail is indexed as though it were never written.
    #
    # HuggingFace stays selectable: it needs no API key and no network, which
    # keeps tests, offline work, and the container build hermetic. Switching
    # providers changes the vector dimensionality, so an existing collection
    # must be rebuilt, not appended to.
    embedding_provider: EmbeddingProvider = EmbeddingProvider.OPENAI
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # huggingface provider only

    # Vector Store
    vector_store_type: VectorStoreType = VectorStoreType.CHROMA
    chroma_persist_dir: str = "./chroma_data"
    # Configurable so an alternate index (e.g. semantically chunked) can live
    # alongside the shipped one for A/B comparison.
    chroma_collection: str = "rag_enterprise"
    opensearch_url: str = ""
    opensearch_index: str = "rag-enterprise"

    # Auth
    jwt_secret_key: str = "change-this-to-a-random-secret-key"
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 60

    # Database
    database_url: str = "sqlite+aiosqlite:///./rag_enterprise.db"

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    environment: Environment = Environment.DEVELOPMENT
    # Where the Streamlit dashboard is served. It is a separate process on a
    # separate port — and, deployed, a separate host — so the landing page
    # cannot link to it with a relative path. Compose and the deploy
    # environment override this; the default is where `make dashboard` puts it.
    dashboard_url: str = "http://localhost:8501"

    # Rate Limiting
    rate_limit: str = "20/minute"

    # Ingestion
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Floor on chunk length, applied after splitting. The recursive splitter
    # cannot merge a short line into a neighbour that is already near the
    # budget, so a bare heading or a leftover page marker comes out as its own
    # chunk and then competes for a top-k slot on equal terms with a paragraph.
    #
    # 80 is measured, not guessed. Below it the corpus is section headings
    # ("Competition", "Human Capital"), repeated document furniture ("PART II
    # Item 8", "THE GOLDMAN SACHS GROUP, INC. AND SUBSIDIARIES"), and splitter
    # fragments (".", "million and $", "actors") — 618 chunks, none of which
    # can answer anything. Immediately above it the text is real and worth
    # keeping, which is why the floor is not higher: at 200 it would discard
    # sentences like "As of September 27, 2025, the Company had approximately
    # 166,000 full-time equivalent employees."
    min_chunk_chars: int = 80

    # Chunking strategy: "fixed", "recursive", or "semantic" — see the enum for
    # what each does. Changing this only affects new ingestion; an existing
    # index keeps whatever chunks it was built with, so comparing strategies
    # means building one CHROMA_COLLECTION per strategy, not flipping this on
    # a live index. Every chunk records the strategy that produced it, so a
    # collection can always say which one it is.
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    semantic_breakpoint_percentile: float = 95.0
    semantic_buffer_size: int = 1
    # Guard against a uniform section becoming one giant chunk; oversized
    # chunks are re-split with the fixed-size splitter.
    #
    # 1200 was originally forced by all-MiniLM-L6-v2's 256-token window
    # (~1300 chars of filing prose), past which text was silently dropped at
    # embedding time and became unretrievable. text-embedding-3-small takes
    # 8k tokens, so that ceiling is gone — the cap is now a retrieval-quality
    # choice, not a hard limit. Kept at 1200 so the semantic-vs-recursive
    # ablation compares chunking strategy rather than chunk size. Raise it
    # only as its own measured experiment, and only on the OpenAI provider.
    semantic_max_chunk_chars: int = 1200

    # Near-duplicate suppression at ingestion. Filings repeat themselves —
    # boilerplate risk language, the same figure restated in MD&A and again in
    # the notes — and a duplicate chunk is worse than useless: it occupies one
    # of the top_k slots the LLM gets, displacing something new. Cosine
    # similarity against already-indexed chunks; anything above the threshold
    # is skipped and counted.
    #
    # 0.95 is deliberately high. Two chunks discussing the same topic in
    # different words sit well below it; near-identical text sits above. A
    # lower threshold starts discarding genuinely distinct disclosures, which
    # is a silent recall loss — the opposite of the problem being fixed.
    dedup_enabled: bool = True
    dedup_threshold: float = 0.95

    # Retrieval
    retrieval_top_k: int = 5

    # Re-ranking — a cross-encoder scores (query, doc) pairs jointly, which is
    # slower but substantially more precise than the bi-encoder cosine
    # similarity ChromaDB returns. Cheap because it only runs on a small
    # candidate set, not the whole corpus.
    rerank_enabled: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidate_k: int = 20

    # Hybrid search — pairs dense embedding search with a BM25 lexical index
    # and fuses the rankings (RRF). Dense handles paraphrase; BM25 handles the
    # exact-match terms filings are full of (tickers, "Item 7A", dollar
    # figures) that a 384-dim embedding blurs together.
    #
    # Off by default: the ablation in README ("Retrieval Pipeline Ablation")
    # measured it lifting Faithfulness (+0.09) and Precision, but costing
    # Context Recall -0.23 — five times the run-to-run noise floor. Lexical
    # hits crowd semantically-relevant chunks out of a fixed fusion budget.
    # Turn on once the candidate budget or fusion weighting is tuned.
    hybrid_search_enabled: bool = False

    # Per-entity retrieval — when a question names two or more of the five
    # companies in the corpus, retrieve for each of them separately and merge,
    # instead of ranking one global top-k.
    #
    # On by default because a global budget cannot answer a comparison. Every
    # comparative question in the eval set scored exactly 0.000 answer
    # relevancy under a global top-5: the ranker filled all five slots from
    # whichever filing scored better overall, the model correctly said it
    # could not compare, and a refusal has no relevance to the question asked.
    # Each named company gets its own retrieval_top_k slots, so a two-company
    # question sends roughly twice the context — that is the cost of asking
    # two questions at once, and it is paid only on questions that do.
    multi_entity_retrieval_enabled: bool = True

    # HyDE — ask the LLM to write the passage that would answer the question,
    # then embed that instead of the question. Retrieval becomes document-to-
    # document rather than question-to-document. Costs one extra LLM call per
    # query, on the critical path before retrieval.
    hyde_enabled: bool = False
    # Pure HyDE embeds only the hypothetical passage; this keeps the original
    # question alongside it so literal terms (tickers, "Item 7A") aren't lost
    # if the generated passage happens not to echo them.
    hyde_include_query: bool = False

    # Citation verification — pair each cited claim with the chunk it points
    # at and ask a judge whether that chunk supports it. Off at runtime: it
    # costs one LLM call per citation (5-15 per answer) on the critical path,
    # and the signals that feed the confidence score — citation coverage,
    # out-of-range references — come from parsing alone and are always on.
    # The eval harness turns it on, which is where the cost buys something:
    # the citation_accuracy metric.
    citation_verification_enabled: bool = False
    # A different judge from the RAGAS one, and deliberately not pinned the
    # same way. EVAL_JUDGE_MODEL is frozen because every result already in
    # evaluation/results/ was measured with it; this judge is new apparatus
    # with no history to invalidate, so it can be changed on its merits.
    citation_judge_provider: LLMProvider = LLMProvider.OPENAI
    citation_judge_model: str = "gpt-4o-mini"

    # Below this retrieval confidence the system returns a structured account
    # of what it searched instead of generating an answer (Project 6 §3.4).
    #
    # The scale is the normalized relevance in src/retrieval/scores.py. On
    # the default reranked pipeline that is a logistic over cross-encoder
    # logits, which is strongly bimodal — a relevant passage lands near 1.0
    # and an irrelevant one near 0.0 — so the threshold has a wide basin to
    # sit in and 0.15 is conservative. It is not equally calibrated for the
    # plain-dense path, where Chroma relevance scores cluster far lower; the
    # effect there is that the gate rarely fires, which is the safe direction
    # to be wrong in.
    insufficient_context_threshold: float = 0.15

    # RAGAS eval judge — separate from LLM_PROVIDER (the runtime generation
    # model) so switching the app's provider doesn't silently change what
    # judges eval scores. Defaults to OpenAI gpt-4o-mini, matching the
    # historical baseline in evaluation/results/. Override for a run where
    # OpenAI isn't available (e.g. EVAL_JUDGE_PROVIDER=gemini) — just note
    # scores from a different judge aren't directly comparable to the baseline.
    eval_judge_provider: LLMProvider = LLMProvider.OPENAI
    # Pinned to gpt-4o-mini even though runtime generation is gpt-4o. The
    # judge is measurement apparatus: changing it invalidates comparison with
    # every result already in evaluation/results/, and RAGAS issues several
    # judged calls per question per metric, so gpt-4o here would cost more
    # than the runs it is grading.
    eval_judge_model: str = "gpt-4o-mini"
    # Output budget for the judge, separate from the 1024 tokens generation
    # gets. RAGAS faithfulness extracts the statements in an answer and emits
    # a structured verdict for each, so its output scales with the answer
    # length; at 1024 it overran on 4 of 20 questions in the first measured
    # run, and RAGAS drops an overrun row rather than truncating it. The
    # dropped rows are the long multi-claim answers — precisely the ones most
    # at risk of being unfaithful — so the mean was taken over the easy ones
    # and moved depending on which rows happened to overrun. Raising this
    # changes measured faithfulness by fixing coverage, not by changing the
    # judge: see evaluation/results/README.md.
    eval_judge_max_tokens: int = 4096

    # SEC EDGAR
    edgar_user_agent: str = "RAGEnterprise murali140824@gmail.com"
    edgar_rate_limit: int = 10
    edgar_data_dir: str = "./data/edgar"

    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION

    @property
    def project_root(self) -> Path:
        return Path(__file__).parent.parent


settings = Settings()
