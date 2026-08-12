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
    # Recursive splitting plus a post-pass that re-attaches to each table
    # chunk the words the split severed from it: the caption sentence
    # stranded in the previous chunk's tail, and the filing's provenance.
    # Exists because the ground-truth figures for Goldman's market-risk
    # question sit in bare tables ("CET1 capital | $ | 104,297 ...") whose
    # captions — "The table below presents inf" — end mid-word in the chunk
    # before, leaving the table with no thematic vocabulary for an embedder
    # to match against a question. See src/ingestion/table_context.py.
    TABLE_CONTEXT = "table_context"


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
    #
    # Enforced per client IP, on POST /query and the two auth routes only.
    # Deliberately not a blanket default limit: the landing page pulls three
    # static assets plus a favicon per load, so a global 20/minute would spend
    # a visitor's budget on stylesheets and lock them out of the demo they came
    # to try. What is worth limiting is what costs money or can be brute
    # forced, and that is these three routes.
    rate_limit: str = "20/minute"
    # Login and registration. A different surface: this is where a password is
    # guessed at, not where money is spent.
    #
    # Higher than /query rather than lower, which looks backwards until you
    # watch someone use the demo. The landing page logs in once per role-button
    # click and switching roles *is* the demonstration — it is how a visitor
    # sees the information barriers move. Five roles, clicked through a couple
    # of times while comparing what each can retrieve, is ordinary use and must
    # not trip anything. Meanwhile the demo credentials are published on that
    # same page, so throttling guesses at *those* accounts protects nothing;
    # what this is worth keeping for is any account that is not one of the five.
    auth_rate_limit: str = "30/minute"
    # Escape hatch for tests and for anyone running locally who does not want
    # to think about it. The image leaves it on.
    rate_limit_enabled: bool = True

    # A coupling worth knowing about before it surprises someone: slowapi's
    # default storage is in-memory and per process, so these numbers are per
    # *instance*, not per service. The Cloud Run service runs
    # `autoscaling.knative.dev/maxScale = 1`, which is the only reason the
    # limit above is exact rather than approximate.
    #
    # Raising maxScale to N silently multiplies every limit here by N, because
    # each instance counts on its own and a visitor's requests land wherever
    # the load balancer sends them. Nothing errors and nothing logs; the limit
    # just quietly stops meaning what it says. If maxScale ever goes above 1,
    # the limiter needs shared storage (`storage_uri=` on the Limiter, backed
    # by Redis or similar) — which is a bigger change than raising the number
    # and should not be discovered afterwards.

    # Ceiling on POST /documents/ingest, in bytes. UploadFile spools to disk
    # past a threshold, so an unbounded upload is a disk-fill rather than a
    # memory exhaustion — slower to notice and no less effective. Ingest is off
    # in the deployment (see allow_runtime_ingest below), so this guards the
    # from-source and compose paths.
    max_upload_bytes: int = 10 * 1024 * 1024

    # Whether POST /documents/ingest may write to the vector store.
    #
    # False in the image (see Dockerfile). The deployment ships a corpus
    # verified to a specific digest and the demo's admin credentials are
    # published — on the landing page, as a button — so anyone could otherwise
    # add documents to the index the README's numbers describe. A demo whose
    # corpus a visitor can edit cannot honestly claim a fingerprint.
    #
    # True by default, so running from source (`make dev`) keeps document
    # upload. The image sets it false, which means the compose stack is
    # read-only too — set ALLOW_RUNTIME_INGEST=true there if you need uploads.
    allow_runtime_ingest: bool = True

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
    # Off by default: measured against dense + rerank on the same corpus, it
    # lost on all four metrics with every delta at or inside the n=20 noise
    # floor. That supports "not established as helpful" — not "hybrid is
    # worse", which one run at n=20 cannot support either way.
    #
    # This comment used to cite a Context Recall cost of -0.23, "five times the
    # noise floor", from the README ablation. **That claim was retracted** — it
    # did not reproduce on the fixed corpus, and the ablation table it came
    # from is itself obsolete (measured on 9,572 chunks). It survived here
    # after being withdrawn everywhere else, which is the hazard of putting a
    # measurement in a comment: nothing re-runs it.
    #
    # Turn on once the candidate budget or fusion weighting is tuned.
    hybrid_search_enabled: bool = False

    # How much each retriever counts for in the RRF sum. Only the ratio
    # matters — the output is a ranking, so scaling both weights scales every
    # score identically and reorders nothing.
    #
    # Both default to 1.0 rather than to the 0.7/0.3 Project 6 suggests,
    # because the hybrid-vs-dense comparison in the README was measured at
    # equal weighting. Shipping a different default would leave the one
    # measurement of this pipeline describing a configuration the code no
    # longer runs, and the result would read as a tuning gain rather than as
    # an unmeasured change. 0.7/0.3 is a reasonable first thing to try; it is
    # a starting point for an experiment, not a known-better setting.
    #
    # Raising the dense weight favours paraphrase, raising the sparse weight
    # favours exact matches — tickers, "Item 7A", dollar figures — which is
    # the trade this corpus actually presents.
    hybrid_dense_weight: float = 1.0
    hybrid_sparse_weight: float = 1.0

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

    # Load the cross-encoder at startup instead of on the first query that
    # needs it. Both services scale to zero to stay inside the budget, so a
    # visitor arriving after an idle period pays the cold start: measured
    # against the live deployment, 50.9s for the landing page (image pull and
    # boot) and then 23.1s for their first query, against 0.10s and 3.4s once
    # warm. Nearly all of that 23s is this model.
    #
    # Off by default so the test suite and local runs do not load a model they
    # may never use. The Dockerfile turns it on, the same way it pins
    # ALLOW_RUNTIME_INGEST, so the property cannot be lost by a deploy that
    # forgets a flag.
    #
    # This only moves the cost off the first *query*. The container boot itself
    # is unchanged, and the thing that hides that from visitors is keeping an
    # instance warm — a scheduled ping, not min-instances, which would cost
    # ~$50/month against a $10 ceiling.
    warm_models_on_startup: bool = False

    # Below this retrieval confidence the system returns a structured account
    # of what it searched instead of generating an answer (Project 6 §3.4).
    #
    # The scale is the normalized relevance in src/retrieval/scores.py. On the
    # default reranked pipeline that is a logistic over a cross-encoder logit.
    #
    # This was 0.15 on the reasoning that the logit is bimodal, so the gate had
    # a wide basin to sit in. Measured, it does not: what the cross-encoder
    # scores is *topic* match, and it has no notion of which company a passage
    # is about. On the 49-item held-out probe set in
    # evaluation/datasets/gate_calibration_v1.json (run scripts/calibrate_gate.py):
    #
    #   Citigroup's CET1 ratio          0.9750   out of corpus
    #   Bank of America net interest    0.8994   out of corpus
    #   the current price of Bitcoin    0.2962   out of corpus
    #   ...
    #   Microsoft's dividend            0.0031   in corpus, answerable
    #
    # Ten of 24 out-of-corpus probes cleared 0.15, because the corpus does
    # discuss CET1 ratios and net interest income — for other banks. Meanwhile
    # 0.15 refused questions retrieving context that scores 1.000 context
    # recall. The two labels overlap across almost the whole range, so no
    # threshold on this number separates answerable from unanswerable.
    #
    # What settles the value is scripts/probe_refusal.py: with the gate open,
    # the model refused 24 of 24 out-of-corpus questions on its own, including
    # the one scoring 0.975. The gate was never what made those refusals
    # correct. So it stops being a correctness mechanism and becomes a cost
    # guard — decline to pay for a generation that is certain to be a refusal —
    # and the binding constraint on the number is that it must never be the
    # reason an answerable question is declined.
    #
    # 0.001 is the most it can be while satisfying that: the lowest in-corpus
    # probe scored 0.0031, and this leaves a margin below it. It still
    # short-circuits 12 of the 24 out-of-corpus probes without an LLM call.
    # Raising it trades answerable questions for savings on refusals the model
    # already gets right.
    insufficient_context_threshold: float = 0.001

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
