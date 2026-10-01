import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration for the RAG API"""

    # API Keys
    LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    APP_API_KEY = os.getenv("APP_API_KEY")

    # LlamaCloud Index Configuration
    INDEX_NAMES = os.getenv("INDEX_NAMES", "docs-1,docs-2,docs-3").split(
        ","
    )
    PROJECT_NAME = os.getenv("PROJECT_NAME", "Default")

    # Retrieval Configuration (per LlamaCloud managed index)
    #
    # ACCURACY (R3): per-index reranking is DISABLED by default. With three indexes
    # and multi-query fusion we issue (indexes x queries) managed-retrieval calls; with
    # enable_reranking=True each call previously truncated to RERANK_TOP_N (30) *before*
    # fusion, discarding candidates Cohere never got to see and re-ranking redundantly.
    # We want maximum recall flowing into the single global reranker, so we keep the raw
    # hybrid (dense+sparse) candidates here and let Cohere be the only reranker downstream.
    # (LlamaCloud's own composite "FULL" mode follows the same shape: query each index,
    # then globally rerank once at the end.)
    RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "chunks")
    DENSE_SIMILARITY_TOP_K = int(os.getenv("DENSE_SIMILARITY_TOP_K", "50"))
    SPARSE_SIMILARITY_TOP_K = int(os.getenv("SPARSE_SIMILARITY_TOP_K", "50"))
    ALPHA = float(os.getenv("ALPHA", "0.35"))
    ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "False").lower() == "true"
    # Only consulted when ENABLE_RERANKING is true; kept env-overridable for experiments.
    RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "30"))

    # Fusion Retriever Configuration
    #
    # FUSION_NUM_QUERIES (R5): 4 = 1 original query + 3 LLM-generated paraphrases. Multi-query
    # fusion lifts recall (each paraphrase surfaces chunks the original phrasing misses), which
    # is the dominant goal here. 4 is the documented sweet spot in LlamaIndex's reciprocal-rerank
    # fusion guide; going higher (e.g. 6) mostly adds LLM latency for diminishing recall gains.
    #
    # FUSION_SIMILARITY_TOP_K (R3): this is the candidate pool handed to the single Cohere
    # reranker. Reranker-depth ablations show recall keeps climbing up to ~50+ candidates, and
    # Cohere rerank is cheap/fast, so we cast a wide net (72) and let the cross-encoder recover
    # precision. Must stay comfortably above COHERE_TOP_N or the final reranker has nothing to
    # choose from.
    FUSION_SIMILARITY_TOP_K = int(os.getenv("FUSION_SIMILARITY_TOP_K", "72"))
    FUSION_NUM_QUERIES = int(os.getenv("FUSION_NUM_QUERIES", "4"))
    FUSION_MODE = os.getenv("FUSION_MODE", "reciprocal_rerank")

    # Cohere Rerank Configuration
    #
    # The single, global cross-encoder reranker (R3). It is the highest-impact precision stage in
    # the pipeline: it re-scores the fused candidate pool by deep query-document relevance and
    # returns the final ordered top-N. rerank-v4 is the strongest current Cohere reranker.
    COHERE_MODEL = os.getenv("COHERE_MODEL", "rerank-v4.0-pro")
    COHERE_TOP_N = int(os.getenv("COHERE_TOP_N", "20"))
    # Cohere payload caps. Only the first COHERE_MAX_DOCS fused candidates are
    # sent, each truncated to COHERE_DOC_CHARS for the API call; full node text
    # is restored in the response. Defaults favour accuracy (wider pool + enough
    # chars for the cross-encoder); dial down only if an HTTPS proxy stalls.
    COHERE_MAX_DOCS = int(os.getenv("COHERE_MAX_DOCS", "40"))
    COHERE_DOC_CHARS = int(os.getenv("COHERE_DOC_CHARS", "1000"))

    # OpenAI Configuration
    #
    # LLM is used by QueryFusionRetriever to generate the paraphrased sub-queries (R5) — keep it.
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    #
    # EMBEDDING_MODEL / EMBEDDING_DIMENSIONS (R4): these DO NOT affect managed retrieval. With
    # LlamaCloud managed indexes the query embedding is computed server-side using the model the
    # index was *ingested* with; that model is fixed at ingestion time and cannot be overridden
    # from the client. These vars are retained only for documentation/parity and have no effect
    # on retrieval accuracy — changing them here changes nothing. To change retrieval embeddings
    # you must re-ingest the index with a different embedding model in LlamaCloud.
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "3072"))

    # Retrieval guardrails
    MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "2000"))

    # Debugging
    # Gates the verbose LlamaDebugHandler callback. Defaults OFF: in a long-running server the
    # handler records an unbounded in-memory event map (README notes 2-4 GB/worker growth) and
    # prints a full event trace on every operation. Enable only for local debugging.
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"

    # Workflow Configuration
    WORKFLOW_TIMEOUT = int(os.getenv("WORKFLOW_TIMEOUT", "180"))

    # API Configuration
    API_TITLE = "RAG Retrieval API"
    API_DESCRIPTION = "Advanced RAG retrieval system using LlamaIndex with fusion retrieval and Cohere reranking"
    API_VERSION = "1.0.0"
    API_PREFIX = os.getenv("API_PREFIX", "/rag")
