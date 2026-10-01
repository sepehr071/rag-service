import asyncio
import logging
import os
import time
from typing import List

import httpx
from cohere import Client as CohereClient
from llama_cloud_services import LlamaCloudIndex
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle
from llama_index.llms.openai import OpenAI

from app.core.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGService:
    """RAG service that runs the two-stage retrieval pipeline.

    Pipeline shape (accuracy-first, recall -> precision):

        multi-query hybrid fusion retrieval  ->  single global Cohere rerank

    Stage 1 issues a wide, high-recall hybrid (dense+sparse) retrieval across all
    LlamaCloud indexes, expanded with LLM-paraphrased sub-queries and fused via RRF.
    Stage 2 hands the fused candidate pool to ONE strong cross-encoder reranker
    (Cohere) which recovers precision and returns the final ordered top-N.

    There is intentionally no LongContextReorder step: that postprocessor only helps
    when nodes are stuffed into an LLM context window ("lost-in-the-middle"). Here the
    output is a score-ranked JSON list returned to a calling agent, and any ordering it
    produced would be immediately overwritten by the Cohere rerank — it was pure waste.
    """

    def __init__(self) -> None:
        """Initialize retrievers, the fusion layer, and the single reranker once."""
        logger.info("Initializing RAG Service...")

        # Configure LlamaIndex global settings.
        #
        # LlamaDebugHandler is gated behind DEBUG: it records an unbounded in-memory event
        # map and prints a full trace on every op, which leaks memory and spams logs in a
        # long-running server. Off by default for production.
        if Config.DEBUG:
            Settings.callback_manager = CallbackManager(
                [LlamaDebugHandler(print_trace_on_end=True)]
            )

        # The LLM is used by QueryFusionRetriever to generate paraphrased sub-queries.
        Settings.llm = OpenAI(
            model=Config.LLM_MODEL, temperature=Config.LLM_TEMPERATURE
        )
        #
        # NOTE: we deliberately do NOT set Settings.embed_model. For LlamaCloud *managed*
        # retrieval the query is embedded server-side with the model the index was ingested
        # with; a local embed_model is never consulted, so setting it here would be
        # misleading dead config. See Config.EMBEDDING_MODEL for the full explanation.

        # Build one hybrid retriever per LlamaCloud index.
        # Cold-start / flaky egress: retry each index a few times so one ReadTimeout
        # does not take down the whole worker boot.
        logger.info("Building retrievers from LlamaCloud indexes...")
        self.sub_retrievers = []
        for name in Config.INDEX_NAMES:
            last_err: Exception | None = None
            for attempt in range(1, 6):
                try:
                    index = LlamaCloudIndex(name=name, project_name=Config.PROJECT_NAME)
                    retriever = index.as_retriever(
                        retrieval_mode=Config.RETRIEVAL_MODE,
                        dense_similarity_top_k=Config.DENSE_SIMILARITY_TOP_K,
                        sparse_similarity_top_k=Config.SPARSE_SIMILARITY_TOP_K,
                        alpha=Config.ALPHA,
                        # Per-index reranking is OFF: we want raw hybrid candidates to flow into
                        # fusion so the single global Cohere reranker sees the widest pool. Enabling
                        # it here would prematurely truncate each index's results and rerank twice.
                        enable_reranking=Config.ENABLE_RERANKING,
                        rerank_top_n=Config.RERANK_TOP_N,
                    )
                    self.sub_retrievers.append(retriever)
                    logger.info(f"   -> Index '{name}' loaded and retriever ready")
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    logger.warning(
                        "   -> Index '%s' attempt %d/5 failed: %s: %s",
                        name,
                        attempt,
                        type(e).__name__,
                        e,
                    )
                    time.sleep(min(2 * attempt, 8))
            if last_err is not None:
                raise last_err

        # Fuse the per-index retrievers with multi-query RRF.
        self.fusion_retriever = QueryFusionRetriever(
            retrievers=self.sub_retrievers,
            similarity_top_k=Config.FUSION_SIMILARITY_TOP_K,
            num_queries=Config.FUSION_NUM_QUERIES,
            mode=Config.FUSION_MODE,
            use_async=True,
            verbose=Config.DEBUG,
        )

        # Cohere can be routed through an optional HTTPS proxy. We do NOT keep a
        # long-lived shared httpx.Client: httpx clients are not thread-safe, and rerank
        # runs in asyncio.to_thread — a shared client wedges under concurrency.
        self._cohere_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get(
            "https_proxy"
        )
        self._cohere_timeout = float(os.getenv("COHERE_HTTP_TIMEOUT", "25"))
        # Marker used by health checks — client is built per-request.
        self._cohere_client = True
        logger.info(
            "Cohere rerank configured (proxy=%s timeout=%ss)",
            self._cohere_proxy or "none",
            self._cohere_timeout,
        )

        logger.info("RAG Service initialized successfully")

    def _cohere_rerank_proxy_safe(
        self, fused_nodes: List[NodeWithScore], query_bundle: QueryBundle
    ) -> List[NodeWithScore]:
        """Rerank with a small Cohere payload; return full-text nodes with new scores.

        When Cohere is reached through an HTTPS proxy, large rerank bodies can
        hang/time out on that path. We also bypass LlamaIndex's CohereRerank helper
        because it builds document strings with MetadataMode.EMBED (metadata bloat).

        Cap candidate count, truncate plain text only, call Cohere with an explicit
        proxied httpx client + timeout, then map scores back onto original nodes.
        """
        if not fused_nodes:
            return []

        max_docs = max(1, Config.COHERE_MAX_DOCS)
        doc_chars = max(50, Config.COHERE_DOC_CHARS)
        candidates = fused_nodes[:max_docs]
        top_n = min(Config.COHERE_TOP_N, len(candidates))

        documents: List[str] = []
        for nws in candidates:
            # NONE: plain passage text only — EMBED mode injects metadata and balloons
            # the Cohere body until a proxy stalls.
            text = nws.node.get_content(metadata_mode=MetadataMode.NONE) or ""
            if len(text) > doc_chars:
                text = text[:doc_chars]
            documents.append(text)

        payload_chars = sum(len(d) for d in documents)
        logger.info(
            "[Rerank] Cohere payload caps: docs=%d char_limit=%d top_n=%d "
            "payload_chars=%d (of %d fused)",
            len(documents),
            doc_chars,
            top_n,
            payload_chars,
            len(fused_nodes),
        )

        # Fresh client per call (httpx.Client is not thread-safe; rerank runs in a
        # worker thread). Explicit proxy — do not rely on ambient trust_env.
        try:
            with httpx.Client(
                proxy=self._cohere_proxy,
                timeout=httpx.Timeout(self._cohere_timeout),
                trust_env=False,
            ) as http_client:
                client = CohereClient(
                    api_key=Config.COHERE_API_KEY,
                    httpx_client=http_client,
                    timeout=self._cohere_timeout,
                )
                results = client.rerank(
                    model=Config.COHERE_MODEL,
                    query=query_bundle.query_str,
                    documents=documents,
                    top_n=top_n,
                )
        except Exception as e:
            # Proxy to Cohere is flaky under load; return fusion-order candidates so
            # the RAG endpoint still serves rather than 500ing the whole chat path.
            logger.warning(
                "[Rerank] Cohere failed (%s: %s) — returning top %d fusion nodes unranked",
                type(e).__name__,
                e,
                top_n,
            )
            return candidates[:top_n]

        final: List[NodeWithScore] = []
        for result in results.results:
            idx = result.index
            if idx < 0 or idx >= len(candidates):
                continue
            original = candidates[idx]
            original.score = result.relevance_score
            final.append(original)
        return final

    async def retrieve(self, query: str) -> List[NodeWithScore]:
        """Run the retrieve -> rerank pipeline for a query and return ranked nodes.

        Transient LlamaCloud/OpenAI timeouts on multi-index fusion are retried once
        (common under network jitter). Persistent infrastructure failures still
        propagate as 5xx. Cohere failures never 500 — they fall back to fusion order.
        A genuine zero-hit returns an empty list (valid 200).
        """
        logger.info("=" * 100)
        logger.info(f"STARTING ADVANCED RAG RETRIEVAL ({len(query)} chars)")
        logger.info("=" * 100)

        query_bundle = QueryBundle(query_str=query)

        # Stage 1: multi-query hybrid fusion retrieval (high recall).
        logger.info("[Retrieval] Running fusion retriever...")
        fused_nodes: List[NodeWithScore] = []
        last_err: Exception | None = None
        for attempt in range(1, 3):
            try:
                fused_nodes = await self.fusion_retriever.aretrieve(query_bundle)
                last_err = None
                break
            except Exception as e:
                last_err = e
                transient = any(
                    s in type(e).__name__.lower() or s in str(e).lower()
                    for s in ("timeout", "timed out", "temporarily", "connection reset")
                )
                if attempt < 2 and transient:
                    logger.warning(
                        "[Retrieval] fusion attempt %d/2 failed (%s: %s) — retrying",
                        attempt,
                        type(e).__name__,
                        e,
                    )
                    await asyncio.sleep(0.4 * attempt)
                    continue
                raise
        if last_err is not None:
            raise last_err
        logger.info(f"   Retrieved {len(fused_nodes)} nodes after fusion retrieval")

        # Stage 2: single global Cohere rerank (precision). Uses HTTPS_PROXY if set.
        # Cap payload for proxy stability.
        logger.info(f"[Rerank] Running Cohere rerank (top_n={Config.COHERE_TOP_N})...")
        final_nodes = await asyncio.to_thread(
            self._cohere_rerank_proxy_safe, fused_nodes, query_bundle
        )

        logger.info(
            f" FINAL TOP {len(final_nodes)} NODES AFTER ALL STAGES ".center(100, "=")
        )
        for i, node in enumerate(final_nodes, 1):
            file = node.metadata.get("file_name", "unknown")
            score = node.score if node.score is not None else 0.0
            logger.info(f"{i:2d}. [{file}]  score = {score:.5f}")
            logger.debug(f"     {node.get_content().strip()[:600]}...")

        return final_nodes

    def get_health_status(self) -> dict:
        """Liveness + cheap readiness check.

        Does NOT call any external API (so it is safe to hit frequently). It reports whether
        the service finished wiring up its retrievers and reranker; "ready" only when both
        the per-index retrievers and the global reranker were successfully constructed.
        """
        retrievers_ready = bool(getattr(self, "sub_retrievers", None)) and len(
            self.sub_retrievers
        ) == len(Config.INDEX_NAMES)
        reranker_ready = getattr(self, "_cohere_client", None) is not None
        ready = retrievers_ready and reranker_ready

        retrievers_built = len(getattr(self, "sub_retrievers", []) or [])
        return {
            "status": "ready" if ready else "initializing",
            # indexes_loaded = retrievers actually built (satisfies HealthResponse and is
            # more honest than counting configured names). The extra diagnostic keys below
            # are ignored by the HealthResponse model (pydantic extra='ignore').
            "indexes_loaded": retrievers_built,
            "indexes_configured": len(Config.INDEX_NAMES),
            "retrievers_built": retrievers_built,
            "reranker_ready": reranker_ready,
            "index_names": Config.INDEX_NAMES,
        }
