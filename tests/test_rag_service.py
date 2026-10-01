"""Unit tests for the retrieval pipeline logic. All external calls are mocked."""

import asyncio
from types import SimpleNamespace

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from app.core.config import Config
from app.services import rag_service as rs


def make_nodes(n, text_len=10):
    return [
        NodeWithScore(
            node=TextNode(text=f"{i}" * text_len, metadata={"file_name": f"doc{i}.md"}),
            score=1.0 / (i + 1),
        )
        for i in range(n)
    ]


def bare_service():
    # Skip __init__ (it talks to LlamaCloud); set only what the methods use.
    svc = rs.RAGService.__new__(rs.RAGService)
    svc._cohere_proxy = None
    svc._cohere_timeout = 5.0
    return svc


class FakeCohere:
    calls = []
    results = []
    error = None

    def __init__(self, **kwargs):
        pass

    def rerank(self, **kwargs):
        FakeCohere.calls.append(kwargs)
        if FakeCohere.error:
            raise FakeCohere.error
        return SimpleNamespace(results=FakeCohere.results)


@pytest.fixture
def cohere(monkeypatch):
    FakeCohere.calls, FakeCohere.results, FakeCohere.error = [], [], None
    monkeypatch.setattr(rs, "CohereClient", FakeCohere)
    monkeypatch.setattr(Config, "COHERE_MAX_DOCS", 3)
    monkeypatch.setattr(Config, "COHERE_DOC_CHARS", 50)
    monkeypatch.setattr(Config, "COHERE_TOP_N", 2)
    return FakeCohere


def q(text="warranty?"):
    return rs.QueryBundle(query_str=text)


def test_rerank_reorders_by_cohere_and_drops_bad_indexes(cohere):
    nodes = make_nodes(5)
    cohere.results = [
        SimpleNamespace(index=2, relevance_score=0.9),
        SimpleNamespace(index=7, relevance_score=0.8),  # out of candidate range
        SimpleNamespace(index=0, relevance_score=0.4),
    ]
    out = bare_service()._cohere_rerank_proxy_safe(nodes, q())

    assert [n.metadata["file_name"] for n in out] == ["doc2.md", "doc0.md"]
    assert [n.score for n in out] == [0.9, 0.4]


def test_rerank_caps_docs_truncates_text_and_top_n(cohere):
    nodes = make_nodes(5, text_len=80)
    bare_service()._cohere_rerank_proxy_safe(nodes, q("x"))

    call = cohere.calls[0]
    assert len(call["documents"]) == 3  # COHERE_MAX_DOCS
    assert all(len(d) == 50 for d in call["documents"])  # COHERE_DOC_CHARS
    assert call["top_n"] == 2
    assert call["query"] == "x"
    # Full text is kept on the node itself; only the API payload is truncated.
    assert len(nodes[0].node.get_content()) == 80


def test_rerank_top_n_never_exceeds_candidates(cohere, monkeypatch):
    monkeypatch.setattr(Config, "COHERE_TOP_N", 20)
    bare_service()._cohere_rerank_proxy_safe(make_nodes(2), q())
    assert cohere.calls[0]["top_n"] == 2


def test_rerank_failure_falls_back_to_fusion_order(cohere):
    cohere.error = TimeoutError("proxy stalled")
    nodes = make_nodes(5)
    out = bare_service()._cohere_rerank_proxy_safe(nodes, q())
    assert out == nodes[:2]


def test_rerank_empty_input_skips_cohere(cohere):
    assert bare_service()._cohere_rerank_proxy_safe([], q()) == []
    assert cohere.calls == []


class FakeFusion:
    def __init__(self, errors, result):
        self.errors, self.result, self.calls = list(errors), result, 0

    async def aretrieve(self, query_bundle):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return self.result


def service_with_fusion(fusion):
    svc = bare_service()
    svc.fusion_retriever = fusion
    svc._cohere_rerank_proxy_safe = lambda nodes, qb: nodes  # identity rerank
    return svc


def test_retrieve_retries_once_on_transient_error():
    nodes = make_nodes(3)
    fusion = FakeFusion([TimeoutError("read timed out")], nodes)
    out = asyncio.run(service_with_fusion(fusion).retrieve("q"))
    assert out == nodes
    assert fusion.calls == 2


def test_retrieve_does_not_retry_non_transient_error():
    fusion = FakeFusion([ValueError("bad index")], [])
    with pytest.raises(ValueError):
        asyncio.run(service_with_fusion(fusion).retrieve("q"))
    assert fusion.calls == 1


def test_retrieve_gives_up_after_second_transient_error():
    fusion = FakeFusion([TimeoutError("t1"), TimeoutError("t2")], [])
    with pytest.raises(TimeoutError):
        asyncio.run(service_with_fusion(fusion).retrieve("q"))
    assert fusion.calls == 2


def test_health_status_ready_only_when_all_indexes_built(monkeypatch):
    monkeypatch.setattr(Config, "INDEX_NAMES", ["a", "b"])
    svc = bare_service()
    svc._cohere_client = True
    svc.sub_retrievers = [object()]
    assert svc.get_health_status()["status"] == "initializing"
    svc.sub_retrievers = [object(), object()]
    assert svc.get_health_status()["status"] == "ready"
