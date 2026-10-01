"""API tests: auth, validation, rate limit and response mapping, with a fake RAG service."""

import pytest
from fastapi.testclient import TestClient
from llama_index.core.schema import NodeWithScore, TextNode

from app.core.config import Config
from app.dependencies import set_rag_service
from app.main import app
from app.routers import rag as rag_router

KEY = "test-key"


class FakeService:
    def __init__(self, nodes=None):
        self.nodes = nodes or []
        self.queries = []

    async def retrieve(self, query):
        self.queries.append(query)
        return self.nodes

    def get_health_status(self):
        return {"status": "ready", "indexes_loaded": 2, "index_names": ["a", "b"]}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(Config, "APP_API_KEY", KEY)
    monkeypatch.setattr(Config, "DEBUG", False)
    rag_router._rate_state.clear()
    svc = FakeService(
        [
            NodeWithScore(node=TextNode(text="hit", metadata={"file_name": "a.md"}), score=0.8),
            NodeWithScore(node=TextNode(text="no score"), score=None),
        ]
    )
    set_rag_service(svc)
    # No `with`: lifespan (which connects to LlamaCloud) is not run.
    c = TestClient(app)
    c.svc = svc
    yield c
    set_rag_service(None)


def post(client, body, key=KEY):
    headers = {"X-API-Key": key} if key else {}
    return client.post(f"{Config.API_PREFIX}/query", json=body, headers=headers)


def test_query_maps_nodes_and_strips_query(client):
    r = post(client, {"query": "  warranty?  "})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 2
    assert body["query"] == "warranty?"
    assert body["nodes"][0] == {"text": "hit", "score": 0.8, "file_name": "a.md"}
    assert body["nodes"][1]["score"] == 0.0  # None score -> 0.0
    assert body["nodes"][1]["file_name"] is None
    assert client.svc.queries == ["warranty?"]


@pytest.mark.parametrize("key,status", [(None, 403), ("wrong", 403)])
def test_query_requires_valid_api_key(client, key, status):
    assert post(client, {"query": "q"}, key=key).status_code == status


def test_query_fails_closed_without_server_key(client, monkeypatch):
    monkeypatch.setattr(Config, "APP_API_KEY", None)
    assert post(client, {"query": "q"}).status_code == 503


@pytest.mark.parametrize(
    "body,status",
    [
        ({}, 422),
        ({"query": ""}, 422),
        ({"query": "x" * (Config.MAX_QUERY_LENGTH + 1)}, 422),
        ({"query": "   "}, 400),
    ],
)
def test_query_validation(client, body, status):
    assert post(client, body).status_code == status
    assert client.svc.queries == []


def test_rate_limit_returns_429_with_retry_after(client, monkeypatch):
    monkeypatch.setattr(rag_router, "_RATE_LIMIT_PER_MINUTE", 2)
    assert post(client, {"query": "q"}).status_code == 200
    assert post(client, {"query": "q"}).status_code == 200
    r = post(client, {"query": "q"})
    assert r.status_code == 429
    assert int(r.headers["Retry-After"]) >= 1


def test_internal_error_is_not_leaked(client):
    async def boom(query):
        raise RuntimeError("secret upstream detail")

    client.svc.retrieve = boom
    r = post(client, {"query": "q"})
    assert r.status_code == 500
    assert "secret" not in r.text
    assert "correlation_id=" in r.text


def test_health_hides_index_names_unless_debug(client, monkeypatch):
    url = f"{Config.API_PREFIX}/health"
    assert client.get(url).json() == {"status": "ready", "indexes_loaded": 2}
    monkeypatch.setattr(Config, "DEBUG", True)
    assert client.get(url).json()["index_names"] == ["a", "b"]
