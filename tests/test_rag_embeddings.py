from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, APIStatusError

from backend.app.config import Settings
from backend.app.rag.embeddings import ProductionEmbeddingGateway
from backend.app.rag.schemas import RetrievalError
from tests.test_rag_retrieval import offline


def config(**kwargs):
    return Settings(_env_file=None, LLM_PROVIDER="siliconflow", LLM_API_KEY="fake-test-credential", **kwargs)


def stub_client(monkeypatch, *, error=None, indexes=(1, 0)):
    calls = []

    class Client:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.embeddings = self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def create(self, **kwargs):
            calls.append(kwargs)
            if error:
                raise error
            return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=[float(i), 1.0]) for i in indexes])

    monkeypatch.setattr("backend.app.rag.embeddings.OpenAI", Client)
    return calls


def test_production_protocol_and_same_provider_credential_reuse(monkeypatch):
    calls = stub_client(monkeypatch)
    assert ProductionEmbeddingGateway(config()).embed(["first", "second"]) == [[0.0, 1.0], [1.0, 1.0]]
    assert calls[0]["api_key"] == "fake-test-credential"
    assert calls[0]["max_retries"] == 0
    assert calls[1] == dict(model="BAAI/bge-m3", input=["first", "second"], encoding_format="float")


def test_separate_provider_requires_separate_key(monkeypatch):
    calls = stub_client(monkeypatch)
    with pytest.raises(RetrievalError):
        ProductionEmbeddingGateway(config(EMBEDDING_PROVIDER="other")).embed(["first", "second"])
    assert not calls
    ProductionEmbeddingGateway(config(EMBEDDING_PROVIDER="other", EMBEDDING_API_KEY="fake-embedding-test")).embed(["first", "second"])
    assert calls[0]["api_key"] == "fake-embedding-test"


@pytest.mark.parametrize("status,retryable", [(401, False), (400, False), (429, True), (503, True)])
def test_provider_errors_are_safe_and_not_retried(monkeypatch, status, retryable):
    response = httpx.Response(status, request=httpx.Request("POST", "https://example.invalid/embeddings"))
    calls = stub_client(monkeypatch, error=APIStatusError("unsafe-provider-body", response=response, body=None))
    with pytest.raises(RetrievalError) as caught:
        ProductionEmbeddingGateway(config()).embed(["first", "second"])
    assert str(caught.value) == "EMBEDDING_PROVIDER_ERROR"
    assert caught.value.retryable is retryable
    assert len(calls) == 2


def test_connection_error_classification(monkeypatch):
    stub_client(monkeypatch, error=APIConnectionError(request=httpx.Request("POST", "https://example.invalid")))
    with pytest.raises(RetrievalError) as caught:
        ProductionEmbeddingGateway(config()).embed(["first", "second"])
    assert caught.value.retryable is True


def test_duplicate_vector_indexes_are_rejected(monkeypatch):
    stub_client(monkeypatch, indexes=(0, 0))
    with pytest.raises(RetrievalError, match="EMBEDDING_PROVIDER_ERROR"):
        ProductionEmbeddingGateway(config()).embed(["first", "second"])
