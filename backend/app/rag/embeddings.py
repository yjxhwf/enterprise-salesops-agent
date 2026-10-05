"""Backend-only credentials; no hidden retry and no Chat runtime accounting."""
import hashlib
import math
import re
from typing import Protocol
from urllib.parse import urlsplit
from openai import OpenAI, APIConnectionError, APITimeoutError, APIStatusError
from backend.app.config import get_settings
from backend.app.rag.schemas import RetrievalError


class EmbeddingGateway(Protocol):
    provider: str
    model: str
    def embed(self, texts: list[str]) -> list[list[float]]: ...


def validate_vectors(vectors, count):
    if len(vectors) != count or not vectors or not vectors[0]:
        raise RetrievalError("EMBEDDING_PROVIDER_ERROR")
    size = len(vectors[0])
    if any(len(v) != size or any(not math.isfinite(x) for x in v) or not any(v) for v in vectors):
        raise RetrievalError("EMBEDDING_PROVIDER_ERROR")
    return vectors


class ProductionEmbeddingGateway:
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.provider, self.model = self.settings.EMBEDDING_PROVIDER, self.settings.EMBEDDING_MODEL

    def embed(self, texts):
        config = self.settings
        credential = config.EMBEDDING_API_KEY
        if not credential or not credential.get_secret_value().strip():
            credential = config.LLM_API_KEY if self.provider.lower() == config.LLM_PROVIDER.lower() else None
        parsed = urlsplit(config.EMBEDDING_BASE_URL)
        if not credential or not credential.get_secret_value().strip() or not self.model or parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise RetrievalError("EMBEDDING_PROVIDER_ERROR")
        try:
            with OpenAI(api_key=credential.get_secret_value(), base_url=config.EMBEDDING_BASE_URL, max_retries=0, timeout=6) as client:
                response = client.embeddings.create(model=self.model, input=texts, encoding_format="float")
            data = sorted(response.data, key=lambda item: item.index)
            if [v.index for v in data] != list(range(len(texts))):
                raise RetrievalError("EMBEDDING_PROVIDER_ERROR")
            return validate_vectors([item.embedding for item in data], len(texts))
        except (APIConnectionError, APITimeoutError):
            raise RetrievalError("EMBEDDING_PROVIDER_ERROR", retryable=True) from None
        except APIStatusError as error:
            raise RetrievalError("EMBEDDING_PROVIDER_ERROR", retryable=error.status_code == 429 or error.status_code >= 500) from None
        except RetrievalError:
            raise
        except Exception:
            raise RetrievalError("EMBEDDING_PROVIDER_ERROR") from None


class FakeEmbeddingGateway:
    """Deterministic offline feature hash, not a production semantic model."""
    provider, model = "fake", "deterministic-hash-v1"

    def embed(self, texts):
        vectors = []
        for text in texts:
            vector = [0.0]*256
            for token in re.findall(r"[\u4e00-\u9fff]|[a-z0-9-]+", text.lower()):
                index = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big") % len(vector)
                vector[index] += 1
            norm = math.sqrt(sum(x*x for x in vector))
            vectors.append([x/norm for x in vector] if norm else [1.0]+vector[1:])
        return vectors
