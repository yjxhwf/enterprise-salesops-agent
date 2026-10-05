"""Explicit vectors only. No default embedding function, telemetry or model downloads."""
from pathlib import Path
import chromadb
from chromadb.config import Settings

STORAGE_ROOT = Path(__file__).resolve().parents[2] / "storage" / "chroma"
VECTOR_TOP_K = 8


class VectorStore:
    def __init__(self, path=STORAGE_ROOT):
        self.client = chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))

    def create(self, name, chunks, vectors):
        collection = self.client.create_collection(name, embedding_function=None, metadata={"hnsw:space": "cosine"})
        collection.add(ids=[c.chunk_id for c in chunks], embeddings=vectors)
        return collection

    def open(self, name):
        return self.client.get_collection(name, embedding_function=None)

    def search(self, name, vector, top_k=VECTOR_TOP_K):
        collection = self.open(name)
        if not collection.count():
            return []
        result = collection.query(query_embeddings=[vector], n_results=min(top_k, collection.count()), include=["distances"])
        pairs = sorted(zip(result["ids"][0], result["distances"][0]), key=lambda x: (x[1], x[0]))
        return [dict(chunk_id=cid, distance=float(distance), rank=i) for i, (cid, distance) in enumerate(pairs, 1)]
