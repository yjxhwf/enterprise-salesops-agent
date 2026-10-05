"""Explicit index build; manifest publishes only a completely written generation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from uuid import uuid4
from backend.app.rag import chunking
from backend.app.rag.documents import CORPUS_ROOT, load_documents
from backend.app.rag.embeddings import ProductionEmbeddingGateway, validate_vectors
from backend.app.rag.schemas import RetrievalError, PolicyChunk
from backend.app.rag.vector_store import STORAGE_ROOT, VectorStore

INDEX_VERSION = "policy-index-v1"


def signature(documents, gateway):
    return dict(index_version=INDEX_VERSION, chunking_version=chunking.CHUNKING_VERSION,
        max_chunk_chars=chunking.MAX_CHUNK_CHARS, overlap_chars=chunking.CHUNK_OVERLAP_CHARS,
        embedding_provider=gateway.provider, embedding_model=gateway.model,
        documents=[dict(doc_id=d.doc_id, content_hash=d.content_hash, source_path=d.source_path) for d in documents])


def open_index(gateway, root=CORPUS_ROOT, storage=STORAGE_ROOT):
    storage = Path(storage)
    try:
        manifest = json.loads((storage/"manifest.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise RetrievalError("KNOWLEDGE_INDEX_NOT_READY") from None
    except (ValueError, OSError):
        raise RetrievalError("RETRIEVAL_ERROR") from None
    if not isinstance(manifest, dict):
        raise RetrievalError("RETRIEVAL_ERROR")
    if manifest.get("embedding_model") != gateway.model or manifest.get("embedding_provider") != gateway.provider:
        raise RetrievalError("INDEX_MODEL_MISMATCH")
    try:
        expected = signature(load_documents(root), gateway)
        if any(manifest.get(k) != v for k,v in expected.items()):
            raise RetrievalError("INDEX_REBUILD_REQUIRED")
        filename = manifest["chunks_file"]
        if Path(filename).name != filename:
            raise RetrievalError("RETRIEVAL_ERROR")
        raw = (storage/filename).read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest["chunks_hash"]:
            raise RetrievalError("INDEX_REBUILD_REQUIRED")
        chunks = [PolicyChunk.model_validate(c) for c in json.loads(raw)]
        store = VectorStore(storage)
        collection = store.open(manifest["collection"])
        if collection.count() != manifest["chunk_count"] or set(collection.get()["ids"]) != {c.chunk_id for c in chunks}:
            raise RetrievalError("INDEX_REBUILD_REQUIRED")
        return manifest, chunks, store
    except RetrievalError:
        raise
    except Exception:
        raise RetrievalError("INDEX_REBUILD_REQUIRED") from None


def build_index(gateway, *, root=CORPUS_ROOT, storage=STORAGE_ROOT, rebuild=False):
    storage = Path(storage)
    if (storage/"manifest.json").exists() and not rebuild:
        return open_index(gateway, root, storage)[0]
    documents = load_documents(root)
    chunks = chunking.chunk_documents(documents)
    vectors = validate_vectors(gateway.embed([c.text for c in chunks]), len(chunks))
    storage.mkdir(parents=True, exist_ok=True)
    generation = "sales-policy-"+uuid4().hex
    store = VectorStore(storage)
    collection = store.create(generation, chunks, vectors)
    if collection.count() != len(chunks):
        raise RetrievalError("RETRIEVAL_ERROR")
    raw = json.dumps([c.model_dump(mode="json") for c in chunks], ensure_ascii=False).encode()
    filename = generation+".json"
    (storage/filename).write_bytes(raw)
    manifest = dict(**signature(documents, gateway), collection=generation,
        chunk_count=len(chunks), vector_count=collection.count(), embedding_dimension=len(vectors[0]),
        chunks_file=filename, chunks_hash=hashlib.sha256(raw).hexdigest(), built_at=datetime.now(timezone.utc).isoformat())
    temporary = storage/(generation+".manifest.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(storage/"manifest.json")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Explicitly build synthetic enterprise policy index")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    try:
        result = build_index(ProductionEmbeddingGateway(), rebuild=args.rebuild)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as error:
        print(json.dumps({"success": False, "code": error.code if isinstance(error, RetrievalError) else "INDEX_BUILD_ERROR"}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
