from backend.app.rag.bm25 import LexicalIndex
from backend.app.rag.embeddings import ProductionEmbeddingGateway, validate_vectors
from backend.app.rag.fusion import reciprocal_rank_fusion
from backend.app.rag.index import open_index
from backend.app.rag.documents import CORPUS_ROOT
from backend.app.rag.vector_store import STORAGE_ROOT
from backend.app.rag.schemas import PolicySearchInput, PolicySearchResult, PolicyCitation, PolicyHit, RetrievalError

MAX_EXCERPT_CHARS = 800
# Cosine distance gate rejects orthogonal/unrelated hits, not a cross-model score fusion.
MAX_VECTOR_DISTANCE = 0.65


class PolicyRetriever:
    def __init__(self, gateway=None, *, root=CORPUS_ROOT, storage=STORAGE_ROOT):
        self.gateway = gateway or ProductionEmbeddingGateway()
        self.root, self.storage = root, storage

    def search(self, query, top_k=5):
        args = PolicySearchInput(query=query, top_k=top_k)
        try:
            manifest, chunks, store = open_index(self.gateway, self.root, self.storage)
            vector = validate_vectors(self.gateway.embed([args.query]), 1)[0]
            if len(vector) != manifest["embedding_dimension"]:
                raise RetrievalError("INDEX_MODEL_MISMATCH")
            vector_results = [r for r in store.search(manifest["collection"], vector) if r["distance"] <= MAX_VECTOR_DISTANCE]
            lexical_results = LexicalIndex(chunks).search(args.query)
            # Both paths executed successfully; either can legitimately have zero hits.
            fused = reciprocal_rank_fusion(vector_results, lexical_results, args.top_k)
            if not fused:
                raise RetrievalError("NO_RETRIEVAL_RESULTS")
            catalog = {c.chunk_id: c for c in chunks}
            vectors = {r["chunk_id"]: r for r in vector_results}
            lexical = {r["chunk_id"]: r for r in lexical_results}
            hits = []
            for rank, result in enumerate(fused, 1):
                c = catalog[result["chunk_id"]]
                citation = PolicyCitation(policy_evidence_id="POLICY::"+c.chunk_id,
                    **c.model_dump(include={"chunk_id", "doc_id", "title", "section_title", "source_path", "version", "effective_date"}))
                v, b = vectors.get(c.chunk_id, {}), lexical.get(c.chunk_id, {})
                hits.append(PolicyHit(rank=rank, chunk_id=c.chunk_id, doc_id=c.doc_id, title=c.title,
                    section_title=c.section_title, excerpt=c.text[:MAX_EXCERPT_CHARS], vector_rank=v.get("rank"),
                    bm25_rank=b.get("rank"), vector_distance=v.get("distance"), bm25_score=b.get("score"),
                    rrf_score=result["rrf_score"], metadata=dict(department=c.department, policy_type=c.policy_type,
                        content_hash=c.content_hash, excerpt_truncated=len(c.text)>MAX_EXCERPT_CHARS), citation=citation))
            return PolicySearchResult(query=args.query, hits=hits)
        except RetrievalError:
            raise
        except Exception:
            raise RetrievalError("RETRIEVAL_ERROR") from None
