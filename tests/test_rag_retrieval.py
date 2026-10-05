import json
from pathlib import Path
import socket
import shutil
import pytest
from backend.app.rag.documents import CORPUS_ROOT, load_documents
from backend.app.rag.chunking import chunk_documents
from backend.app.rag.embeddings import FakeEmbeddingGateway, ProductionEmbeddingGateway
from backend.app.rag.index import build_index, open_index
from backend.app.rag.retriever import PolicyRetriever
from backend.app.rag.bm25 import LexicalIndex
from backend.app.rag.fusion import reciprocal_rank_fusion
from backend.app.rag.schemas import RetrievalError


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args,**kwargs): raise AssertionError('Offline RAG tests cannot use network')
    monkeypatch.setattr(socket.socket,'connect',forbidden)


@pytest.fixture
def rag_index(tmp_path):
    gateway=FakeEmbeddingGateway()
    storage=tmp_path/'chroma'
    manifest=build_index(gateway,storage=storage)
    yield gateway,storage,manifest
    # Chroma embedded client lifecycle; stop test-local server before tmpdir cleanup on Windows.
    from chromadb.api.shared_system_client import SharedSystemClient
    system=SharedSystemClient._identifier_to_system.get(str(storage))
    if system:
        system.stop()
        SharedSystemClient._identifier_to_system.pop(str(storage),None)
    shutil.rmtree(storage)


def test_hybrid_reference_recall_and_real_excerpts(rag_index):
    gateway,storage,m=rag_index
    retriever=PolicyRetriever(gateway,storage=storage)
    chunks={c.chunk_id:c for c in chunk_documents(load_documents())}
    cases=json.loads(Path('tests/fixtures/rag_queries.json').read_text(encoding='utf-8'))
    hits=0
    for case in cases:
        if case['expected_doc_id'] is None:
            with pytest.raises(RetrievalError,match='NO_RETRIEVAL_RESULTS'):retriever.search(case['query'])
            continue
        result=retriever.search(case['query'])
        assert result.retrieval_mode=='HYBRID'
        assert case['expected_doc_id'] in [h.doc_id for h in result.hits[:3]]
        hits+=1
        for h in result.hits:
            assert h.excerpt in chunks[h.chunk_id].text and len(h.excerpt)<=800
            assert h.citation.policy_evidence_id=='POLICY::'+h.chunk_id
    assert hits==10
    lexical=LexicalIndex(list(chunks.values())).search('POL-SALES-001')
    assert chunks[lexical[0]['chunk_id']].doc_id=='POL-SALES-001'


def test_idempotence_and_explicit_rebuild(rag_index):
    gateway,storage,m=rag_index
    first=open_index(gateway,storage=storage)[1]
    assert build_index(gateway,storage=storage)==m
    second=build_index(gateway,storage=storage,rebuild=True)
    assert second['chunk_count']==m['chunk_count']==12
    assert [c.chunk_id for c in first]==[c.chunk_id for c in open_index(gateway,storage=storage)[1]]


@pytest.mark.parametrize('change',['model','chunking','corpus','stored_chunks'])
def test_index_mismatch_requires_rebuild(rag_index,monkeypatch,tmp_path,change):
    gateway,storage,m=rag_index
    root=CORPUS_ROOT
    if change=='model': gateway.model='other-model'
    elif change=='chunking':
        from backend.app.rag import chunking
        monkeypatch.setattr(chunking,'CHUNKING_VERSION','changed')
    elif change=='corpus':
        root=tmp_path/'corpus';shutil.copytree(CORPUS_ROOT,root)
        path=next(root.glob('*.md'));path.write_text(path.read_text(encoding='utf-8')+'\n新条款',encoding='utf-8')
    else: (storage/m['chunks_file']).write_text('[]',encoding='utf-8')
    with pytest.raises(RetrievalError,match='INDEX_MODEL_MISMATCH|INDEX_REBUILD_REQUIRED'):
        open_index(gateway,root,storage)


def test_no_index_does_not_call_embeddings(tmp_path):
    class Forbidden(FakeEmbeddingGateway):
        def embed(self,texts):raise AssertionError('No implicit rebuild')
    with pytest.raises(RetrievalError,match='KNOWLEDGE_INDEX_NOT_READY'):
        PolicyRetriever(Forbidden(),storage=tmp_path/'absent').search('折扣审批')


def test_embedding_failure_no_lexical_fallback(rag_index):
    gateway,storage,m=rag_index
    class Failed(FakeEmbeddingGateway):
        def embed(self,texts):raise RetrievalError('EMBEDDING_PROVIDER_ERROR',retryable=True)
    with pytest.raises(RetrievalError,match='EMBEDDING_PROVIDER_ERROR'):
        PolicyRetriever(Failed(),storage=storage).search('折扣审批')


def test_rrf_rank_only_tiebreak():
    a=[dict(chunk_id='b',rank=1,score=999),dict(chunk_id='a',rank=2,score=0)]
    b=[dict(chunk_id='a',rank=1,distance=100),dict(chunk_id='b',rank=2,distance=-100)]
    result=reciprocal_rank_fusion(a,b)
    assert [r['chunk_id'] for r in result]==['a','b']
    assert result[0]['rrf_score']==1/61+1/62


def test_semantic_paraphrase_controlled_fake(tmp_path):
    # Test-only semantic oracle. Production does not read this mapping or test fixtures.
    class Controlled(FakeEmbeddingGateway):
        model='test-semantic-v1'
        def embed(self,texts):
            return [[1.,0.] if '利润率与低毛利销售治理' in t or t=='营收热闹却赚得不多该看什么' else [0.,1.] for t in texts]
    gateway=Controlled();storage=tmp_path/'semantic'
    m=build_index(gateway,storage=storage)
    manifest,chunks,store=open_index(gateway,storage=storage)
    results=store.search(m['collection'],gateway.embed(['营收热闹却赚得不多该看什么'])[0])
    catalog={c.chunk_id:c for c in chunks}
    assert catalog[results[0]['chunk_id']].doc_id=='POL-SALES-002'
    from chromadb.api.shared_system_client import SharedSystemClient
    SharedSystemClient._identifier_to_system[str(storage)].stop()
    SharedSystemClient._identifier_to_system.pop(str(storage))
    shutil.rmtree(storage)


def test_fake_determinism():
    gateway=FakeEmbeddingGateway()
    assert gateway.embed(['折扣审批'])==gateway.embed(['折扣审批'])
