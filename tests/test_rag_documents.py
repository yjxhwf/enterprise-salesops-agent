import hashlib
from pathlib import Path
import pytest
from pydantic import ValidationError
from backend.app.rag.documents import CORPUS_ROOT, load_documents
from backend.app.rag.chunking import chunk_documents, MAX_CHUNK_CHARS
from backend.app.rag.bm25 import tokenize


def test_metadata_hash_and_stable_section_chunks():
    documents = load_documents()
    assert len(documents) == 6
    chunks = chunk_documents(documents)
    assert len(chunks) == 12 and chunks == chunk_documents(documents)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert all(c.title in c.text and c.section_title in c.text and len(c.text)<=MAX_CHUNK_CHARS for c in chunks)
    assert all(c.content_hash == hashlib.sha256(c.text.encode()).hexdigest() for c in chunks)
    assert all(d.source_path.startswith('knowledge_base/policies/') for d in documents)


@pytest.mark.parametrize('replacement', ['title: \n', 'title: X\ntitle: Duplicate\n', 'unknown: value\n', 'version: not-version\n', 'effective_date: impossible\n'])
def test_strict_metadata(tmp_path, replacement):
    source=next(CORPUS_ROOT.glob('*.md')).read_text(encoding='utf-8')
    key=replacement.split(':')[0]
    import re
    source=re.sub(r'^'+(key if key!='unknown' else 'title')+r':.*\n',replacement,source,flags=re.M)
    (tmp_path/'policy.md').write_text(source,encoding='utf-8')
    with pytest.raises((ValueError,ValidationError)):
        load_documents(tmp_path)


def test_long_section_overlap_and_no_loss():
    doc=load_documents()[0].model_copy(update={'body':'## Long\n'+''.join(chr(0x4e00+i%3000) for i in range(3500))})
    chunks=chunk_documents([doc])
    assert len(chunks)>2 and max(map(lambda c:len(c.text),chunks))<=1200
    bodies=[c.text.split('\n',2)[2] for c in chunks]
    assert bodies[0][-150:]==bodies[1][:150]
    assert ''.join([bodies[0]]+[b[150:] for b in bodies[1:]]) == doc.body.split('\n',1)[1]


def test_content_change_changes_id_and_hash(tmp_path):
    source=next(CORPUS_ROOT.glob('*.md')).read_text(encoding='utf-8')
    path=tmp_path/'policy.md';path.write_text(source,encoding='utf-8')
    first=load_documents(tmp_path)
    path.write_text(source+'\n新的通用条款。',encoding='utf-8')
    second=load_documents(tmp_path)
    assert first[0].content_hash != second[0].content_hash
    assert chunk_documents(first)[-1].chunk_id != chunk_documents(second)[-1].chunk_id


@pytest.mark.parametrize('text,token',[('折扣审批','折扣'),('特殊报价','报价'),('订单复核','复核'),('POL-SALES-001','pol-sales-001')])
def test_mixed_tokenizer(text,token):
    assert token in tokenize(text)


def test_corpus_contains_no_golden_entities():
    import re
    for doc in load_documents():
        assert not re.search(r'EAST_CHINA|SKU-A12|SKU-B07|SKU-C03|C102|C207|2026年8月利润下降原因',doc.body)


def test_source_resolving_outside_root_is_rejected(tmp_path, monkeypatch):
    path = tmp_path/'policy.md'
    path.write_text('unused', encoding='utf-8')
    original = Path.resolve
    def resolve(value, *args, **kwargs):
        return tmp_path.parent/'outside.md' if value == path else original(value, *args, **kwargs)
    monkeypatch.setattr(Path, 'resolve', resolve)
    with pytest.raises(ValueError, match='escapes corpus root'):
        load_documents(tmp_path)
