import pytest
from pydantic import ValidationError
from backend.app.rag.schemas import PolicySearchInput, RetrievalError
from backend.app.rag.retriever import PolicyRetriever
from backend.app.tools.registry import build_default_registry
from tests.test_rag_retrieval import rag_index, offline


@pytest.mark.parametrize('args',[{'query':' '},{'query':'x'},{'query':'x'*501},{'query':'折扣','top_k':0},{'query':'折扣','top_k':9},{'query':'折扣','path':'../secret'},{'query':'折扣','top_k':True}])
def test_policy_input_strict(args):
    with pytest.raises(ValidationError):PolicySearchInput.model_validate(args)


def test_policy_tool_is_additive_read_and_does_not_open_db(rag_index):
    gateway,storage,_=rag_index
    def forbidden():raise AssertionError('Policy tool must not use business DB')
    registry=build_default_registry(forbidden,policy_retriever=PolicyRetriever(gateway,storage=storage))
    assert len(registry.list_tools(permission='READ'))==6
    tool=registry.get_tool('search_sales_policy')
    assert tool.permission_level=='READ' and tool.input_schema['additionalProperties'] is False
    result=registry.invoke('search_sales_policy',dict(query='折扣审批'))
    assert result.success and result.data.retrieval_mode=='HYBRID'
    assert result.evidence_ids==[h.citation.policy_evidence_id for h in result.data.hits]
    assert 'POLICY::' in result.model_dump_json()


@pytest.mark.parametrize('code,retryable',[('KNOWLEDGE_INDEX_NOT_READY',False),('INDEX_MODEL_MISMATCH',False),('NO_RETRIEVAL_RESULTS',False),('RETRIEVAL_ERROR',False),('EMBEDDING_PROVIDER_ERROR',True)])
def test_policy_safe_errors(code,retryable):
    class Failed:
        calls=0
        def search(self,*args):
            self.calls+=1
            raise RetrievalError(code,retryable=retryable)
    retriever=Failed()
    result=build_default_registry(lambda:None,policy_retriever=retriever).invoke('search_sales_policy',{'query':'折扣审批'})
    assert not result.success and result.error.code==code and result.error.retryable==retryable
    assert result.data is None and not result.evidence_ids and retriever.calls==1
