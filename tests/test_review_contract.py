"""Review wrapper contract: strict canonical object, never permissive normalization."""
from copy import deepcopy
from pathlib import Path
import json
import pytest
from pydantic import ValidationError
from langchain_core.messages import AIMessage
from backend.app.agent.schemas import ProgressReview
from backend.app.agent import prompts
from backend.app.agent.guards.context import active_runtime
from backend.app.agent.guards.runtime import Runtime
from backend.app.agent.guards.limits import RuntimeLimits
from backend.app.agent.state import initial_state
from backend.app.llm.provider import ProductionGateway
from backend.app.llm.review_diagnostics import ReviewContractError, safe_plan_shape
from tests.agent_fakes import goal, plan
from tests.test_agent_investigation import offline


def valid(decision='CONTINUE'):
    return dict(decision=decision,selected_evidence_ids=['test-evidence'],
        updated_plan={'steps':[dict(objective='Check order authorization',expected_output='Order authorization evidence')]} if decision=='CONTINUE' else None,
        remaining_evidence_gap='Missing order authorization evidence' if decision=='CONTINUE' else None,
        next_objective='Check order authorization' if decision=='CONTINUE' else None)


@pytest.mark.parametrize('decision',['CONTINUE','COMPLETE'])
def test_canonical_valid(decision):
    value=valid(decision)
    assert ProgressReview.model_validate(value).model_dump(mode='json')==value


@pytest.mark.parametrize('value',['wrong',7,False,[],[{'objective':'x','expected_output':'y'}],{}, {'steps':[]},{'steps':[{'objective':'x'}]},{'steps':[{'objective':'x','expected_output':'y','extra':'z'}]},{'steps':[{'objective':'x','expected_output':'y'}],'extra':'z'}])
def test_invalid_shapes_rejected_without_normalization(value):
    output=valid();output['updated_plan']=value
    with pytest.raises(ValidationError):ProgressReview.model_validate(output)


@pytest.mark.parametrize('field',['remaining_evidence_gap','next_objective','updated_plan'])
def test_continue_still_requires_all_work_fields(field):
    output=valid();output[field]=None
    with pytest.raises(ValidationError):ProgressReview.model_validate(output)


def test_complete_must_not_create_new_work():
    output=valid();output.update(decision='COMPLETE',remaining_evidence_gap=None,next_objective=None)
    with pytest.raises(ValidationError):ProgressReview.model_validate(output)


class Client:
    def __init__(self,output):self.output=output;self.calls=0;self.parameters=None
    def bind_tools(self,tools,**kwargs):self.parameters=tools[0]['function']['parameters'];return self
    def invoke(self,*args,**kwargs):
        self.calls+=1
        return AIMessage(content='',tool_calls=[dict(name='ProgressReview',args=self.output,id='offline')],usage_metadata=dict(input_tokens=10,output_tokens=5,total_tokens=15))


def test_three_contract_layers_and_real_bound_schema():
    client=Client(valid());gateway=ProductionGateway();gateway._model=client
    gateway.review_progress(goal(),plan(),plan()['steps'][0],{},[],[],[])
    expected=ProgressReview.model_json_schema()
    assert client.parameters==expected
    assert expected['properties']['updated_plan']['anyOf']==[{'$ref':'#/$defs/RemainingPlan'},{'type':'null'}]
    assert expected['$defs']['RemainingPlan']['type']=='object'
    assert expected['$defs']['RemainingPlan']['required']==['steps']
    assert expected['$defs']['RemainingPlan']['additionalProperties'] is False
    assert '"steps": [{"objective": "...", "expected_output": "..."}]' in prompts.REVIEW
    assert 'Never return a bare array' in prompts.REVIEW


@pytest.mark.parametrize('bad,kind',[('secret-string','string'),(42,'number'),([{'objective':'private','expected_output':'private'}],'array'),({'steps':[{'objective':'private'}]},'object')])
def test_value_free_diagnostic_and_no_transport_retry(bad,kind):
    output=valid();output['updated_plan']=bad
    client=Client(output);gateway=ProductionGateway();gateway._model=client
    state=initial_state('Review');runtime=Runtime(state,'review_progress',RuntimeLimits(),lambda _:pytest.fail('No retry allowed'))
    token=active_runtime.set(runtime)
    try:
        with pytest.raises(ReviewContractError) as exc:gateway.review_progress(goal(),plan(),plan()['steps'][0],{},[],[],[])
        diagnostic=exc.value.structured_output_diagnostic
        assert diagnostic['actual_type']==kind and diagnostic['field_path']=='updated_plan'
        assert diagnostic['expected_type'].startswith('object|null')
        assert diagnostic['validation_code']=='MODEL_OUTPUT_VALIDATION_ERROR'
        text=json.dumps(diagnostic)
        assert 'secret-string' not in text and 'private' not in text
        assert client.calls==state['llm_call_count']==1 and state['retry_count']==0
        assert state['context_manifests'][0]['node']=='review_progress'
    finally:active_runtime.reset(token)


def test_historical_fixture_does_not_invent_actual_shape():
    data=json.loads(Path('tests/fixtures/review_updated_plan_diagnostic.json').read_text(encoding='utf-8'))
    assert data['validation_rule']=='model_type' and data['actual_type']=='NOT_RETAINED'
    assert data['safe_shape'] is None


def test_safe_shape_redacts_unknown_keys_and_values():
    data={'steps':[{'objective':'sensitive','expected_output':'sensitive','api-secret-as-key':'secret'}]*10,'Authorization':'Bearer secret'}
    shape=safe_plan_shape(data);text=json.dumps(shape)
    assert 'sensitive' not in text and 'secret' not in text and 'Authorization' not in text
    assert shape['unknown_field_count']==1 and shape['fields']['steps']['omitted_item_count']==7
