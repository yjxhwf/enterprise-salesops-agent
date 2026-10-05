import json
import socket
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.messages import AIMessage
from openai import APIConnectionError, AuthenticationError, BadRequestError

from backend.app.agent.graph import build_agent_graph
from backend.app.agent.schemas import GoalUnderstanding, Plan
from backend.app.agent.state import initial_state
from backend.app.agent.tooling import ToolCatalog
from backend.app.config import Settings
from backend.app.llm import provider
from backend.app.llm.gateway import ErrorCode, GatewayError
from backend.app.llm.provider import ProductionGateway
from tests.agent_fakes import DATES, QUERY, goal, plan


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Live network is prohibited in pytest")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")


def settings(**changes):
    return Settings(_env_file=None, **{"LLM_PROVIDER": "compatible", "LLM_MODEL": "test-model",
                    "LLM_BASE_URL": "https://model.invalid/v1", "LLM_API_KEY": "test-only-not-a-real-credential", **changes})


class StubNativeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.bindings = []
        self.requests = []

    def bind_tools(self, tools, **kwargs):
        self.bindings.append((tools, kwargs))
        return self

    def invoke(self, messages, **kwargs):
        self.requests.append((messages, kwargs))
        output = next(self.responses)
        if isinstance(output, Exception):
            raise output
        return output


def native(name, arguments):
    return AIMessage(content="", tool_calls=[{"name": name, "args": arguments, "id": "test-call", "type": "tool_call"}])


def inject(monkeypatch, responses):
    client = StubNativeClient(responses)
    constructors = []
    def build(**kwargs):
        constructors.append(kwargs)
        return client
    monkeypatch.setattr(provider, "ChatOpenAI", build)
    return client, constructors


def test_production_native_protocol_and_lazy_client(monkeypatch):
    client, constructors = inject(monkeypatch, [native("GoalUnderstanding", goal()), native("Plan", plan()), native("query_orders", DATES)])
    gateway = ProductionGateway(settings())
    graph = build_agent_graph(gateway, raise_unexpected_errors=True)
    assert not constructors
    result = graph.invoke(initial_state(QUERY))
    assert result["status"] == "TOOL_SELECTED"
    assert result["selected_tool"]["tool_name"] == "query_orders"
    assert result["llm_call_count"] == len(client.requests) == 3
    assert len(constructors) == 1
    config = constructors[0]
    assert config["max_retries"] == 0 and config["timeout"] == 30 and config["temperature"] == 0
    assert config["base_url"] == "https://model.invalid/v1" and config["use_responses_api"] is False
    assert client.bindings[0][0][0]["function"]["parameters"] == GoalUnderstanding.model_json_schema()
    assert client.bindings[1][0][0]["function"]["parameters"] == Plan.model_json_schema()
    assert client.bindings[2][0] == ToolCatalog().tool_schemas()
    assert client.bindings[2][1] == {"tool_choice": "required", "parallel_tool_calls": False}
    assert all(request[0][0][0] == "system" and request[0][1][0] == "human" for request in client.requests)
    assert "tool_name" not in json.loads(client.requests[2][0][1][1])["current_step"]
    assert config["api_key"] not in json.dumps(result)
    assert config["api_key"] not in json.dumps(client.requests)


@pytest.mark.parametrize("changes", [
    {"LLM_API_KEY": None}, {"LLM_API_KEY": ""}, {"LLM_MODEL": ""}, {"LLM_PROVIDER": ""},
    {"LLM_BASE_URL": ""}, {"LLM_BASE_URL": "file:///private"},
    {"LLM_BASE_URL": "https://username:password@model.invalid/v1"},
])
def test_missing_config_is_lazy_and_safe(monkeypatch, changes):
    client, constructors = inject(monkeypatch, [])
    result = build_agent_graph(ProductionGateway(settings(**changes))).invoke(initial_state(QUERY))
    assert result["status"] == "ERROR"
    assert result["errors"][0]["code"] == "MODEL_CONFIGURATION_ERROR"
    assert not constructors and not client.requests
    assert "password" not in json.dumps(result)


def test_openai_base_url_optional(monkeypatch):
    client, constructors = inject(monkeypatch, [native("GoalUnderstanding", goal())])
    ProductionGateway(settings(LLM_PROVIDER="openai", LLM_BASE_URL="")).understand_goal(QUERY)
    assert constructors[0]["base_url"] is None


@pytest.mark.parametrize("response,code", [
    (AIMessage(content="I choose a tool"), "MODEL_CAPABILITY_ERROR"),
    (AIMessage(content="", invalid_tool_calls=[{"name": "query_orders", "args": "broken", "id": "t", "error": "parse error", "type": "invalid_tool_call"}]), "MODEL_OUTPUT_VALIDATION_ERROR"),
    (AIMessage(content="", tool_calls=[{"name": "query_orders", "args": {}, "id": str(i), "type": "tool_call"} for i in range(2)]), "MODEL_OUTPUT_VALIDATION_ERROR"),
    (native("wrong_schema", goal()), "MODEL_OUTPUT_VALIDATION_ERROR"),
    (native("GoalUnderstanding", {"unexpected": "sensitive-output-marker"}), "MODEL_OUTPUT_VALIDATION_ERROR"),
    (NotImplementedError("native unsupported sensitive marker"), "MODEL_CAPABILITY_ERROR"),
])
def test_model_response_rejection(monkeypatch, response, code):
    client, _ = inject(monkeypatch, [response])
    result = build_agent_graph(ProductionGateway(settings())).invoke(initial_state(QUERY))
    assert result["status"] == "ERROR" and result["errors"][0]["code"] == code
    assert len(client.requests) == 1
    assert "sensitive" not in json.dumps(result)


@pytest.mark.parametrize("kind,message,code", [
    (BadRequestError, "tools not supported secret-marker", "MODEL_CAPABILITY_ERROR"),
    (BadRequestError, "temperature invalid secret-marker", "MODEL_REQUEST_ERROR"),
    (AuthenticationError, "invalid credential secret-marker", "MODEL_CONFIGURATION_ERROR"),
])
def test_provider_error_classification_and_no_retry(monkeypatch, kind, message, code):
    response = httpx.Response(400, request=httpx.Request("POST", "https://model.invalid/v1"))
    error = kind(message, response=response, body={"sensitive": "secret-marker"})
    client, _ = inject(monkeypatch, [error])
    result = build_agent_graph(ProductionGateway(settings())).invoke(initial_state(QUERY))
    assert result["errors"][0]["code"] == code
    assert len(client.requests) == 1 and "secret-marker" not in json.dumps(result)


def test_connection_error_and_malformed_plan(monkeypatch):
    client, _ = inject(monkeypatch, [APIConnectionError(request=httpx.Request("POST", "https://model.invalid/v1"))])
    with pytest.raises(GatewayError) as caught:
        ProductionGateway(settings()).understand_goal(QUERY)
    assert caught.value.code == ErrorCode.MODEL_REQUEST_ERROR and len(client.requests) == 1
    client, _ = inject(monkeypatch, [native("GoalUnderstanding", goal()), native("Plan", {"steps": []})])
    result = build_agent_graph(ProductionGateway(settings())).invoke(initial_state(QUERY))
    assert result["errors"][0]["code"] == "MODEL_OUTPUT_VALIDATION_ERROR"
    assert result["llm_call_count"] == 2 and len(client.requests) == 2


def test_real_binding_accepts_registry_schemas_without_network():
    # Exercise the installed LangChain constructor/adapter, without invoke or HTTP.
    gateway = ProductionGateway(settings())
    client = gateway._client()
    assert client.max_retries == 0
    bound = client.bind_tools(ToolCatalog().tool_schemas(), tool_choice="required", parallel_tool_calls=False)
    assert len(bound.kwargs["tools"]) == 6
    assert bound.kwargs["tool_choice"] == "required"


def test_explicit_request_timeout_does_not_enable_retry(monkeypatch):
    _, constructors = inject(monkeypatch, [native("GoalUnderstanding", goal())])
    ProductionGateway(settings(), request_timeout_seconds=120).understand_goal(QUERY)
    assert constructors[0]["timeout"] == 120
    assert constructors[0]["max_retries"] == 0


def test_investigation_gateway_passes_real_context(monkeypatch):
    review = {"decision": "COMPLETE", "selected_evidence_ids": ["test-evidence"],
              "next_objective": None, "updated_plan": None, "remaining_evidence_gap": None}
    draft = {"executive_interpretation": "Measured revenue", "findings": [{"title": "Revenue", "interpretation": "Measured",
              "claim_type": "OBSERVATION", "supporting_evidence_ids": ["test-evidence"]}],
              "recommendations": [{"title": "Review", "action": "Advisory", "related_finding_index": 0}],
              "limitations": ["Limited scope"]}
    client, _ = inject(monkeypatch, [native("query_orders", DATES), native("ProgressReview", review),
                                    native("GroundedAnalysisDraft", draft)])
    gateway = ProductionGateway(settings())
    history = [{"tool_name": "get_sales_overview", "success": True, "evidence_ids": ["test-evidence"], "data": {"current": {"revenue": "1.00"}}}]
    observations = [{"summary": "Prior observation"}]
    gateway.select_investigation_tool(goal(), plan(), plan()["steps"][0], observations, ToolCatalog().tool_schemas())
    assert gateway.review_progress(goal(), plan(), plan()["steps"][0], history[0], observations, history, []) == review
    from backend.app.agent.schemas import GroundedAnalysisDraft
    assert gateway.synthesize(goal(), observations, history) == GroundedAnalysisDraft.model_validate(draft).model_dump(mode="json")
    payloads = [json.loads(messages[1][1]) for messages, _ in client.requests]
    assert len(payloads[0]["pending_objectives"]) == 2 and "observations" not in payloads[0]
    assert payloads[1]["latest_source_tool"] == history[0]["tool_name"]
    assert payloads[2]["executive_facts"][0]["facts"][0]["metric"] == "revenue"
    for payload in payloads[1:]:
        assert "tool_history" not in payload and "latest_tool_result" not in payload
        assert "internal_reference" not in json.dumps(payload)
        assert "json_pointer" not in json.dumps(payload)
