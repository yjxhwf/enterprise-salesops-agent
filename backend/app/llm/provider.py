from backend.app.agent.guards.context import active_runtime, build_llm_context
from backend.app.observability.integration import provider_identity
"""Lazy OpenAI-compatible native tool calling. Runtime-managed transport retries; no text or model fallback."""

import json
from urllib.parse import urlsplit

from langchain_openai import ChatOpenAI
from openai import APIError, AuthenticationError, BadRequestError, NotFoundError, PermissionDeniedError
from pydantic import ValidationError

from backend.app.agent import prompts
from backend.app.agent.schemas import GoalUnderstanding, Plan, ProgressReview, GroundedAnalysisDraft
from backend.app.config import Settings, get_settings
from backend.app.llm.gateway import ErrorCode, GatewayError


class ProductionGateway:
    runtime_transport_managed = True
    def __init__(self, settings: Settings | None = None, *, request_timeout_seconds: float = 30):
        self._settings = settings
        self._model = None
        self._request_timeout_seconds = request_timeout_seconds

    def _client(self):
        if self._model is not None:
            return self._model
        try:
            settings = self._settings if self._settings is not None else get_settings()
            key = settings.LLM_API_KEY.get_secret_value().strip() if settings.LLM_API_KEY else ""
            provider, model, base_url = settings.LLM_PROVIDER.strip(), settings.LLM_MODEL.strip(), settings.LLM_BASE_URL.strip()
            if not key or not model or not provider or (provider.lower() != "openai" and not base_url):
                raise GatewayError(ErrorCode.MODEL_CONFIGURATION_ERROR)
            if base_url:
                url = urlsplit(base_url)
                if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
                    raise GatewayError(ErrorCode.MODEL_CONFIGURATION_ERROR)
            self._model = ChatOpenAI(model=model, api_key=key, base_url=base_url or None,
                                     temperature=0, timeout=self._request_timeout_seconds, max_retries=0, use_responses_api=False)
            provider_identity(settings, self._model)
        except (ValidationError, ValueError):
            raise GatewayError(ErrorCode.MODEL_CONFIGURATION_ERROR) from None
        return self._model

    def _native_call(self, system: str, payload: dict, tools: list[dict], *, tool_choice, multiple_proposals=False):
        try:
            model = self._client()
            provider_identity(self._settings, model, tool_choice)
            client = model.bind_tools(tools, tool_choice=tool_choice, parallel_tool_calls=False)
            def request():
                return client.invoke([("system", system), ("human", json.dumps(payload, ensure_ascii=False))],
                                     config={"callbacks": []})
            runtime = active_runtime.get()
            response = runtime.llm(request) if runtime else request()
        except (AuthenticationError, PermissionDeniedError, NotFoundError):
            raise GatewayError(ErrorCode.MODEL_CONFIGURATION_ERROR) from None
        except BadRequestError as error:
            # Classify only explicit unsupported-tool errors as capability errors.
            # Never retain or log the provider body, headers or exception text.
            message = str(error).lower()
            capability = (any(word in message for word in ("tool", "function", "parallel_tool_calls"))
                          and any(word in message for word in ("not support", "unsupported", "not available")))
            raise GatewayError(ErrorCode.MODEL_CAPABILITY_ERROR if capability else ErrorCode.MODEL_REQUEST_ERROR) from None
        except NotImplementedError:
            raise GatewayError(ErrorCode.MODEL_CAPABILITY_ERROR) from None
        except APIError:
            raise GatewayError(ErrorCode.MODEL_REQUEST_ERROR) from None
        validation_code = (ErrorCode.TOOL_SELECTION_VALIDATION_ERROR if multiple_proposals
                           else ErrorCode.MODEL_OUTPUT_VALIDATION_ERROR)
        if response.invalid_tool_calls:
            raise GatewayError(validation_code)
        if not response.tool_calls:
            raise GatewayError(ErrorCode.MODEL_CAPABILITY_ERROR)
        if not multiple_proposals and len(response.tool_calls) != 1:
            raise GatewayError(ErrorCode.MODEL_OUTPUT_VALIDATION_ERROR)
        for call in response.tool_calls:
            if not isinstance(call.get("name"), str) or not isinstance(call.get("args"), dict):
                raise GatewayError(validation_code)
        return response.tool_calls if multiple_proposals else response.tool_calls[0]

    def _structured(self, schema, system, payload):
        name = schema.__name__
        tools = [{"type": "function", "function": {"name": name, "description": "Return the requested structured business output.",
                  "parameters": schema.model_json_schema()}}]
        call = self._native_call(system, payload, tools, tool_choice=name)
        if call["name"] != name:
            raise GatewayError(ErrorCode.MODEL_OUTPUT_VALIDATION_ERROR)
        try:
            return schema.model_validate(call["args"]).model_dump(mode="json")
        except ValidationError as error:
            if schema.__name__ == "ActionDecision":
                from backend.app.actions.diagnostics import pydantic_diagnostic
                wrapped = GatewayError(ErrorCode.MODEL_OUTPUT_VALIDATION_ERROR)
                wrapped.action_proposal_diagnostic = pydantic_diagnostic(error, call["args"], schema=schema)
                raise wrapped from None
            if schema is ProgressReview:
                from backend.app.llm.review_diagnostics import ReviewContractError
                raise ReviewContractError(call["args"], error) from None
            raise GatewayError(ErrorCode.MODEL_OUTPUT_VALIDATION_ERROR) from None

    def understand_goal(self, query):
        return self._structured(GoalUnderstanding, prompts.UNDERSTAND, build_llm_context("understand_goal", user_query=query))

    def create_plan(self, goal, capabilities):
        return self._structured(Plan, prompts.PLAN, build_llm_context("plan", goal=goal, capabilities=capabilities))

    def select_tool(self, goal, step, tools):
        calls = self._native_call(prompts.SELECT, build_llm_context("select_tool", goal=goal, step=step, tools=tools), tools,
                                  tool_choice="required", multiple_proposals=True)
        return self._proposals(calls)

    def select_investigation_tool(self, goal, plan, step, observations, tools):
        calls = self._native_call(prompts.SELECT,
                                 build_llm_context("select_tool", goal=goal, plan=plan, step=step, observations=observations, tools=tools),
                                 tools, tool_choice="required", multiple_proposals=True)
        return self._proposals(calls)

    @staticmethod
    def _proposals(calls):
        return {"candidates": [{"tool_name": call["name"], "arguments": call["args"], "provider_order": index}
                                for index, call in enumerate(calls)]}

    def review_progress(self, goal, plan, step, latest_result, observations, history, capabilities):
        return self._structured(ProgressReview, prompts.REVIEW,
                                build_llm_context("review_progress", goal=goal, plan=plan, step=step,
                                                  observations=observations, history=history))

    def synthesize(self, goal, observations, history):
        return self._structured(GroundedAnalysisDraft, prompts.SYNTHESIZE,
                                build_llm_context("synthesize", goal=goal, observations=observations, history=history))

    def propose_action(self, context):
        from backend.app.actions.schemas import ActionDecision
        from backend.app.actions.prompts import PROPOSE_ACTION
        try:
            return self._structured(ActionDecision, PROPOSE_ACTION, context)
        except GatewayError as error:
            if error.code == ErrorCode.MODEL_OUTPUT_VALIDATION_ERROR:
                wrapped = GatewayError(ErrorCode.ACTION_PROPOSAL_VALIDATION_ERROR)
                wrapped.action_proposal_diagnostic = getattr(error, "action_proposal_diagnostic", None)
                raise wrapped from None
            raise
