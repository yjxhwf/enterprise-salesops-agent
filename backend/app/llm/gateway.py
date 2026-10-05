from enum import StrEnum
from typing import Protocol


class ErrorCode(StrEnum):
    CONTEXT_BUDGET_EXCEEDED = "CONTEXT_BUDGET_EXCEEDED"
    EVIDENCE_NOT_VISIBLE_IN_CONTEXT = "EVIDENCE_NOT_VISIBLE_IN_CONTEXT"
    MODEL_CONFIGURATION_ERROR = "MODEL_CONFIGURATION_ERROR"
    MODEL_CAPABILITY_ERROR = "MODEL_CAPABILITY_ERROR"
    MODEL_OUTPUT_VALIDATION_ERROR = "MODEL_OUTPUT_VALIDATION_ERROR"
    MODEL_REQUEST_ERROR = "MODEL_REQUEST_ERROR"
    TOOL_SELECTION_VALIDATION_ERROR = "TOOL_SELECTION_VALIDATION_ERROR"
    ACTION_PROPOSAL_VALIDATION_ERROR = "ACTION_PROPOSAL_VALIDATION_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


MESSAGES = {
    ErrorCode.CONTEXT_BUDGET_EXCEEDED: "Mandatory context exceeds the node character budget.",
    ErrorCode.EVIDENCE_NOT_VISIBLE_IN_CONTEXT: "Citation was not visible in this model request.",
    ErrorCode.MODEL_CONFIGURATION_ERROR: "LLM configuration missing or invalid.",
    ErrorCode.MODEL_CAPABILITY_ERROR: "Provider does not support the required native tool calling request.",
    ErrorCode.MODEL_OUTPUT_VALIDATION_ERROR: "Model output does not satisfy the required business schema.",
    ErrorCode.MODEL_REQUEST_ERROR: "Model request failed.",
    ErrorCode.TOOL_SELECTION_VALIDATION_ERROR: "Selected tool or arguments are not permitted by the read catalog.",
    ErrorCode.INTERNAL_ERROR: "An internal agent defect occurred.",
    ErrorCode.ACTION_PROPOSAL_VALIDATION_ERROR: "Action proposal does not satisfy grounded write constraints.",
}


class GatewayError(Exception):
    def __init__(self, code: ErrorCode):
        self.code = code
        super().__init__(MESSAGES[code])


class ModelGateway(Protocol):
    def understand_goal(self, query: str) -> dict: ...
    def create_plan(self, goal: dict, capabilities: list[dict]) -> dict: ...
    def select_tool(self, goal: dict, step: dict, tools: list[dict]) -> dict: ...


class InvestigationGateway(ModelGateway, Protocol):
    def select_investigation_tool(self, goal: dict, plan: dict, step: dict,
                                  observations: list[dict], tools: list[dict]) -> dict: ...
    def review_progress(self, goal: dict, plan: dict, step: dict, latest_result: dict,
                        observations: list[dict], history: list[dict], capabilities: list[dict]) -> dict: ...
    def synthesize(self, goal: dict, observations: list[dict], history: list[dict]) -> dict: ...


class ActionGateway(InvestigationGateway, Protocol):
    def propose_action(self, context: dict) -> dict: ...
