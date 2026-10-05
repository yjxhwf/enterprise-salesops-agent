from dataclasses import dataclass
from enum import StrEnum


@dataclass(frozen=True)
class RuntimeLimits:
    max_agent_steps: int = 30
    max_llm_calls: int = 20
    max_tool_calls: int = 8
    max_retries_per_llm_operation: int = 2
    max_total_retries: int = 4
    max_repeated_tool_args: int = 2
    max_stagnant_cycles: int = 2
    soft_token_budget: int = 120000

    def __post_init__(self):
        for value in vars(self).values():
            if type(value) is not int or value < 0:
                raise ValueError("Runtime limits must be nonnegative integers")


class StopReason(StrEnum):
    COMPLETED = "COMPLETED"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    UNSUPPORTED = "UNSUPPORTED"
    MODEL_ERROR = "MODEL_ERROR"
    TOOL_ERROR = "TOOL_ERROR"
    MODEL_RETRY_EXHAUSTED = "MODEL_RETRY_EXHAUSTED"
    MAX_AGENT_STEPS_EXCEEDED = "MAX_AGENT_STEPS_EXCEEDED"
    MAX_LLM_CALLS_EXCEEDED = "MAX_LLM_CALLS_EXCEEDED"
    MAX_TOOL_CALLS_EXCEEDED = "MAX_TOOL_CALLS_EXCEEDED"
    MAX_RETRIES_EXCEEDED = "MAX_RETRIES_EXCEEDED"
    REPEATED_ACTION_LOOP = "REPEATED_ACTION_LOOP"
    STAGNANT_LOOP = "STAGNANT_LOOP"
    TOOL_TIMEOUT = "TOOL_TIMEOUT"
    TOKEN_BUDGET_EXCEEDED = "TOKEN_BUDGET_EXCEEDED"
