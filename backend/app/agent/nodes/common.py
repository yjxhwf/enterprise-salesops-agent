from backend.app.agent.validation_diagnostics import SemanticValidationError
from backend.app.agent.guards.context import active_runtime, RuntimeStop
from functools import wraps
from backend.app.observability.integration import validation_error

from pydantic import ValidationError

from backend.app.llm.gateway import ErrorCode, GatewayError, MESSAGES
from backend.app.observability.logger import get_logger

logger = get_logger("agent")


def model_node(name, *, validation_code=ErrorCode.MODEL_OUTPUT_VALIDATION_ERROR, strict=False):
    """Count gateway attempts once and preserve only safe classified errors."""
    def decorate(function):
        @wraps(function)
        def run(state):
            update = {"visited_nodes": [*state.get("visited_nodes", []), name],
                      "llm_call_count": state.get("llm_call_count", 0) + (0 if active_runtime.get() else 1)}
            diagnostic = None
            context_diagnostic = None
            structured_diagnostic = None
            captured_error = None
            try:
                update.update(function(state))
                return update
            except RuntimeStop:
                raise
            except GatewayError as error:
                captured_error = error
                code = error.code
                context_diagnostic = getattr(error, "context_diagnostic", None)
                structured_diagnostic = getattr(error, "structured_output_diagnostic", None)
                if code == ErrorCode.CONTEXT_BUDGET_EXCEEDED:
                    update["llm_call_count"] = state.get("llm_call_count", 0)
                if isinstance(error, SemanticValidationError):
                    diagnostic = error.diagnostic
                    runtime = active_runtime.get()
                    if runtime:
                        runtime.event("SEMANTIC_VALIDATION_REJECTED", **diagnostic)
            except ValidationError as error:
                captured_error = error
                code = validation_code
            except Exception as error:
                captured_error = error
                logger.error("Agent defect node=%s type=%s", name, type(error).__name__)
                if strict:
                    raise
                code = ErrorCode.INTERNAL_ERROR
            validation_error(name, captured_error, code.value)
            update.update(status="ERROR", errors=[*state.get("errors", []),
                          {"node": name, "code": code.value, "message": MESSAGES[code], **({"validation_diagnostic": diagnostic} if diagnostic else {}), **({"context_diagnostic": context_diagnostic} if context_diagnostic else {}), **({"structured_output_diagnostic": structured_diagnostic} if structured_diagnostic else {})}])
            return update
        return run
    return decorate
