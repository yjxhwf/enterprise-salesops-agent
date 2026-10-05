from backend.app.agent.nodes.common import model_node
from backend.app.agent.schemas import ToolProposals, ToolSelection
from backend.app.llm.gateway import ErrorCode, GatewayError


def make_select_node(gateway, catalog, *, strict=False, investigation=False):
    @model_node("select_tool", validation_code=ErrorCode.TOOL_SELECTION_VALIDATION_ERROR, strict=strict)
    def select_tool(state):
        step = state["plan"]["steps"][state["current_step_index"]]
        if investigation:
            output = gateway.select_investigation_tool(state["goal"], state["plan"], step,
                                                       state["observations"], catalog.tool_schemas())
        else:
            output = gateway.select_tool(state["goal"], step, catalog.tool_schemas())
        # Preserve the original single-selection gateway contract for existing callers.
        if isinstance(output, dict) and "candidates" not in output:
            output = {"candidates": [{**ToolSelection.model_validate(output).model_dump(mode="json"), "provider_order": 0}]}
        proposals = ToolProposals.model_validate(output)
        validated = []
        for order, candidate in enumerate(proposals.candidates):
            if candidate.provider_order != order:
                raise GatewayError(ErrorCode.TOOL_SELECTION_VALIDATION_ERROR)
            selection = catalog.validate(ToolSelection.model_validate(candidate.model_dump(exclude={"provider_order"})))
            if selection.tool_name != "search_sales_policy" and any(selection.arguments[field] != state["goal"][field] for field in ("start_date", "end_date")):
                raise GatewayError(ErrorCode.TOOL_SELECTION_VALIDATION_ERROR)
            validated.append({**selection.model_dump(mode="json"), "provider_order": order})
        # Only after ALL candidates pass may the first proposal become executable.
        immediate = {key: value for key, value in validated[0].items() if key != "provider_order"}
        return {"selected_tool": immediate, "tool_proposal_count": len(validated),
                "deferred_tool_proposals": validated[1:], "status": "TOOL_SELECTED"}
    return select_tool
