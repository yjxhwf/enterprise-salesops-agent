from copy import deepcopy

from backend.app.agent.schemas import ToolExecutionRecord, ToolSelection
from backend.app.llm.gateway import ErrorCode, GatewayError, MESSAGES


def make_execute_node(registry, catalog):
    def execute_tool(state):
        update = {"visited_nodes": [*state["visited_nodes"], "execute_tool"]}
        # Revalidate at the execution boundary, even if called outside the graph.
        try:
            selection = catalog.validate(ToolSelection.model_validate(state["selected_tool"]))
            if selection.tool_name != "search_sales_policy" and any(selection.arguments[field] != state["goal"][field] for field in ("start_date", "end_date")):
                raise GatewayError(ErrorCode.TOOL_SELECTION_VALIDATION_ERROR)
        except (GatewayError, ValueError):
            code = ErrorCode.TOOL_SELECTION_VALIDATION_ERROR
            return {**update, "status": "ERROR", "errors": [*state["errors"],
                    {"node": "execute_tool", "code": code.value, "message": MESSAGES[code]}]}
        result = registry.invoke(selection.tool_name, selection.arguments).model_dump(mode="json")
        plan = deepcopy(state["plan"])
        step = plan["steps"][state["current_step_index"]]
        step["status"] = "IN_PROGRESS" if result["success"] else "FAILED"
        record = ToolExecutionRecord(tool_name=selection.tool_name, arguments=selection.arguments,
                                     success=result["success"], summary=result["summary"],
                                     evidence_ids=result["evidence_ids"], meta=result["meta"], data=result["data"],
                                     error_code=result["error"]["code"] if result["error"] else None,
                                     plan_step_id=step["step_id"]).model_dump(mode="json")
        update.update(latest_tool_result=result, tool_history=[*state["tool_history"], record],
                      tool_call_count=state["tool_call_count"] + 1, plan=plan,
                      status="TOOL_EXECUTED" if result["success"] else "TOOL_ERROR")
        if not result["success"]:
            update["errors"] = [*state["errors"], {"node": "execute_tool", "code": record["error_code"],
                                 "message": "Read tool failed after the applicable runtime policy."}]
        return update
    return execute_tool
