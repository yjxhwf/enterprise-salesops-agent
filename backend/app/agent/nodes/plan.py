from backend.app.agent.nodes.common import model_node
from backend.app.agent.schemas import Plan


def make_plan_node(gateway, catalog, *, strict=False):
    @model_node("plan", strict=strict)
    def plan(state):
        result = Plan.model_validate(gateway.create_plan(state["goal"], catalog.capabilities()))
        return {"plan": result.model_dump(mode="json"), "current_step_index": 0, "status": "PLANNED"}
    return plan
