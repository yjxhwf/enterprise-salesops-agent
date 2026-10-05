from backend.app.agent.nodes.common import model_node
from backend.app.agent.schemas import GoalUnderstanding


def make_understand_node(gateway, *, strict=False):
    @model_node("understand_goal", strict=strict)
    def understand(state):
        goal = GoalUnderstanding.model_validate(gateway.understand_goal(state["user_query"]))
        status = ("UNSUPPORTED" if not goal.supported else
                  "NEEDS_CLARIFICATION" if goal.needs_clarification else "GOAL_UNDERSTOOD")
        return {"goal": goal.model_dump(mode="json"), "status": status}
    return understand
