"""Scripted test dependency only. These responses do not measure live model quality."""
from copy import deepcopy


class FakeGateway:
    def __init__(self, goal, plan=None, selection=None):
        self.outputs = {"understand_goal": goal, "plan": plan, "select_tool": selection}
        self.calls = []

    def _respond(self, name, **inputs):
        self.calls.append((name, deepcopy(inputs)))
        result = self.outputs[name]
        if isinstance(result, Exception):
            raise result
        return deepcopy(result)

    def understand_goal(self, query):
        return self._respond("understand_goal", query=query)

    def create_plan(self, goal, capabilities):
        return self._respond("plan", goal=goal, capabilities=capabilities)

    def select_tool(self, goal, step, tools):
        return self._respond("select_tool", goal=goal, step=step, tools=tools)


QUERY = "分析2026年8月为什么销售额增长但利润下降，找出最主要的三个原因，并给出具体行动建议。"
DATES = {"start_date": "2026-08-01", "end_date": "2026-08-31"}


def goal(**changes):
    return {"supported": True, "task_type": "ROOT_CAUSE_ANALYSIS",
            "normalized_goal": "解释目标期间收入增长而利润下降的主要原因，并提出行动建议。",
            **DATES, "requested_outputs": ["three main root causes", "actionable recommendations"],
            "needs_clarification": False, "clarification_question": None, **changes}


def plan(first="确认整体经营指标及与前期变化"):
    return {"steps": [
        {"step_id": "s1", "objective": first, "expected_output": "目标期间与前期的经营指标", "status": "PENDING"},
        {"step_id": "s2", "objective": "定位利润变化明显的业务维度", "expected_output": "需要进一步调查的维度", "status": "PENDING"},
        {"step_id": "s3", "objective": "按未来调查发现下钻客户和订单", "expected_output": "后续证据需求", "status": "PENDING"},
    ]}


def selection(name="get_sales_overview", **arguments):
    return {"tool_name": name, "arguments": {**DATES, **arguments}}


class FakeInvestigationGateway(FakeGateway):
    """Explicit scripted queues; callable fixtures can inspect actual read results."""
    def __init__(self, goal, plan, selections, reviews, draft):
        super().__init__(goal, plan)
        self.selections = iter(selections)
        self.reviews = iter(reviews)
        self.draft = draft

    def _scripted(self, name, output, **inputs):
        self.calls.append((name, deepcopy(inputs)))
        if isinstance(output, Exception):
            raise output
        return deepcopy(output(**inputs) if callable(output) else output)

    def select_investigation_tool(self, goal, plan, step, observations, tools):
        return self._scripted("select_tool", next(self.selections), goal=goal, plan=plan,
                              step=step, observations=observations, tools=tools)

    def review_progress(self, goal, plan, step, latest_result, observations, history, capabilities):
        return self._scripted("review_progress", next(self.reviews), goal=goal, plan=plan, step=step,
                              latest_result=latest_result, observations=observations,
                              history=history, capabilities=capabilities)

    def synthesize(self, goal, observations, history):
        return self._scripted("synthesize", self.draft, goal=goal, observations=observations, history=history)
