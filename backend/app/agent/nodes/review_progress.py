from copy import deepcopy
from backend.app.agent.guards.context import build_llm_context, visible_ids
from backend.app.agent.policy_evidence import policy_catalog

from backend.app.agent.grounding import invalid_output
from backend.app.agent.evidence import extract_evidence, validate_citations
from backend.app.agent.semantics import validate_review
from backend.app.agent.facts import canonical_facts
from backend.app.agent.nodes.common import model_node
from backend.app.agent.schemas import InvestigationObservation, ProgressReview


def make_review_node(gateway, catalog, *, strict=False):
    @model_node("review_progress", strict=strict)
    def review_progress(state):
        step = state["plan"]["steps"][state["current_step_index"]]
        review = ProgressReview.model_validate(gateway.review_progress(
            state["goal"], state["plan"], step, state["latest_tool_result"],
            state["observations"], state["tool_history"], catalog.capabilities()))
        policies = policy_catalog(state["tool_history"])
        business_ids = [eid for eid in review.selected_evidence_ids if not eid.startswith("POLICY::")]
        policy_ids = [eid for eid in review.selected_evidence_ids if eid.startswith("POLICY::")]
        if business_ids:
            business_ids = validate_citations(business_ids, state["tool_history"])
        if any(eid not in policies for eid in policy_ids):
            invalid_output()
        from backend.app.agent.guards.context import require_visible
        require_visible("review_progress", review.selected_evidence_ids)
        evidence = extract_evidence(state["tool_history"])
        validate_review(review, evidence)
        if not state["latest_tool_result"]["success"] or state["tool_history"][-1]["plan_step_id"] != step["step_id"]:
            invalid_output()
        finished = {**deepcopy(step), "status": "COMPLETED"}
        completed = [*deepcopy(state["completed_steps"]), finished]
        next_plan = None
        if review.updated_plan:
            used = {item["step_id"] for item in completed}
            remaining = []
            for index, objective in enumerate(review.updated_plan.steps, 1):
                step_id = f"replan-{len(state['observations']) + 1}-{index}"
                while step_id in used:
                    step_id += "-next"
                used.add(step_id)
                remaining.append({**objective.model_dump(mode="json"), "step_id": step_id, "status": "PENDING"})
            next_plan = {"steps": [*completed, *remaining]}
        observation = InvestigationObservation(
            observation_id=f"observation-{len(state['observations']) + 1}",
            completed_step_id=step["step_id"],
            source_tool=state["latest_tool_result"]["tool_name"],
            facts=canonical_facts(evidence, business_ids), policy_evidence=[policies[eid] for eid in policy_ids],
            evidence_ids=review.selected_evidence_ids,
            decision=review.decision, remaining_evidence_gap=review.remaining_evidence_gap,
            next_objective=review.next_objective, plan_before=state["plan"], plan_after=next_plan).model_dump(mode="json")
        update = {"observations": [*state["observations"], observation], "selected_tool": None,
                  "completed_steps": completed}
        if next_plan:
            # Replace remaining objectives based on evidence; never index++.
            update.update(plan=next_plan, current_step_index=len(completed), status="INVESTIGATING")
        else:
            plan = deepcopy(state["plan"])
            plan["steps"][state["current_step_index"]] = finished
            for remaining in plan["steps"]:
                if remaining["status"] == "PENDING":
                    remaining["status"] = "SKIPPED"
                    remaining["skip_reason"] = "SUFFICIENT_EVIDENCE"
            observation["plan_after"] = deepcopy(plan)
            update.update(plan=plan, status="INVESTIGATION_COMPLETE")
        return update
    return review_progress
