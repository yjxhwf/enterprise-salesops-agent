"""Explicit production-only CLI; never substitutes a fake model."""
import argparse
import json

from backend.app.agent.graph import build_agent_graph, build_investigation_graph
from backend.app.agent.state import initial_state
from backend.app.llm.provider import ProductionGateway
from backend.app.tools.registry import build_default_registry


def main():
    parser = argparse.ArgumentParser(description="Read-only business investigation with a production model")
    parser.add_argument("--query", required=True)
    parser.add_argument("--planning-only", action="store_true", help="Stop at the Task 03 tool proposal")
    parser.add_argument("--recursion-limit", type=int, default=50,
                        help="LangGraph development harness safety, above the application runtime step guard")
    parser.add_argument("--request-timeout-seconds", type=float, default=30,
                        help="SDK single-request timeout; runtime controls bounded retries")
    args = parser.parse_args()
    if args.request_timeout_seconds <= 0:
        parser.error("request timeout must be positive")
    gateway = ProductionGateway(request_timeout_seconds=args.request_timeout_seconds)
    if args.planning_only:
        result = build_agent_graph(gateway).invoke(initial_state(args.query))
    else:
        with build_default_registry() as registry:
            result = build_investigation_graph(gateway, registry).invoke(
                initial_state(args.query), config={"recursion_limit": args.recursion_limit})
    print(json.dumps({key: result[key] for key in ("run_id", "goal", "plan", "selected_tool", "status",
                                                  "tool_call_count", "llm_call_count", "analysis_draft")},
                     ensure_ascii=False, indent=2))
    for error in result["errors"]:
        print(error["code"] + ": " + error["message"])
    return 1 if result["status"] in {"ERROR", "TOOL_ERROR", "RUNTIME_STOPPED"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
