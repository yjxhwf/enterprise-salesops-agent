"""Run-local counters, centralized transport retry, and read-only execution boundaries."""
from copy import deepcopy
from uuid import uuid4
from queue import Queue, Empty
from threading import Thread
import time
from backend.app.observability.collector import observe
from backend.app.observability.integration import runtime_event, llm_start, llm_end, tool_start, tool_end, node_finished

from backend.app.agent.guards.context import active_runtime, RuntimeStop
from backend.app.agent.guards.limits import RuntimeLimits, StopReason
from backend.app.agent.guards.loops import action_signature, progress_fingerprint
from backend.app.agent.guards.retry import backoff, transient_code


def runtime_defaults():
    return dict(agent_step_count=0, retry_count=0, repeated_action_counts={}, stagnant_cycle_count=0,
                stop_reason=None, runtime_events=[], last_action_signature=None,
                last_progress_fingerprint=None,
                token_usage=dict(input_tokens=0, output_tokens=0, total_tokens=0,
                                 usage_available=False, reported_responses=0, missing_responses=0))


def soft_call(function, timeout):
    """Return at the deadline; a daemon READ worker may still finish afterward."""
    queue = Queue(maxsize=1)
    def work():
        try:
            queue.put((True, function()))
        except BaseException as error:
            queue.put((False, error))
    Thread(target=work, daemon=True, name="salesops-read-tool").start()
    try:
        success, result = queue.get(timeout=timeout)
    except Empty:
        raise RuntimeStop(StopReason.TOOL_TIMEOUT) from None
    if not success:
        raise result
    return result


class Runtime:
    def __init__(self, state, node, limits, sleeper):
        self.state, self.node, self.limits, self.sleeper = state, node, limits, sleeper
        self.prepared_manifest = self.current_manifest = None

    def event(self, event_type, **fields):
        self.state["runtime_events"].append(dict(event_type=event_type, node=self.node, **fields))
        runtime_event(self.node, event_type, fields)

    def stop(self, reason):
        raise RuntimeStop(reason)

    def llm_budget(self):
        if self.state["llm_call_count"] >= self.limits.max_llm_calls:
            self.stop(StopReason.MAX_LLM_CALLS_EXCEEDED)
        if self.state["token_usage"]["total_tokens"] >= self.limits.soft_token_budget:
            self.stop(StopReason.TOKEN_BUDGET_EXCEEDED)

    def usage(self, response, attempt=1):
        usage = getattr(response, "usage_metadata", None)
        target = self.state["token_usage"]
        keys = ("input_tokens", "output_tokens", "total_tokens")
        reliable = isinstance(usage, dict) and all(type(usage.get(k)) is int and usage[k] >= 0 for k in keys)
        reliable = reliable and usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]
        if reliable:
            for key in keys:
                target[key] += usage[key]
            target["reported_responses"] += 1
            target["usage_available"] = target["missing_responses"] == 0
        else:
            target["usage_available"] = False
            target["missing_responses"] += 1

        self.event("LLM_USAGE", attempt=attempt, context_id=(self.current_manifest or {}).get("context_id"), usage_available=bool(reliable),
                   **({k: usage[k] for k in keys} if reliable else {}))

    def llm(self, function):
        retries = 0
        while True:
            self.llm_budget()
            if retries:
                self.state["retry_count"] += 1
            self.state["llm_call_count"] += 1
            if self.prepared_manifest is not None:
                self.current_manifest = {**deepcopy(self.prepared_manifest), "context_id": str(uuid4()), "attempt": retries+1}
                self.state.setdefault("context_manifests", []).append(deepcopy(self.current_manifest))
                self.event("CONTEXT_MANIFEST", **{k:v for k,v in self.current_manifest.items() if k != "node"})
            started = time.perf_counter()
            llm_start(self, retries + 1)
            try:
                response = function()
            except Exception as error:
                llm_end(self, retries + 1, started, error=error)
                code = transient_code(error)
                if code is None:
                    raise
                self.event("LLM_FAILURE", operation=self.node, attempt=retries + 1, error_code=code, retryable=True,
                           context_id=(self.current_manifest or {}).get("context_id"))
                self.llm_budget()
                if retries >= self.limits.max_retries_per_llm_operation:
                    self.stop(StopReason.MODEL_RETRY_EXHAUSTED)
                if self.state["retry_count"] >= self.limits.max_total_retries:
                    self.stop(StopReason.MAX_RETRIES_EXCEEDED)
                retries += 1
                delay = backoff(retries)
                self.event("LLM_RETRY", operation=self.node, attempt=retries + 1,
                           error_code=code, retryable=True, delay_ms=int(delay * 1000))
                self.sleeper(delay)
                continue
            llm_end(self, retries + 1, started, response=response)
            self.usage(response, retries + 1)
            return response

    def tool(self, registry, name, arguments):
        metadata = registry.get_tool(name)
        signature = action_signature(name, arguments)
        for attempt in (1, 2):
            if self.state["tool_call_count"] >= self.limits.max_tool_calls:
                self.stop(StopReason.MAX_TOOL_CALLS_EXCEEDED)
            if self.state["repeated_action_counts"].get(signature, 0) >= self.limits.max_repeated_tool_args:
                self.stop(StopReason.REPEATED_ACTION_LOOP)
            if attempt == 2:
                if self.state["retry_count"] >= self.limits.max_total_retries:
                    self.stop(StopReason.MAX_RETRIES_EXCEEDED)
                self.state["retry_count"] += 1
                self.event("TOOL_RETRY", operation=name, attempt=2, error_code="DATABASE_ERROR", retryable=True, delay_ms=0)
            self.state["tool_call_count"] += 1
            self.state["repeated_action_counts"][signature] = self.state["repeated_action_counts"].get(signature, 0) + 1
            self.state["last_action_signature"] = signature
            call_id, started = str(uuid4()), time.perf_counter()
            tool_start(self.node, name, arguments, metadata.permission_level, call_id, attempt)
            try:
                result = soft_call(lambda: registry.invoke(name, deepcopy(arguments)), metadata.timeout_seconds)
            except Exception as error:
                tool_end(self.node, name, metadata.permission_level, call_id, started, error=error)
                raise
            tool_end(self.node, name, metadata.permission_level, call_id, started, result=result)
            if result.success or attempt == 2 or not (
                metadata.permission_level == "READ" and result.error and
                result.error.retryable and result.error.code == "DATABASE_ERROR"
            ):
                return result
            self.event("TOOL_FAILURE", operation=name, attempt=attempt, error_code="DATABASE_ERROR", retryable=True)


class GuardedGateway:
    def __init__(self, gateway):
        self.gateway = gateway

    def __getattr__(self, name):
        operation = getattr(self.gateway, name)
        def call(*args, **kwargs):
            runtime = active_runtime.get()
            # ProductionGateway counts actual client.invoke, after configuration succeeds.
            if getattr(self.gateway, "runtime_transport_managed", False):
                return operation(*args, **kwargs)
            from backend.app.agent.guards.context import prepare_gateway_context
            prepare_gateway_context(name, args)
            return runtime.llm(lambda: operation(*args, **kwargs))
        return call


class GuardedRegistry:
    def __init__(self, registry):
        self.registry = registry

    def invoke(self, name, arguments):
        return active_runtime.get().tool(self.registry, name, arguments)


def guarded_node(name, function, *, limits=None, sleeper=time.sleep):
    limits = limits or RuntimeLimits()
    def run(state):
        working = deepcopy(state)
        for key, value in runtime_defaults().items():
            working.setdefault(key, value)
        runtime = Runtime(working, name, limits, sleeper)
        token = active_runtime.set(runtime)
        started = time.perf_counter()
        node_error = None
        observe("emit", "NODE_STARTED", node=name, status="STARTED")
        try:
            if working["status"] == "RUNTIME_STOPPED":
                return working
            if working["agent_step_count"] >= limits.max_agent_steps:
                runtime.stop(StopReason.MAX_AGENT_STEPS_EXCEEDED)
            working["agent_step_count"] += 1
            update = function(working)
            # Counters are owned here, not by the frozen business nodes.
            update.pop("llm_call_count", None)
            update.pop("tool_call_count", None)
            working.update(update)
            if name in {"review_progress", "synthesize", "propose_action"}:
                from backend.app.agent.context.evidence import EvidenceLedger
                working["evidence_usage"] = EvidenceLedger(working).usage()
            if name == "review_progress" and working["status"] == "INVESTIGATING":
                fingerprint = progress_fingerprint(working)
                working["stagnant_cycle_count"] = working["stagnant_cycle_count"] + 1 if fingerprint == working["last_progress_fingerprint"] else 0
                working["last_progress_fingerprint"] = fingerprint
                if working["stagnant_cycle_count"] >= limits.max_stagnant_cycles:
                    runtime.stop(StopReason.STAGNANT_LOOP)
            terminal = {"DRAFT_READY": StopReason.COMPLETED, "NEEDS_CLARIFICATION": StopReason.NEEDS_CLARIFICATION,
                        "UNSUPPORTED": StopReason.UNSUPPORTED, "ERROR": StopReason.MODEL_ERROR, "TOOL_ERROR": StopReason.TOOL_ERROR}
            if working["status"] in terminal:
                working["stop_reason"] = terminal[working["status"]].value
        except RuntimeStop as error:
            runtime.event("LOOP_DETECTED" if error.reason in {StopReason.REPEATED_ACTION_LOOP, StopReason.STAGNANT_LOOP}
                          else "TOOL_TIMEOUT" if error.reason == StopReason.TOOL_TIMEOUT else "GUARD_STOP",
                          stop_reason=error.reason.value)
            working.update(status="RUNTIME_STOPPED", stop_reason=error.reason.value, analysis_draft=None)
        except BaseException as error:
            node_error = error
            raise
        finally:
            node_finished(name, state, working, started, error=node_error)
            active_runtime.reset(token)
        return working
    return run
