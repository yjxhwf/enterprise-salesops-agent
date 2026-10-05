"""JSON-only trace records. These types are deliberately absent from AgentState."""
from dataclasses import dataclass, field


@dataclass
class TraceEvent:
    event_id: str
    run_id: str
    sequence: int
    timestamp: str
    event_type: str
    node: str | None
    duration_ms: float | None
    status: str | None
    safe_metadata: dict = field(default_factory=dict)


@dataclass
class RunTrace:
    run_id: str
    started_at: str
    finished_at: str | None = None
    duration_ms: float = 0
    final_status: str | None = None
    stop_reason: str | None = None
    events: list[TraceEvent] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    llm_totals: dict = field(default_factory=dict)
    tool_totals: dict = field(default_factory=dict)
    evidence_totals: dict = field(default_factory=dict)
    approval_summary: dict = field(default_factory=dict)
    error_summary: dict = field(default_factory=dict)
    trace_truncated: bool = False
    dropped_events: int = 0
    observability_errors: int = 0
