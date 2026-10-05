"""Two explicit INSERTs on a separate session, called only inside ApprovalStore's gate."""
from contextlib import contextmanager
from time import perf_counter
from backend.app.data.database import make_engine, session_factory
from backend.app.data.models import CRMTask, BusinessAlert, Customer, Product, REGIONS
from backend.app.tools.schemas import ToolResult, ToolError, ToolMeta
from backend.app.tools.write_schemas import WriteResultData
from backend.app.actions.schemas import ActionError
from backend.app.actions.validation import validate_tool


def lazy_write_sessions(database_url):
    @contextmanager
    def open_session():
        # Engine/session creation happens only after the approval gate.
        engine = make_engine(database_url)
        try:
            with session_factory(engine)() as session:
                yield session
        finally:
            engine.dispose()
    return open_session


def execute_write(pending, registry, write_session_factory, clock):
    started = perf_counter()
    try:
        args = validate_tool(registry, pending.tool_name, pending.arguments)
        if args != pending.arguments:
            raise ActionError("ACTION_TAMPERED")
        model = CRMTask if pending.tool_name == "create_crm_task" else BusinessAlert
        identity = ("crm-" if model is CRMTask else "alt-")+pending.action_fingerprint[:36]
        with write_session_factory() as session:
            with session.begin():
                if model is CRMTask:
                    if session.get(Customer, args["customer_id"]) is None:
                        raise ActionError("ENTITY_NOT_FOUND")
                else:
                    typ, target = args["target_type"], args["target_id"]
                    if typ in {"CUSTOMER", "PRODUCT"} and session.get(Customer if typ == "CUSTOMER" else Product, target) is None:
                        raise ActionError("ENTITY_NOT_FOUND")
                    if typ == "REGION" and target not in REGIONS:
                        raise ActionError("ENTITY_NOT_FOUND")
                row = session.get(model, identity)
                if row is None:
                    row = model(**args, **{"task_id" if model is CRMTask else "alert_id": identity},
                                status="OPEN", created_at=clock().replace(tzinfo=None))
                    session.add(row)
                    session.flush()
                elif any(getattr(row, k) != value for k, value in args.items()):
                    raise ActionError("ACTION_TAMPERED")
                data = WriteResultData(action_id=pending.action_id, object_id=identity,
                    object_type="CRM_TASK" if model is CRMTask else "BUSINESS_ALERT", created_at=row.created_at)
        return ToolResult(success=True, tool_name=pending.tool_name, data=data, summary="Approved local action recorded.",
            evidence_ids=["ACTION::"+identity], error=None,
            meta=ToolMeta(latency_ms=(perf_counter()-started)*1000, returned_count=1))
    except Exception:
        return ToolResult(success=False, tool_name=pending.tool_name, data=None, summary=None, evidence_ids=[],
            error=ToolError(code="WRITE_EXECUTION_FAILED", message="Controlled write failed; no automatic retry.", retryable=False),
            meta=ToolMeta(latency_ms=(perf_counter()-started)*1000))
