from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import event, text, update
from sqlalchemy.exc import OperationalError

from backend.app.data.database import session_factory
from backend.app.data.models import Order
from backend.app.data.validation import dataset_hashes
from backend.app.tools import base
from backend.app.tools.base import read_session
from backend.app.tools.registry import build_default_registry

DATES = {"start_date": "2026-08-01", "end_date": "2026-08-31"}
NAMES = {"get_sales_overview", "analyze_region_performance", "analyze_product_performance", "analyze_customer_performance", "query_orders"}


def test_static_registry_metadata(seeded_engine):
    registry = build_default_registry(session_factory(seeded_engine))
    metadata = registry.list_tools(permission="READ")
    assert len(registry.list_tools()) == 8
    assert {item.name for item in metadata} == NAMES | {"search_sales_policy"}
    assert len(metadata) == 6
    for item in metadata:
        assert item.permission_level == "READ" and item.timeout_seconds == 8
        assert item.input_schema["additionalProperties"] is False
        assert ({"query"} if item.name == "search_sales_policy" else {"start_date", "end_date"}) <= set(item.input_schema["required"])
        assert "session_factory" not in item.input_schema["properties"]
        assert "database_url" not in item.input_schema["properties"]
        assert len(item.description) > 80
        assert registry.get_tool(item.name) == item
    metadata[0].description = "caller edit"
    assert registry.list_tools()[0].description != "caller edit"
    assert registry.get_tool("delete_orders") is None


@pytest.mark.parametrize("name", ["delete_orders", "__import__", "os.system", "", None, {}])
def test_unknown_tool_never_calls_factory(name):
    def forbidden():
        raise AssertionError("Unknown tools must not access the DB")
    result = build_default_registry(forbidden).invoke(name, {})
    assert result.error.code == "TOOL_NOT_FOUND" and not result.error.retryable


@pytest.mark.parametrize("arguments", [[], None, "not a dict", {**DATES, "extra": "value"}])
def test_invalid_argument_shape(arguments):
    result = build_default_registry(lambda: None).invoke("get_sales_overview", arguments)
    assert result.error.code == "INVALID_ARGUMENT"


def test_database_failure_is_safe_and_not_retried():
    calls = []
    def broken():
        calls.append(1)
        raise OperationalError("SELECT private", {}, Exception("test-only-private-marker"))
    result = build_default_registry(broken).invoke("get_sales_overview", DATES)
    assert result.error.code == "DATABASE_ERROR" and result.error.retryable
    assert len(calls) == 1
    assert "private" not in result.model_dump_json()


def test_database_execution_failure(empty_engine):
    result = build_default_registry(session_factory(empty_engine)).invoke("get_sales_overview", DATES)
    assert result.error.code == "DATABASE_ERROR" and result.error.retryable


def test_programming_bugs_remain_visible():
    def buggy():
        raise RuntimeError("test programming defect")
    with pytest.raises(RuntimeError, match="programming defect"):
        build_default_registry(buggy, raise_unexpected_errors=True).invoke("get_sales_overview", DATES)
    safe = build_default_registry(buggy).invoke("get_sales_overview", DATES)
    assert safe.error.code == "INTERNAL_ERROR" and not safe.error.retryable
    assert "test programming defect" not in safe.model_dump_json()


def test_all_tools_are_readonly_and_no_n_plus_one(seeded_engine):
    factory = session_factory(seeded_engine)
    with factory() as session:
        before = dataset_hashes(session)
    statements = []
    def capture(_conn, _cursor, statement, _parameters, _context, _many):
        statements.append(statement)
    event.listen(seeded_engine, "before_cursor_execute", capture)
    try:
        registry = build_default_registry(factory)
        for name in NAMES:
            assert registry.invoke(name, DATES).success
    finally:
        event.remove(seeded_engine, "before_cursor_execute", capture)
    selects = [sql for sql in statements if sql.lstrip().upper().startswith("SELECT")]
    assert len(selects) == 10  # Two fixed SELECTs per unfiltered tool, no per-row SQL.
    assert sum("GROUP BY" in sql for sql in selects) == 6
    assert not any(sql.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE", "CREATE", "DROP")) for sql in statements)
    with factory() as session:
        assert dataset_hashes(session) == before  # All seven tables, including empty system tables.
        assert session.execute(text("PRAGMA query_only")).scalar_one() == 0


def test_injected_session_rejects_writes_and_restores_pool(seeded_engine):
    factory = session_factory(seeded_engine)
    with pytest.raises(OperationalError, match="readonly"):
        with read_session(factory) as session:
            session.execute(update(Order).values(quantity=999))
    with factory() as session:
        assert session.execute(text("PRAGMA query_only")).scalar_one() == 0


def test_default_file_connection_is_read_only(seeded_engine, monkeypatch):
    monkeypatch.setattr(base, "get_settings", lambda: SimpleNamespace(DATABASE_URL=str(seeded_engine.url)))
    with build_default_registry() as registry:
        assert registry.invoke("get_sales_overview", DATES).success
        with registry._factory() as session:
            session.execute(text("PRAGMA query_only=OFF"))
            with pytest.raises(OperationalError, match="readonly"):
                session.execute(update(Order).values(quantity=999))


def test_missing_default_database_not_created(tmp_path, monkeypatch):
    missing = tmp_path / "missing.db"
    monkeypatch.setattr(base, "get_settings", lambda: SimpleNamespace(DATABASE_URL=f"sqlite:///{missing.as_posix()}"))
    with build_default_registry() as registry:
        result = registry.invoke("get_sales_overview", DATES)
        assert result.error.code == "DATABASE_ERROR"
    assert not missing.exists()


def test_runtime_isolation_and_no_dynamic_execution():
    root = Path(__file__).resolve().parents[1] / "backend/app"
    for directory in (root / "tools", root / "data/repositories"):
        for path in directory.glob("*.py"):
            source = path.read_text(encoding="utf-8")
            for forbidden in ("ground_truth", "SKU-A12", "C102", "C207", "data.validation", "eval(", "exec(", "importlib", "__import__"):
                assert forbidden not in source, (path, forbidden)
