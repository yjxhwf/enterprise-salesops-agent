from datetime import datetime
from types import SimpleNamespace

import pytest
from sqlalchemy import select, text

from backend.app.data import seed as seed_module
from backend.app.data.database import create_tables, drop_tables, make_engine, session_factory
from backend.app.data.generators import generate_dataset
from backend.app.data.models import AgentRun
from backend.app.data.seed import ExistingDataError, check_cli_database_url, seed_database
from backend.app.data.validation import DatasetValidationError, ValidationReport, dataset_hashes


def hashes(engine):
    with session_factory(engine)() as session:
        return dataset_hashes(session)


def test_seed_protection_reset_and_determinism(tmp_path):
    engines = [make_engine(f"sqlite:///{(tmp_path / name).as_posix()}") for name in ("one.db", "two.db")]
    try:
        first = seed_database(engines[0])
        before = hashes(engines[0])
        with pytest.raises(ExistingDataError, match="Use --reset"):
            seed_database(engines[0])
        assert hashes(engines[0]) == before
        second = seed_database(engines[1])
        assert hashes(engines[1]) == before
        assert first.metrics == second.metrics and first.checks == second.checks
        reset = seed_database(engines[0], reset=True)
        assert hashes(engines[0]) == before and reset.metrics == first.metrics
        # Re-create from scratch, comparing every business/system table content.
        drop_tables(engines[1])
        engines[1].dispose()
        (tmp_path / "two.db").unlink()
        rebuilt = seed_database(engines[1])
        assert hashes(engines[1]) == before and rebuilt.metrics == first.metrics
    finally:
        for engine in engines:
            engine.dispose()


def test_seed_respects_independent_random_instance():
    import random

    random.seed(1)
    first = generate_dataset(123)
    random.seed(999)
    second = generate_dataset(123)
    third = generate_dataset(124)
    rows = lambda dataset: [(o.order_id, o.order_date, o.quantity, o.sale_price) for o in dataset.orders]
    assert rows(first) == rows(second)
    assert rows(first) != rows(third)


def test_reset_rolls_back_on_validation_failure(empty_engine, monkeypatch):
    seed_database(empty_engine)
    before = hashes(empty_engine)
    monkeypatch.setattr(seed_module, "validate_dataset", lambda *args: ValidationReport({}, {"forced_failure": False}))
    with pytest.raises(DatasetValidationError, match="forced_failure"):
        seed_database(empty_engine, reset=True)
    assert hashes(empty_engine) == before


def test_system_data_also_protected(empty_engine):
    create_tables(empty_engine)
    with session_factory(empty_engine).begin() as session:
        session.add(AgentRun(run_id="test-run", user_query="test fixture only", start_time=datetime(2026, 1, 1), status="TEST"))
    with pytest.raises(ExistingDataError, match="Use --reset"):
        seed_database(empty_engine)
    report = seed_database(empty_engine, reset=True)
    assert report.checks["system_tables_empty"]
    with session_factory(empty_engine)() as session:
        assert session.scalar(select(AgentRun)) is None


def test_refuses_unknown_database(empty_engine):
    with empty_engine.begin() as connection:
        connection.execute(text("CREATE TABLE unrelated (value TEXT)"))
        connection.execute(text("INSERT INTO unrelated VALUES ('keep')"))
    with pytest.raises(ExistingDataError, match="unknown"):
        seed_database(empty_engine, reset=True)
    with empty_engine.connect() as connection:
        assert connection.execute(text("SELECT value FROM unrelated")).scalar_one() == "keep"


@pytest.mark.parametrize("url", ["sqlite:///../outside.db", "sqlite:///:memory:", "sqlite:///./README.md", "sqlite:///./backend/storage/../../../outside.db", "sqlite:///./backend/storage/test.db?mode=rw", "postgresql://localhost/example"])
def test_cli_reset_path_guard(url):
    with pytest.raises(ValueError):
        check_cli_database_url(url)


def test_cli_exit_status_and_custom_seed(tmp_path, monkeypatch, capsys):
    # Redirect the CLI's project boundary into a temporary project, never dev storage.
    monkeypatch.setattr(seed_module, "PROJECT_ROOT", tmp_path)
    url = f"sqlite:///{(tmp_path / 'backend/storage/cli.db').as_posix()}"
    monkeypatch.setattr(seed_module, "get_settings", lambda: SimpleNamespace(DATABASE_URL=url))
    assert seed_module.main(["--seed", "20260917", "--reset"]) == 0
    assert "Ground Truth Validation: PASS" in capsys.readouterr().out
    assert seed_module.main([]) == 1
    assert "Use --reset" in capsys.readouterr().out
    monkeypatch.setattr(seed_module, "validate_dataset", lambda *args: ValidationReport({}, {"broken": False}))
    assert seed_module.main(["--reset"]) == 1
    assert "broken" in capsys.readouterr().out
