"""Explicit deterministic dataset CLI. No directory/file deletion is performed."""

import argparse
from pathlib import Path

from sqlalchemy import Engine, delete, inspect, select
from sqlalchemy.engine import make_url

from backend.app.config import get_settings
from backend.app.data.database import Base, PROJECT_ROOT, create_tables, make_engine, session_factory
from backend.app.data.generators import DATASET_VERSION, DEFAULT_SEED, generate_dataset
from backend.app.data.ground_truth import load_ground_truth
from backend.app.data.validation import DatasetValidationError, ValidationReport, validate_dataset


class ExistingDataError(ValueError):
    pass


def check_cli_database_url(database_url: str) -> None:
    """The CLI may reset only the explicitly configured project storage DB."""
    url = make_url(database_url)
    if url.drivername not in {"sqlite", "sqlite+pysqlite"} or url.query or not url.database:
        raise ValueError("Seed CLI requires a local SQLite file in backend/storage")
    storage = (PROJECT_ROOT / "backend" / "storage").resolve()
    path = Path(url.database)
    path = (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
    if not path.is_relative_to(storage) or path.suffix not in {".db", ".sqlite", ".sqlite3"}:
        raise ValueError("Seed CLI only operates on the configured file within backend/storage")


def seed_database(engine: Engine, *, seed: int = DEFAULT_SEED, reset: bool = False) -> ValidationReport:
    """Seed a caller-owned engine (tests pass temporary DBs).

    Reset deletes only known table rows in one transaction; validation failure
    rolls back both deletion and insertion, preserving any previous dataset.
    """
    truth = load_ground_truth()
    if truth.dataset_version != DATASET_VERSION:
        raise DatasetValidationError("Generator and ground-truth dataset versions differ")
    schema = inspect(engine)
    existing = set(schema.get_table_names())
    unknown = existing - set(Base.metadata.tables)
    if unknown or schema.get_view_names():
        raise ExistingDataError("Database has unknown tables/views; refusing to seed or reset")
    for name in existing:
        if {c["name"] for c in schema.get_columns(name)} != set(Base.metadata.tables[name].columns.keys()):
            raise ExistingDataError("Existing schema differs; refusing to overwrite")
    create_tables(engine)
    dataset = generate_dataset(seed)
    with session_factory(engine).begin() as session:
        populated = any(session.execute(select(table).limit(1)).first() is not None for table in Base.metadata.sorted_tables)
        if populated and not reset:
            raise ExistingDataError("Existing database contains seeded data. Use --reset to rebuild.")
        if reset:
            for table in reversed(Base.metadata.sorted_tables):
                session.execute(delete(table))
        session.add_all(dataset.sales_reps)
        session.flush()
        session.add_all(dataset.customers + dataset.products)
        session.flush()
        session.add_all(dataset.orders)
        session.flush()
        report = validate_dataset(session, truth)
        report.require_valid()
    # Re-read the committed database, rather than trusting in-memory generation.
    with session_factory(engine)() as session:
        persisted_report = validate_dataset(session, truth)
        persisted_report.require_valid()
    return persisted_report


def print_summary(report: ValidationReport, seed: int) -> None:
    m = report.metrics
    print(f"Dataset version: {DATASET_VERSION}\nDataset seed: {seed}")
    for key, count in m["counts"].items():
        print(f"{key}: {count}")
    for month, values in m["monthly"].items():
        print(f"{month}: orders={values['orders']}, revenue={values['revenue']:.2f}, "
              f"profit={values['profit']:.2f}, margin={values['margin_pct']:.2f}%")
    print(f"Revenue Growth: {m['revenue_growth_pct']:.2f}%")
    print(f"Profit Growth: {m['profit_growth_pct']:.2f}%")
    print(f"Margin Change: {m['margin_change_pp']:.2f} percentage points")
    for name in ("east_china", "sku_a12", "c102", "c207", "b07_mix_shift", "c03_decline"):
        print(f"{name}: {'PASS' if report.checks[name] else 'FAIL'}")
    print(f"Missing approvals: {m['missing_approvals']}")
    print(f"Ground Truth Validation: {'PASS' if report.passed else 'FAIL'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed the fictional SalesOps SQLite dataset")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--reset", action="store_true", help="Replace known dataset rows transactionally")
    args = parser.parse_args(argv)
    engine = None
    try:
        database_url = get_settings().DATABASE_URL
        check_cli_database_url(database_url)
        engine = make_engine(database_url)
        report = seed_database(engine, seed=args.seed, reset=args.reset)
        print_summary(report, args.seed)
        return 0
    except ValueError as error:
        print(f"Seed failed: {error}")
        return 1
    finally:
        if engine is not None:
            engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
