"""Internal execution helpers. No dependency here is a model-visible argument."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import sqlite3

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from backend.app.config import get_settings
from backend.app.data.database import PROJECT_ROOT
from backend.app.tools.schemas import Contract, DateInput

SessionFactory = Callable[[], Session]


def readonly_engine() -> Engine:
    """Independent read-only connections: missing DBs are never silently created."""
    url = make_url(get_settings().DATABASE_URL)
    if url.drivername not in {"sqlite", "sqlite+pysqlite"} or url.query or not url.database or url.database == ":memory:":
        raise ValueError("Default tools require a configured local SQLite file")
    path = Path(url.database)
    path = (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
    uri = path.as_uri() + "?mode=ro"
    engine = create_engine("sqlite+pysqlite://", creator=lambda: sqlite3.connect(uri, uri=True, check_same_thread=False))

    @event.listens_for(engine, "connect")
    def protect(connection, _record):
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA query_only=ON")

    return engine


def readonly_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@contextmanager
def read_session(factory: SessionFactory) -> Iterator[Session]:
    """Also protect injected SQLite sessions; restore pooled connection settings."""
    with factory() as session:
        with session.no_autoflush:
            connection = session.connection()
            if connection.dialect.name != "sqlite":
                raise ValueError("Task 02 supports SQLite sessions only")
            previous = connection.exec_driver_sql("PRAGMA query_only").scalar_one()
            connection.exec_driver_sql("PRAGMA query_only=ON")
            try:
                # Python's sqlite3 legacy mode does not BEGIN for SELECT. Start a
                # read transaction so current/previous/count rows share a snapshot.
                driver = connection.connection.driver_connection
                if not driver.in_transaction:
                    connection.exec_driver_sql("BEGIN")
                yield session
            finally:
                try:
                    if not connection.closed and not connection.invalidated:
                        connection.exec_driver_sql("PRAGMA query_only=ON" if previous else "PRAGMA query_only=OFF")
                finally:
                    session.rollback()


@dataclass
class ToolPayload:
    data: Contract
    summary: str
    evidence_ids: list[str]
    meta: dict = field(default_factory=dict)


def aggregate_evidence(name: str, arguments: DateInput) -> list[str]:
    canonical = json.dumps(arguments.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(canonical.encode()).hexdigest()[:20]
    return [f"aggregate:{name}:{arguments.start_date}:{arguments.end_date}:{fingerprint}"]


def comparison_dates(arguments: DateInput) -> tuple:
    previous = arguments.comparison_period()
    return arguments.start_date, arguments.end_date, previous.start_date, previous.end_date


def period_meta(arguments: DateInput, available: bool) -> dict:
    return {"current_period": arguments.current_period(), "comparison_period": arguments.comparison_period(),
            "comparison_available": available}
