"""Explicit SQLite lifecycle; importing this module never opens or resets a DB."""

from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.config import get_settings

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Base(DeclarativeBase):
    pass


def make_engine(database_url: str | None = None) -> Engine:
    url = make_url(database_url if database_url is not None else get_settings().DATABASE_URL)
    if url.drivername not in {"sqlite", "sqlite+pysqlite"} or url.query:
        raise ValueError("Only local SQLite URLs without query parameters are supported")
    memory = url.database in {None, "", ":memory:"}
    if not memory:
        path = Path(url.database)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        url = url.set(database=str(path))
    options = {"poolclass": StaticPool} if memory else {}
    engine = create_engine(url, connect_args={"check_same_thread": False}, **options)

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(connection, _record):
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine


def create_tables(engine: Engine) -> None:
    from backend.app.data import models  # noqa: F401: register all tables

    Base.metadata.create_all(engine)


def drop_tables(engine: Engine) -> None:
    """Explicit helper for caller-owned test databases; never called on import."""
    from backend.app.data import models  # noqa: F401

    Base.metadata.drop_all(engine)


def session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)
