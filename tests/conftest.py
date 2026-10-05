from collections.abc import Iterator

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from backend.app.data.database import make_engine, session_factory
from backend.app.data.seed import seed_database


@pytest.fixture
def empty_engine(tmp_path) -> Iterator[Engine]:
    engine = make_engine(f"sqlite:///{(tmp_path / 'empty.db').as_posix()}")
    yield engine
    engine.dispose()


@pytest.fixture(scope="module")
def seeded_engine(tmp_path_factory) -> Iterator[Engine]:
    path = tmp_path_factory.mktemp("salesops") / "seeded.db"
    engine = make_engine(f"sqlite:///{path.as_posix()}")
    seed_database(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def data_session(seeded_engine) -> Iterator[Session]:
    with session_factory(seeded_engine)() as session:
        yield session
        # Each corruption/constraint test rolls back; the development DB is never used.
        session.rollback()
