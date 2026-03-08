"""
Database engine and session management.

Uses SQLite via SQLAlchemy Core + ORM.

The database file is created at the project root (news2alpha.db) so it
is easy to inspect with any SQLite browser during development.

Usage in services:
    from app.db.database import get_session

    with get_session() as session:
        session.add(some_model)
        session.commit()
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Database location — project root / news2alpha.db
# ---------------------------------------------------------------------------
_DB_PATH = Path(__file__).parent.parent.parent / "news2alpha.db"
_DB_URL = f"sqlite:///{_DB_PATH}"

# ---------------------------------------------------------------------------
# Engine — check_same_thread=False is required for SQLite when FastAPI
# handles requests across threads (even in sync mode).
# ---------------------------------------------------------------------------
engine = create_engine(
    _DB_URL,
    connect_args={"check_same_thread": False},
    echo=False,  # Set to True to log every SQL statement during debugging
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


# ---------------------------------------------------------------------------
# Declarative base — imported by app/db/models.py
# ---------------------------------------------------------------------------
class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Session context manager
# ---------------------------------------------------------------------------
@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Provide a transactional database session.

    Commits on clean exit, rolls back on any exception, always closes.

    Example:
        with get_session() as session:
            session.add(record)
            # commit happens automatically on __exit__
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Table initialisation — called once from main.py on startup
# ---------------------------------------------------------------------------
def init_db() -> None:
    """
    Create all tables defined in app/db/models.py if they do not exist.

    Safe to call on every startup — SQLAlchemy uses CREATE TABLE IF NOT EXISTS.
    """
    # Import models here to ensure they are registered on Base.metadata
    import app.db.models  # noqa: F401
    import app.db.cluster_models  # noqa: F401
    import app.db.cluster_analysis_models  # noqa: F401
    import app.db.market_context_models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    logger.info("Database initialised at %s", _DB_PATH)
