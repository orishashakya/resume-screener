"""
Database connection setup using SQLAlchemy.
Uses SQLite for simplicity - swap the DATABASE_URL to Postgres later if needed,
the rest of the code doesn't change.
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite:///./screener.db"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}  # needed for SQLite + FastAPI
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency - provides a DB session per request, closes it after."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()